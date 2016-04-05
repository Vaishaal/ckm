package nodes.images

import breeze.linalg._
import breeze.numerics._
import nodes.learning.ZCAWhitener
import nodes.stats.Fastfood
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}
import workflow.Transformer
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister

/**
 * Convolves images with a bank of convolution filters. Convolution filters must be square.
 * Used for using the same label for all patches from an image.
 * TODO: Look into using Breeze's convolve
 *
 * @param filters Bank of convolution filters to apply - each filter is an array in row-major order.
 * @param imgWidth Width of images in pixels.
 * @param imgHeight Height of images in pixels.
 */
class CC(
    numInputFeatures: Int,
    numOutputFeatures: Int,
    seed: Int,
    bandwidth: Double,
    imgWidth: Int,
    imgHeight: Int,
    imgChannels: Int,
    whitener: Option[ZCAWhitener] = None,
    whitenerOffset: Double = 1e-12,
    poolSize: Int = 1,
    insanity: Boolean = false,
    fastfood: Boolean = false
    )
  extends Transformer[Image, Image] {

  val convSize = math.sqrt(numInputFeatures/imgChannels).toInt

  val resWidth = imgWidth - convSize + 1
  val resHeight = imgHeight - convSize + 1
  val outX = math.ceil((resWidth - (poolSize/2)).toDouble / poolSize).toInt
  val outY = math.ceil((resHeight- (poolSize/2)).toDouble / poolSize).toInt
  implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
  val gaussian = new Gaussian(0, 1)
  val uniform = new Uniform(0, 1)
  val convolutionsDouble = (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth).t
  val convolutions = convert(convolutionsDouble, Float)
  val phaseDouble = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
  val phase = convert(phaseDouble, Float)
  var patchMat = new DenseMatrix[Float](resWidth*resHeight, convSize*convSize*imgChannels)

  override def apply(in: RDD[Image]): RDD[Image] = {
    println(s"Convolve: ${resWidth}, ${resHeight}, ${numOutputFeatures}")
    println(s"Input: ${imgWidth}, ${imgHeight}, ${imgChannels}")
    println(s"First pixel ${in.take(1)(0).get(0,0,0)}")

    in.mapPartitions(CC.convolvePartitions(_, resWidth, resHeight, imgChannels, convSize,
      whitener, whitenerOffset, numInputFeatures, numOutputFeatures, seed, bandwidth, insanity, fastfood))
  }

  def apply(in: Image): Image = {
    CC.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, convSize, whitener, whitenerOffset, convolutions.data, phase, insanity, None, numOutputFeatures, numInputFeatures)
  }
}

object CC {
  /**
    * Given an array of filters, packs the filters into a DenseMatrix[Float] which has the following form:
    * for a row i, column c+y*numChannels+x*numChannels*yDim corresponds to the pixel value at (x,y,c) in image i of
    * the filters array.
    *
    * @param filters Array of filters.
    * @return DenseMatrix of filters, as described above.
    */
  def packFilters(filters: Array[Image]): DenseMatrix[Float] = {
    val (xDim, yDim, numChannels) = (filters(0).metadata.xDim, filters(0).metadata.yDim, filters(0).metadata.numChannels)
    val filterSize = xDim*yDim*numChannels
    val res = DenseMatrix.zeros[Float](filters.length, filterSize)

    var i,x,y,c = 0
    while(i < filters.length) {
      x = 0
      while(x < xDim) {
        y = 0
        while(y < yDim) {
          c = 0
          while (c < numChannels) {
            val rc = c + x*numChannels + y*numChannels*xDim
            res(i, rc) = filters(i).get(x,y,c).toFloat

            c+=1
          }
          y+=1
        }
        x+=1
      }
      i+=1
    }

    res
  }


  def convolve(img: Image,
      patchMat: DenseMatrix[Float],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      whitener: Option[ZCAWhitener],
      whitenerOffset: Double,
      convolutions: Array[Float],
      phase: DenseVector[Float],
      insanity: Boolean,
      fastfood: Option[Fastfood],
      out: Int,
      in: Int
      ): Image = {

    val imgMat = makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize,
      whitener)

    val whitenedImage: DenseMatrix[Float] =
    whitener match  {
      case None => {
        imgMat
      }
      case Some(whitener) => {
        convert(whitener(convert(imgMat, Double)) :+ whitenerOffset, Float)
      }
    }

    val patchNorms = convert(norm(convert(whitenedImage, Double) :+ whitenerOffset, Axis._1), Float)
    val normalizedPatches = whitenedImage(::, *) :/ patchNorms
    var convRes:DenseMatrix[Float] =
    fastfood.map { ff =>
      val ff_out = MatrixUtils.matrixToRowArray(normalizedPatches).map((x:DenseVector[Float]) => convert(ff(convert(x, Double)), Float))
      MatrixUtils.rowsToMatrix(ff_out)
    } getOrElse {
      val convRes = normalizedPatches * (new DenseMatrix(out, in, convolutions)).t
      convRes(*, ::) :+= phase
      cos.inPlace(convRes)
      if (insanity) {
        convRes(::,*) :*= patchNorms
      }
      convRes
    }

    val res = new RowMajorArrayVectorizedImage(
      convert(convRes, Double).toArray,
      ImageMetadata(resWidth, resHeight, out))
    res
  }
  
  /**
   * This function takes an image and generates a matrix of all of its patches. Patches are expected to have indexes
   * of the form: c + x*numChannels + y*numChannels*xDim
   *
   * @param img
   * @return
   */
  def makePatches(img: Image,
      patchMat: DenseMatrix[Float],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      whitener: Option[ZCAWhitener]
      ): DenseMatrix[Float] = {
    var x,y,chan,pox,poy,py,px = 0

    poy = 0
    while (poy < convSize) {
      pox = 0
      while (pox < convSize) {
        y = 0
        while (y < resHeight) {
          x = 0
          while (x < resWidth) {
            chan = 0
            while (chan < imgChannels) {
              px = chan + pox*imgChannels + poy*imgChannels*convSize
              py = x + y*resWidth

              patchMat(py, px) = (img.get(x+pox, y+poy, chan)).toFloat

              chan+=1
            }
            x+=1
          }
          y+=1
        }
        pox+=1
      }
      poy+=1
    }

    patchMat
  }

  def convolvePartitions(
      imgs: Iterator[Image],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      whitener: Option[ZCAWhitener],
      whitenerOffset: Double,
      numInputFeatures: Int,
      numOutputFeatures: Int,
      seed: Int,
      bandwidth: Double,
      insanity: Boolean,
      fastfood: Boolean
      ): Iterator[Image] = {

    var patchMat = new DenseMatrix[Float](resWidth*resHeight, convSize*convSize*imgChannels)
      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutionsDouble = 
    if (!fastfood) {
      (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth).data
    } else {
      (DenseVector.rand(numOutputFeatures, gaussian) :* bandwidth).data 
    }

    val phaseDouble = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val ff = 
      if (fastfood) {
        Some(new Fastfood(DenseVector(convolutionsDouble), phaseDouble, numOutputFeatures))
      } else {
        None
      }
    val convolutions = convert(convolutionsDouble, Float)
    val phase = convert(phaseDouble, Float)
    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, convSize,
      whitener, whitenerOffset, convolutions, phase, insanity, ff, numOutputFeatures, numInputFeatures))
  }
}
