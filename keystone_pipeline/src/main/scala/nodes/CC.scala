package nodes.images

import breeze.linalg._
import breeze.numerics._
import nodes.learning.ZCAWhitener
import nodes.stats.Fastfood
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.Accumulator
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}
import utils.external.NativeRoutines
import workflow.Transformer
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.util.FastMath

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
    sc: SparkContext,
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
  val make_patches_accum = sc.accumulator(0.0, "Make patches:")
  val norm_accum = sc.accumulator(0.0, "Norm")
  val whitening_accum = sc.accumulator(0.0, "Whitening")
  val dgemm_accum = sc.accumulator(0.0, "Dgemm")
  val phase_accum = sc.accumulator(0.0, "Phase add")
  val cosine_accum = sc.accumulator(0.0, "Cosine")
  val insanity_accum = sc.accumulator(0.0, "Insanity")
  val image_create_accum = sc.accumulator(0.0, "Image Create")
  val accs = List(make_patches_accum, whitening_accum, norm_accum, dgemm_accum, phase_accum, cosine_accum, insanity_accum, image_create_accum)

  override def apply(in: RDD[Image]): RDD[Image] = {
    println(s"Convolve: ${resWidth}, ${resHeight}, ${numOutputFeatures}")
    println(s"Input: ${imgWidth}, ${imgHeight}, ${imgChannels}")
    println(s"First pixel ${in.take(1)(0).get(0,0,0)}")

    in.mapPartitions(CC.convolvePartitions(_, resWidth, resHeight, imgChannels, convSize,
      whitener, whitenerOffset, numInputFeatures, numOutputFeatures, seed, bandwidth, insanity, fastfood, accs))
  }

  def apply(in: Image): Image = {
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutionsDouble = (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth).t
    val phaseDouble = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    CC.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, convSize, whitener, whitenerOffset, convolutionsDouble.data, phaseDouble, insanity, None, numOutputFeatures, numInputFeatures, accs)
  }
  }

object CC {
  /**
    * Given an array of filters, packs the filters into a DenseMatrix[Double] which has the following form:
    * for a row i, column c+y*numChannels+x*numChannels*yDim corresponds to the pixel value at (x,y,c) in image i of
    * the filters array.
    *
    * @param filters Array of filters.
    * @return DenseMatrix of filters, as described above.
    */
  def packFilters(filters: Array[Image]): DenseMatrix[Double] = {
    val (xDim, yDim, numChannels) = (filters(0).metadata.xDim, filters(0).metadata.yDim, filters(0).metadata.numChannels)
    val filterSize = xDim*yDim*numChannels
    val res = DenseMatrix.zeros[Double](filters.length, filterSize)

    var i,x,y,c = 0
    while(i < filters.length) {
      x = 0
      while(x < xDim) {
        y = 0
        while(y < yDim) {
          c = 0
          while (c < numChannels) {
            val rc = c + x*numChannels + y*numChannels*xDim
            res(i, rc) = filters(i).get(x,y,c)

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
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      whitener: Option[ZCAWhitener],
      whitenerOffset: Double,
      convolutions: Array[Double],
      phase: DenseVector[Double],
      insanity: Boolean,
      fastfood: Option[Fastfood],
      out: Int,
      in: Int,
      accs: List[Accumulator[Double]]
      ): Image = {
    val makePatchesStart = System.nanoTime()
    val imgMat = makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize,
      whitener)
    accs(0) += timeElapsed(makePatchesStart)


    val whiteningStart = System.nanoTime()
    val whitenedImage: DenseMatrix[Double] =
    whitener match  {
      case None => {
        imgMat
      }
      case Some(whitener) => {
        val W = whitener.whitener
        val means = whitener.means
        imgMat(*,::) :-= means
        imgMat * W
      }
    }
    accs(1) += timeElapsed(whiteningStart)

    val normStart = System.nanoTime()
    whitenedImage :+= whitenerOffset
    val patchNorms = norm(whitenedImage, Axis._1)
    whitenedImage(::, *) :/= patchNorms
    accs(2) += timeElapsed(normStart)
    var convRes:DenseMatrix[Double] =
    fastfood.map { ff =>
      val ff_out = MatrixUtils.matrixToRowArray(whitenedImage).map(ff(_))
      MatrixUtils.rowsToMatrix(ff_out)
    } getOrElse {
      val dgemmStart = System.nanoTime()
      val convRes = whitenedImage * (new DenseMatrix(out, in, convolutions)).t
      accs(3) += timeElapsed(dgemmStart)
      val phaseStart = System.nanoTime()
      convRes(*, ::) :+= phase
      accs(4) += timeElapsed(phaseStart)

      val convRes_data = convRes.data
      println("COSINE SIZE " + convRes_data.size)
      val cosStart = System.nanoTime()
      var j = 0
      while (j < convRes_data.size) {
          convRes_data(j) = FastMath.cos(convRes_data(j))
          j += 1
      }
      accs(5) += timeElapsed(cosStart)

      if (insanity) {
        val insanityStart = System.nanoTime()
        convRes(::,*) :*= patchNorms
        accs(6) += timeElapsed(insanityStart)
      }
      convRes
    }

    val imCreateStart = System.nanoTime()
    val res = new RowMajorArrayVectorizedImage(
      convRes.data,
      ImageMetadata(resWidth, resHeight, out))
    accs(7) += timeElapsed(imCreateStart)
    res
  }

def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9

  /**
   * This function takes an image and generates a matrix of all of its patches. Patches are expected to have indexes
   * of the form: c + x*numChannels + y*numChannels*xDim
   *
   * @param img
   * @return
   */
  def makePatches(img: Image,
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      whitener: Option[ZCAWhitener]
      ): DenseMatrix[Double] = {
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

              patchMat(py, px) = img.get(x+pox, y+poy, chan)

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
      fastfood: Boolean,
      accs: List[Accumulator[Double]]
      ): Iterator[Image] = {

    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutions = 
    if (!fastfood) {
      (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth).data
    } else {
      (DenseVector.rand(numOutputFeatures, gaussian) :* bandwidth).data 
    }

    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val ff = 
      if (fastfood) {
        Some(new Fastfood(DenseVector(convolutions), phase, numOutputFeatures))
      } else {
        None
      }

    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, convSize,
      whitener, whitenerOffset, convolutions, phase, insanity, ff, numOutputFeatures, numInputFeatures, accs))
  }
}
