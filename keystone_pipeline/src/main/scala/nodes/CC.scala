package nodes.images

import breeze.linalg._
import breeze.numerics._
import nodes.learning.ZCAWhitener
import nodes.stats.FastfoodBatch
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.Accumulator
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}
import utils.external.NativeRoutines
import workflow.Transformer
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import net.jafama.FastMath

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
    insanity: Boolean = false,
    fastfood: Boolean = false,
    patchStride: Int = 1,
    zeroPad: Boolean = false,
    normalize: Boolean = true
    )
  extends Transformer[Image, Image] {

  val convSize = math.sqrt(numInputFeatures/imgChannels).toInt

  val resWidth = imgWidth - convSize + 1
  val resHeight = imgHeight - convSize + 1

  val padding = if (zeroPad) convSize - 1 else 0

  val outWidth = math.ceil(resWidth/patchStride.toFloat).toInt + padding
  val outHeight = math.ceil(resHeight/patchStride.toFloat).toInt + padding

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
    println(s"Convolve: ${outWidth}, ${outHeight}, ${numOutputFeatures}")
    println(s"Reswidth: ${resWidth}, ${resWidth}, ${numOutputFeatures}")
    println(s"Input: ${imgWidth}, ${imgHeight}, ${imgChannels}")
    println(s"First pixel ${in.take(1)(0).get(0,0,0)}")

    in.mapPartitions(CC.convolvePartitions(_, resWidth, resHeight, imgChannels, convSize,
      whitener, whitenerOffset, numInputFeatures, numOutputFeatures, seed, bandwidth, insanity, fastfood, patchStride, accs, zeroPad, normalize))
  }

  def apply(in: Image): Image = {
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutionsDouble = (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth).t
    val phaseDouble = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val outWidth = math.ceil(resWidth/patchStride).toInt + (convSize - 1)

    val outHeight = math.ceil(resHeight/patchStride).toInt + (convSize - 1)

    var patchMat = new DenseMatrix[Double](outWidth*outHeight, convSize*convSize*imgChannels)
    CC.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, convSize, whitener, whitenerOffset, convolutionsDouble.data, phaseDouble, insanity, None, numOutputFeatures, numInputFeatures, patchStride, accs, zeroPad, normalize)
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
      fastfood: Option[FastfoodBatch],
      out: Int,
      in: Int,
      patchStride: Int,
      accs: List[Accumulator[Double]],
      zeroPad: Boolean,
      normalize: Boolean
      ): Image = {
    val makePatchesStart = System.nanoTime()
    val imgMat = MyConvolver.makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize, patchStride, zeroPad)
    accs(0) += timeElapsed(makePatchesStart)

    val padding = if (zeroPad) convSize - 1  else 0

    val outWidth = math.ceil(resWidth/patchStride).toInt + padding
    val outHeight = math.ceil(resHeight/patchStride).toInt + padding

    val whiteningStart = System.nanoTime()
    var whitenedImage: DenseMatrix[Double] =
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
    val patchNorms =
      if (normalize) {
        l2Normalize(whitenedImage, whitenerOffset)
      } else {
        DenseVector.ones[Double](whitenedImage.rows)
      }

    accs(2) += timeElapsed(normStart)
    var convRes:DenseMatrix[Double] =
    fastfood.map { ff =>
      val dgemmStart = System.nanoTime()
      val convRes = ff(whitenedImage)
      accs(3) += timeElapsed(dgemmStart)
      convRes
    } getOrElse {
      val dgemmStart = System.nanoTime()
      val convRes = whitenedImage * (new DenseMatrix(out, in, convolutions)).t
      accs(3) += timeElapsed(dgemmStart)
      convRes
    }
      val cosStart = System.nanoTime()
      var j = 0
      while (j < convRes.cols) {
        var i = 0
        val pj = phase(j)
        while (i < convRes.rows) {
          convRes(i,j) = FastMath.cos(convRes(i,j) + pj)
          i += 1
        }
        j += 1
      }


      accs(5) += timeElapsed(cosStart)

      if (insanity) {
        val insanityStart = System.nanoTime()
        convRes(::,*) :*= patchNorms
        accs(6) += timeElapsed(insanityStart)
      }
      convRes

    val imCreateStart = System.nanoTime()
    println("CONV RES SIZE " + convRes.size)
    val res = new RowMajorArrayVectorizedImage(
      convRes.data,
      ImageMetadata(outWidth, outHeight, out))
    accs(7) += timeElapsed(imCreateStart)
    res
  }

def l2Norms(X: DenseMatrix[Double], offset: Double): DenseVector[Double] = {
  var i = 0;
  val out = DenseVector.zeros[Double](X.rows)
  while (i < X.rows) {
    var j = 0
    var norm = 0.0;
    while (j < X.cols) {
      val xij = X(i,j) + offset
      norm += xij * xij
      j += 1
    }
    out(i) = FastMath.sqrt(norm)
    i += 1
  }
  out
}

def l2Normalize(X: DenseMatrix[Double], offset: Double): DenseVector[Double] =  {
  var i = 0;
  val norms = DenseVector.zeros[Double](X.rows)
  while (i < X.rows) {
    var j = 0
    var norm = 0.0;
    while (j < X.cols) {
      val xij = X(i,j) + offset
      norm += xij * xij
      j += 1
    }
    norm = FastMath.sqrt(norm)
    j = 0
    while (j < X.cols) {
      X(i,j) = X(i,j)/norm
      j += 1
    }
    norms(i) = norm
    i += 1
  }
  norms
}


def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9

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
      patchStride: Int,
      accs: List[Accumulator[Double]],
      zeroPad: Boolean,
      normalize: Boolean
      ): Iterator[Image] = {
    val padding = if (zeroPad) convSize - 1 else 0

    val outWidth = math.ceil(resWidth/patchStride.toFloat).toInt + padding
    val outHeight = math.ceil(resHeight/patchStride.toFloat).toInt + padding
    var patchMat = new DenseMatrix[Double](outWidth*outHeight, convSize*convSize*imgChannels)
      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutions =
    if (!fastfood) {
      val W = DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian)
      /* TODO: KLUDGE */
      val X = whitener.map(_.samples).getOrElse(DenseMatrix.eye[Double](108) :* bandwidth)
      val means = whitener.map(_.means).getOrElse(DenseVector.zeros[Double](108))
      var covMatrix = convert(convert(1.0/X.rows * X.t * X, Float), Double) - means * means.t

      val covMatrixRoot = cholesky(covMatrix)
      val filters = (W * covMatrixRoot.t)
      filters.data
    } else {
      DenseVector.rand(numOutputFeatures, gaussian).data
    }

    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val ff =
      if (fastfood) {
        val sigma = bandwidth
        Some(new FastfoodBatch(DenseVector(convolutions), phase, numOutputFeatures, seed, sigma))
      } else {
        None
      }
    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, convSize,
      whitener, whitenerOffset, convolutions, phase, insanity, ff, numOutputFeatures, numInputFeatures, patchStride, accs, zeroPad, normalize))
  }
}
