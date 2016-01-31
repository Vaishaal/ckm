package nodes.images

import breeze.linalg._
import nodes.learning.ZCAWhitener
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}
import workflow.Transformer

/**
 * Convolves images with a bank of convolution filters. Convolution filters must be square.
 * Used for using the same label for all patches from an image.
 * TODO: Look into using Breeze's convolve
 *
 * @param filters Bank of convolution filters to apply - each filter is an array in row-major order.
 * @param imgWidth Width of images in pixels.
 * @param imgHeight Height of images in pixels.
 */
class MyConvolver(
    filters: DenseMatrix[Double],
    imgWidth: Int,
    imgHeight: Int,
    imgChannels: Int,
    whitener: Option[ZCAWhitener] = None,
    normalizePatches: Boolean = true,
    varConstant: Double = 10.0)
  extends Transformer[Image, Image] {

  val convSize = math.sqrt(filters.cols/imgChannels).toInt
  val convolutions = filters.t

  val resWidth = imgWidth - convSize + 1
  val resHeight = imgHeight - convSize + 1

  override def apply(in: RDD[Image]): RDD[Image] = {
    in.mapPartitions(MyConvolver.convolvePartitions(_, resWidth, resHeight, imgChannels, convSize,
      normalizePatches, whitener, convolutions, varConstant))
  }

  def apply(in: Image): Image = {
    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    MyConvolver.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, convSize, normalizePatches, whitener, convolutions)
  }
}

object MyConvolver {
  def convolve(img: Image,
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      convolutions: DenseMatrix[Double],
      varConstant: Double = 10.0): Image = {

    val imgMat = makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize,
      normalizePatches, whitener, varConstant)

    val convRes: DenseMatrix[Double] = imgMat * convolutions

    val res = new ChannelMajorArrayVectorizedImage(
      new Array[Double](resWidth*resHeight*convolutions.cols),
      ImageMetadata(resWidth, resHeight, convolutions.cols))

    // val res = new RowMajorArrayVectorizedImage(
    //   new Array[Double](resWidth*resHeight*convolutions.cols),
    //   ImageMetadata(resWidth, resHeight, convolutions.cols))

    //Now pack the convolved features into the result.
    var x, y, chan = 0
    chan = 0
    while (chan < convolutions.cols) {
      x = 0
      while (x < resWidth) {
        y = 0
        while ( y < resHeight) {
          res.put(x, y, chan, convRes(x + y*resWidth, chan))
          y += 1
        }
        x += 1
      }
      chan += 1
    }

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
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      varConstant: Double): DenseMatrix[Double] = {
    var x,y,chan,pox,poy,py,px = 0

    x = 0
    while (x < resWidth) {
      y = 0
      while (y < resHeight) {
          poy = 0
          while (poy < convSize) {
            pox = 0
            while (pox < convSize) {
              chan = 0
              while (chan < imgChannels) {

                px = chan + pox*imgChannels + poy*imgChannels*convSize
                py = x + y*resWidth

                patchMat(py, px) = img.get(x+pox, y+poy, chan)

                chan+=1
              }
              pox+=1
            }
            poy+=1

          }
        y+=1
      }
      x+=1
    }

    val patchMatN = if(normalizePatches) Stats.normalizeRows(patchMat, varConstant) else patchMat

    val res = whitener match {
      case None => patchMatN
      case Some(whiteness) => patchMatN(*, ::) - whiteness.means
    }

    res
  }

  def convolvePartitions(
      imgs: Iterator[Image],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      convolutions: DenseMatrix[Double],
      varConstant: Double): Iterator[Image] = {

    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, convSize, normalizePatches,
      whitener, convolutions, varConstant))

  }
}
