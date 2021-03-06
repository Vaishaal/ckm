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
 *
 * @param filters Bank of convolution filters to apply - each filter is an array in row-major order.
 * @param imgWidth Width of images in pixels.
 * @param imgHeight Height of images in pixels.
 * @param imgChannels Number of channels in input images.
 * @param whitener An optional whitening matrix to apply to the image patches.
 * @param varConstant
 */
class MyConvolver(
    filters: DenseMatrix[Double],
    imgWidth: Int,
    imgHeight: Int,
    imgChannels: Int,
    whitener: Option[ZCAWhitener] = None,
    normalizePatches: Boolean = false,
    varConstant: Double = 10.0,
    patchStride: Int = 1,
    zeroPad: Boolean = false)
  extends Transformer[Image, Image] {

  val convSize = math.sqrt(filters.cols/imgChannels).toInt
  val convolutions = filters.t

  val (resWidth, resHeight) = ((imgWidth - convSize + 1), (imgHeight - convSize + 1))

  override def apply(in: RDD[Image]): RDD[Image] = {
    in.mapPartitions(MyConvolver.convolvePartitions(_, resWidth, resHeight, imgChannels, convSize,
      normalizePatches, whitener, convolutions, varConstant, patchStride, zeroPad))
  }

  def apply(in: Image): Image = {

    val padding = if (zeroPad) convSize - 1  else 0

    val outWidth = math.ceil((resWidth)/patchStride).toInt + padding
    val outHeight = math.ceil((resWidth)/patchStride).toInt + padding

    var patchMat = DenseMatrix.zeros[Double](outWidth*outHeight, convSize*convSize*imgChannels)
    MyConvolver.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, convSize, normalizePatches, whitener, convolutions, varConstant, patchStride, zeroPad)
  }
}

object MyConvolver {
  /**
    * User-friendly constructor interface for the Conovler.
    *
    * @param filters An array of images with which we convolve each input image. These images should *not* be pre-whitened.
    * @param imgInfo Metadata of a typical image we will be convolving. All images must have the same size/shape.
    * @param whitener Whitener to be applied to both the input images and the filters before convolving.
    * @param normalizePatches Should the patches be normalized before convolution?
    * @param varConstant Constant to be used in scaling.
    * @param flipFilters Should the filters be flipped before convolution is applied (used for comparability to MATLAB's
    *                    convnd function.)
    */
  def apply(filters: Array[Image],
           imgInfo: ImageMetadata,
           whitener: Option[ZCAWhitener] = None,
           normalizePatches: Boolean = true,
           varConstant: Double = 10.0,
           flipFilters: Boolean = false) = {

    //If we are told to flip the filters, invert their indexes.
    val filterImages = if (flipFilters) {
      filters.map(ImageUtils.flipImage)
    } else filters

    //Pack the filter array into a dense matrix of the right format.
    val packedFilters = packFilters(filterImages)

    //If the whitener is not empty, construct a new one:
    val whitenedFilterMat = whitener match {
      case Some(x) => x.apply(packedFilters) * x.whitener.t
      case None => packedFilters
    }

    new MyConvolver(
      whitenedFilterMat,
      imgInfo.xDim,
      imgInfo.yDim,
      imgInfo.numChannels,
      whitener,
      normalizePatches,
      varConstant)
  }

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
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      convolutions: DenseMatrix[Double],
      varConstant: Double = 10.0,
      patchStride: Int = 1,
      zeroPad: Boolean = false): Image = {


    val imgMat = makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize,
      patchStride, zeroPad)

    val convRes: DenseMatrix[Double] = imgMat * convolutions

    val res = new RowMajorArrayVectorizedImage(
      convRes.toArray,
      ImageMetadata(math.sqrt(patchMat.rows).toInt, math.sqrt(patchMat.rows).toInt, convolutions.cols))

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
      patchStride: Int,
      zeroPad: Boolean): DenseMatrix[Double] = {
    var x,y,chan,pox,poy,py,px = 0
    println(zeroPad)
    poy = 0
    val padding = if (zeroPad) convSize/2 else 0
    val even = if (zeroPad && convSize % 2 == 0) 1 else 0
    while (poy < convSize) {
      pox = 0
      while (pox < convSize) {
        y = 0  - padding
        while (y < resHeight + padding - even) {
          x = 0 - padding
          while (x < resWidth + padding - even) {
            chan = 0
            while (chan < imgChannels) {
              val xNew = x + padding
              val yNew = y + padding
              px = chan + pox*imgChannels + poy*imgChannels*convSize
              py = math.ceil((xNew/patchStride + (yNew*(resWidth + (2*padding - even))/(patchStride*patchStride)))).toInt

              val imx = x + pox
              val imy = y + poy

              if (imx >= img.metadata.xDim || imy >= img.metadata.yDim) {
                patchMat(py, px) = 0.0
              } else if (imx < 0 || imy < 0) {
                patchMat(py, px) = 0.0
              } else {
                patchMat(py, px) = img.get(imx, imy, chan)
              }
              chan+=1
            }
            x += patchStride
          }
          y += patchStride
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
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      convolutions: DenseMatrix[Double],
      varConstant: Double,
      patchStride: Int,
      zeroPad: Boolean): Iterator[Image] = {

    val padding = if (zeroPad) convSize - 1  else 0

    val outWidth = math.ceil(resWidth/patchStride).toInt + padding
    val outHeight = math.ceil(resWidth/patchStride).toInt + padding
    var patchMat = new DenseMatrix[Double](outWidth*outHeight, convSize*convSize*imgChannels)
    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, convSize, normalizePatches,
      whitener, convolutions, varConstant, patchStride, zeroPad))

  }
}
