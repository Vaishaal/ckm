package nodes.images

import breeze.linalg.DenseVector
import org.apache.spark.Accumulator
import org.apache.spark.SparkContext
import pipelines._
import utils.{ImageMetadata, ChannelMajorArrayVectorizedImage, Image}
import workflow.Transformer

/**
 * This node takes an image and performs pooling on regions of the image.
 *
 * Divides images into fixed size pools, but when fed with images of various
 * sizes may produce a varying number of pools.
 *
 * NOTE: By default strides start from poolSize/2.
 *
 * @param stride x and y stride to get regions of the image
 * @param poolSize size of the patch to perform pooling on
 * @param pixelFunction function to apply on every pixel before pooling
 * @param poolFunction pooling function to use on every region.
 */
class MyPoolerOld(
    stride: Int,
    poolSize: Int,
    pixelFunction: Double => Double,
    poolFunction: DenseVector[Double] => Double,
    sc: SparkContext)
  extends Transformer[Image, Image] {

  val strideStart = poolSize / 2
  val pooling_accum = sc.accumulator(0.0, "Pool")

  def apply(image: Image) = {
    val poolStart = System.nanoTime()
    val xDim = image.metadata.xDim
    val yDim = image.metadata.yDim
    val numChannels = image.metadata.numChannels

    val numPoolsX = math.ceil((xDim - strideStart).toDouble / stride).toInt
    val numPoolsY = math.ceil((yDim - strideStart).toDouble / stride).toInt
    val patch = new Array[Double]( numPoolsX * numPoolsY * numChannels)

    // Start at strideStart in (x, y) and
    var x = strideStart
    while (x < xDim) {
      var y = strideStart
      while (y < yDim) {
      val pool = DenseVector.zeros[Double](poolSize * poolSize)
      val startX = x - poolSize/2
      val endX = math.min(x + poolSize/2, xDim)
      val startY = y - poolSize/2
      val endY = math.min(y + poolSize/2, yDim)

      var c = 0
      while (c < numChannels) {
        var s = startX
        while (s < endX) {
          var b = startY
          while (b < endY) {
            pool((s-startX) + (b-startY)*(endX-startX)) =
              pixelFunction(image.get(s, b, c))
            b = b + 1
          }
          s = s + 1
        }
        patch(c + (x - strideStart)/stride * numChannels +
          (y - strideStart)/stride * numPoolsX * numChannels) = poolFunction(pool)
        c = c + 1
      }
      y += stride
    }
    x += stride
    }
    val out = ChannelMajorArrayVectorizedImage(patch, ImageMetadata(numPoolsX, numPoolsY, numChannels))
    pooling_accum += timeElapsed(poolStart)
    out
  }
def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9
}
