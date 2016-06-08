package nodes.images

import java.io.File

import breeze.linalg._
import breeze.stats._
import org.scalatest.FunSuite

import pipelines.LocalSparkContext
import pipelines.Logging
import utils._
import org.apache.spark.SparkContext
import utils.ChannelMajorArrayVectorizedImage
import utils.ImageMetadata

class MyConvolverSuite extends FunSuite with LocalSparkContext with Logging {
  test("5x5 patches convolutions no stride") {
    val imgWidth = 256
    val imgHeight = 256
    val imgChannels = 3
    val numFilters = 2
    val convSize = 5

    val imageBGR  = TestUtils.loadTestImage("images/test.jpeg")

    val imArray = imageBGR.toArray

    val imArrayRGB  = TestUtils.BGRtoRGB(new DenseMatrix(3, 256*256, imArray)).data

     val image = new ChannelMajorArrayVectorizedImage(imArrayRGB, imageBGR.metadata)
    //val image = imageBGR

    val theanoFile = new File(TestUtils.getTestResourceFileName("theano_conv2d_nostride.csv"))

    val convolvedImgRaw:Array[Double] = csvread(theanoFile).data

    val convolvedImg = new ColumnMajorArrayVectorizedImage(convolvedImgRaw, ImageMetadata(imgWidth - convSize + 1, imgHeight - convSize + 1, numFilters))

    val convBank = convert(new DenseMatrix(2, 5*5*3, (0 until 2*5*5*3).toArray.map(x => 1)), Double)

    val convolver = new MyConvolver(convBank, imgWidth, imgHeight, imgChannels)

    val poolImage = convolver(image)

    logInfo(s"Image Dimensions ${poolImage.metadata.xDim} ${poolImage.metadata.yDim} ${poolImage.metadata.numChannels}")

    assert(poolImage.metadata.xDim == image.metadata.xDim - convSize + 1, "Convolved image should have the right xDims.")
    assert(poolImage.metadata.yDim == image.metadata.yDim - convSize + 1, "Convolved image should have the right yDims.")
    assert(poolImage.metadata.numChannels == convBank.rows, "Convolved image should have the right num channels.")
    assert(poolImage.equals(convolvedImg), "Convolved image should match theano convolution")
  }

  test("5x5 patches convolutions with stride = 2") {
    val imgWidth = 256
    val imgHeight = 256
    val imgChannels = 3
    val numFilters = 2
    val convSize = 5

    val imageBGR  = TestUtils.loadTestImage("images/test.jpeg")

    val imArray = imageBGR.toArray

    val imArrayRGB  = TestUtils.BGRtoRGB(new DenseMatrix(3, 256*256, imArray)).data

    val image = new ChannelMajorArrayVectorizedImage(imArrayRGB, imageBGR.metadata)

    val theanoFile = new File(TestUtils.getTestResourceFileName("theano_conv2d_2stride.csv"))

    val convolvedImgRaw:Array[Double] = csvread(theanoFile).data

    val convolvedImg = new ColumnMajorArrayVectorizedImage(convolvedImgRaw, ImageMetadata((imgWidth - convSize + 1)/2, (imgHeight - convSize + 1)/2, numFilters))

    val convBank = convert(new DenseMatrix(2, 5*5*3, (0 until 2*5*5*3).toArray.map(x => 1)), Double)

    val convolver = new MyConvolver(convBank, imgWidth, imgHeight, imgChannels, patchStride=2)

    val poolImage = convolver(image)

    //logInfo(s"Image: ${poolImage.toArray.mkString(",")}")
    logInfo(s"Image Dimensions ${poolImage.metadata.xDim} ${poolImage.metadata.yDim} ${poolImage.metadata.numChannels}")

    assert(poolImage.metadata.xDim == (image.metadata.xDim - convSize + 1)/2, "Convolved image should have the right xDims.")
    assert(poolImage.metadata.yDim == (image.metadata.yDim - convSize + 1)/2, "Convolved image should have the right yDims.")
    assert(poolImage.metadata.numChannels == convBank.rows, "Convolved image should have the right num channels.")
    assert(poolImage.equals(convolvedImg), "Convolved image should match theano convolution")
  }

  test("5x5 patches convolutions with zero padding") {
    val imgWidth = 256
    val imgHeight = 256
    val imgChannels = 3
    val numFilters = 2
    val convSize = 5

    val imageBGR  = TestUtils.loadTestImage("images/test.jpeg")

    val imArray = imageBGR.toArray

    val imArrayRGB  = TestUtils.BGRtoRGB(new DenseMatrix(3, 256*256, imArray)).data

    val image = new ChannelMajorArrayVectorizedImage(imArrayRGB, imageBGR.metadata)

    val theanoFile = new File(TestUtils.getTestResourceFileName("theano_conv2d_zeropad.csv"))

    val convolvedImgRaw:Array[Double] = csvread(theanoFile).data

    val convolvedImg = new ColumnMajorArrayVectorizedImage(convolvedImgRaw, ImageMetadata(imgWidth, imgHeight, numFilters))

    val convBank = DenseMatrix.ones[Double](2, 5*5*3)

    val convolver = new MyConvolver(convBank, imgWidth, imgHeight, imgChannels, zeroPad=true)

    val poolImage = convolver(image)

    logInfo(s"Image Dimensions ${poolImage.metadata.xDim} ${poolImage.metadata.yDim} ${poolImage.metadata.numChannels}")


    assert(poolImage.metadata.xDim == (image.metadata.xDim), "Convolved image should have the right xDims.")
    assert(poolImage.metadata.yDim == (image.metadata.yDim), "Convolved image should have the right yDims.")
    assert(poolImage.metadata.numChannels == convBank.rows, "Convolved image should have the right num channels.")

    val poolArray = poolImage.toArray
    val imageArray = convolvedImg.toArray

    assert(poolImage.equals(convolvedImg), "Convolved image should match theano convolution")
  }
}
