package loaders


import java.io.{ File, FileInputStream, FileOutputStream, DataInputStream }
import java.net.URL
import java.nio.file.{ Files, Paths }

import java.util.zip.GZIPInputStream
import breeze.linalg._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils.{Image, ImageMetadata, LabeledImage, ChannelMajorArrayVectorizedImage}
import pipelines.CKMConf
import scala.reflect._


case class Dataset(
  val train: RDD[LabeledImage],
  val test: RDD[LabeledImage])

object CKMLayerLoader {

  def convertVectorToImage(in: DenseVector[Double], xDim: Int, yDim: Int, numChannels: Int): Image = {
    /* TODO: If conf.pool(layer) was 1 output should be RowMajor image */
    val meta = ImageMetadata(xDim, yDim, numChannels)
    ChannelMajorArrayVectorizedImage(in.data, meta)
  }

  /* TODO: Hardcoded for imagenet, fix later */
  def computeLayerSpatialInfo(layer:Int, conf: CKMConf): (Int, Int, Int) = {
    var xDim = 256
    var yDim = 256
    for (i <- 0 until layer) {
        xDim = math.ceil(((xDim  - conf.patch_sizes(i) + 1) - conf.pool(i)/2.0)/conf.poolStride(0)).toInt
        yDim = math.ceil(((yDim  - conf.patch_sizes(i) + 1) - conf.pool(i)/2.0)/conf.poolStride(0)).toInt
    }
    val numChannels = conf.filters(layer)
    (xDim, yDim, numChannels)
  }

  def apply(sc: SparkContext, layer: Int, featureId: String, conf: CKMConf, partitions: Option[Int] = None): Dataset = {
      val featurized = CKMFeatureLoader(sc, conf.featureDir, featureId, partitions)
      val trainFeatures= featurized.XTrain
      val testFeatures = featurized.XTest
      val yTrain = featurized.yTrain
      val yTest = featurized.yTrain
      val (xDim, yDim, numChannels) = computeLayerSpatialInfo(layer, conf)
      val trainImages:RDD[Image] = trainFeatures.map(convertVectorToImage(_, xDim, yDim, numChannels))
      val testImages:RDD[Image] = testFeatures.map(convertVectorToImage(_, xDim, yDim, numChannels))
      val trainLabeledImages:RDD[LabeledImage] = trainImages.zip(yTrain).map(x => LabeledImage(x._1, x._2))
      val testLabeledImages:RDD[LabeledImage] = testImages.zip(yTest).map(x => LabeledImage(x._1, x._2))
      Dataset(trainLabeledImages, testLabeledImages)
  }

}
