package loaders

import java.io.FileInputStream

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import nodes.images._
import pipelines.CKMConf
import utils.{Image, ImageMetadata, LabeledImage, RowMajorArrayVectorizedImage, ColumnMajorArrayVectorizedImage}


/**
 * Loads images from the CIFAR-10 Dataset.
 */
object CifarWhitenedLoader {
  // We hardcode this because these are properties of the CIFAR-10 dataset.
  val xDim = 32
  val yDim = 32
  val numChannels = 3

  val labelSize = 1

  def apply(sc: SparkContext, path: String): Dataset= {
      val featurized = CKMFeatureLoader(sc, path, "cifar_whitened")
      val trainFeatures= featurized.XTrain
      val testFeatures = featurized.XTest
      val yTrain = featurized.yTrain
      val yTest = featurized.yTest
      val trainImages:RDD[Image] = trainFeatures.map(CKMLayerLoader.convertVectorToImage(_, xDim, yDim, numChannels))
      val testImages:RDD[Image] = testFeatures.map(CKMLayerLoader.convertVectorToImage(_, xDim, yDim, numChannels))
      val trainLabeledImages:RDD[LabeledImage] = trainImages.zip(yTrain).map(x => LabeledImage(x._1, x._2))
      val testLabeledImages:RDD[LabeledImage] = testImages.zip(yTest).map(x => LabeledImage(x._1, x._2))
      Dataset(trainLabeledImages, testLabeledImages)
  }
}
