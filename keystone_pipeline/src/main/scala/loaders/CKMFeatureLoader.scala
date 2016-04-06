package loaders


import java.io.{ File, FileInputStream, FileOutputStream, DataInputStream }
import java.net.URL
import java.nio.file.{ Files, Paths }

import java.util.zip.GZIPInputStream
import breeze.linalg._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils.{ImageMetadata, LabeledImage, ColumnMajorArrayVectorizedImage}
import scala.reflect._


/* Loads (Feature vector, label) pairs written to disk by CKM.scala
 * If spark is not running in local mode, path refers to an hdfs path
 * @param sc The spark context
 * @param path directory to find the feature folders
 * @param feature_id the featurization to load
 * @param partitions number of partitions of data
 */

object CKMFeatureLoader {

  def convertFeature(featureString: String): (DenseVector[Double], Int) = {
     /* Maintain backward compatibility with old featurizations */
     val splitX = featureString.replace("(","").replace(")","").split(",")
     val y = splitX.last.toInt
     val x = DenseVector(splitX.slice(0,splitX.size).map(_.toDouble))
     (x,y)
  }

  def apply(sc: SparkContext, path: String,  feature_id: String, partitions: Option[Int] = None): FeaturizedDataset =
  {
   val trainPath = s"${path}/ckn_${feature_id}_train_features"
   println(trainPath)
   val testPath = s"${path}/ckn_${feature_id}_test_features"
   val (trainText, testText) =
   if (partitions.getOrElse(-1) == -1) {
     val trainText:RDD[String] = sc.textFile(trainPath)
     val testText:RDD[String] = sc.textFile(testPath)
     (trainText, testText)
  } else {
     println("1000 partitions")
     val trainText:RDD[String] = sc.textFile(trainPath, 1000)
     val testText:RDD[String] = sc.textFile(testPath, 1000)
     (trainText, testText)
  }

   val trainPairs: RDD[(DenseVector[Double], Int)] = trainText.map(convertFeature)
   val testPairs: RDD[(DenseVector[Double], Int)] = testText.map(convertFeature)
   val XTrain = trainPairs.map(_._1).cache
   val XTest = testPairs.map(_._1).cache
   val yTrain = trainPairs.map(_._2).cache
   val yTest = testPairs.map(_._2).cache
   XTrain.count()
   XTest.count()
   new FeaturizedDataset(XTrain, XTest, yTrain, yTest)
   }
  }

case class FeaturizedDataset(val XTrain: RDD[DenseVector[Double]],
  val XTest: RDD[DenseVector[Double]],
  val yTrain: RDD[Int],
  val yTest: RDD[Int])




