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

  def convertFeature(featureString: String): (DenseVector[Float], Int) = {
     /* Maintain backward compatibility with old featurizations */
     val splitX = featureString.replace("(","").replace(")","").split(",")
     val y = splitX.last.toInt
     val x = DenseVector(splitX.slice(0,splitX.size - 1).map(_.toFloat))
     (x,y)
  }

  def apply(sc: SparkContext, path: String,  feature_id: String, partitions: Option[Int] = None): FeaturizedDataset =
  {
   val trainPath = s"${path}/ckn_${feature_id}_train_features"
   println(trainPath)
   val testPath = s"${path}/ckn_${feature_id}_test_features"
   val (trainText, testText) =
   if (partitions.isEmpty) {
     val count = sc.defaultParallelism
     val trainText:RDD[String] = sc.textFile(trainPath, count).coalesce(count)
     val testText:RDD[String] = sc.textFile(testPath, count).coalesce(count)
     (trainText, testText)
  } else {
     val trainText:RDD[String] = sc.textFile(trainPath, partitions.get).coalesce(partitions.get)
     val testText:RDD[String] = sc.textFile(testPath, partitions.get).coalesce(partitions.get)
     (trainText, testText)
  }

   val trainPairs: RDD[(DenseVector[Float], Int)] = trainText.map(convertFeature).setName("train").cache
   val testPairs: RDD[(DenseVector[Float], Int)] = testText.map(convertFeature).setName("test").cache
   val trainCount = trainPairs.count()
   val testCount = testPairs.count()
   println(s"NUM TRAIN EXAMPLES: ${trainCount}, NUM TEST EXAMPLES ${testCount}")
   val XTrain = trainPairs.map(x => convert(x._1, Double))
   val XTest = testPairs.map(x => convert(x._1, Double))
   val yTrain = trainPairs.map(_._2)
   val yTest = testPairs.map(_._2)
   new FeaturizedDataset(XTrain, XTest, yTrain, yTest)
   }
  }

case class FeaturizedDataset(val XTrain: RDD[DenseVector[Double]],
  val XTest: RDD[DenseVector[Double]],
  val yTrain: RDD[Int],
  val yTest: RDD[Int])




