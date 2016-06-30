package pipelines

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import breeze.stats.{mean, median}
import evaluation.{AugmentedExamplesEvaluator, MulticlassClassifierEvaluator}
import loaders._
import nodes.images._
import nodes.learning._
import nodes.stats.{StandardScaler, Sampler, SeededCosineRandomFeatures, BroadcastCosineRandomFeatures, CosineRandomFeatures}
import nodes.util.{Identity, Cacher, ClassLabelIndicatorsFromIntLabels, TopKClassifier, MaxClassifier, VectorCombiner}
import org.apache.spark.Accumulator
import workflow.Transformer

import net.jafama.FastMath
import org.apache.commons.math3.random.MersenneTwister
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import utils.{Image, MatrixUtils, Stats, ImageMetadata, LabeledImage, RowMajorArrayVectorizedImage, ChannelMajorArrayVectorizedImage}
import workflow.Pipeline

import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.Yaml
import scala.reflect.{BeanProperty, ClassTag}

import java.io.{File, BufferedWriter, FileWriter}

object CifarWhiten extends Serializable with Logging {

  def run(sc: SparkContext) = {
    //Set up some constants.
    val numClasses = 10
    val imageSize = 32
    val numChannels = 3
    val whitenerSize = 100000

    // Load up training data, and optionally sample.
    //
    val trainData = CifarLoader(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_train.bin").cache

    val trainImages = ImageExtractor(trainData)

    val labelExtractor = LabelExtractor andThen
      ClassLabelIndicatorsFromIntLabels(numClasses) andThen
      new Cacher[DenseVector[Double]]

    val trainLabels = labelExtractor(trainData)


    val whiten = new Preprocess(trainImages, 1.0) andThen ImageVectorizer


    val testData = CifarLoader(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_test.bin").cache
    val testImages = ImageExtractor(testData)
    val testLabels = labelExtractor(testData)

    val XTrain = whiten(trainImages).get()
    var XTest = whiten(testImages).get()
    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(numClasses)
    val yTrain = labelVectorizer(LabelExtractor(trainData))
    val yTest = labelVectorizer(LabelExtractor(testData))

    XTrain.count()
    XTest.count()

      println("Saving TRAIN CIFAR")
      XTrain.zip(LabelExtractor(trainData)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"/user/vaishaal/cifar_whitened/ckn_cifar_whitened_train_features")
      println("Finished saving TRAIN CIFAR")

      println("Saving TEST CIFAR")
      XTest.zip(LabelExtractor(testData)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"/user/vaishaal/cifar_whitened/ckn_cifar_whitened_test_features")
      println("Finished saving TEST CIFAR")
}
  def main(args: Array[String]) = {
      val conf = new SparkConf()
      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
      // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
      conf.remove("spark.jars")
      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      conf.setAppName("CifarWhitener")
      val sc = new SparkContext(conf)
      run(sc)
      sc.stop()
  }
}
