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

object CKMDeepImageNetSolve extends Serializable with Logging {
  val appName = "CKMDeepImageNetSolve"

  def run(sc: SparkContext, conf: CKMConf) {
    println("RUNNING CKMDeepImageNetSolve")

    val featureId = CKMConf.genFeatureId(conf, conf.seed < CKMConf.LEGACY_CUTOFF)
    println("FeatureID: " + featureId)
    val featurized =
    if (conf.float.contains(conf.layers - 1)) {
      CKMFloatFeatureLoader(sc, conf.featureDir, featureId)
    } else {
      CKMFeatureLoader(sc, conf.featureDir, featureId)
    }

    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
    val yTrain = labelVectorizer(featurized.yTrain)
    val yTest = labelVectorizer(featurized.yTest)
    val XTrain = featurized.XTrain
    val XTest = featurized.XTest
    val n = XTrain.count()
    val d = XTrain.first.size
    var lambda = conf.reg * 1.0/(n) * sum(XTrain.map(x => x :* x ).reduce(_ :+ _))
    lambda = conf.reg
    println("FIRST 4 FEATURES" + XTrain.take(4).map(_.slice(0,4)).mkString("\n"))
    println("Feature Dim: " + XTrain.first.size)
    println("LAMBDA " + lambda)
    val model =
    if (conf.solver == "ls") {
      new BlockLeastSquaresEstimator(conf.blockSize, conf.numIters, lambda).fit(XTrain, yTrain)
    } else if (conf.solver == "wls") {
      new BlockWeightedLeastSquaresEstimator(conf.blockSize, conf.numIters, lambda, conf.solverWeight).fit(XTrain, yTrain)
    } else {
      throw new IllegalArgumentException("Unknown Solver")
    }

    println("Training finish!")
    val trainPredictions = model.apply(XTrain).cache()

      val yTrainPred = MaxClassifier.apply(trainPredictions)

      val top1TrainActual = TopKClassifier(1)(yTrain)
      if (conf.numClasses >= 5) {
        val top5TrainPredicted = TopKClassifier(5)(trainPredictions)
        println("Top 5 train acc is " + (100 - Stats.getErrPercent(top5TrainPredicted, top1TrainActual, trainPredictions.count())) + "%")
      }

      val top1TrainPredicted = TopKClassifier(1)(trainPredictions)
      println("Top 1 train acc is " + (100 - Stats.getErrPercent(top1TrainPredicted, top1TrainActual, trainPredictions.count())) + "%")

      val testPredictions = model.apply(XTest).cache()

      val yTestPred = MaxClassifier.apply(testPredictions)

      val numTestPredict = testPredictions.count()
      println("NUM TEST PREDICT " + numTestPredict)

      val top1TestActual = TopKClassifier(1)(yTest)
      if (conf.numClasses >= 5) {
        val top5TestPredicted = TopKClassifier(5)(testPredictions)
        println("Top 5 test acc is " + (100 - Stats.getErrPercent(top5TestPredicted, top1TestActual, numTestPredict)) + "%")
      }

      val top1TestPredicted = TopKClassifier(1)(testPredictions)
      println("Top 1 test acc is " + (100 - Stats.getErrPercent(top1TestPredicted, top1TestActual, testPredictions.count())) + "%")

  }

  def loadWhitener(patchSize: Double, modelDir: String): ZCAWhitener = {
    val matrixPath = s"${modelDir}/${patchSize.toInt}.whitener.matrix"
    val meansPath = s"${modelDir}/${patchSize.toInt}.whitener.means"
    val whitenerVector = loadDenseVector(matrixPath)
    val whitenSize = math.sqrt(whitenerVector.size).toInt
    val whitener = whitenerVector.toDenseMatrix.reshape(whitenSize, whitenSize)
    val means = loadDenseVector(meansPath)
    new ZCAWhitener(whitener, means, DenseMatrix.zeros[Double](means.size, means.size))
  }

  def loadDenseVector(path: String): DenseVector[Double] = {
    DenseVector(scala.io.Source.fromFile(path).getLines.toArray.flatMap(_.split(",")).map(_.toDouble))
  }

  def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9
  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {

    if (args.size < 1) {
      println("Incorrect number of arguments...Exiting now.")
    } else {
      val configfile = scala.io.Source.fromFile(args(0))
      val configtext = try configfile.mkString finally configfile.close()
      println(configtext)
      val yaml = new Yaml(new Constructor(classOf[CKMConf]))
      val appConfig = yaml.load(configtext).asInstanceOf[CKMConf]
      val conf = new SparkConf().setAppName(appConfig.expid)
      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
      // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
      conf.remove("spark.jars")
      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      val featureId = CKMConf.genFeatureId(appConfig, appConfig.seed < CKMConf.LEGACY_CUTOFF)
      conf.setAppName(featureId)
      val sc = new SparkContext(conf)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
