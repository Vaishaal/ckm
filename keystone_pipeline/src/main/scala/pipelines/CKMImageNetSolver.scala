package pipelines

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import breeze.stats.{mean, median}
import evaluation.{AugmentedExamplesEvaluator, MulticlassClassifierEvaluator}
import loaders._
import nodes.images._
import nodes.learning._
import nodes.util.{Identity, Cacher, ClassLabelIndicatorsFromIntLabels, TopKClassifier, MaxClassifier, VectorCombiner}
import org.apache.spark.Accumulator
import workflow.Transformer

import org.apache.commons.math3.random.MersenneTwister
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import utils.{Image, MatrixUtils, Stats}
import workflow.Pipeline

import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.Yaml
import scala.reflect.{BeanProperty, ClassTag}

import java.io.{File, BufferedWriter, FileWriter}

object CKMImageNetSolver extends Serializable with Logging {
  val appName = "CKMImageNetSolver"

  def run(sc: SparkContext, conf: CKMConf) {
    println("RUNNING CKMImageNetSolver")
    val featureId = conf.seed + "_" + conf.dataset + "_" +  conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" + conf.bandwidth.mkString("-") + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + "_" + conf.filters.mkString("-")

    println(featureId)
    val featurized = CKMFeatureLoader(sc, conf.featureDir, featureId,  Some(conf.numClasses))
    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
    println(conf.numClasses)
    println("VECTORIZING LABELS")

    val yTrain = labelVectorizer(featurized.yTrain)
    val yTest = labelVectorizer(featurized.yTest)
    val XTrain = featurized.XTrain
    val XTest = featurized.XTest
    val model = new BlockWeightedLeastSquaresEstimator(conf.blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)

    println("Training finish!")

    /*
    println("Saving model")
    val xs = model.xs.zipWithIndex
    xs.map(mi => breeze.linalg.csvwrite(new File(s"${conf.modelDir}/${featureId}.reg.${conf.reg}model.weights.${mi._2}"), mi._1, separator = ','))
    model.bOpt.map(b => breeze.linalg.csvwrite(new File(s"${conf.modelDir}/${featureId}.model.reg.${conf.reg}.intercept"),b.toDenseMatrix, separator = ','))
    */
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
      // Logger.getLogger("org").setLevel(Level.WARN)
      // Logger.getLogger("akka").setLevel(Level.WARN)

      // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
      conf.remove("spark.jars")

      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      conf.setAppName(appConfig.expid)
      val sc = new SparkContext(conf)
      //sc.setCheckpointDir(appConfig.checkpointDir)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
