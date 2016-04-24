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
    val hadoopConf = sc.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val exists = fs.exists(new org.apache.hadoop.fs.Path(s"/${conf.featureDir}/ckn_${featureId}_train_features"))
    println("EXISTS " + exists)
    println(s"/${conf.featureDir}/ckn_${featureId}_train_features")

    if (!exists) {
      System.err.println(s"Features not found at ${conf.featureDir}/ckn_${featureId}_train_features")
    }

    val featurized = CKMFeatureLoader(sc, "/" + conf.featureDir, featureId,  Some(conf.numClasses))

    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
    println(conf.numClasses)
    println("VECTORIZING LABELS")

    val yTrain = labelVectorizer(featurized.yTrain)
    val yTest = labelVectorizer(featurized.yTest).map(convert(_, Int).toArray)
    val XTrain = featurized.XTrain
    val XTest = featurized.XTest

    val reg:Double = conf.reg*XTrain.map(x => sum(x :* x)).reduce(_ + _) * 1.0/XTrain.count()

    val model = new BlockWeightedLeastSquaresEstimator(conf.blockSize, conf.numIters, reg, conf.solverWeight).fit(XTrain, yTrain)

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
