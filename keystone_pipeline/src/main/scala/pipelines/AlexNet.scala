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
import pipelines.Logging
import scopt.OptionParser
import utils.{Image, MatrixUtils, Stats, ImageMetadata, LabeledImage, RowMajorArrayVectorizedImage, ChannelMajorArrayVectorizedImage}
import workflow.Pipeline

import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.Yaml
import scala.reflect.{BeanProperty, ClassTag}

import java.io.{File, BufferedWriter, FileWriter}

object AlexNet extends Serializable with Logging {
  val appName = "AlexNet"

  def run(sc: SparkContext, conf: CKMConf) {

    /* This is a CKM flavor of the AlexNet architechture */
    println("RUNNING Alexnet")
    val featureId = conf.seed + "_" + "ALEXNET" + "_" + conf.filters.mkString("-") +
    conf.bandwidth.mkString("-")

    /* Imagenet constants */
    val xDim = 256
    val yDim = 256
    val numChannels = 3

    /* First load ze data */
    val train = ImageNetLoader(sc, s"${conf.featureDir}/imagenet-train-brewed",
      s"${conf.labelDir}/imagenet-labels").cache

    val test = ImageNetLoader(sc, s"${conf.featureDir}/imagenet-validation-brewed",
      s"${conf.labelDir}/imagenet-labels").cache


    /* Layer 1
     * 11 x 11 patches, stride of 4, pool by 4
     */

    val layer1Patch = 11
    val layer1Pool = 4
    val layer1Stride = 4
    val layer1Channels = numChannels
    val layer1InputFeatures = numChannels*layer1Patch*layer1Patch
    val layer1OutputFeatures = conf.filters(0)

    val layer1PatchExtractor = new Windower(1, layer1Patch)
                              .andThen(ImageVectorizer.apply)
                              .andThen(new Sampler(10000, conf.seed))
    val layer1Samples = MatrixUtils.rowsToMatrix(layer1PatchExtractor(train.map(_.image)))
    println("Whitening Layer 1")
    val layer1Whitener = new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(layer1Samples)
    val layer1Convolver = new CC(layer1InputFeatures,
                                 layer1OutputFeatures,
                                 conf.seed,
                                 conf.bandwidth(0),
                                 xDim,
                                 yDim,
                                 layer1Channels,
                                 sc,
                                 Some(layer1Whitener),
                                 conf.whitenerOffset,
                                 layer1Pool,
                                 conf.insanity,
                                 false,
                                 layer1Stride)

    val layer1Pooler = new MyPooler(layer1Pool, layer1Pool, identity, (x:DenseVector[Double]) => mean(x), sc)

    val layer1 = ImageExtractor andThen layer1Convolver andThen layer1Pooler
    val layer1train = layer1(train)
    val layer1test = layer1(test)

    /* Layer 2
     * 5 x 5 patches, stride of 1, pool by 2
     */

    val layer2Patch = 5
    val layer2Pool = 2
    val layer2Stride = 1
    val layer2Channels = conf.filters(0)
    val layer2InputFeatures = layer2Channels*layer2Patch*layer2Patch
    val layer2OutputFeatures = conf.filters(1)

    val layer2PatchExtractor = new Windower(1, layer2Patch)
                              .andThen(ImageVectorizer.apply)
                              .andThen(new Sampler(10000, conf.seed))
    val layer2Samples = MatrixUtils.rowsToMatrix(layer2PatchExtractor(layer1train))
    println("Whitening Layer 2")
    val layer2Whitener = new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(layer2Samples)

    val layer2Convolver = new CC(layer2InputFeatures,
                                 layer2OutputFeatures,
                                 conf.seed,
                                 conf.bandwidth(1),
                                 xDim,
                                 yDim,
                                 layer2Channels,
                                 sc,
                                 Some(layer2Whitener),
                                 conf.whitenerOffset,
                                 layer2Pool,
                                 conf.insanity,
                                 false,
                                 layer2Stride)

    val layer2Pooler = new MyPooler(layer2Pool, layer2Pool, identity, (x:DenseVector[Double]) => mean(x), sc)

    val layer2 =  layer2Convolver andThen layer2Pooler
    val layer2train = layer2(layer1train)
    val layer2test = layer2(layer1test)

    /* Layer 3
     * 3 x 3 patches, stride of 3, pool by 12
     */

    val layer3Patch = 3
    val layer3Pool = 12
    val layer3Stride = 1
    val layer3Channels = conf.filters(2)
    val layer3InputFeatures = layer3Channels*layer3Patch*layer3Patch
    val layer3OutputFeatures = conf.filters(2)

    val layer3PatchExtractor = new Windower(1, layer3Patch)
                              .andThen(ImageVectorizer.apply)
                              .andThen(new Sampler(10000, conf.seed))
    val layer3Samples = MatrixUtils.rowsToMatrix(layer3PatchExtractor(layer2train))
    println("Whitening Layer 3")
    val layer3Whitener = new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(layer3Samples)

    val layer3Convolver = new CC(layer3InputFeatures,
                                 layer3OutputFeatures,
                                 conf.seed,
                                 conf.bandwidth(2),
                                 xDim,
                                 yDim,
                                 layer3Channels,
                                 sc,
                                 Some(layer3Whitener),
                                 conf.whitenerOffset,
                                 layer3Pool,
                                 conf.insanity,
                                 false,
                                 layer3Stride)

    val layer3Pooler = new MyPooler(layer3Pool, layer3Pool, identity, (x:DenseVector[Double]) => mean(x), sc)

    val layer3 =  layer3Convolver andThen layer3Pooler
    val layer3train = layer3(layer2train)
    val layer3test = layer3(layer2test)


    /* Layer 4
     * Non convolutional layer just lift
     */

    val layer4Patch = 1
    val layer4Stride = 1
    val layer4Channels = conf.filters(3)
    val layer4InputFeatures = layer4Channels*layer4Patch*layer4Patch
    val layer4OutputFeatures = conf.filters(2)

    val layer4PatchExtractor = new Windower(1, layer4Patch)
                              .andThen(ImageVectorizer.apply)
                              .andThen(new Sampler(10000, conf.seed))
    val layer4Samples = MatrixUtils.rowsToMatrix(layer4PatchExtractor(layer3train))

    val lift = SeededCosineRandomFeatures(layer3OutputFeatures, conf.filters(3), conf.bandwidth(3), conf.seed)

    val finalLayer = ImageVectorizer andThen lift andThen new Cacher[DenseVector[Double]]
    val XTrain = finalLayer(layer3train)
    val XTest = finalLayer(layer3test)

    XTrain.count()
    XTest.count()

    if (conf.saveFeatures) {
      println("Saving Features")
      XTrain.zip(LabelExtractor(train)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"${conf.featureDir}/ckn_${featureId}_train_features")
      XTest.zip(LabelExtractor(train)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"${conf.featureDir}/ckn_${featureId}_test_features")
    }
      val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
      val yTrain = labelVectorizer(LabelExtractor(train))
      val yTest = labelVectorizer(LabelExtractor(test))

      val model = new BlockWeightedLeastSquaresEstimator(conf.blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)

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
      val featureId = appConfig.seed + "_" + appConfig.dataset + "_" +  appConfig.expid  + "_" + appConfig.layers + "_" + appConfig.patch_sizes.mkString("-") + "_" + appConfig.bandwidth.mkString("-") + "_" + appConfig.pool.mkString("-") + "_" + appConfig.poolStride.mkString("-") + "_" + appConfig.filters.mkString("-")
      conf.setAppName(featureId)
      val sc = new SparkContext(conf)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
