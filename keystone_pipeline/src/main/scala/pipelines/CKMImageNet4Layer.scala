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
import CKMImageNetTest.loadTest

object CKMImageNet4Layer extends Serializable with Logging {
  val appName = "CKMImageNet4Layer"

  def run(sc: SparkContext, conf: CKMConf) {
    var data = loadTrain(sc, conf.dataset, conf.featureDir, conf.labelDir)
    var dataTest = loadTest(sc, conf.dataset, conf.featureDir, conf.labelDir)
    println("RUNNING CKMImageNet4Layer")
    val featureId = conf.seed + "_" + conf.dataset + "_" +  conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" + conf.bandwidth.mkString("-") + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + "_" + conf.filters.mkString("-")

    println("BLAS TEST")
    val x = DenseMatrix.rand(100,100)
    val y = x*x

    var convKernel: Pipeline[Image, Image] = new Identity()
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    var numOutputFeatures = 0

    val numTrain = data.count()

    val (xDim, yDim, numChannels) = getInfo(data)
    println(s"Info ${xDim}, ${yDim}, ${numChannels}")

    var numInputFeatures = numChannels
    var currX = xDim
    var currY = yDim

    var accs: List[Accumulator[Double]] = List()
    var pool_accum = sc.accumulator(0.0)

    val whitener = if (!conf.loadWhitener) {
      val patchExtractor = new Windower(1, conf.patch_sizes(0))
        .andThen(ImageVectorizer.apply)
        .andThen(new Sampler(10000, conf.seed))

      val baseFilters = patchExtractor(data.map(_.image))
      val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
      new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(baseFilterMat)
    } else {
      CKMImageNetTest.loadWhitener(conf.patch_sizes(0), conf.modelDir)
    }

    val rows = whitener.whitener.rows
    val cols = whitener.whitener.cols
    println(s"Whitener Rows :${rows}, Cols: ${cols}")

    numOutputFeatures = conf.filters(0)
    val patchSize = math.pow(conf.patch_sizes(0), 2).toInt
    val seed = conf.seed

    val ccap = new CC(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(0), currX, currY, numInputFeatures, sc, Some(whitener), conf.whitenerOffset, conf.pool(0), conf.insanity, conf.fastfood)
    accs =  ccap.accs
    var pooler =  new MyPooler(conf.poolStride(0), conf.pool(0), identity, (x:DenseVector[Double]) => mean(x), sc)
    pool_accum = pooler.pooling_accum
    convKernel = convKernel andThen ccap andThen pooler

    currX = math.ceil(((currX  - conf.patch_sizes(0) + 1) - conf.pool(0)/2.0)/conf.poolStride(0)).toInt
    currY = math.ceil(((currY  - conf.patch_sizes(0) + 1) - conf.pool(0)/2.0)/conf.poolStride(0)).toInt

    numInputFeatures = numOutputFeatures

    for (i <- 1 until conf.layers) {
      numOutputFeatures = conf.filters(i)
      val patchSize = math.pow(conf.patch_sizes(i), 2).toInt
      val seed = conf.seed + i
      val ccap = new CC(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(i), currX, currY, numInputFeatures, sc, None, conf.whitenerOffset, conf.pool(i), conf.insanity, conf.fastfood)

      if (conf.pool(i) > 1) {
        var pooler =  new Pooler(conf.poolStride(i), conf.pool(i), identity, (x:DenseVector[Double]) => mean(x))
        convKernel = convKernel andThen ccap andThen pooler
      } else {
        convKernel = convKernel andThen ccap
      }
      // (8 - 3 + 1)
      currX = math.ceil(((currX  - conf.patch_sizes(i) + 1) - conf.pool(i)/2.0)/conf.poolStride(i)).toInt
      currY = math.ceil(((currY  - conf.patch_sizes(i) + 1) - conf.pool(i)/2.0)/conf.poolStride(i)).toInt
      println(s"Layer ${i} output, Width: ${currX}, Height: ${currY}")
      numInputFeatures = numOutputFeatures
    }
    val outFeatures = currX * currY * numOutputFeatures
    val featurizer1 = ImageExtractor  andThen convKernel
    val featurizer2 = ImageVectorizer andThen new Cacher[DenseVector[Double]]
    println(s"Layer 0 output, Width: ${currX}, Height: ${currY}")
    println("OUT FEATURES " +  outFeatures)

    val featurizer = ImageExtractor andThen convKernel andThen ImageVectorizer andThen new Cacher[DenseVector[Double]]
    val convTrainBegin = System.nanoTime
    var XTrain = featurizer(data)
    val count = XTrain.count()
    val convTrainTime  = timeElapsed(convTrainBegin)
    println(s"Generating train features took ${convTrainTime} secs")

    println(s"Per image metric breakdown (for last layer):")
    accs.map(x => println(x.name.get + ":" + x.value/(count)))
    println(pool_accum.name.get + ":" + pool_accum.value/(count))
    println("Total Time (for last layer): " + (accs.map(x => x.value/(count)).reduce(_ + _) +  pool_accum.value/(count)))

    // val numFeatures = XTrain.take(1)(0).size
    println(s"NUM TRAIN FEATURES: ${count}")
    println(s"NUM TEST FEATURES ${count}")

    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
    val yTrain = labelVectorizer(LabelExtractor(data))
    var XTest = featurizer(dataTest)
    val yTest = labelVectorizer(LabelExtractor(dataTest))
    val count2 = XTest.count


    if (conf.saveFeatures) {
      println("Saving Features")
      XTrain.zip(LabelExtractor(data)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"${conf.featureDir}/ckn_${featureId}_train_features")
      XTest.zip(LabelExtractor(data)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"${conf.featureDir}/ckn_${featureId}_train_features")
      breeze.linalg.csvwrite(new File(s"${conf.modelDir}/${conf.patch_sizes(0).toInt}.whitener.matrix"),whitener.whitener, separator = ',')
      breeze.linalg.csvwrite(new File(s"${conf.modelDir}/${conf.patch_sizes(0).toInt}.whitener.means"),whitener.means.toDenseMatrix, separator = ',')
    }
    if (conf.solve) {

      val model = new BlockWeightedLeastSquaresEstimator(conf.blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)

      println("Training finish!")
      val trainPredictions = model.apply(XTrain).cache()

      val yTrainPred = MaxClassifier.apply(trainPredictions)

      val top1TrainActual = TopKClassifier(1)(yTrain)
      val top5TrainPredicted = TopKClassifier(5)(trainPredictions)
      println("Top 5 train acc is " + (100 - Stats.getErrPercent(top5TrainPredicted, top1TrainActual, trainPredictions.count())) + "%")

      val top1TrainPredicted = TopKClassifier(1)(trainPredictions)
      println("Top 1 train acc is " + (100 - Stats.getErrPercent(top1TrainPredicted, top1TrainActual, trainPredictions.count())) + "%")
      val testPredictions = model.apply(XTest).cache()

      val yTestPred = MaxClassifier.apply(testPredictions)

      val numTestPredict = testPredictions.count()
      println("NUM TEST PREDICT " + numTestPredict)

      val top1TestActual = TopKClassifier(1)(yTest)
      val top5TestPredicted = TopKClassifier(5)(testPredictions)
      println("Top 5 test acc is " + (100 - Stats.getErrPercent(top5TestPredicted, top1TestActual, numTestPredict)) + "%")
      val top1TestPredicted = TopKClassifier(1)(testPredictions)
      println("Top 1 test acc is " + (100 - Stats.getErrPercent(top1TestPredicted, top1TestActual, testPredictions.count())) + "%")
    }
  }

  def loadTrain(sc: SparkContext, dataset: String, dataRoot: String = "/", labelsRoot: String = "/"): RDD[LabeledImage] = {
    if (dataset == "imagenet") {
      ImageNetLoader(sc, s"${dataRoot}/imagenet-train-brewed",
        s"${labelsRoot}/imagenet-labels").cache
    } else if (dataset == "imagenet-small") {
      ImageNetLoader(sc, s"${dataRoot}/imagenet-train-brewed-small",
        s"${labelsRoot}/imagenet-small-labels").repartition(200).cache()
    } else {
        throw new IllegalArgumentException("Only Imagenet allowed")
    }
  }

  def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9

  def getInfo(data: RDD[LabeledImage]): (Int, Int, Int) = {
    val image = data.take(1)(0).image
    (image.metadata.xDim, image.metadata.yDim, image.metadata.numChannels)
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
      // Logger.getLogger("org").setLevel(Level.WARN)
      // Logger.getLogger("akka").setLevel(Level.WARN)

      // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
      conf.remove("spark.jars")

      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      val featureId = appConfig.seed + "_" + appConfig.dataset + "_" +  appConfig.expid  + "_" + appConfig.layers + "_" + appConfig.patch_sizes.mkString("-") + "_" + appConfig.bandwidth.mkString("-") + "_" + appConfig.pool.mkString("-") + "_" + appConfig.poolStride.mkString("-") + "_" + appConfig.filters.mkString("-")
      conf.setAppName(featureId)
      val sc = new SparkContext(conf)
      //sc.setCheckpointDir(appConfig.checkpointDir)
      run(sc, appConfig)
      sc.stop()
    }
  }
}