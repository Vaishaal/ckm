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

class LabelAugmenter[T: ClassTag](mult: Int) extends FunctionNode[RDD[T], RDD[T]] {
  def apply(in: RDD[T]) = in.flatMap(x => Seq.fill(mult)(x))
}

object CKM extends Serializable with Logging {
  val appName = "CKM"

  def pairwiseMedian(data: DenseMatrix[Double]): Double = {
      val x = data(0 until data.rows/2, *)
      val y = data(data.rows/2 to -1, *)
      val x_norm = norm(x :+ 1e-13, Axis._1)
      val y_norm = norm(y :+ 1e-13, Axis._1)
      val x_normalized = x / x_norm
      val y_normalized = y / y_norm
      val diff = (x_normalized - y_normalized)
      val diff_norm = norm(diff, Axis._1)
      val diff_norm_median = median(diff_norm)
      diff_norm_median
  }

  def samplePairwiseMedian(data: RDD[Image], patchSize: Int = 0): Double = {
      val baseFilters =
      if (patchSize == 0) {
        new Sampler(1000)(ImageVectorizer(data))
      } else {
        val patchExtractor = new Windower(1, patchSize)
                                              .andThen(ImageVectorizer.apply)
                                              .andThen(new Sampler(1000))
        patchExtractor(data)
      }

      val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
      pairwiseMedian(baseFilterMat)
  }

  def augmentData(data: Dataset, conf: CKMConf): Dataset = {
    /* Augment always blows up data set by 10 (for now)
     * TODO: Make this more modular?
     * */
    val labelAugmenter = new LabelAugmenter[Int](10)
    val trainAugment =
      if (conf.augmentType  == "random") {
        RandomFlipper(0.5).apply(
          RandomPatcher(10, conf.augmentPatchSize, conf.augmentPatchSize).apply(
            ImageExtractor(data.train)))
      } else {
        CenterCornerPatcher(conf.augmentPatchSize, conf.augmentPatchSize, true).apply(ImageExtractor(data.train))
      }
      val testAugment = CenterCornerPatcher(conf.augmentPatchSize, conf.augmentPatchSize, true).apply(ImageExtractor(data.test))

      val trainLabelsAugmented = labelAugmenter.apply(LabelExtractor(data.train))
      val testLabelsAugmented = labelAugmenter.apply(LabelExtractor(data.test))
      val augmentedTrain = trainAugment.zip(trainLabelsAugmented).map(x => LabeledImage(x._1,x._2))
      val augmentedTest = testAugment.zip(testLabelsAugmented).map(x => LabeledImage(x._1,x._2))
      new Dataset(augmentedTrain, augmentedTest)

  }

  def run(sc: SparkContext, conf: CKMConf) {
    var data: Dataset = loadData(sc, conf.dataset)
    println("RUNNING BENCHMARK")
    val augmentString =
      if (conf.augment) {
        "Augmented"
      } else {
        ""
      }

    val feature_id = conf.seed + "_" + conf.dataset + augmentString + "_" +  conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" +
      conf.bandwidth.mkString("-") + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + "_" + conf.filters.mkString("-")
    println(feature_id)
    val hadoopConf = sc.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val exists = fs.exists(new org.apache.hadoop.fs.Path(s"/${conf.featureDir}/ckn_${feature_id}_train_features"))
    println(s"/${conf.featureDir}/ckn_${feature_id}_train_features")


    var trainIds = data.train.zipWithUniqueId.map(x => x._2.toInt)
    var testIds = data.test.zipWithUniqueId.map(x => x._2.toInt)
    if (conf.augment) {
        val labelAugmenter = new LabelAugmenter[Int](10)
        trainIds = labelAugmenter(trainIds)
        testIds = labelAugmenter(testIds)
    }

      var convKernel: Pipeline[Image, Image] = new Identity()
      var convKernel_old: Pipeline[Image, Image] = new Identity()
      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))
      val gaussian = new Gaussian(0, 1)
      val uniform = new Uniform(0, 1)
      var numOutputFeatures = 0
      data =
      if (conf.augment) {
        augmentData(data, conf)
      } else {
        data
      }

      val (xDim, yDim, numChannels) = getInfo(data)
      println(s"Info ${xDim}, ${yDim}, ${numChannels}")
      var numInputFeatures = numChannels
      var currX = xDim
      var currY = yDim

      var accs: List[Accumulator[Double]] = List()
      var pool_accum = sc.accumulator(0.0)
        val start = System.nanoTime()
        // Whiten top level
        val patchExtractor = new Windower(1, conf.patch_sizes(0))
                                                .andThen(ImageVectorizer.apply)
                                                .andThen(new Sampler(100000, conf.seed))
        val baseFilters = patchExtractor(data.train.map(_.image))
        val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
        val whitener = new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(baseFilterMat)
        val whitenedBase = whitener(baseFilterMat)

        val rows = whitener.whitener.rows
        val cols = whitener.whitener.cols
        println(s"Whitener Rows :${rows}, Cols: ${cols}")
        println(s"Whitening took ${timeElapsed(start)} secs")

        numOutputFeatures = conf.filters(0)
        val patchSize = math.pow(conf.patch_sizes(0), 2).toInt
        val seed = conf.seed
        val ccap = new CC(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(0), currX, currY, numInputFeatures, sc, Some(whitener), conf.whitenerOffset, conf.pool(0), conf.insanity, conf.fastfood)
        accs =  ccap.accs
        val ccap_old = new CC_old(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(0), currX, currY, numInputFeatures, Some(whitener), conf.whitenerOffset, conf.pool(0), conf.insanity, conf.fastfood)
        if (conf.pool(0) > 1) {
          var pooler =  new MyPooler(conf.poolStride(0), conf.pool(0), identity, (x:DenseVector[Double]) => mean(x), sc)
          pool_accum = pooler.pooling_accum
          convKernel = convKernel andThen ccap andThen pooler
          convKernel_old = convKernel_old andThen ccap_old andThen pooler
        } else {
          convKernel = convKernel andThen ccap
          convKernel_old = convKernel_old andThen ccap
        }
        currX = math.ceil(((currX  - conf.patch_sizes(0) + 1) - conf.pool(0)/2.0)/conf.poolStride(0)).toInt
        currY = math.ceil(((currY  - conf.patch_sizes(0) + 1) - conf.pool(0)/2.0)/conf.poolStride(0)).toInt

        println(s"Layer 0 output, Width: ${currX}, Height: ${currY}")
        numInputFeatures = numOutputFeatures

      val outFeatures = currX * currY * numOutputFeatures

      val meta = data.train.take(1)(0).image.metadata
      val featurizer = ImageExtractor  andThen convKernel andThen ImageVectorizer andThen new Cacher[DenseVector[Double]]
      val featurizer_old = ImageExtractor  andThen convKernel_old andThen ImageVectorizer andThen new Cacher[DenseVector[Double]]

      val dataSample = ImageVectorizer(ImageExtractor(data.train))
      println(dataSample.collect().map(sum(_)).reduce(_ + _))
      val convTrainBegin = System.nanoTime
      var XTrain = featurizer(data.train)
      val count = XTrain.count()
      val convTrainTime  = timeElapsed(convTrainBegin)

      val convTrainBegin_old = System.nanoTime
      var XTrain_old = featurizer_old(data.train)
      val count_old = XTrain_old.count()
      val convTrainTime_old  = timeElapsed(convTrainBegin_old)
      val correct = XTrain_old.zip(XTrain).map(x => Stats.aboutEq(x._1, x._2, 1e-4)).reduce(_ && _)
      println("XTrain sum" + XTrain.map(sum(_)).reduce(_ + _))
      println("XTrain old sum" + XTrain_old.map(sum(_)).reduce(_ + _))
      assert(correct)
      println(s"Correctness is: ${correct}")
      println(s"Old featurization took ${convTrainTime_old} secs, new featurizaiton ook ${convTrainTime} secs")
      println(s"Per image metric breakdown:")
      accs.map(x => println(x.name.get + ":" + x.value/(count)))
      println(pool_accum.name.get + ":" + pool_accum.value/(count))
      val numFeatures = XTrain.take(1)(0).size

  }

  def loadData(sc: SparkContext, dataset: String):Dataset = {
    val (train, test) =
      if (dataset == "imagenet-small") {
        val train = ImageNetLoader(sc, "/user/vaishaal/imagenet-train-brewed-small",
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache
        val test = ImageNetLoader(sc, "/user/vaishaal/imagenet-validation-brewed-small",
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache

        (train.repartition(384), test.repartition(384))
      } else if (dataset == "imagenet-tiny") {
        val train = ImageNetLoader(sc, "/user/vaishaal/imagenet-tiny",
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache
        (train.repartition(50), train.repartition(50))
      } else if (dataset == "imagenet-tiny-local") {
        val train = ImageNetLoader(sc, "/home/eecs/vaishaal/ckm/mldata/imagenet-tiny",
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache
        (train.repartition(50), train.repartition(50))
      } else {
        throw new IllegalArgumentException("Unknown Dataset")
      }
      train.checkpoint()
      test.checkpoint()
      return new Dataset(train, test)
  }
  def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9

  def getInfo(data: Dataset): (Int, Int, Int) = {
    val image = data.train.take(1)(0).image
    (image.metadata.xDim, image.metadata.yDim, image.metadata.numChannels)
  }

  class CKMConf {
    @BeanProperty var  dataset: String = "mnist_small"
    @BeanProperty var  expid: String = "mnist_small_simple"
    @BeanProperty var  mode: String = "scala"
    @BeanProperty var  seed: Int = 0
    @BeanProperty var  layers: Int = 1
    @BeanProperty var  filters: Array[Int] = Array(1)
    @BeanProperty var  bandwidth : Array[Double] = Array(1.8)
    @BeanProperty var  patch_sizes: Array[Int] = Array(5)
    @BeanProperty var  loss: String = "WeightedLeastSquares"
    @BeanProperty var  reg: Double = 0.001
    @BeanProperty var  numClasses: Int = 10
    @BeanProperty var  yarn: Boolean = true
    @BeanProperty var  solverWeight: Double = 0
    @BeanProperty var  cosineSolver: Boolean = false
    @BeanProperty var  cosineFeatures: Int = 40000
    @BeanProperty var  cosineGamma: Double = 1e-8
    @BeanProperty var  kernelGamma: Double = 5e-5
    @BeanProperty var  blockSize: Int = 4000
    @BeanProperty var  numBlocks: Int = 2
    @BeanProperty var  numIters: Int = 2
    @BeanProperty var  whiten: Boolean = false
    @BeanProperty var  whitenerValue: Double =  0.1
    @BeanProperty var  whitenerOffset: Double = 0.001
    @BeanProperty var  solve: Boolean = true
    @BeanProperty var  solver: String = "kernel"
    @BeanProperty var  insanity: Boolean = false
    @BeanProperty var  saveFeatures: Boolean = false
    @BeanProperty var  pool: Array[Int] = Array(2)
    @BeanProperty var  poolStride: Array[Int] = Array(2)
    @BeanProperty var  checkpointDir: String = "/tmp/spark-checkpoint"
    @BeanProperty var  augment: Boolean = false
    @BeanProperty var  augmentPatchSize: Int = 24
    @BeanProperty var  augmentType: String = "random"
    @BeanProperty var  fastfood: Boolean = false
    @BeanProperty var  featureDir: String = "/"
  }


  case class Dataset(
    val train: RDD[LabeledImage],
    val test: RDD[LabeledImage])


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
      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      conf.setAppName(appConfig.expid)
      val sc = new SparkContext(conf)
      sc.setCheckpointDir(appConfig.checkpointDir)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
