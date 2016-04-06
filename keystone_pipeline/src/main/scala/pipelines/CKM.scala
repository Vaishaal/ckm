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

    val feature_id = conf.seed + "_" + conf.dataset + "_" +  conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" +
      conf.bandwidth.mkString("-") + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + "_" + conf.filters.mkString("-")
    println(feature_id)
    val hadoopConf = sc.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)


    val featurized: FeaturizedDataset = CKMFeatureLoader(sc, conf.featureDir, feature_id,  Some(conf.numClasses))

    val trainIds = featurized.XTrain.zipWithUniqueId.map(x => x._2.toInt)
    val testIds = featurized.XTest.zipWithUniqueId.map(x => x._2.toInt)

    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
    println(conf.numClasses)
    println("VECTORIZING LABELS")

    val yTrain = labelVectorizer(featurized.yTrain)
    val yTest = labelVectorizer(featurized.yTest).map(convert(_, Int).toArray)
    val XTrain = featurized.XTrain
    val XTest = featurized.XTest

    if (conf.solve) {
      val model =
      if (conf.solver ==  "kernel" ) {
      val kernelGen = new GaussianKernelGenerator(conf.kernelGamma, XTrain)
       new KernelRidgeRegression(kernelGen, conf.reg, conf.blockSize, conf.numIters, Some(895423832L)).fit(XTrain, yTrain)
     } else {
      new BlockWeightedLeastSquaresEstimator(conf.blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)
    }
        val trainPredictions = model.apply(XTrain).cache()
        val testPredictions =  model.apply(XTest).cache()
        val testLabels = labelVectorizer(featurized.yTest)
        val trainLabels = labelVectorizer(featurized.yTrain)

        val yTrainPred = MaxClassifier.apply(trainPredictions)
        val yTestPred =  MaxClassifier.apply(testPredictions)

        val top1Actual = TopKClassifier(1)(testLabels)
        val top1TrainActual = TopKClassifier(1)(trainLabels)

          if (!conf.augment) {
            if (conf.numClasses >= 5) {
              val top5Predicted = TopKClassifier(5)(testPredictions)
              println("Top 5 test acc is " + (100 - Stats.getErrPercent(top5Predicted, top1Actual, testPredictions.count())) + "%")
              val top5TrainPredicted = TopKClassifier(5)(trainPredictions)
              println("Top 5 train acc is " + (100 - Stats.getErrPercent(top5TrainPredicted, top1TrainActual, trainPredictions.count())) + "%")
            }

            val top1Predicted = TopKClassifier(1)(testPredictions)
            val top1TrainPredicted = TopKClassifier(1)(trainPredictions)
            println("Top 1 test acc is " + (100 - Stats.getErrPercent(top1Predicted, top1Actual, testPredictions.count())) + "%")
            println("Top 1 train acc is " + (100 - Stats.getErrPercent(top1TrainPredicted, top1TrainActual, trainPredictions.count())) + "%")
          } else {
            val trainEval = AugmentedExamplesEvaluator(
              trainIds, trainPredictions, featurized.yTrain, conf.numClasses)
            val testEval = AugmentedExamplesEvaluator(
              testIds, testPredictions, featurized.yTest, conf.numClasses)
            println(s"total training accuracy ${1 - trainEval.totalError}")
            println(s"total testing accuracy ${1 - testEval.totalError}")
          }



        val out_train = new BufferedWriter(new FileWriter("/tmp/ckm_train_results"))
        val out_test = new BufferedWriter(new FileWriter("/tmp/ckm_test_results"))

        trainPredictions.zip(featurized.yTrain.zip(trainIds)).map {
            case (weights, (label, id)) => s"$id,$label," + weights.toArray.mkString(",")
          }.collect().foreach{x =>
            out_train.write(x)
            out_train.write("\n")
          }
          out_train.close()

        testPredictions.zip(featurized.yTest.zip(testIds)).map {
            case (weights, (label, id)) => s"$id,$label," + weights.toArray.mkString(",")
          }.collect().foreach{x =>
            out_test.write(x)
            out_test.write("\n")
          }
          out_test.close()
      }
  }

  def loadData(sc: SparkContext, dataset: String):Dataset = {
    val (train, test) =
      if (dataset == "cifar") {
        val train = CifarLoader2(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_train.bin").cache
        val test = CifarLoader2(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_test.bin").cache
        (train, test)
      } else if (dataset == "mnist") {
        val train = MnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist", 10, "train").cache
        val test = MnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist", 10, "test").cache
        (train, test)
      } else if (dataset == "mnist_small") {
        val train = SmallMnistLoader(sc, "/Users/vaishaal/research/ckm/mldata/mnist_small", 10, "train").cache
        val test = SmallMnistLoader(sc, "/Users/vaishaal/research/ckm/mldata/mnist_small", 10, "test").cache
        (train, test)
      } else if (dataset == "imagenet") {
        val train = ImageNetLoader(sc, "/user/vaishaal/imagenet-train-brewed",
          "/home/eecs/vaishaal/ckm/mldata/imagenet/imagenet-labels").cache
        val test = ImageNetLoader(sc, "/user/vaishaal/imagenet-validation-brewed",
          "/home/eecs/vaishaal/ckm/mldata/imagenet/imagenet-labels").cache
        (train, test)
      } else if (dataset == "imagenet-small") {
        val train = ImageNetLoader(sc, "/user/vaishaal/imagenet-train-brewed-small",
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache
        val test = ImageNetLoader(sc, "/user/vaishaal/imagenet-validation-brewed-small",
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache

        (train.repartition(200), test.repartition(200))
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
    @BeanProperty var  featureDir: String = "s3n://ckmfeatures/"
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
      conf.set("spark.hadoop.fs.s3n.awsSecretAccessKey", "pcJoXFSbsDBHFW7jIxZbeudetLgx4WgqqT/OV85J")
      conf.set("spark.hadoop.fs.s3n.awsAccessKeyId", "AKIAJ5XDCFWZOHFC4ESA")
      conf.setAppName(appConfig.expid)
      val sc = new SparkContext(conf)
      sc.setCheckpointDir(appConfig.checkpointDir)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
