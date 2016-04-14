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

import java.io.{File, BufferedWriter, FileWriter}
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file._
import scala.collection.mutable.ArrayBuffer


object CKMImageNetTest extends Serializable with Logging {
  val appName = "CKMImageNetTest"

  def run(sc: SparkContext, conf: CKMConf) {
    var data = loadTest(sc, conf.dataset, conf.featureDir, conf.labelDir)
    println("RUNNING CKMImageNetTest")
    val featureId = conf.seed + "_" + conf.dataset + "_" +  conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" + conf.bandwidth.mkString("-") + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + "_" + conf.filters.mkString("-")

    var convKernel: Pipeline[Image, Image] = new Identity()
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    var numOutputFeatures = 0
    val (xDim, yDim, numChannels) = getInfo(data)
    println(s"Info ${xDim}, ${yDim}, ${numChannels}")
    var numInputFeatures = numChannels
    var currX = xDim
    var currY = yDim

    var accs: List[Accumulator[Double]] = List()
    var pool_accum = sc.accumulator(0.0)
    val patchExtractor = new Windower(1, conf.patch_sizes(0))
      .andThen(ImageVectorizer.apply)
      .andThen(new Sampler(100000, conf.seed))
      val baseFilters = patchExtractor(data.map(_.image))
      val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
      val whitener = loadWhitener(featureId, conf.modelDir)
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
      data.count()
      val outFeatures = currX * currY * numOutputFeatures
      println(s"Layer 0 output, Width: ${currX}, Height: ${currY}")
      println("OUT FEATURES " +  outFeatures)

      val featurizer = ImageExtractor andThen convKernel andThen ImageVectorizer andThen new Cacher[DenseVector[Double]]
      val convTestBegin = System.nanoTime
      var XTest = featurizer(data)
      val count = XTest.count()
      val convTestTime  = timeElapsed(convTestBegin)
      println(s"Generating test features took ${convTestTime} secs")

      println(s"Per image metric breakdown (for last layer):")
      accs.map(x => println(x.name.get + ":" + x.value/(count)))
      println(pool_accum.name.get + ":" + pool_accum.value/(count))
      println("Total Time (for last layer): " + (accs.map(x => x.value/(count)).reduce(_ + _) +  pool_accum.value/(count)))
      val numFeatures = XTest.take(1)(0).size
      println(s"numFeatures: ${numFeatures}, count: ${count}")
      val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
      println(conf.numClasses)
      println("VECTORIZING LABELS")

      val yTest = labelVectorizer(LabelExtractor(data))
      val model = loadModel(featureId, conf.modelDir, conf)
      println("Testing finish!")
      val testPredictions = model.apply(XTest).cache()

      val yTestPred = MaxClassifier.apply(testPredictions)

      val top1TestActual = TopKClassifier(1)(yTest)
      if (conf.numClasses >= 5) {
        val top5TestPredicted = TopKClassifier(5)(testPredictions)
        println("Top 5 test acc is " + (100 - Stats.getErrPercent(top5TestPredicted, top1TestActual, testPredictions.count())) + "%")
      }

      val top1TestPredicted = TopKClassifier(1)(testPredictions)
      println("Top 1 test acc is " + (100 - Stats.getErrPercent(top1TestPredicted, top1TestActual, testPredictions.count())) + "%")
  }

  def loadModel(featureId: String, modelDir: String, conf: CKMConf): BlockLinearMapper = {
    val files = ArrayBuffer.empty[Path]
    val blockSize = conf.blockSize
    val numClasses = conf.numClasses
    val root = Paths.get(modelDir)

    Files.walkFileTree(root, new SimpleFileVisitor[Path] {
      override def visitFile(file: Path, attrs: BasicFileAttributes) = {
        if (file.getFileName.toString.startsWith(s"${featureId}.model.weights")) {
          files += file
        }
      FileVisitResult.CONTINUE
      }
    })

    val xs: Seq[DenseMatrix[Double]] = files.map { f =>
      val xVector = loadDenseVector(f.toString)
      /* This is usually blocksize, but the last block may be smaller */
      val rows = xVector.size/numClasses
      xVector.toDenseMatrix.reshape(rows, numClasses)
    }
    val interceptPath = s"${modelDir}/${featureId}.model.intercept"
    val bOpt =
      if (Files.exists(Paths.get(interceptPath))) {
        Some(loadDenseVector(interceptPath))
      } else {
        None
      }
    new BlockLinearMapper(xs, blockSize, bOpt)
  }

  def loadDenseVector(path: String): DenseVector[Double] = {
    DenseVector(scala.io.Source.fromFile(path).getLines.toArray.flatMap(_.split(",")).map(_.toDouble))
  }


  def loadWhitener(featureId: String, modelDir: String): ZCAWhitener = {
    val matrixPath = s"${modelDir}/${featureId}.whitener.matrix"
    val meansPath = s"${modelDir}/${featureId}.whitener.means"
    val whitenerVector = loadDenseVector(matrixPath)
    val whitenSize = math.sqrt(whitenerVector.size).toInt
    val whitener = whitenerVector.toDenseMatrix.reshape(whitenSize, whitenSize)
    val means = loadDenseVector(meansPath)
    new ZCAWhitener(whitener, means)
  }

  def loadTest(sc: SparkContext, dataset: String, dataRoot: String = "/", labelsRoot: String = "/"): RDD[LabeledImage] = {
    if (dataset == "imagenet") {
      ImageNetLoader(sc, s"${dataRoot}/imagenet-validation-brewed",
        s"${labelsRoot}/imagenet-labels").cache
    } else if (dataset == "imagenet-small") {
      ImageNetLoader(sc, s"${dataRoot}/imagenet-validation-brewed-small",
        s"${labelsRoot}/imagenet-small-labels").cache.repartition(200)
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
      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
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
