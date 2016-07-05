package pipelines

import breeze.linalg.{Transpose => _, _}
import breeze.numerics._
import breeze.stats.{mean, median}
import evaluation.MulticlassClassifierEvaluator
import loaders._
import nodes.images._
import nodes.learning._
import nodes.stats.{Sampler, StandardScaler}
import nodes.util._
import org.apache.spark.{SparkConf, SparkContext}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.Yaml
import pipelines.Logging
import scopt.OptionParser
import utils.{Image, MatrixUtils, Stats}
import workflow.Pipeline

import org.apache.log4j.Level
import org.apache.log4j.Logger

object BenCifar  extends Serializable with Logging {
  val appName = "BenCifar"

  def run(sc: SparkContext, conf: CKMConf) {
    var featureId = CKMConf.genFeatureId(conf, conf.seed < CKMConf.LEGACY_CUTOFF)
    println("FEATURE ID IS " + featureId)
    //Set up some constants.
    val numClasses = 10
    val imageSize = 32
    val numChannels = 3
    val whitenerSize = 100000

    // Load up training data, and optionally sample.
    //

    val data = CifarWhitenedLoader(sc, "/user/vaishaal/cifar_whitened")
    val trainData = data.train
    val testData =  data.test
    var trainImages = ImageExtractor(trainData)
    var testImages = ImageExtractor(testData)

    val sampled = trainImages.map(x => norm(new DenseVector(x.toArray)))
    val means = 1.0/(sampled.count()) * sampled.reduce(_ + _)
    println("MEAN IS " + means)

    val patchExtractor = new Windower(1, conf.patch_sizes(0))
      .andThen(ImageVectorizer.apply)
      .andThen(new Sampler(10000))

      val filters: DenseMatrix[Double] = {
        val baseFilters = patchExtractor(trainImages)
        val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
        val sampleFilters = MatrixUtils.sampleRows(baseFilterMat, conf.filters(0))
        sampleFilters
      }

    val alphaVal = 1.0

    val featurizer =
        new Convolver(filters, imageSize, imageSize, numChannels, None, false) andThen
        new SymmetricRectifier(alpha=alphaVal) andThen
        new MyPooler(conf.poolStride(0), conf.pool(0), identity, (x:DenseVector[Double]) => mean(x), sc) andThen
        ImageVectorizer andThen
        new Cacher[DenseVector[Double]]



    // Do testing.
    var XTrain = featurizer(trainImages).get()
    var XTest = featurizer(testImages).get()
    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
    val yTrain = labelVectorizer(LabelExtractor(trainData))
    val yTest = labelVectorizer(LabelExtractor(testData))



    val n = XTrain.count()
    val d = XTrain.first.size
    println("DATA MEAN " + 1.0/(n*d) * sum(XTrain.reduce(_ + _)))
    XTest.count()
    if (conf.saveFeatures) {
      println("Saving TRAIN Features")
      XTrain.zip(LabelExtractor(trainData)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"${conf.featureDir}/ckn_${featureId}_train_features")
      println("Finished saving TRAIN Features")

      println("Saving TEST Features")
      XTest.zip(LabelExtractor(testData)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"${conf.featureDir}/ckn_${featureId}_test_features")
      println("Finished saving TEST Features")
    }
      println("Feature Dim: " + XTrain.first.size)

      println("FIRST 4 FEATURES" + XTrain.take(4).map(_.slice(0,4)).mkString("\n"))
      var lambda = conf.reg * 1.0/(d) * sum(XTrain.map(x => x :* x ).reduce(_ :+ _))
      lambda = conf.reg
      println("LAMBDA IS " + lambda)
      val model = new LocalDualLeastSquaresEstimator(conf.blockSize, lambda).fit(XTrain, yTrain)

      println("Training finish!")
      val trainPredictions = model.apply(XTrain).cache()

      val yTrainPred = MaxClassifier.apply(trainPredictions)

      val top1TrainActual = TopKClassifier(1)(yTrain)
      val top1TrainPredicted = TopKClassifier(1)(trainPredictions)
      println("Top 1 train acc is " + (100 - Stats.getErrPercent(top1TrainPredicted, top1TrainActual, trainPredictions.count())) + "%")

      val testPredictions = model.apply(XTest).cache()

      val yTestPred = MaxClassifier.apply(testPredictions)

      val numTestPredict = testPredictions.count()
      println("NUM TEST PREDICT " + numTestPredict)

      val top1TestActual = TopKClassifier(1)(yTest)

      val top1TestPredicted = TopKClassifier(1)(testPredictions)
      println("Top 1 test acc is " + (100 - Stats.getErrPercent(top1TestPredicted, top1TestActual, testPredictions.count())) + "%")
  }

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
      conf.setAppName("BenCifar")
      val sc = new SparkContext(conf)
      sc.setCheckpointDir(appConfig.checkpointDir)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
