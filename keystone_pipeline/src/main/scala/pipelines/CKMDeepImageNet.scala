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

object CKMDeepImageNet extends Serializable with Logging {
  val appName = "CKMDeepImageNet"

  def run(sc: SparkContext, conf: CKMConf) {

    /* This is a CKM flavor of the CKMImageNet4Layer.scala architechture */
    println(s"Running a ${conf.layers} layer network")
    val featureId = CKMConf.genFeatureId(conf, conf.seed < CKMConf.LEGACY_CUTOFF)
    println(s"Legacy mode is: ${conf.seed < CKMConf.LEGACY_CUTOFF}")
    println(s"Running a ${conf.layers} layer network")
    println(s"Feature ID is ${featureId}")

    /* Imagenet constants */
    var xDim = 256
    var yDim = 256
    var numChannels = 3
    println("LAYER TO LOAD " + conf.layerToLoad)
    println("LOAD LAYER " + conf.loadLayer)
    val (train, test, startLayer) =
      if (conf.loadLayer) {
        val numLayers = conf.layers
        conf.layers = (conf.layerToLoad + 1)
        val oldFeatureId = CKMConf.genFeatureId(conf, conf.seed < CKMConf.LEGACY_CUTOFF)

        val oldLayers = CKMLayerLoader(sc, conf.layerToLoad, oldFeatureId, conf, None)
        val train = oldLayers.train
        val test = oldLayers.test
        conf.layers = numLayers
        (train, test, conf.layerToLoad)
      } else {
        /* First load ze data */
       val train = ImageNetLoader(sc, s"${conf.featureDir}/imagenet-train-brewed",
         s"${conf.labelDir}/imagenet-labels").cache

       val test = ImageNetLoader(sc, s"${conf.featureDir}/imagenet-validation-brewed",
         s"${conf.labelDir}/imagenet-labels").cache
       (train, test, 0)
      }
      /* Layer 1
       * 11 x 11 patches, stride of 4, pool by 4
       */

      var layerPatch = conf.patch_sizes(0)
      var layerPool = conf.pool(0)
      var layerStride = conf.convStride.getOrElse(0, 1)
      var layerChannels = numChannels
      var layerInputFeatures = numChannels*layerPatch*layerPatch
      var layerOutputFeatures = conf.filters(0)
      var layerWhitener =
        if (conf.whiten.contains(0)) {
          if (conf.loadWhitener) {
            /* Only the first layer whitener can be read from disk */
            Some(loadWhitener(layerPatch, conf.modelDir))
          } else {
            val layerPatchExtractor = new Windower(1, layerPatch)
              .andThen(ImageVectorizer.apply)
              .andThen(new Sampler(10000, conf.seed))
              val layerSamples = MatrixUtils.rowsToMatrix(layerPatchExtractor(train.map(_.image)))
              println("Whitening Layer 1")
              val layerWhitener = Some(new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(layerSamples))
              val whitener = layerWhitener.get
              breeze.linalg.csvwrite(new File(s"${conf.modelDir}/${conf.patch_sizes(0).toInt}.whitener.matrix"),whitener.whitener, separator = ',')
              breeze.linalg.csvwrite(new File(s"${conf.modelDir}/${conf.patch_sizes(0).toInt}.whitener.means"),whitener.means.toDenseMatrix, separator = ',')
              layerWhitener
          }
      } else {
        None
      }
      var layerConvolver = new CC(layerInputFeatures,
                                   layerOutputFeatures,
                                   conf.seed,
                                   conf.bandwidth(0),
                                   xDim,
                                   yDim,
                                   numChannels,
                                   sc,
                                   layerWhitener,
                                   conf.whitenerOffset,
                                   conf.insanity,
                                   conf.fastfood.contains(0),
                                   layerStride)
      var layerPooler = new MyPooler(layerPool, layerPool, identity, (x:DenseVector[Double]) => mean(x), sc)

      var layer =
        if (startLayer == 0) {

          if (layerPool == 1) {
            xDim = math.ceil(((xDim - layerPatch + 1)/layerStride)).toInt
            yDim = math.ceil(((yDim - layerPatch + 1)/layerStride)).toInt
          } else {
            xDim = math.ceil(((xDim - layerPatch + 1)/layerStride - layerPool/2.0)/layerPool).toInt
            yDim = math.ceil(((yDim - layerPatch + 1)/layerStride - layerPool/2.0)/layerPool).toInt
          }
          ImageExtractor andThen layerConvolver andThen layerPooler
      } else {
          val metadata = train.first.image.metadata
          xDim = metadata.xDim
          yDim = metadata.yDim
          layerChannels = metadata.numChannels
          ImageExtractor
      }

    /* Layer 2 - N
     */
    println("START LAYER " + startLayer)
    for (i <- (startLayer + 1  until conf.layers)) {
      numChannels = conf.filters(i - 1)
      println(s"LAYER ${i + 1} input: ${xDim} x ${yDim} x ${numChannels}")

      layerPatch = conf.patch_sizes(i)
      layerPool = conf.pool(i)
      layerStride = conf.convStride.getOrElse(i, 1)
      layerChannels = numChannels
      layerInputFeatures = layerChannels*layerPatch*layerPatch
      layerOutputFeatures = conf.filters(i)

      layerWhitener =
        if (conf.whiten.contains(i)) {
          val layerPatchExtractor = new Windower(1, layerPatch)
            .andThen(ImageVectorizer.apply)
            .andThen(new Sampler(10000, conf.seed))
            val layerSamples = MatrixUtils.rowsToMatrix(layerPatchExtractor(layer(train)))
            println("Whitening Layer " + (i+1))
            Some(new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(layerSamples))
          } else {
            None
          }

          layerConvolver = new CC(layerInputFeatures,
            layerOutputFeatures,
            conf.seed,
            conf.bandwidth(i),
            xDim,
            yDim,
            numChannels,
            sc,
            layerWhitener,
            conf.whitenerOffset,
            conf.insanity,
            conf.fastfood.contains(i),
            layerStride)

          layerPooler = new MyPooler(layerPool, layerPool, identity, (x:DenseVector[Double]) => mean(x), sc)
          if (conf.pool(i) == 1) {
            layer =  layer andThen layerConvolver
          } else {
            layer =  layer andThen layerConvolver andThen layerPooler
          }

          if (layerPool == 1) {
            xDim = math.ceil(((xDim - layerPatch + 1)/layerStride)).toInt
            yDim = math.ceil(((yDim - layerPatch + 1)/layerStride)).toInt
          } else {
            xDim = math.ceil(((xDim - layerPatch + 1)/layerStride - layerPool/2.0)/layerPool).toInt
            yDim = math.ceil(((yDim - layerPatch + 1)/layerStride - layerPool/2.0)/layerPool).toInt
          }
          numChannels = conf.filters(conf.layers - 1)
    }

    println(s"Final Layer dimensions: ${xDim} x ${yDim} x ${numChannels}")

    val finalLayer = ImageVectorizer andThen new Cacher[DenseVector[Double]]

    val featurizer = layer andThen finalLayer
    var XTrain = featurizer(train)
    var XTest = featurizer(test)


    train.count()
    test.count()
    val trainStart = System.nanoTime()
    println("STARTING TRAIN CONVOLUTIONS")
    XTrain.count()
    println(s"FINISHED TRAIN CONVOLUTIONS in took ${timeElapsed(trainStart)} secs")
    if (conf.saveFeatures) {
      println("Saving TRAIN Features")
     if (conf.float.contains(conf.layers - 1)) {
     XTrain.zip(LabelExtractor(train)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
     s"${conf.featureDir}/ckn_${featureId}_train_features")
     } else {
      XTrain.zip(LabelExtractor(train)).map(xy => xy._1.toArray.mkString(",") + "," + xy._2).saveAsTextFile(
     s"${conf.featureDir}/ckn_${featureId}_train_features")
    }
     println("Finished saving TRAIN Features")
    }
    val testStart = System.nanoTime()
    println("STARTING TEST CONVOLUTIONS")
    XTest.count()
    println(s"FINISHED TEST CONVOLUTIONS in took ${timeElapsed(testStart)} secs")
    if (conf.saveFeatures) {
      if (conf.float.contains(conf.layers - 1)) {
        XTest.zip(LabelExtractor(test)).map(xy => xy._1.map(_.toFloat).toArray.mkString(",") + "," + xy._2).saveAsTextFile(
          s"${conf.featureDir}/ckn_${featureId}_test_features")
      } else {
      XTest.zip(LabelExtractor(test)).map(xy => xy._1.toArray.mkString(",") + "," + xy._2).saveAsTextFile(
        s"${conf.featureDir}/ckn_${featureId}_test_features")
      }
    }
    println("FIRST 4 FEATURES" + XTrain.take(4).map(_.slice(0,4)).mkString("\n"))
    println("Feature Dim: " + XTrain.first.size)



      val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)
      val yTrain = labelVectorizer(LabelExtractor(train))
      val yTest = labelVectorizer(LabelExtractor(test))

      val zippedTrain = XTrain.zip(yTrain).coalesce(sc.defaultParallelism)
      val model = 
      if (conf.solverWeight == 0) {
        new BlockLeastSquaresEstimator(conf.blockSize, conf.numIters, conf.reg).fit(XTrain, yTrain)
      } else {
        new BlockWeightedLeastSquaresEstimator(conf.blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)
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
    new ZCAWhitener(whitener, means)
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
