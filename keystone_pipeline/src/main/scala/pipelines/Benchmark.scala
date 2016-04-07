package pipelines

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import evaluation.{AugmentedExamplesEvaluator, MulticlassClassifierEvaluator}
import loaders.{CifarLoader, CifarLoader2, MnistLoader, SmallMnistLoader, ImageNetLoader}
import nodes.images._
import workflow.Transformer
import nodes.learning._
import nodes.stats.{StandardScaler, Sampler, SeededCosineRandomFeatures, BroadcastCosineRandomFeatures, CosineRandomFeatures}
import nodes.util.{Identity, Cacher, ClassLabelIndicatorsFromIntLabels, TopKClassifier, MaxClassifier, VectorCombiner}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pipelines.Logging
import scopt.OptionParser
import workflow.Pipeline
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.commons.math3.random.MersenneTwister
import utils.{Image, MatrixUtils, Stats, ImageMetadata, LabeledImage, RowMajorArrayVectorizedImage, ChannelMajorArrayVectorizedImage}

import scala.reflect.{BeanProperty, ClassTag}
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor

import java.io.{File, BufferedWriter, FileWriter}
import utils.external.NativeRoutines
import scala.util.Random

object Benchmark extends Serializable with Logging {

case class BenchmarkParam(name: String, size: (Int, Int, Int), kernelSize: Int, numKernels: Int, poolSize: Int, poolStride: Int)

  def genData(x: Int, y: Int, z: Int, inorder: Boolean=false): Array[Double] = {
    if (!inorder) Array.fill(x*y*z)(Random.nextDouble) else (0 until x*y*z).map(_.toDouble).toArray
  }

  /** Generate a random `ChannelMajroArrayVectorizedImage` */
  def genChannelMajorArrayVectorizedImage(x: Int, y: Int, z: Int): ChannelMajorArrayVectorizedImage = {
    ChannelMajorArrayVectorizedImage(genData(x, y, z), ImageMetadata(x,y,z))
  }

  def convTime(x: Image, t: BenchmarkParam) = {
      val filters = DenseMatrix.rand[Double](t.numKernels, t.kernelSize*t.kernelSize*t.size._3)
      val conv = new Convolver(filters, x.metadata.xDim, x.metadata.yDim, x.metadata.numChannels, normalizePatches = false)

      val start = System.nanoTime
      val res = conv(x)
      val elapsed = System.nanoTime - start

      elapsed
    }

  def ckmTime(x: Image, t: BenchmarkParam) = {
      val filters = DenseMatrix.rand[Double](t.numKernels, t.kernelSize*t.kernelSize*t.size._3)
      val cc = new CC(3*25, t.numKernels, 0, 1, 256, 256, 3, None, 0.1, 14, false, false)
      val start = System.nanoTime
      val res = cc(x)
      val elapsed = System.nanoTime - start
      elapsed
    }

  def convolutionBench() = {
    val t = BenchmarkParam("imagenet", (256,256,3), 5, 2048, 14, 14)

    val res = for(
      iter <- 1 to 100) yield {
      val img = genChannelMajorArrayVectorizedImage(t.size._1, t.size._2, t.size._3) //Standard grayScale format.

      val flops = (t.size._1.toLong-t.kernelSize+1)*(t.size._2-t.kernelSize+1)*
        t.size._3*t.kernelSize*t.kernelSize*
        t.numKernels

      val t1 = convTime(img, t)

      logDebug(s"${t.name},$t1,$flops,${2.0*flops.toDouble/t1}")
      (t1, flops, (2.0*flops.toDouble/t1))
    }
    val flops = DenseVector(res.map(_._3):_*)
    val times = DenseVector(res.map(_._1):_*)
    val maxf = max(flops)
    val medf = median(flops)
    val stddevf = stddev(flops)

    val maxt = max(times)
    val medt = median(times)

    println(f"FLOPS $maxf%2.3f,$medf%2.3f,$stddevf%2.3f,")
    println(f"TIME $maxt%2.3f,$medt%2.3f,")
    medt
  }

  def ckmBench() = {
    val t = BenchmarkParam("imagenet", (256,256,3), 5, 1024, 14, 14)

    val res = for(
      iter <- 1 to 10) yield {
      val img = genChannelMajorArrayVectorizedImage(t.size._1, t.size._2, t.size._3) //Standard grayScale format.

      val flops = (t.size._1.toLong-t.kernelSize+1)*(t.size._2-t.kernelSize+1)*
        t.size._3*t.kernelSize*t.kernelSize*
        t.numKernels

      val t1 = ckmTime(img, t)

      logDebug(s"${t.name},$t1,$flops,${2.0*flops.toDouble/t1}")
      (t1, flops, (2.0*flops.toDouble/t1))
    }
    val flops = DenseVector(res.map(_._3):_*)
    val times = DenseVector(res.map(_._1):_*)
    val maxf = max(flops)
    val medf = median(flops)
    val stddevf = stddev(flops)

    val maxt = max(times)
    val medt = median(times)

    println(f"FLOPS $maxf%2.3f,$medf%2.3f,$stddevf%2.3f,")
    println(f"TIME $maxt%2.3f,$medt%2.3f,")
    medt
  }


  def main(args: Array[String]) = {
    println("Hello World")
    //val convTime = convolutionBench()

    val t = BenchmarkParam("imagenet", (256,256,3), 5, 1024, 14, 14)

    val img = genChannelMajorArrayVectorizedImage(t.size._1, t.size._2, t.size._3) //Standard grayScale format.
    var start = System.nanoTime()
    val extLib = new NativeRoutines()
    var i = 0.0
    for(iter <- 1 to 10e7.toInt)  {
      i += math.cos(1)
    }
    println(i)
    var end = System.nanoTime()
    println(s"Took ${end - start} ns")
    println(s"Took ${(end - start)/10e7} ns per cal")

    start = System.nanoTime()
    i = 0.0
    for(iter <- 1 to 10e7.toInt)  {
      i += extLib.cosine(1)
    }
    println(i)
    end = System.nanoTime()
    println(s"Took ${end - start} ns")
    println(s" native Took ${end - start} ns")
    println(s"Took ${(end - start)/10e7} ns per cal")

    //println(s"RATIO: ${ckmTime*1.0/convTime}")
  }
}
