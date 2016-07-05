package nodes.learning

import org.scalatest.FunSuite

import java.io._

import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.stats._
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import nodes.util._

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import pipelines._
import utils.{Stats, MatrixUtils, TestUtils}

import org.apache.log4j.Level
import org.apache.log4j.Logger

class LocalDualLeastSquaresEstimatorSuite extends FunSuite with Logging with PipelineContext {

  test("Local Dual Least Squares Solver should match primal") {
    val n = 500
    val d = 1024
    val k = 1000
    val lambda = 0.0
    val seed = 0

    val conf = new SparkConf().setAppName("LocalDualLeastSquaresEstimatorSuite")
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
    conf.remove("spark.jars")
    conf.set("spark.master", "local[16]")
    conf.set("spark.driver.maxResultSize", "0")
    sc = new SparkContext(conf)

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(k)
    val data = sc.parallelize((0 until n)).map(x => DenseVector.rand[Double](d)).cache()
    val rand = scala.util.Random
    val labels = labelVectorizer(sc.parallelize((0 until n).map { x => rand.nextInt(k)
    })).cache()
    val libPathProperty = System.getProperty("java.library.path");
    println(libPathProperty);

    /* first fit a linear model in primal */
   data.count()
   labels.count()

   val primalSolver = new BlockLeastSquaresEstimator(d, 1, 0.0)
   val primalModel = primalSolver.fit(data, labels)
   val primalPredictions = MaxClassifier(primalModel.apply(data))

   val dualSolver = new LocalDualLeastSquaresEstimator(d, 0.0)
   val dualModel = dualSolver.fit(data, labels)
   val dualPredictions = MaxClassifier(dualModel.apply(data))

   val primalDualMatch = primalPredictions.zip(dualPredictions).map( x => x._1 == x._2).reduce(_ && _)
   assert(primalDualMatch)
   sc.stop()
  }
}
