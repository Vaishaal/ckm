package nodes.learning

import nodes.util.VectorSplitter
import workflow.{WeightedNode, LabelEstimator}

import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._


import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._

import org.apache.spark.rdd.RDD
import org.apache.spark.HashPartitioner

import edu.berkeley.cs.amplab.mlmatrix.{RowPartition, NormalEquations, BlockCoordinateDescent, RowPartitionedMatrix}
import edu.berkeley.cs.amplab.mlmatrix.util.{Utils => MLMatrixUtils}

import nodes.stats.StandardScaler
import pipelines.Logging
import workflow._
import utils.{MatrixUtils, Stats}
import utils.external.NativeRoutines

class DualLeastSquaresModel(model: DenseMatrix[Double], train: RDD[DenseVector[Double]])
  extends Transformer[DenseVector[Double], DenseVector[Double]] {

    val d = train.first.size
    override def apply(test: DenseVector[Double]) = {
      val testX  = new DenseVector(train.map(x => x.t * test).collect())
      testX :/= (1.0*d)
      model * testX
    }
  }

class LocalDualLeastSquaresEstimator(lambda: Double, blockSize: Int)
  extends LabelEstimator[DenseVector[Double], DenseVector[Double], DenseVector[Double]] {

  @transient lazy val extLib = new NativeRoutines()
  var ptr:Long = 0

  /*
   * Split features into appropriate blocks and fit a weighted least squares model.
   *
   * NOTE: This function makes multiple passes over the training data. Caching
   *
   * @trainingFeatures and @trainingLabels before calling this function is recommended.
   * @param trainingFeatures training data RDD
   * @param trainingLabels training labels RDD
   * @returns DualLeastSquares Model
   */
  override def fit(
      trainingFeatures: RDD[DenseVector[Double]],
      trainingLabels: RDD[DenseVector[Double]]
      ): DualLeastSquaresModel = {
        val d = trainingFeatures.first.size
        val n = trainingFeatures.count().toInt
        val k = trainingLabels.first.size
        val y = MatrixUtils.rowsToMatrix(trainingLabels.collect())

        /* Create the C++ object */
        ptr = extLib.NewDualLeastSquaresEstimator(n, k, d, lambda)

        /* Accumulate the gram matrix in pieces */
        accumulateGram(trainingFeatures)

        println("EIGEN SOLVE START ")
        val solveStart = System.nanoTime()
        val modelRaw = extLib.DualLeastSquaresEstimatorSolve(ptr,y.data)
        println(s"Solve took ${timeElapsed(solveStart)} secs")
        val model = new DenseMatrix(n, k, modelRaw)
        new DualLeastSquaresModel(model, trainingFeatures)
    }

  def accumulateGram(trainingFeatures: RDD[DenseVector[Double]]) {
      val n = trainingFeatures.count().toInt
      println(s"n :${n}")
      val d = trainingFeatures.first.size
      var i = 0;
      var block = 0;
      println("ACCUMULATING GRAM MATRIX")
      val gramStart = System.nanoTime()
      while (i < d) {
        println(s"Block ${block}")
        block += 1
        val X:DenseMatrix[Double] = MatrixUtils.rowsToMatrix(trainingFeatures.map(x => x(i until i+blockSize)).collect())
        extLib.DualLeastSquaresEstimatorAccumulateGram(ptr, X.data)
      }
      println(s"Gram matrix accumulation took ${timeElapsed(gramStart)} secs")
  }

def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9

}

