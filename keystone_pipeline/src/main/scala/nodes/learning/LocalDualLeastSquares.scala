package nodes.learning

import nodes.util.VectorSplitter
import workflow.{WeightedNode, LabelEstimator}

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{Transpose => _, _}
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
    val C = model
    println(s"DUAL MODEL  ${C.rows} x ${C.cols}")
    /* KLUDGE */
    override def apply(test: RDD[DenseVector[Double]]) = {
      test
    }

    override def apply(test: DenseVector[Double]) = {
      val t = new DenseVector(train.map(test.t * _).collect())
      model.t * t
    }
  }

class LocalDualLeastSquaresEstimator(blockSize: Int, lambda: Double)
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
        val n = trainingFeatures.count().toInt
        val d = trainingFeatures.first.size
        val k = trainingLabels.first.size
        val y = MatrixUtils.rowsToMatrix(trainingLabels.collect())

        ptr = extLib.NewDualLeastSquaresEstimator(n, k, d, lambda)

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
        val XB = trainingFeatures.map(x => x(i until i+blockSize))
        val X:DenseMatrix[Double] = MatrixUtils.rowsToMatrix(XB.collect())
        println("X Collected")
        extLib.DualLeastSquaresEstimatorAccumulateGram(ptr, X.data)
        i += blockSize;
      }
      println(s"Gram matrix accumulation took ${timeElapsed(gramStart)} secs")
  }

def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9

}

