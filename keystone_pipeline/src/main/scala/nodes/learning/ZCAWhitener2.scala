package nodes.learning

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.spark.rdd.RDD
import org.netlib.util.intW
import pipelines._
import workflow.{Transformer, Estimator}

class ZCAWhitener(val whitener: DenseMatrix[Double], val means: DenseVector[Double], val samples: DenseMatrix[Double])
  extends Transformer[DenseMatrix[Double],DenseMatrix[Double]] {

  def apply(in: DenseMatrix[Double]): DenseMatrix[Double] = {
    (in(*, ::) - means) * whitener
  }
}

/**
  * Computes a ZCA Whitener, which is intended to rotate an input dataset to identity covariance.
  * The "Z" in ZCA Whitening means that the solution will be as close to the original dataset as possible while having
  * this identity covariance property.
  *
  * See here for more details:
  * http://ufldl.stanford.edu/wiki/index.php/Whitening
  *
  * @param eps Regularization Parameter
  */
class ZCAWhitenerEstimator2(val eps: Double = 0.1)
  extends Estimator[DenseMatrix[Double],DenseMatrix[Double]] {

  def fit(in: RDD[DenseMatrix[Double]]): ZCAWhitener = {
    fitSingle(in.first)
  }

  def fitSingle(in: DenseMatrix[Double]): ZCAWhitener = {
    val means = (mean(in(::, *))).toDenseVector
    val whiteningMeans = mean(in, Axis._0)

    val X = in
    val covMatrix = 1.0/(X.rows) * (X.t * X) - (whiteningMeans.t * whiteningMeans)
    val es  = eigSym(covMatrix)
    val E = es.eigenvalues
    val V = es.eigenvectors
    val invSqrtEvals = diag(sqrt(E :+ eps) :^ -1.0)
    val whitener = V * invSqrtEvals * V.t
    new ZCAWhitener(whitener, means, in)
  }
}


