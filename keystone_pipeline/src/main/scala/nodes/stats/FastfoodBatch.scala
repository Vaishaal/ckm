package nodes.stats

import breeze.linalg._
import breeze.numerics._
import breeze.numerics.cos
import breeze.stats.distributions.{Rand, ChiSquared, Bernoulli}
import breeze.stats.mean
import org.apache.spark.rdd.RDD
import utils.{MatrixUtils, FWHT}
import workflow.Transformer
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import utils.external.NativeRoutines




class FastfoodBatch(
  val g: DenseVector[Double], // should be out long
  val b: DenseVector[Double], // should be out long
  val out: Int, // Num output features
  val seed: Int = 10 // rng seed
  ) // should be numOutputFeatures by 1
  extends Transformer[DenseMatrix[Double], DenseMatrix[Double]] {

  assert(g.size == out)
  assert(FWHT.isPower2(out))
  @transient lazy val extLib = new NativeRoutines()

  implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
  var B = DenseVector.rand(out, new Bernoulli(0.5, randBasis)).map(if (_) -1.0 else 1.0)
  val P:IndexedSeq[Int] = randBasis.permutation(out).draw()
  val Gnorm:Double = pow(norm(g), -0.5)
  val S = (DenseVector.rand(out, ChiSquared(out)) :^ 0.5) * Gnorm

  override def apply(in: DenseMatrix[Double]): DenseMatrix[Double] =  {
    assert(FWHT.isPower2(in.cols))
    val outArray = extLib.fastfood(g.data, B.data, b.data, S.data, in.t.toArray, seed, out, in.cols, in.rows)
    println(outArray.size)
    (new DenseMatrix(out, in.rows, outArray)).t
  }
}

