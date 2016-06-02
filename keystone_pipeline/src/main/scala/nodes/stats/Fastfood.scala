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




class Fastfood(
  val g: DenseVector[Double], // should be out long
  val b: DenseVector[Double], // should be out long
  val out: Int, // Num output features
  val seed: Int = 10, // rng seed
  val sigma: Double = 1.0 // rng seed
  ) // should be numOutputFeatures by 1
  extends Transformer[DenseVector[Double], DenseVector[Double]] {

  assert(g.size == out)
  assert(FWHT.isPower2(out))
  implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
  var B = DenseVector.rand(out, new Bernoulli(0.5, randBasis)).map(if (_) -1.0 else 1.0)
  val P:IndexedSeq[Int] = randBasis.permutation(out).draw()
  val S = (DenseVector.rand(out, ChiSquared(out)) :^ 0.5) :*  1.0/norm(g)
  override def apply(in: DenseVector[Double]): DenseVector[Double] =  {
    val d = FWHT.nextPower2(in.size).toInt
    val inPad = padRight(in, d, 0.0)
    val blocks =
      for (i <- List.range(0, out/d)) yield processBlock(inPad, g(i*d until (i+1)*d), B(i*d until (i+1)*d), P.slice(i*d,(i+1)*d).map(_ % d), S(i*d until (i+1)*d))
    var outVector = DenseVector.vertcat(blocks:_*)
    outVector
  }

  def processBlock(in: DenseVector[Double], G: DenseVector[Double], B: DenseVector[Double], P: IndexedSeq[Int], S: DenseVector[Double]): DenseVector[Double] = {
    val d = in.size
    var W:DenseVector[Double] = FWHT(B :* in)
    1.0/sigma :* S :* FWHT(G :* W)
  }
}

