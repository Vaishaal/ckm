package nodes.stats

import java.io.File

import breeze.linalg._
import breeze.stats._
import breeze.numerics._
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.scalatest.FunSuite
import utils.Stats


class MyFastfoodSuite extends FunSuite {
  test("Sanity test: RKS should be close to gaussian kernel") {
    val numPatches = 28*28
    val patchSize = 2
    val imgChannels = 1
    val numInputFeatures = patchSize*patchSize*imgChannels
    val numOutputFeatures = 512
    val bandwidth = 1e-8

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)(randBasis)
    val patchMat = DenseMatrix.rand(28*28, numInputFeatures, gaussian)
    val w = DenseMatrix.rand(numInputFeatures, numOutputFeatures, gaussian) :* sqrt(2*bandwidth)
    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)

    /* Compute gaussian gram matrix */
    val X2: DenseVector[Double] = sum(patchMat :* patchMat, Axis._1)
    println("NORMS SIZE IS " + X2.size)
    val XXT = patchMat * patchMat.t
    XXT :*= -2.0
    XXT(::, *) :+= X2
    XXT(*, ::) :+= X2

    val K0 =  -1 * bandwidth * XXT
    val gaussianGramMatrix = exp(K0)

    println("gaussian GRAM SIZE " + gaussianGramMatrix.size)

    println("FIRST GAUSSIAN " + gaussianGramMatrix(121,74))

    val randomProduct = patchMat * w
    val outFeatures =  sqrt(2.0/numOutputFeatures) * cos(randomProduct(*, ::) :+ phase)
    val outGramMatrix = outFeatures * outFeatures.t
    println("cos GRAM SIZE " + outGramMatrix.size)
    println("FIRST COSINE " + outGramMatrix(121, 74))

    println(sum(abs(outGramMatrix - gaussianGramMatrix)))
    assert(Stats.aboutEq(outGramMatrix, gaussianGramMatrix, 1e-1))
  }

  test("Fastfood should be close to RKS") {
    val numPatches = 28*28
    val patchSize = 2
    val imgChannels = 1
    val numInputFeatures = patchSize*patchSize*imgChannels
    val numOutputFeatures = 8192
    val bandwidth = 1e-3

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)(randBasis)
    val patchMat = DenseMatrix.rand(28*28, numInputFeatures, gaussian)

    /* Compute gaussian gram matrix */
    val X2: DenseVector[Double] = sum(patchMat :* patchMat, Axis._1)
    println("NORMS SIZE IS " + X2.size)
    val XXT = patchMat * patchMat.t
    XXT :*= -2.0
    XXT(::, *) :+= X2
    XXT(*, ::) :+= X2
    val K0 =  -1 * bandwidth * XXT
    val gaussianGramMatrix = exp(K0)

    val w = DenseMatrix.rand(numInputFeatures, numOutputFeatures, gaussian) :* sqrt(2*bandwidth)
    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)

    val ff = new FastfoodBatch(w(0, ::).inner, phase, numOutputFeatures, 0)

    val randomProduct = patchMat * w
    val outFeatures =  sqrt(2.0/numOutputFeatures) * cos(randomProduct(*, ::) :+ phase)
    val outGramMatrix = outFeatures * outFeatures.t

    val ffOut = ff(patchMat)

    val ffOutFeatures =  sqrt(2.0/numOutputFeatures) * cos(ffOut(*, ::) :+ phase)
    val ffOutGramMatrix = ffOutFeatures * ffOutFeatures.t
    println("FIRST RBF " + gaussianGramMatrix(121, 74))
    println("FIRST RKS " + outGramMatrix(121, 74))
    println("FIRST FF " + ffOutGramMatrix(121, 74))
  }
}

