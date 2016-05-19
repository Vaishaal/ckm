package nodes.stats

import java.io.File

import breeze.linalg._
import breeze.stats._
import breeze.numerics._
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.scalatest.FunSuite
import utils._


class MyFastfoodSuite extends FunSuite {
  test("Sanity test: RKS should be close to gaussian kernel") {
    val numPatches = 8*8
    val patchSize = 2
    val imgChannels = 1
    val numInputFeatures = patchSize*patchSize*imgChannels
    val numOutputFeatures = 8192
    val sigma = 0.1

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(3)))
    val gaussian = new Gaussian(0, 1)
    val gaussian2 = new Gaussian(17, 32)
    val uniform = new Uniform(0, 1)(randBasis)
    val patchMat = DenseMatrix.rand(numPatches, numInputFeatures, gaussian2)

    val w = DenseMatrix.rand(numInputFeatures, numOutputFeatures, gaussian) :* 1.0/(sqrt(2)*sigma)
    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)

    /* Compute gaussian gram matrix */
    val X2: DenseVector[Double] = sum(patchMat :* patchMat, Axis._1)
    println("NORMS SIZE IS " + X2.size)
    val XXT = patchMat * patchMat.t
    XXT :*= -2.0
    XXT(::, *) :+= X2
    XXT(*, ::) :+= X2

    val K0 =  -1 * 1.0/(2*sigma*sigma) * XXT
    val gaussianGramMatrix = exp(K0)

    val randomProduct = patchMat * w
    val outFeatures =  sqrt(2.0/numOutputFeatures) * cos(randomProduct(*, ::) :+ phase)
    val outGramMatrix = outFeatures * outFeatures.t

    val errorMatrix = abs(outGramMatrix - gaussianGramMatrix)
    val avgError = 1.0/(outGramMatrix.size) * sum(errorMatrix)
    val maxError = max(errorMatrix)

    println("AVERAGE ERROR (RKS) IS " + avgError)
    println("MAX ERROR (RKS) IS " + maxError)
    assert(avgError <= 1e-2)
  }

  test("Fastfood scala should be close to gaussian kernel") {
    val numPatches = 8*8
    val patchSize = 2
    val imgChannels = 1
    val numInputFeatures = patchSize*patchSize*imgChannels
    val numOutputFeatures = 8192
    val sigma = 0.1

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(3)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)(randBasis)
    val gaussian2 = new Gaussian(17, 32)
    val patchMat = DenseMatrix.rand(numPatches, numInputFeatures, gaussian2)

    val gaussian3 = new Gaussian(0, 1)
    val wf = DenseVector.rand(numOutputFeatures, gaussian)
    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val ff = new Fastfood(wf, phase, numOutputFeatures, 0, sigma)
    val ffOutArray  = MatrixUtils.matrixToRowArray(patchMat).map(ff(_))
    val ffOut = MatrixUtils.rowsToMatrix(ffOutArray)
    val ffOutFeatures =  sqrt(2.0/numOutputFeatures) * cos(ffOut(*, ::) :+ phase)
    val ffOutGramMatrix = ffOutFeatures * ffOutFeatures.t

    /* Compute gaussian gram matrix */
    val X2: DenseVector[Double] = sum(patchMat :* patchMat, Axis._1)
    val XXT = patchMat * patchMat.t
    XXT :*= -2.0
    XXT(::, *) :+= X2
    XXT(*, ::) :+= X2
    val K0 =  -1 * 1.0/(2*sigma*sigma) * XXT
    val gaussianGramMatrix = exp(K0)


    val errorMatrix = abs(ffOutGramMatrix - gaussianGramMatrix)
    val avgError = 1.0/(ffOutGramMatrix.size) * sum(errorMatrix)
    val maxError = max(errorMatrix)
    val maxIndex = argmax(errorMatrix)
    println("AVERAGE ERROR (FF) IS " + avgError)
    println("MAX ERROR (FF) IS " + maxError)
    assert(maxError <= 1e-1)
    }

  test("Fastfood Batch scala should be close to gaussian kernel") {
    val numPatches = 8*8
    val patchSize = 2
    val imgChannels = 1
    val numInputFeatures = patchSize*patchSize*imgChannels
    val numOutputFeatures = 8192
    val sigma = 0.1

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(3)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)(randBasis)
    val gaussian2 = new Gaussian(17, 32)
    val patchMat = DenseMatrix.rand(numPatches, numInputFeatures, gaussian2)

    val gaussian3 = new Gaussian(0, 1)
    val wf = DenseVector.rand(numOutputFeatures, gaussian)
    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val ff = new FastfoodBatch(wf, phase, numOutputFeatures, 0, sigma)
    val ffOut = ff(patchMat)
    val ffOutFeatures =  sqrt(2.0/numOutputFeatures) * cos(ffOut(*, ::) :+ phase)
    val ffOutGramMatrix = ffOutFeatures * ffOutFeatures.t

    /* Compute gaussian gram matrix */
    val X2: DenseVector[Double] = sum(patchMat :* patchMat, Axis._1)
    val XXT = patchMat * patchMat.t
    XXT :*= -2.0
    XXT(::, *) :+= X2
    XXT(*, ::) :+= X2
    val K0 =  -1 * 1.0/(2*sigma*sigma) * XXT
    val gaussianGramMatrix = exp(K0)


    val errorMatrix = abs(ffOutGramMatrix - gaussianGramMatrix)
    val avgError = 1.0/(ffOutGramMatrix.size) * sum(errorMatrix)
    val maxError = max(errorMatrix)
    val maxIndex = argmax(errorMatrix)
    println("AVERAGE ERROR (FF BATCH) IS " + avgError)
    println("MAX ERROR (FF BATCH) IS " + maxError)
    assert(maxError <= 1e-1)
    }
}

