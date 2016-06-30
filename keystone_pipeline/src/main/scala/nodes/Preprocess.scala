package nodes.images

import pipelines._
import nodes.learning._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.{mean, median}
import nodes.images._
import utils.{Image, MatrixUtils, Stats, ImageMetadata, LabeledImage, RowMajorArrayVectorizedImage, ChannelMajorArrayVectorizedImage}
import nodes.stats.{Sampler, StandardScaler}
import org.apache.spark.rdd.RDD
import workflow.Transformer

case class Preprocess(train: RDD[Image], sampleFrac: Double = 0.1)
  extends Transformer[Image, Image] {

  println("TRAINING PRE PROCESSING UNIT")
  val count = train.count()
  val metadata = train.first.metadata
  val trainArray:RDD[DenseVector[Double]] = train.map(x => new DenseVector(x.toArray))

  val min_divisor = 1e-8;
  val scale = 55.0
  val normalized = trainArray.map { x =>
    val xCentered = x - mean(x)
    var xNorm = norm(xCentered)/scale
    if (xNorm <= min_divisor) {
      xNorm = 1
    }
    val xNormalized = xCentered/xNorm
    xNormalized
  }

  @transient
  val X = MatrixUtils.rowsToMatrix(normalized.sample(false, sampleFrac).collect())
  val whiteningMeans = mean(X, Axis._0)
  val covMatrix = 1.0/(X.rows) * (X.t * X) - (whiteningMeans.t * whiteningMeans)
  val es  = eigSym(covMatrix)
  val E = es.eigenvalues
  val V = es.eigenvectors
  val invSqrtEvals = diag(sqrt(E :+ 0.1) :^ -1.0)
  val whitening = V * invSqrtEvals * V.t


  def apply(img: Image): Image = {
    /* Channel Major array */
    val imArray = new DenseVector(img.toArray)

    /* Subtract da mean */
    imArray :-= mean(imArray)

    var imNorm = norm(imArray)/scale
    if (imNorm <= min_divisor) {
      imNorm = 1
    }

    /* Normalize da image */
    imArray :/= imNorm

    /* ZCA */
   val whitenedIm = whitening * (imArray :- whiteningMeans.toDenseVector)
   val res = ChannelMajorArrayVectorizedImage(whitenedIm.data, img.metadata)
   res
  }
}
