package pipelines
import scala.reflect.{BeanProperty, ClassTag}

class CKMConf {
  @BeanProperty var  dataset: String = "imagenet-small"
  @BeanProperty var  expid: String = "imagenet-small-run"
  @BeanProperty var  mode: String = "scala"
  @BeanProperty var  seed: Int = 0
  @BeanProperty var  layers: Int = 1
  /* Mandatory Architechture params must be layers long*/
  @BeanProperty var  filters: Array[Int] = Array(1)
  @BeanProperty var  bandwidth : Array[Double] = Array(1.8)
  @BeanProperty var  pool: Array[Int] = Array(2)
  @BeanProperty var  patch_sizes: Array[Int] = Array(5)

  /* Optional Bells and whistles */

 /* Whether the output of this layer should be serialized as float */
  @BeanProperty var  float: Array[Int] = Array()

 /* Whether this convolution layer should be zero padded*/
  @BeanProperty var  zeroPad: Array[Int] = Array()

  /* Whether to use FWHT as opposed to regular matrix multiply */
  @BeanProperty var  fastfood: Array[Int] = Array()

  /* If stride is not provided default stride of 1 will be used */
  @BeanProperty var  convStride: Map[Int, Int] = Map()

  /* If stride is not provided default stride of poolSize will be used (for that layer) */
 /* TODO: THIS IS IGNORED RIGHT NOW */
  @BeanProperty var  poolStride: Array[Int] = Array(2)

  @BeanProperty var  loss: String = "WeightedLeastSquares"
  @BeanProperty var  reg: Double = 0.001
  @BeanProperty var  numClasses: Int = 1000
  @BeanProperty var  yarn: Boolean = true
  @BeanProperty var  solverWeight: Double = 0
  @BeanProperty var  kernelGamma: Double = 5e-5
  @BeanProperty var  blockSize: Int = 4000
  @BeanProperty var  numIters: Int = 1
  @BeanProperty var  whiten: Array[Int] = Array()
  @BeanProperty var  whitenerValue: Double =  0.1
  @BeanProperty var  whitenerOffset: Double = 0.001
  @BeanProperty var  solve: Boolean = true
  @BeanProperty var  solver: String = "BlockWeightedLeastSquares"
  @BeanProperty var  insanity: Boolean = false
  @BeanProperty var  saveFeatures: Boolean = false
  @BeanProperty var  saveModel: Boolean = false
  @BeanProperty var  checkpointDir: String = "/tmp/spark-checkpoint"
  @BeanProperty var  augment: Boolean = false
  @BeanProperty var  augmentPatchSize: Int = 24
  @BeanProperty var  augmentType: String = "random"
  @BeanProperty var  featureDir: String = "/"
  @BeanProperty var  labelDir: String = "/"
  @BeanProperty var  modelDir: String = "/tmp"
  @BeanProperty var  loadWhitener: Boolean = false
  @BeanProperty var  loadLayer: Boolean = false
  @BeanProperty var  hashFeatureId: Boolean = false
  @BeanProperty var  layerToLoad: Int = 0
}

object CKMConf { val LEGACY_CUTOFF: Int = 1250

  def genFeatureId(conf: CKMConf, legacy:Boolean = false) = {
    /* Any random seed below 1250 is considered legacy mode */
   val featureId =
     if (legacy) {
       conf.seed + "_" +
       conf.dataset + "_" +
       conf.expid  + "_" +
       conf.layers + "_" +
       conf.patch_sizes.mkString("-") + "_" +
       conf.bandwidth.mkString("-") + "_" +
       conf.pool.mkString("-") + "_" +
       conf.poolStride.mkString("-") + "_" +
       conf.filters.mkString("-")
     } else {
       val fastFood = if (conf.fastfood.filter(_ < conf.layers).size != 0 ) "ff_" + conf.fastfood.filter(_ < conf.layers).mkString("-") + "_"  else ""
       val augment = if (conf.augment) "Augment_" else ""
       val float = if (conf.float.filter(_ < conf.layers).size != 0 ) "ff_" + conf.float.filter(_ < conf.layers).mkString("-") + "_"  else ""
       val zeroPad = if (conf.zeroPad.filter(_ < conf.layers).size != 0 ) "ff_" + conf.zeroPad.filter(_ < conf.layers).mkString("-") + "_"  else ""
       conf.seed + "_" +
       conf.dataset + "_" +
       conf.layers + "_" +
       float +
       fastFood +
       zeroPad +
       augment +
       conf.patch_sizes.slice(0,conf.layers).mkString("-") + "_" +
       conf.convStride.keys.filter(_ < conf.layers).mkString("-") + "_" +
       conf.bandwidth.slice(0,conf.layers).mkString("-") + "_" +
       conf.pool.slice(0,conf.layers).mkString("-") + "_" +
       conf.poolStride.filter(_ < conf.layers).mkString("-") + "_" +
       conf.filters.slice(0,conf.layers).mkString("-") + "_" +
       conf.whiten.filter(_ < conf.layers).mkString("-")
     }

     if(conf.hashFeatureId) {
       println("HASHING FEATURE ID " + featureId)
       Math.abs(featureId.hashCode()).toString()
     } else {
       featureId
     }
  }
}


