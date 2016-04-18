package pipelines
import scala.reflect.{BeanProperty, ClassTag}

class CKMConf {
  @BeanProperty var  dataset: String = "imagenet-small"
  @BeanProperty var  expid: String = "imagenet-small-run"
  @BeanProperty var  mode: String = "scala"
  @BeanProperty var  seed: Int = 0
  @BeanProperty var  layers: Int = 1
  @BeanProperty var  filters: Array[Int] = Array(1)
  @BeanProperty var  bandwidth : Array[Double] = Array(1.8)
  @BeanProperty var  patch_sizes: Array[Int] = Array(5)
  @BeanProperty var  loss: String = "WeightedLeastSquares"
  @BeanProperty var  reg: Double = 0.001
  @BeanProperty var  numClasses: Int = 10
  @BeanProperty var  yarn: Boolean = true
  @BeanProperty var  solverWeight: Double = 0
  @BeanProperty var  cosineSolver: Boolean = false
  @BeanProperty var  cosineFeatures: Int = 40000
  @BeanProperty var  cosineGamma: Double = 1e-8
  @BeanProperty var  kernelGamma: Double = 5e-5
  @BeanProperty var  blockSize: Int = 4000
  @BeanProperty var  numBlocks: Int = 2
  @BeanProperty var  numIters: Int = 2
  @BeanProperty var  whiten: Boolean = false
  @BeanProperty var  whitenerValue: Double =  0.1
  @BeanProperty var  whitenerOffset: Double = 0.001
  @BeanProperty var  solve: Boolean = true
  @BeanProperty var  solver: String = "linear"
  @BeanProperty var  insanity: Boolean = false
  @BeanProperty var  saveFeatures: Boolean = false
  @BeanProperty var  pool: Array[Int] = Array(2)
  @BeanProperty var  poolStride: Array[Int] = Array(2)
  @BeanProperty var  checkpointDir: String = "/tmp/spark-checkpoint"
  @BeanProperty var  augment: Boolean = false
  @BeanProperty var  augmentPatchSize: Int = 24
  @BeanProperty var  augmentType: String = "random"
  @BeanProperty var  fastfood: Boolean = false
  @BeanProperty var  featureDir: String = "/"
  @BeanProperty var  labelDir: String = "/"
  @BeanProperty var  modelDir: String = "/tmp"
  @BeanProperty var  loadWhitener: Boolean = false
  @BeanProperty var  loadLayer: Boolean = false
  @BeanProperty var  layerToLoad: Int = 0
}

