package utils.external

class NativeRoutines extends Serializable {
  System.loadLibrary("NativeRoutines") // This will load libNativeRoutines.{so,dylib} from the library path.

  /**
   * Sum-Pools and Symmetric rectifies an image
   * Input image must be in column major vectorized format. (See image.scala in
   * keystone-ml for more details)
   */

  @native
  def poolAndRectify(stride: Int, poolSize: Int,
    numChannels: Int = 3, xDim: Int, yDim: Int,
    maxVal: Double = 0.0, alpha: Double = 0.0, image: Array[Double]): Array[Double]

  @native
  def fwht(in: Array[Double], length: Int) : Array[Double]
}
