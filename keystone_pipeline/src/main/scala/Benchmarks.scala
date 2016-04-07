package pipelines

import utils.{ImageMetadata, ColumnMajorArrayVectorizedImage}
import nodes.images.{SymmetricRectifier, Pooler}
import nodes.images.{MyPooler, MyConvolver, Convolver, CCaP}

import breeze.linalg._
import breeze.stats._
import utils._

import scala.util.Random


/*

Remember to run a few times to let JIT work

java -cp target/scala-2.10/solarflares-assembly-0.1.jar:target/scala-2.10/solarflares-assembly-0.1-deps.jar pipelines.Benchmark BenchmarkName

 */
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, Image}
import workflow.Transformer

object Benchmarks {

  def convolverBench() = {

    println("Convolver Benchmark")

    val imgsizeX = 256
    val imgsizeY = 256
    val imgChan = 3
    val patchSize = 10
    val filterN = 1024

    val alpha = 1.0
    val metadata = ImageMetadata(imgsizeX, imgsizeY,  imgChan)
    val data = DenseVector.rand(imgsizeX*imgsizeY*imgChan)

    var myimg = ChannelMajorArrayVectorizedImage(data.toArray, metadata)


    var filters = DenseMatrix.rand(filterN, patchSize * patchSize * imgChan)

    val c1 = new Convolver(filters, imgsizeX, imgsizeY, imgChan)
    val c2 = new MyConvolver(filters, imgsizeX, imgsizeY, imgChan) 
    for (i <- 0 to 10) {

      val start1 = System.nanoTime()
      val res1 = c1.apply(myimg)
      val stop1 =     System.nanoTime()
      println(s"Time 1 was ${(stop1 - start1)/1e9}")

      val start2 = System.nanoTime()
      val res2 = c2.apply(myimg)
      val stop2 =     System.nanoTime()
      println(s"Time 2 was ${(stop2 - start2)/1e9}")

    }
  }
  def CCaPBench() = {

    println("CCaP Benchmark")
    /*
    val imgsizeX = 256
    val imgsizeY = 256
    val imgChan = 12
    val patchSize = 5
    val filterN = 2048

    val alpha = 1.0
    val poolSize = 14
    val metadata = ImageMetadata(imgsizeX, imgsizeY,  imgChan)
    val data = DenseVector.rand(imgsizeX*imgsizeY*imgChan)

    var myimg = ChannelMajorArrayVectorizedImage(data.toArray, metadata)


    var filters = DenseMatrix.rand(filterN, patchSize * patchSize * imgChan)


    val c1 = new MyConvolver(filters, imgsizeX, imgsizeY, imgChan, None, false).andThen(new MyPooler(poolSize, poolSize, 0.0, alpha))

    val c2 = new CCaP(filters, imgsizeX, imgsizeY, imgChan, poolSize, poolSize, 0.0, alpha, false)
    val res1 = c1.apply(myimg).toArray
    val res2 = c2.apply(myimg).toArray

    val delta = DenseVector(res1) - DenseVector(res2)
    val alldelta = sum(abs(delta))

    println(s"Array difference is ${alldelta}")
    var total1 = 0.0
    var total2 = 0.0
    val totalIters = 20
    for (iter <- 0 to totalIters) {
      val start1 = System.nanoTime()
      val res1 = c1.apply(myimg)
      val stop1 =     System.nanoTime()
      total1 += (stop1 - start1)/1e9

      val start2 = System.nanoTime()
      val res2 = c2.apply(myimg)
      val stop2 =     System.nanoTime()
      total2 += (stop2 - start2)/1e9

    }
    println(s"Average for regular convolution + poolRectify is ${total1/totalIters} seconds")
    println(s"Average for CCaP is ${total2/totalIters} seconds")
    */
  }

  def main(args: Array[String]) {
    if (args(0) == "convolverBench") {
      convolverBench()
    }

    if (args(0) == "CCaPBench") {
      CCaPBench()
    }
  }
}
