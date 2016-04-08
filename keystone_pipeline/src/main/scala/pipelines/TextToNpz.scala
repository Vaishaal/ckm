package pipelines

import java.net.URI

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}

import scala.io.Source

import breeze.linalg._
import breeze.numerics._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import nodes._
import pipelines._
import utils.{ImageMetadata, Stats, Image, ImageUtils}

object TextToNpz extends Serializable {

  def main(args: Array[String]) {
    if (args.length < 1) {
      println("Usage: TextToNpz featureids outlocation")
      System.exit(0)
    }
    val featureListFile  = args(0)
    val outLocation = args(1)
    val fileNames = Source.fromFile(featureListFile).getLines.toArray

    val conf = new SparkConf()
      .setAppName("ImageNetBrew")
      .setJars(SparkContext.jarOfObject(this).toSeq)
      .set("spark.hadoop.validateOutputSpecs", "false") // overwrite hadoop files

    val sc = new SparkContext(conf)
    sc.addSparkListener(new org.apache.spark.scheduler.JobLogger())

    Thread.sleep(10000)
    val featureFiles = sc.parallelize(fileNames).cache()
    featureFiles.count()
    val count = featureFiles.pipe(s"/home/eecs/vaishaal/ckm/keystone_pipeline/bin/text-to-npz.sh ${outLocation}").count()

    println(s"COUNT ${count}")

    println("DONE")
    sc.stop()
    sys.exit(0)
  }
}
