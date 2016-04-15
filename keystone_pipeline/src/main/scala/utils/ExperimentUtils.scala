package utils

//import awscala._, s3._
import java.io._

object ExperimentUtils { 
  // From http://stackoverflow.com/a/11458240/1073963
  def cut[A](xs: Seq[A], n: Int) = {
    val m = xs.length
    val targets = (0 to n).map{x => math.round((x.toDouble*m)/n).toInt}
    def snip(xs: Seq[A], ns: Seq[Int], got: Vector[Seq[A]]): Vector[Seq[A]] = {
      if (ns.length<2) got
      else {
        val (i,j) = (ns.head, ns.tail.head)
        snip(xs.drop(j-i), ns.tail, got :+ xs.take(j-i))
      }
    }
    snip(xs, targets, Vector.empty)
  }

  // def s3KeyToInputStream(bucketName: String,
  //   key: String, s3conn : S3) : InputStream = {
  //   /* 
  //    Convenience function to take a s3 bucket name and a s3 key
  //    and return the input stream. 
  //    */ 
  // 
  //   val bucket = s3conn.bucket(bucketName)
  // 
  // 
  //   val RETRIES = 3
  //   var current_retry = 0
  // 
  //   var s3obj  = bucket.get.getObject(key)(s3conn)
  // 
  //   while(s3obj.isEmpty) {
  //     Thread sleep 1000
  //     val s3obj  = bucket.get.getObject(key)(s3conn)
  //     if (current_retry > RETRIES) {
  //       throw new FileNotFoundException(s"Could not find $key in $bucketName and too many retries")
  //     }
  //     current_retry += 1
  //   }
  // 
  //   val s = s3obj.getOrElse(throw new FileNotFoundException(s"Could not find $key in $bucketName"))
  //   s.getObjectContent.asInstanceOf[InputStream]
  // 
  // }

  def saveObject[A](obj : A, filename: String) = {
    val file = new FileOutputStream(filename)
    val buffer = new BufferedOutputStream(file)
    val output = new ObjectOutputStream(buffer)

    output.writeObject(obj)
    output.close

  }

  def loadObject[A](filename : String) : A = { 

    val streamIn = new FileInputStream(filename)
    val objectinputstream = new ObjectInputStream(streamIn)
    objectinputstream.readObject().asInstanceOf[A]
  }

  def saveRestoreState[T](filename: String, save: Boolean) : ( => T) => T = {
    def helper(createObj : => T) : T = {
      if(save) {
        val obj = createObj
        saveObject(obj, filename)
        obj
      } else {
        loadObject[T](filename)
      }
    }
    helper
  }

}
