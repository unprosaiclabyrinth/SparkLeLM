import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import java.nio.file._
import scala.io.Source

object FileIO {
  // Set up Hadoop configuration for S3
  private val conf = new Configuration()
  conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
  conf.set("fs.s3a.endpoint", "s3.amazonaws.com")
  conf.set("fs.defaultFS", "s3a://fa24cs441hw2")

  // Create the file system instance
  val fs: FileSystem = FileSystem.get(conf)

  // Read in a file as a string either from S3 or locally (specified by a boolean flag)
  def readFileContentAsString(uri: String, fromS3: Boolean = true): String = {
    if (fromS3) {
      val inStream = fs.open(new Path(uri))
      val content = Source.fromInputStream(inStream)("UTF-8").mkString
      inStream.close()
      content
    } else {
      new String(Files.readAllBytes(Paths.get(uri)))
    }
  }
}
