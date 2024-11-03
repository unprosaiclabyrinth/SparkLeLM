import Constants.{SPARK_APP_NAME, SPARK_MASTER}
import org.apache.spark.sql.SparkSession

object SparkObj {
  lazy val spark: SparkSession = SparkSession.builder()
    .appName(SPARK_APP_NAME)
    .master(SPARK_MASTER)
    .getOrCreate
}
