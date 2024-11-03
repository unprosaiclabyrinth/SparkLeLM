ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.19"

lazy val root = (project in file("."))
  .settings(
    name := "SparkLeLM"
  )

run / fork := true

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.3",
  "org.apache.hadoop" % "hadoop-client" % "3.3.4",

  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
  "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-M2.1" exclude("org.apache.spark", "spark-core_2.10"),

  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
  "org.datavec" % "datavec-api" % "1.0.0-M2.1",

  "com.typesafe" % "config" % "1.4.3",
  "ch.qos.logback" % "logback-classic" % "1.5.6",
  "org.scalatest" %% "scalatest" % "3.2.19" % Test
)
