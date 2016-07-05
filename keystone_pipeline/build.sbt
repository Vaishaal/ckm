import AssemblyKeys._

assemblySettings

name := "ckm"

version := "0.1"

organization := "edu.berkeley.cs.amplab"

scalaVersion := "2.10.4"

javacOptions ++= Seq("-source", "1.7", "-target", "1.7")

parallelExecution in Test := false

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "1.9.1" % "test",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  // the visualization library is distributed separately as well.
  // It depends on LGPL code.
  "org.scalanlp" %% "breeze-viz" % "0.11.2",
  "io.scif" % "scifio" % "0.27.0",
  "net.jafama" % "jafama" % "2.1.0"
)

libraryDependencies ++= Seq(
  "edu.arizona.sista" % "processors" % "3.0" exclude("ch.qos.logback", "logback-classic"),
  "edu.arizona.sista" % "processors" % "3.0" classifier "models",
  "org.slf4j" % "slf4j-api" % "1.7.2",
  "org.slf4j" % "slf4j-log4j12" % "1.7.2",
  "org.scalatest" %% "scalatest" % "1.9.1" % "test",
  "org.mockito" % "mockito-core" % "1.8.5",
  "org.apache.commons" % "commons-compress" % "1.7",
  "commons-io" % "commons-io" % "2.4",
  "org.scalanlp" % "breeze_2.10" % "0.11.2",
  "com.google.guava" % "guava" % "14.0.1",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
  "com.github.scopt" %% "scopt" % "3.3.0"
)


fork := true

{
  val defaultSparkVersion = "1.5.2"
  val sparkVersion =
    scala.util.Properties.envOrElse("SPARK_VERSION", defaultSparkVersion)
  val excludeHadoop = ExclusionRule(organization = "org.apache.hadoop")
  val excludeSpark = ExclusionRule(organization = "org.apache.spark")
  libraryDependencies ++= Seq(
    "org.apache.spark" % "spark-core_2.10" % sparkVersion excludeAll(excludeHadoop),
    "org.apache.spark" % "spark-mllib_2.10" % sparkVersion excludeAll(excludeHadoop),
    "org.apache.spark" % "spark-sql_2.10" % sparkVersion excludeAll(excludeHadoop),
    "edu.berkeley.cs.amplab" % "keystoneml_2.10" % "0.3.1-SNAPSHOT" excludeAll(excludeHadoop, excludeSpark), 
    "org.yaml" % "snakeyaml" % "1.16",
    "org.apache.commons" % "commons-csv" % "1.2",
    "com.amazonaws" % "aws-java-sdk" % "1.9.40",
    "com.github.seratch" %% "awscala" % "0.5.+",
    "net.jafama" % "jafama" % "2.1.0"
  )
}

libraryDependencies ++= Seq(
    "com.github.melrief" %% "purecsv" % "0.0.4"
  , compilerPlugin("org.scalamacros" % "paradise" % "2.0.1" cross CrossVersion.full)
)

dependencyOverrides ++= Set(
  "com.amazonaws" % "aws-java-sdk" % "1.9.40",
  "com.amazonaws" % "aws-java-sdk-core" % "1.9.40",
  "com.amazonaws" % "aws-java-sdk-s3" % "1.9.40"
)

{
  val defaultHadoopVersion = "2.6.0-cdh5.5.1"
  val hadoopVersion =
    scala.util.Properties.envOrElse("SPARK_HADOOP_VERSION", defaultHadoopVersion)
  libraryDependencies ++= Seq("org.apache.hadoop" % "hadoop-aws" % hadoopVersion,
    "org.apache.hadoop" % "hadoop-client" % hadoopVersion)
}

resolvers ++= Seq(
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Cloudera Repository" at "https://repository.cloudera.com/artifactory/cloudera-repos/",
  "Spray" at "http://repo.spray.cc",
  "Bintray" at "http://dl.bintray.com/jai-imageio/maven/",
  "ImageJ Public Maven Repo" at "http://maven.imagej.net/content/groups/public/"
)

resolvers += Resolver.sonatypeRepo("public")

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
  {
    case PathList("javax", "servlet", xs @ _*)               => MergeStrategy.first
    case PathList(ps @ _*) if ps.last endsWith ".html"       => MergeStrategy.first
    case "application.conf"                                  => MergeStrategy.concat
    case "reference.conf"                                    => MergeStrategy.concat
    case "log4j.properties"                                  => MergeStrategy.first
    case m if m.toLowerCase.endsWith("manifest.mf")          => MergeStrategy.discard
    case m if m.toLowerCase.matches("meta-inf.*\\.sf$")      => MergeStrategy.discard
    case m if m.toLowerCase.startsWith("meta-inf/services/") => MergeStrategy.filterDistinctLines
      // This line required for SCIFIO stuff to work correctly, due to some dependency injection stuff
    case "META-INF/json/org.scijava.plugin.Plugin" => MergeStrategy.concat
    case _ => MergeStrategy.first
  }
}

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false, includeDependency = false)

test in assembly := {}

