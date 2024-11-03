import com.typesafe.config.{Config, ConfigFactory}
import Constants.TRAINING_DATA_PATH
import Preprocessor.{sentTokenize, wordTokenize}

import java.nio.file.{Files, Paths}

object TrainingData {
  lazy val asString: String = new String(Files.readAllBytes(Paths.get(TRAINING_DATA_PATH)))
  lazy val asSentences: Array[String] = sentTokenize(asString)
  lazy val asWords: Array[String] = wordTokenize(asString)
}
