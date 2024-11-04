import Constants.TRAINING_DATA_URI
import Preprocessor.{sentTokenize, wordTokenize}

object TrainingData {
  lazy val asString: String = FileIO.readFileContentAsString(TRAINING_DATA_URI)
  lazy val asSentences: Array[String] = sentTokenize(asString)
  lazy val asWords: Array[String] = wordTokenize(asString)
}
