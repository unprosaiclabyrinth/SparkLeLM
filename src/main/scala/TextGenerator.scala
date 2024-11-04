import Constants.MODEL_SAVE_URI
import FileIO.fs
import org.apache.hadoop.fs.Path
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j

import scala.util.Using

object TextGenerator {
  LLMTrainer.trainLLM()

  private val LLModel = Using(fs.open(new Path(MODEL_SAVE_URI)))(inStream =>
    ModelSerializer.restoreMultiLayerNetwork(inStream)
  ).getOrElse {
    throw new RuntimeException("Failed to restore the model from S3")
  }

  // Initialize vocabulary from the training data
  private val vocabulary = TrainingData.asWords

  // Method to generate the next word based on the query using the pretrained model
  def generateNextWord(context: Array[String]): String = {
    // Tokenize context and convert to embedding
    val contextEmbedding = Preprocessor.encodeAndEmbed(context)
    // Forward pass through the transformer layers (pretrained)
    val output = LLModel.output(contextEmbedding)
    // Find the word with the highest probability (greedy search) or sample
    val predictedWordIndex = Nd4j.argMax(output, 1).getInt(0) // get the index of the predicted word
    convertIndexToWord(predictedWordIndex)
  }

  // Method to generate a full sentence based on the seed text
  def generateSentence(seedText: String, maxWords: Int): String = {
    (1 to maxWords).map(_ =>
      s" ${generateNextWord(Preprocessor.wordTokenize(seedText))}"
    ).mkString
  }

  // Helper function to map word index to actual word
  private def convertIndexToWord(index: Int): String = vocabulary(index % vocabulary.length)

  def main(args: Array[String]): Unit = {
    println("Sample queries:-")
    val sampleQueries = List[String]("The cat", "CS441 is", "I was", "Sherlock Holmes", "Elementary")
    sampleQueries.foreach(query =>
      println(s"$query -> ${generateSentence(query, 5)}")
    )
  }
}
