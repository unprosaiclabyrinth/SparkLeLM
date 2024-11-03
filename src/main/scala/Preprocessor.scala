import Constants._
import SparkObj.spark
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

import java.io.File
import scala.util.matching.Regex

object Preprocessor {
  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  private val sentenceDelimiter: Regex = "(?<=[.!?])\\s+(?=[A-Z])".r
  private val wordDelimiter: Regex = "\\w+".r

  // Train a Word12vec model for vector embeddings
  private lazy val sentenceIterator = new LineSentenceIterator(new File(TRAINING_DATA_PATH))
  private lazy val tokenizerFactory: TokenizerFactory = new DefaultTokenizerFactory()
  tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

  private val w2v = new Word2Vec.Builder()
    .minWordFrequency(5)
    .layerSize(EMBEDDING_DIM)
    .seed(42)
    .windowSize(5)
    .iterate(sentenceIterator)
    .tokenizerFactory(tokenizerFactory)
    .build

  w2v.fit()

  // Tokenize the text into sentences
  def sentTokenize(text: String): Array[String] = sentenceDelimiter.split(text)

  // Tokenize the text into words
  def wordTokenize(text: String): Array[String] = {
    val words = sentTokenize(text).flatMap(sentence => wordDelimiter.findAllIn(sentence))
    logger.info(s"Tokenized given string into ${words.length} tokens.")
    words
  }

  // Compute sliding windows from the tokens with Spark
  def slidingWindowsFrom(tokens: Array[String]): List[DataSet] = {
    // Parallelize the input data
    val tokensRDD = spark.sparkContext.parallelize(tokens)

    // Apply the sliding window logic to create the dataset
    val slidingWindowDataset = tokensRDD.mapPartitions(partition => {
      createSlidingWindows(partition.toArray).iterator
    })

    slidingWindowDataset.collect().toList
  }

  // Logic to create sliding windows from the tokens (with position embeddings)
  private def createSlidingWindows(tokens: Array[String]): List[DataSet] = {
    logger.info(s"Got ${tokens.length} tokens for sliding window sampling.")
    (0 until tokens.length - WINDOW_SIZE by STRIDE).map(i => {
      // Extract the input window (WINDOW_SIZE tokens)
      val inputWindow = new Array[String](WINDOW_SIZE)
      System.arraycopy(tokens, i, inputWindow, 0, WINDOW_SIZE)
      // Extract the target token (the token right after the window)
      val targetToken = tokens(i + WINDOW_SIZE)
      // Convert input tokens into embeddings
      val inputEmbeddings = encodeAndEmbed(inputWindow) // Embedding for words
      // Add positional embeddings to word embeddings
      val positionAwareEmbedding = inputEmbeddings.add(positionalEmbedding)
      // Convert the target token into an embedding
      val targetEmbedding = encodeAndEmbed(Array[String](targetToken))
      logger.info(s"Feature dim: ${positionAwareEmbedding.shape().mkString("Array(", ", ", ")")}, labels dim: ${targetEmbedding.shape().mkString("Array(", ", ", ")")}.")
      new DataSet(positionAwareEmbedding, targetEmbedding)
    }).toList
  }

  def encodeAndEmbed(tokens: Array[String]): INDArray = {
    val ret = Nd4j.zeros(tokens.length, EMBEDDING_DIM)

    tokens.indices.foreach(i => {
      // Use the saved word2vec model
      ret.putRow(i,
        if (w2v.hasWord(tokens(i)))
          w2v.getWordVectorMatrix(tokens(i))
        else
          Nd4j.zeros(1, EMBEDDING_DIM)
      )
    })

    logger.info(s"Produced embeddings of shape ${ret.shape().mkString("Array(", ", ", ")")}" +
      s"for dataset of size ${tokens.length}."
    )
    ret
  }

  // Compute sinusoidal positional embeddings for a given window size
  private def positionalEmbedding: INDArray = {
    val positionalEncoding = Nd4j.zeros(WINDOW_SIZE, EMBEDDING_DIM)

    (0 until WINDOW_SIZE).foreach( pos => {
      (0 until EMBEDDING_DIM by 2).foreach( i => {
        val angle = pos / Math.pow(10000, (2 * i) / EMBEDDING_DIM)
        positionalEncoding.putScalar(Array[Int](pos, i), Math.sin(angle))
        positionalEncoding.putScalar(Array[Int](pos, i + 1), Math.cos(angle))
      })
    })

    positionalEncoding
  }
}
