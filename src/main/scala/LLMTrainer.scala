import Constants.{EMBEDDING_DIM, LAYER0_NUM_NEURONS, LAYER1_NUM_NEURONS, MODEL_SAVE_PATH, TRAINING_DATA_PATH, WINDOW_SIZE}
import SparkObj.spark
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import java.io.File

object LLMTrainer {
  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  private def createLLModel: MultiLayerNetwork = {
     val model = new MultiLayerNetwork(
      new NeuralNetConfiguration.Builder()
        .updater(new Adam(1e-3))
        .list()
        .layer(0, new DenseLayer.Builder().nIn(EMBEDDING_DIM).nOut(LAYER0_NUM_NEURONS).activation(Activation.RELU).build())
        .layer(1, new DenseLayer.Builder().nOut(LAYER1_NUM_NEURONS).activation(Activation.RELU).build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX).nOut(EMBEDDING_DIM).build())
        .build()
    )
    model.init()
    model
  }

  def trainLLM(): Unit = {
    // Create your transformer model using DL4J
    val LLModel = createLLModel
    logger.info("Created the neural network underlying the LLM.")

    // Prepare data (using sliding window)
    val windows = Preprocessor.slidingWindowsFrom(TrainingData.asWords)
    logger.info(s"Created ${windows.length} sliding window data samples from the training data.")

    // Pad labels with replicas to match their shape with the shape of the preOutput
    val windowsWithAlignedLabels = windows.map(window => {
      new DataSet(window.getFeatures, window.getLabels.repeat(0, WINDOW_SIZE))
    })

    val rddData = spark.sparkContext.parallelize(windowsWithAlignedLabels)
    logger.info(s"RDDStats: ${rddData.count()}")

    // TrainingMaster configuration for distributed training
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(WINDOW_SIZE)
      .batchSizePerWorker(WINDOW_SIZE) // Experimentally determined (!)
      .averagingFrequency(5)
      .build()

    // SparkDl4jMultiLayer with the Spark context and model
    val sparkModel = new SparkDl4jMultiLayer(spark.sparkContext, LLModel, trainingMaster)

    // Set listeners to monitor the training progress
    LLModel.setListeners(new ScoreIterationListener(10))

    // Train the model on the distributed RDD dataset
    sparkModel.fit(rddData)
    logger.info("Trained the model using Spark.")

    // Save the model
    ModelSerializer.writeModel(sparkModel.getNetwork, new File(MODEL_SAVE_PATH), true)
    logger.info(s"Wrote the model to $MODEL_SAVE_PATH.")

    // Close resources
    trainingMaster.deleteTempFiles(spark.sparkContext)
    spark.stop() // End of pipeline
  }
}