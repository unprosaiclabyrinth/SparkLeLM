import com.typesafe.config.{Config, ConfigFactory}

object Constants {
  private val CONFIG_FILE = "sparklelm.conf"
  private val config: Config = ConfigFactory.load(CONFIG_FILE)

  lazy val WINDOW_SIZE: Int = config.getInt("app.window_size")
  lazy val STRIDE: Int = config.getInt("app.stride")
  lazy val EMBEDDING_DIM: Int = config.getInt("app.embedding_dim")

  lazy val SPARK_APP_NAME: String = config.getString("app.spark_app_name")
  lazy val SPARK_MASTER: String = config.getString("app.spark_master")
  lazy val TRAINING_DATA_PATH: String = config.getString("app.training_data_path")
  lazy val MODEL_SAVE_PATH: String = config.getString("app.model_save_path")
  lazy val W2V_PATH: String = config.getString("app.w2v_path")

  lazy val LAYER0_NUM_NEURONS: Int = config.getInt("app.layer0_num_neurons")
  lazy val LAYER1_NUM_NEURONS: Int = config.getInt("app.layer1_num_neurons")
  lazy val GENERATION_LENGTH: Int = config.getInt("app.generation_length")
}
