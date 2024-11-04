# **SparkLeLM**

## **Author: Himanshu Dongre**

# Overview

SparkLeLM trains an LLM (Large Language Model) from scratch and demonstrates text generation using it on sample text queries. The major pieces are:-

1. **TextGenerator:** A Scala object that is the entry point for SparkLeLM. It's main role is to produce outputs which is text generated by the LLM for certain input text queries.
2. **LLMTrainer:** A Scala object that defines the neural network underlying the LLM and implements its training.
3. **Preprocessor:** A Scala object that implements the API for preprocessing text including splitting text into sentences, tokenizing text into words, creating sliding window data samples from input text data, and producing embeddings using a pre-trained Word2vec model.
4. **SparkObj:** A Scala object that creates the Spark session for the lifetime of the execution of SparkLeLM. The Spark session is stopped by LLMTrainer at the end of the pipeline, after the LLM has been trained.
5. **Constants:** A Scala object that initializes global constants from the config file.
6. **FileIO:** A Scala object that configures the Hadoop file system API and defines helper functions for file I/O from AWS S3.
7. **TrainingData:** A Scala object that reads in the training data from S3 and stores it in three formats: as a string, as an array of sentences, and as an array of tokens (words).
8. **sparklelm.conf:** The config file which sets global parameters:-
   + `EMBEDDING_DIM`: Size of each vector representing a token in the embedding space.
   + `WINDOW_SIZE`: Size of the sliding window.
   + `STRIDE`: Number of tokens by which the sliding window shifts to generate data samples.
   + `TRAINING_DATA_URI`: URI of the file containing the textual training data for the LLM.
   + `MODEL_SAVE_URI`: URI of the file to which SparkLeLM writes the trained LLM.
   + `LAYER0_NUM_NEURONS`: Number of neurons in the first dense layer of the neural network.
   + `LAYER1_NUM_NEURONS`: Number of neurons in the second dense layer of the neural network.
   + `W2V_TMP_LOCAL_FILE`: Name of the temporary local file used to store data to train the Word2Vec model.

# The Model

SparkLeLM defines and trains a neural network using Deeplearning4j’s `MultiLayerNetwork`. It consists of three layers:-

1. a dense input layer with **ReLU activation**,
2. a second dense layer also with ReLU,
3. and an output layer that uses **softmax activation** and **multiclass cross-entropy loss** for classification.

The model is optimized with the **Adam algorithm**, initialized with a learning rate of $$1 \times 10^{-3}$$, and is designed to handle input embeddings of a dimension specified in the config file.

# Prerequisites and Dependencies

SparkLeLM uses (has been tested on):-

+ **Scala 2.12.19**
+ **SBT 1.10.4** (the build system for SparkLeLM)

SparkLeLM uses (has been tested on) the following versions of the dependencies for core functionality:-

+ **DeepLearning4j**, **ND4j 1.0.0-M2.1**
+ **Apache Spark 3.5.1**

SparkLeLM uses **Apache Hadoop 3.3.4** for file I/O.

# Get Started

**Update `sparklelm.conf` as per your needs. It is currently configured to run on an AWS EMR cluster with custom AWS S3 URIs. Also, reconfigure file I/O using Hadoop in the source code to enable local file I/O. It is currently configured to use the file system of AWS S3.**

Once SparkLeLM has been prepared for local execution, it can be run on the command-line using the following command:-

```shell
sbt clean compile run
```

It will print out a runtime log to `stderr`. It will print out some stats and some LLM text generation outputs for some pre-defined input queries to `stdout`. SparkLeLM can be compiled into a *uber JAR* using the command:-

```shell
sbt clean assembly
```

This *fat JAR* can then be deployed on an AWS EMR cluster. As mentioned before, update `sparklelm.conf` as per your needs.

# Testing

SparkLeLM uses the **Scalatest** framework for unit testing. Tests can be run from the command-line using the command:-

```shell
sbt test
```

The tests mainly verify the correctness of funtions that split text into sentences and tokenize text into words. They also verify whether the vector and positional embeddings are produced with correct dimensions.

# Logging

SparkLeLM uses **SLF4j** (Simple Logging Facade for Java) for logging. It prints out a runtime log to `stderr`.

# Training Data

SparkLeLM trains its LLM on the dataset: the plaintext version of the book, *The Adventures of Sherlock Holmes* by **Sir Arthur Conan Doyle** obtained from Project Gutenberg.

```shell
SparkLeLM/$ wc sherlock.txt
   11922  104506  587719 sherlock.txt
```