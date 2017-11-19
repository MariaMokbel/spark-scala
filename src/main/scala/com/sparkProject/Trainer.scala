package com.sparkProject

import breeze.numerics.log
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   val parquetDF = spark.read.parquet("/home/maria/Desktop/projet_spark/prepared_trainingset")

    /** TF-IDF **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val remover = new StopWordsRemover()
          .setInputCol("tokens")
      .setOutputCol("filtered")

    val vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("result")

    val idf = new IDF()
      .setInputCol("result")
      .setOutputCol("tfidf")


    /** INDEXER **/

    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** VECTOR ASSEMBLER **/

    val assembler = new VectorAssembler().
      setInputCols(Array("tfidf", "days_campaign", "hours_prepa","goal","country_indexed"
      ,"currency_indexed")).
      setOutputCol("features")

    /** MODEL **/
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover,vectorizer,idf,indexer_country,indexer_currency,assembler, lr))

    /** TRAINING AND GRID-SEARCH **/

    val Array(training, test) = parquetDF.randomSplit(Array(0.9, 0.1))

    val paramGrid = new ParamGridBuilder()
      .addGrid(vectorizer.minDF, Array(55.0, 95.0, 20))
      .addGrid(lr.regParam, Array(10e-8, 10e-2,log(2)))
      .build()

    val f1score = new MulticlassClassificationEvaluator()
          .setLabelCol("final_status")
            .setPredictionCol("predictions")
            .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(f1score)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(training)

    val df_WithPredictions = model.transform(test)


    val Test_f1Score = f1score.evaluate(df_WithPredictions)
    println("F1 score on test data: " + Test_f1Score)

    df_WithPredictions.groupBy("final_status","predictions").count.show()

  }
}
