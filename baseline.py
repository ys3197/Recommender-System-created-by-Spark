#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
ALS Model Train Baseline
$ spark-submit baseline.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/path/to/save/model
'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import expr
# TODO: you may need to add imports here


def main(spark, train_data_file, test_data_file, model_file):

    # Use Validation and Test user_id to filter Train data, to get the 110k mandatory users
    # Stored here hdfs:/user/dz584/cf_train_sample.parquet
    """
    training_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
    validation_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_validation.parquet')
    testing_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_test.parquet')

    validandtest_userid = validation_data.union(testing_data).select('user_id').distinct()
    validandtest_userid.createOrReplaceTempView('validandtest_userid')

    training_data.createOrReplaceTempView('training_data')
    training_data = spark.sql("SELECT * FROM training_data WHERE user_id IN (SELECT user_id FROM validandtest_userid GROUP BY user_id)")
    training_data.write.parquet("cf_train_sample.parquet")
    """

    training_data = spark.read.parquet(train_data_file)
    indexer_id = StringIndexer(inputCol="user_id", outputCol="userindex").setHandleInvalid("skip")
    indexer_id_model = indexer_id.fit(training_data)
    indexer_item = StringIndexer(inputCol="track_id", outputCol="itemindex").setHandleInvalid("skip")
    indexer_item_model = indexer_item.fit(training_data)

    training_data = indexer_id_model.transform(training_data)
    training_data = indexer_item_model.transform(training_data)

    testing_data = spark.read.parquet(test_data_file)
    testing_data = indexer_id_model.transform(testing_data)
    testing_data = indexer_item_model.transform(testing_data)

    
    training_data = training_data.select('userindex','itemindex','count')
    testing_data = testing_data.select('userindex','itemindex','count')

    result_dict = {}
    rank_list = []
    reg_param_list = []
    alpha_list = []

    for rank in rank_list:
        for reg_param in reg_param_list:
            for alpha in alpha_list:

                als = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", 
                          ratingCol="count", rank=rank, regParam=reg_param, alpha=alpha)
                model = als.fit(training_data)


                prediction = model.recommendForAllUsers(500).select('userindex', 'recommendations.itemindex')
                testing_df = testing_data.groupBy('userindex').agg(expr('collect_list(itemindex) as item_list'))

                predictionAndLabels = prediction.join(testing_df, 'userindex')
                pred_df = predictionAndLabels.select(['itemindex','item_list']).rdd.map(list)

                metrics = RankingMetrics(pred_df)
                eva = metrics.meanAveragePrecision
                result_dict[current_key] = eva

    best_model_param = max(result_dict, key=result_dict.get)
    als = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count", rank=best_model_param[0], regParam=best_model_param[1], alpha=best_model_param[2])
    als.fit(training_data).write().overwrite().save(model_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').config("spark.sql.broadcastTimeout","36000").getOrCreate()
    
    # Get the filename from the command line
    train_data_file = sys.argv[1]

    # Get the filename from the command line
    test_data_file = sys.argv[2]
    
    # And the location to store the trained model
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, train_data_file, test_data_file, model_file)