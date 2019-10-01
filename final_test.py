#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

Usage:

    $ spark-submit final_test.py hdfs:/path/to/load/model.parquet hdfs:/path/to/file

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import expr
# TODO: you may need to add imports here


def main(spark, model_file, test_data_file):
    '''Main routine for supervised evaluation

    Parameters
    ----------
    spark : SparkSession object

    model_file : string, path to store the serialized model file

    data_file : string, path to the parquet file to load
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###
    testing_data = spark.read.parquet(test_data_file)
    model = PipelineModel.load(model_file)    
    prediction = model.recommendForAllUsers(500).select('userindex', 'recommendations.itemindex')
    
    testing_df = testing_data.groupBy('userindex').agg(expr('collect_list(itemindex) as item_list'))
    predictionAndLabels = prediction.join(testing_df, 'userindex')
    pred_df = predictionAndLabels.select(['itemindex','item_list']).rdd.map(list)

    metrics = RankingMetrics(pred_df)
    eva = metrics.meanAveragePrecision

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('remommendation_test').getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Get the filename from the command line
    test_data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, test_data_file)