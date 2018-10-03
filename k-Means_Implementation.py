#!/usr/bin/env python

# dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

import sys
import os


# Path for spark source folder
os.environ['SPARK_HOME']="/home/aadarsh/spark-2.3.1-bin-hadoop2.7"

# Append pyspark  to Python Path
sys.path.append("/home/aadarsh/spark-2.3.1-bin-hadoop2.7/python")

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.mllib.clustering import KMeans
    from pyspark.mllib.feature import StandardScaler
    from pyspark.ml.evaluation import ClusteringEvaluator
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

from collections import OrderedDict
from numpy import array
from math import sqrt
from time import time

def parse_interaction(line):
    """
    Parses a network data interaction.
    """
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    return (line_split[-1], array([float(x) for x in clean_line_split]))


def distance(a, b):
    """
    Calculates the euclidean distance between two numeric RDDs
    """
    return sqrt(
        a.zip(b)
        .map(lambda x: (x[0]-x[1]))
        .map(lambda x: x*x)
        .reduce(lambda a,b: a+b)
        )


def dist_to_centroid(datum, clusters):
    """
    Determines the distance of a point to its cluster centroid
    """
    cluster = clusters.predict(datum)
    centroid = clusters.centers[cluster]
    return sqrt(sum([x**2 for x in (centroid - datum)]))


def clustering_score(data, k):
    t0 = time()
    clusters = KMeans.train(data, k, maxIterations=10, runs=5, initializationMode="random")
    tt = time() - t0
    print ("Data Train in {} seconds".format(round(tt, 3)))
    result = (k, clusters, data.map(lambda datum: dist_to_centroid(datum, clusters)).mean())

    print ("Clustering score for k=%(k)d is %(score)f" % {"k": k, "score": result[2]})
    return result



if __name__ == "__main__":
    if (len(sys.argv) != 3):

        sys.exit(1)

    # set up environment
    #max_k = int(sys.argv[1])
    #data_file = sys.argv[2]
    max_k=10
    data_file='/home/aadarsh/PycharmProjects/python_project/kdd-cup-99-spark-master/kddcup.data_10_percent_corrected'
    conf = SparkConf().setAppName("KDDCup99")

    sc = SparkContext(conf=conf)

    # load raw data
    print( "Loading RAW data...")
    raw_data = sc.textFile(data_file)

    # count by all different labels and print them decreasingly
    print ("Counting all different labels")
    labels = raw_data.map(lambda line: line.strip().split(",")[-1])
    label_counts = labels.countByValue()
    sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
    print("label, count")
    for label, count in sorted_labels.items():
        print (label, count)

    # Prepare data for clustering input
    # the data contains non-numeric features, we want to exclude them since
    # k-means works with numeric features. These are the first three and the last
    # column in each data row
    print ("Parsing dataset...")
    parsed_data = raw_data.map(parse_interaction)

    parsed_data_values = parsed_data.values().cache()

    # Standardize data
    print ("Standardizing data...")
    standardizer = StandardScaler(True, True)

    standardizer_model = standardizer.fit(parsed_data_values)
    standardized_data_values = standardizer_model.transform(parsed_data_values)

    # Evaluate values of k from 5 to 40
    print ("Calculating total in within cluster distance for different k values (10 to %(max_k)d):" % {"max_k": max_k})
    scores = map(lambda k: clustering_score(standardized_data_values, k), range(10,max_k+1,10))
    scores1 = map(lambda k: clustering_score(standardized_data_values, k), range(10, max_k + 1, 10))

    # Obtain min score k

    best_model = min(scores, key=lambda x: x[2])[1]
    min_k = min(scores1, key=lambda x: x[2])[0]
    print("min_k : ",min_k)
    print("best_model : ",best_model)
    print ("Best k value is %(best_k)d" % {"best_k": min_k})
    # print("Accuracy = %s" % best_model.accuracy)

    # evaluator = ClusteringEvaluator()

    # Use the best model to assign a cluster to each datum
    # We use here standardized data - it is more appropriate for exploratory purposes
    print( "Obtaining clustering result sample for k=%(min_k)d..." % {"min_k": min_k})
    print('Accuracy:{0:f}'.format(standardized_data_values))

    t0 = time()
    cluster_assignments_sample = standardized_data_values.map(lambda datum: str(best_model.predict(datum))+","+",".join(map(str,datum))).sample(False,0.05)
    tt = time() - t0
    print("Data prediction in {} seconds".format(round(tt, 3)))

    # Save assignment sample to file
    # print ("Saving sample to file...",cluster_assignments_sample.saveAsTextFile("sample_standardized_1"))
    # print ("DONE!")
