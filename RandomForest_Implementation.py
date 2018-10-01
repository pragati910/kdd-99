from pyspark.mllib.tree import RandomForest, RandomForestModel
# Path for spark source folder
import sys
import os
from pyspark import SparkContext, SparkConf
from numpy import array
from time import time
from pyspark.mllib.regression import LabeledPoint
os.environ['SPARK_HOME']="/home/aadarsh/spark-2.3.1-bin-hadoop2.7"
# Append pyspark  to Python Path
sys.path.append("/home/aadarsh/spark-2.3.1-bin-hadoop2.7/python")
conf = SparkConf().setAppName("KDDCup99")
sc = SparkContext(conf=conf)

def parsePoint(line):
    line_split = line.split(",")
    clean_line_split = line_split[0:1] + line_split[4:41]
    attack = 0.0
    if line_split[41] == 'normal.':
        attack = 1.0
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))
if __name__ == "__main__":
        data_file="/home/aadarsh/PycharmProjects/python_project/kdd-cup-99-spark-master/kddcup.data_10_percent_corrected"

        # Load and parse the data file into an RDD of LabeledPoint.
        data = sc.textFile(data_file)
        # Split the data into training and test sets (30% held out for testing)
        train_Data = data.map(parsePoint)
        (trainingData, testData) = train_Data.randomSplit([0.7, 0.3])

        # Train a RandomForest model.
        #  Empty categoricalFeaturesInfo indicates all features are continuous.
        #  Note: Use larger numTrees in practice.
        #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
        start_time = time()
        model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                             numTrees=3, featureSubsetStrategy="auto",
                                             impurity='gini', maxDepth=4, maxBins=32)
        end_time = time()
        elapsed_time = end_time - start_time
        print("Time to train model: %.3f seconds" % elapsed_time)

        predictions = model.predict(testData.map(lambda x: x.features))
        start_time1 = time()
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        end_time1 = time()
        elapsed_time1 = end_time1 - start_time1
        print("Time to Predictions model: %.3f seconds" % elapsed_time1)

        Accuracy = labelsAndPredictions.filter(lambda vp: vp[0] == vp[1]).count() / float(testData.count())
        print("Model accuracy: %.3f%%" % (Accuracy * 100))
