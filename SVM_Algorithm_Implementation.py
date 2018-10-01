from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
import sys
import os
from pyspark import SparkContext, SparkConf
from numpy import array
from time import time

# Path for spark source folder
os.environ['SPARK_HOME']="/home/aadarsh/spark-2.3.1-bin-hadoop2.7"

# Append pyspark  to Python Path
sys.path.append("/home/aadarsh/spark-2.3.1-bin-hadoop2.7/python")
conf = SparkConf().setAppName("KDDCup99")
sc = SparkContext(conf=conf)
# Load and parse the data
def parsePoint(line):

        line_split = line.split(",")
        clean_line_split = line_split[0:1] + line_split[4:41]
        attack = 0.0
        if line_split[41] == 'normal.':
            attack = 1.0
        return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

if __name__ == "__main__":
        data_file="/home/aadarsh/PycharmProjects/python_project/kdd-cup-99-spark-master/kddcup.data_10_percent_corrected"

        train_raw_data = sc.textFile(data_file)

        train_Data = train_raw_data.map(parsePoint)

        (trainingData, test_data) = train_Data.randomSplit([0.7, 0.3])

        # Build the model
        t0 = time()
        model = SVMWithSGD.train(trainingData, iterations=10)
        tt = time() - t0
        print("Time to train model: %.3f seconds" % tt)

        predictions = model.predict(test_data.map(lambda x: x.features))
        start_time1 = time()
        labelsAndPredictions = test_data.map(lambda lp: lp.label).zip(predictions)
        end_time1 = time()
        elapsed_time1 = end_time1 - start_time1
        print("Time to Predictions model: %.3f seconds" % elapsed_time1)

        Accuracy = labelsAndPredictions.filter(lambda vp: vp[0] == vp[1]).count() / float(test_data.count())
        print("Model accuracy: %.3f%%" % (Accuracy * 100))

