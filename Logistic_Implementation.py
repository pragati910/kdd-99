from pyspark.mllib.classification import LogisticRegressionWithLBFGS
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
        clean_line_split = line_split[0:1] + line_split[4:41]  #check the 1'st and last element
        attack = 0.0   #set the attack value
        if line_split[41] == 'normal.':#check the last pramater in the dataset file
            attack = 1.0   #set the attack value
        return LabeledPoint(attack, array([float(x) for x in clean_line_split]))#given the label point


if __name__ == "__main__":
        # path for the dataset file
        data_file="/home/aadarsh/PycharmProjects/python_project/kdd-cup-99-spark-master/kddcup.data_10_percent_corrected"

        train_raw_data = sc.textFile(data_file) #read the data in spark text

        train_Data = train_raw_data.map(parsePoint)#map the data of dataset for the traning data

        (trainingData, test_data) = train_Data.randomSplit([0.7, 0.3])#split the data for 70  to 30 % for the traning and testing
        # start time for bulid model
        t0 = time()
        model = LogisticRegressionWithLBFGS.train(trainingData)   # Build the model
        tt = time() - t0   #total time for bulid model
        print("Time to train model: %.3f seconds" % tt)

        predictions = model.predict(test_data.map(lambda x: x.features)) #using test data given input as feature
        start_time1 = time()#start time
        labelsAndPredictions = test_data.map(lambda lp: lp.label).zip(predictions)#prediction the data according to train model
        end_time1 = time()#end time
        elapsed_time1 = end_time1 - start_time1#total time
        print("Time to Predictions model: %.3f seconds" % elapsed_time1)

        # check the accuracy using the filter function
        Accuracy = labelsAndPredictions.filter(lambda vp: vp[0] == vp[1]).count() / float(test_data.count())
        print("Model accuracy: %.3f%%" % (Accuracy * 100))
