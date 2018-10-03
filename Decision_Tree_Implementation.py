from pyspark.mllib.tree import DecisionTree
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

        clean_line_split = line[0:41]
        # convert protocol to numeric categorical variable
        try:
            clean_line_split[1] = protocols.index(clean_line_split[1])
        except:
            clean_line_split[1] = len(protocols)

        # convert service to numeric categorical variable
        try:
            clean_line_split[2] = services.index(clean_line_split[2])
        except:
            clean_line_split[2] = len(services)

        # convert flag to numeric categorical variable
        try:
            clean_line_split[3] = flags.index(clean_line_split[3])
        except:
            clean_line_split[3] = len(flags)

        # convert label to binary label
        attack = 0.0
        if line[41] == 'normal.': #check the last pramater in the dataset file
            attack = 1.0 #set the attack value
        return LabeledPoint(attack, array([float(x) for x in clean_line_split]))#given the label point


if __name__ == "__main__":
    # path for the dataset file
        data_file="/home/aadarsh/PycharmProjects/python_project/kdd-cup-99-spark-master/kddcup.data_10_percent_corrected"

        train_raw_data = sc.textFile(data_file) #read the data in spark text

        data = train_raw_data.map(lambda x: x.split(",")) #map the train data

        protocols = data.map(lambda x: x[1]).distinct().collect() #check the protocols in the dataset file
        services = data.map(lambda x: x[2]).distinct().collect()  #check the services in the dataset file
        flags = data.map(lambda x: x[3]).distinct().collect()  #check the flag in the dataset file

        train_Data = data.map(parsePoint) #map the dataset file and call the parespoint function

        (trainingData, test_data) = train_Data.randomSplit([0.7, 0.3]) #split the data 70 to 30% ratio for traning and testing

        # start time of build model
        t0 = time()
        model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
                                             impurity='gini', maxDepth=4, maxBins=100)  # Build the model

        tt = time() - t0  #total time of build model
        print("Time to train model: %.3f seconds" % tt)

        predictions = model.predict(test_data.map(lambda x: x.features)) #given the input as a feature
        start_time1 = time() #start time for prediction
        labelsAndPredictions = test_data.map(lambda lp: lp.label).zip(predictions) #predict the label of data
        end_time1 = time() #end time of prediction
        elapsed_time1 = end_time1 - start_time1 #total time of prediction
        print("Time to Predictions model: %.3f seconds" % elapsed_time1)


        Accuracy = labelsAndPredictions.filter(lambda vp: vp[0] == vp[1]).count() / float(test_data.count())
        #check the accurany of model
        print("Model accuracy: %.3f%%" % (Accuracy * 100))
