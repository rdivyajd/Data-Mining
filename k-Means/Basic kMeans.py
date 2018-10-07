#!/usr/bin/env python3
# Divya Rajendran 2018
# I have taken the data sets from UCI machine learning repository. http://archive.ics.uci.edu/ml/datasets/
# All the data is saved into a csv.

# Data Sets:
# Below are the Data Sets in order taken from UCI Machine Learning Repository
# 1. Iris-Satosa Data Set from http://archive.ics.uci.edu/ml/datasets/Iris
# 2. Wines Data Set from http://archive.ics.uci.edu/ml/datasets/Wine
# 3. Teaching Assistant Evaluation from http://archive.ics.uci.edu/ml/datasets/Teaching+Assistant+Evaluation

# Executing the program
# python ./Q3_Basic.py File_Name function
# File_Names: Name of the file, takes the values [Iris-Data.csv, Wine-Data.csv, TA-Eval.csv]
# function: function names, takes the values [euclidean, cosine, city_block, equation1, equation2]
#

# Algorithm: Basic K-Means algorithm


import math
import time
import numpy as np
import pandas as pd
import sys


# txtdata class to store the text data
def text_process(fileName):  # read the file and pre-process data
    data = pd.read_csv(r".\\"+fileName, header=None) # read

    # Pre process data set - takes care of class labels
    uniq_class, classes = {}, {}

    # Gets the unique class labels
    for index, row in data.iterrows():
        classes[str(row[len(data.columns) - 1])] = 0

    # assigns unique class labels with the cluster number and
    # replaces the class column with actual cluster numbers which can be compared after our predictions
    for key, clas in zip(classes.keys(), range(len(classes.keys()))):
        uniq_class[key] = clas
        data.replace(str(key), float(clas), inplace=True)
    return data.as_matrix(), uniq_class

    # data, uniq_class = data.iloc[:, 0:len(data.columns) - 1]


# Class KMeans to hold the kMeans algorithm and all the functions needed for the algorithm.
class kMeans:
    def __init__(self, k, dataSet=None, function=None, iteration_limit=500, accuracy=0.0001, random_start_limit=10):
        self.k = k
        self.dataSet = dataSet
        self.accuracy = accuracy
        self.iteration_limit = iteration_limit
        self.current_iteration = 0
        self.random_start_limit = random_start_limit
        self.function = function
        self.sse = 0
        self.sse_itr = 0
        self.distances = []

    # calculating the initial centroids
    def initial_centroids(self):  # first k elements assigned
        self.centroids = {}
        random_indices = np.random.choice(len(self.dataSet), self.k, replace = False)
        for i, j in zip(random_indices, range(self.k)):  # for assigning initial array as a centroid
            self.centroids[j] = self.dataSet[i]
            for k in range(len(self.dataSet[i])):
                self.centroids[j][k] = round(self.dataSet[i][k], 2)

    # Euclidean distance
    def euclidean(self, attributes):  # distance function
        self.distances = [math.sqrt(sum([(element - center) ** 2
                                         for element, center in zip(attributes, self.centroids[cntrd])]))
                          for cntrd in self.centroids]

    # Cosine distance
    def cosine(self, attributes):  # distance function
        distances = []
        for cntrd in self.centroids:
            num, dist, point_dist, check_dist = 0, 0, 0, 0
            for element, center in zip(attributes, self.centroids[cntrd]):
                point_dist += element ** 2
                check_dist += center ** 2
                num += element * center
            denom = (check_dist ** 0.5) * (point_dist ** 0.5)

            dist = 1 - float(num) / float(denom)
            distances.append(dist)
        self.distances = distances

    # City Block distance
    def city_block(self, attributes):  # distance function
        dist = 0
        distances = []
        for center in self.centroids:
            for cntrd, element in zip(self.centroids[center], attributes):
                dist += element - cntrd
            distances.append(dist)
        self.distances = distances

    # Equation1 distance
    def equation1(self, attributes):  # distance function
        distances = []
        for cntrd in self.centroids:
            dist1, dist2 = 0, 0
            for element, center in zip(attributes, self.centroids[cntrd]):
                value = element - center
                dist1 += value if value > 0 else 0
                dist2 += -value if -value > 0 else 0
            dist = (dist1 ** 2 + dist2 ** 2) ** 0.5
            distances.append(dist)
        self.distances = distances

    # Equation2 distance
    def equation2(self, attributes):  # distance function
        distances = []
        for cntrd in self.centroids:
            dist1, dist2, dist3 = 0, 0, 0
            for element, center in zip(attributes, self.centroids[cntrd]):
                value = element - center
                dist1 += value if value > 0 else 0
                dist2 += -value if -value > 0 else 0
                dist3 += max([abs(element), abs(center), abs(value)])
            dist = float((dist1 ** 2 + dist2 ** 2) ** 0.5) / float(dist3)
            distances.append(dist)
        self.distances = distances

    # re-calculation of the centroids - average of the points in that cluster
    def recalculate_centroids(self):  # averaging the cluster data points
        for classification in self.classes:
            self.centroids[classification] = np.average(self.classes[classification], axis=0)

    # Sum of Squared errors Calculation - used as an internal quality criteria
    def sse_calculation(self):  # calculates the total SSE value
        self.sse = 0
        for classification in self.centroids.keys():
            centroid = self.centroids[classification]
            for point in self.classes[classification]:
                count = 0
                for i in range(0, len(point)):
                    count += (centroid[i] - point[i])**2
                self.sse += count

    # Basic Cluster fit algorithm
    def clusterfit(self):  # Basic K-Means with SSE objective function criteria
        self.current_iteration = 0
        count = 0
        sse_itr = 0
        for itr in range(self.random_start_limit):  # iteration loop for random starts
            kMeans.initial_centroids(self)  # random initialization of the centroids
            self.is_perfect = False
            # start model fit
            while True:  # iteration for the cluster fit starts
                count += 1
                self.classes = {}
                for clust_itr in range(self.k):
                    self.classes[clust_itr] = []
                for attributes in self.dataSet:  # check distances between points and centroids
                    sse_itr += 1
                    if self.function == "euclidean":
                        kMeans.euclidean(self, attributes)
                    elif self.function == "cosine":
                        kMeans.cosine(self, attributes)
                    elif self.function == "city_block":
                        kMeans.euclidean(self, attributes)
                    elif self.function == "equation1":
                        kMeans.equation1(self, attributes)
                    elif self.function == "equation2":
                        kMeans.equation2(self, attributes)

                    cls = self.distances.index(min(self.distances))  # class assigned # found the closest centroid
                    self.classes[cls].append(attributes)

                # Check to minimize the SSE value
                kMeans.sse_calculation(self)
                previous_sse = self.sse
                kMeans.recalculate_centroids(self)  # recalculate centroids
                kMeans.sse_calculation(self)
                current_sse = self.sse
                if previous_sse - current_sse < self.accuracy or count > self.iteration_limit:  # stopping criteria
                    self.is_perfect = True
                    break

            self.current_iteration = count  # iteration count
            if self.is_perfect:  # stopping criteria for outer loop
                break

        self.sse_itr = sse_itr  # the number of distance calculations
        # specifying the cluster assignment to a data point.
        self.clust_assignment = []
        for key in self.classes.keys():
            clustered = self.classes[key]
            for value in clustered:
                self.clust_assignment.append([key, value])
        self.dataframe = pd.DataFrame(self.clust_assignment,
                                      columns=['Cluster Assigned', 'Data point'])


# Cluster Fitting Program Starts
def main():

    file_name = sys.argv[1]  # "Page-block.csv"
    function = sys.argv[2]  # distance function

    # for naming the output file
    idx = file_name.index('.')
    output = file_name[:idx]

    df, class_labels = text_process(file_name)
    # print(class_labels)
    # dataSet = df.values
    dataSet = df

    km = kMeans(len(class_labels.keys()), dataSet, function)
    start = time.clock()

    km.clusterfit()  # normal cluster fit

    print("Took %f secs" % (time.clock() - start))
    print("The number of iterations taken to converge is ", km.current_iteration)
    print("The number of distance calculations made is ", km.sse_itr)
    print("The total sum of squared errors is, ", round(km.sse, 3))
    print("Cluster assignment to data points array is saved to the csv file in the source location with " +
          "the name Basic_Result-Q3-" + output + ".csv")
    km.dataframe.to_csv(".\\Basic_Result-Q3-"+function+output+".csv", encoding='utf-8', index=False)



# Main Program
if __name__ == "__main__":
    main()