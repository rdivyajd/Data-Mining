#!/usr/bin/env python3
# Divya Rajendran 2018
# I have taken the data sets from UCI machine learning repository. http://archive.ics.uci.edu/ml/datasets/
# All the data is saved into a csv.

# Data Sets:
# Below are the Data Sets in order taken from UCI Machine Learning Repository
# 1. Iris-Satosa Data Set from http://archive.ics.uci.edu/ml/datasets/Iris
# 2. Wines Data Set from http://archive.ics.uci.edu/ml/datasets/Wine
# 3. Teaching Assistant Evaluation from http://archive.ics.uci.edu/ml/datasets/Teaching+Assistant+Evaluation
# A uniform random sample of 10000 data points of 1000 dimensions were considered to test and compare results with
# the results from Elkans paper in https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf

# Executing the program
# python ./Q3_Elkans.py File_Name function
# File_Names: Name of the file, takes the values [Iris-Data.csv, Wine-Data.csv, TA-Eval.csv]
# function: function names, takes the values [euclidean, cosine, city_block, equation1, equation2]
#
# For evaluating random data, please edit the program main function, remove the comment [3], [4] from 341, 342 lines
#  and add a comment on [1], [2] @338 and 340 lines

# Algorithm: Elkans Accelerated K-Means Algorithm


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

        # for ELKAN's accelerator
        self.lower_bound = [[0] * self.k for _ in range(len(self.dataSet))]
        self.upper_bound = [0 for _ in range(len(self.dataSet))]

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

    # calcualting minimum centroid distances
    def min_centroid_distances(self):
        centroid_distances = []
        for key, i in zip(self.centroids, range(len(self.centroids.keys()))):
            v = self.centroids[key]
            rows = len(self.centroids.keys())
            cols = len(self.centroids[key])
            # centroids = np.full((rows, cols), np.asscalar(v))
            centroids = np.full((rows, cols), v)
            centroids[i + 1:] = 0

            if self.function == "euclidean":
                kMeans.euclidean(self, centroids[0])
            elif self.function == "cosine":
                kMeans.cosine(self, centroids[0])
            elif self.function == "city_block":
                kMeans.euclidean(self, centroids[0])
            elif self.function == "equation1":
                kMeans.equation1(self, centroids[0])
            elif self.function == "equation2":
                kMeans.equation2(self, centroids[0])

            centroid_distances.append(self.distances)
        minimum = 99999999999999
        for dist, i in zip(centroid_distances, range(len(centroid_distances[0]))):
            dist[i] = 99999999999999
            if minimum > min(dist):
                minimum = min(dist)
        s_centroid = 0.5 * minimum
        return s_centroid

    # considering cosine distance function
    def inter_dist(self, centroid1, centroid2):
        num, dist, point_dist, check_dist = 0, 0, 0, 0
        for cntrd1, cntrd2 in zip(centroid1, centroid2):
            point_dist += cntrd1 ** 2
            check_dist += cntrd2 ** 2
            num += cntrd1 * cntrd2
        denom = (check_dist ** 0.5) * (point_dist ** 0.5)
        dist = 1 - float(num) / float(denom)
        return dist

    # Elkans cluster fit algorithm
    def elkans_accelerator(self):
        self.current_iteration = 0
        count = 0
        sse_itr = 0
        for itr in range(self.random_start_limit):  # iteration loop for random starts
            kMeans.initial_centroids(self)  # random initialization of the centroids
            self.is_perfect = False
            lower_bound = self.lower_bound
            upper_bound = self.upper_bound
            # start model fit
            while True:  # iteration for the cluster fit starts
                count += 1
                self.classes = {}
                for clust_itr in range(self.k):
                    self.classes[clust_itr] = []
                if count == 1:
                    # check distances between points and centroids
                    for attributes, i in zip(self.dataSet, range(len(self.dataSet))):
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
                        lower_bound[i] = self.distances
                        upper_bound[i] = min(lower_bound[i])
                        cls = self.distances.index(min(lower_bound[i]))  # class assigned # found the closest centroid
                        self.classes[cls].append(attributes)

                        curr_classes = self.classes
                        curr_clust_arrangement = []
                        for key in curr_classes.keys():
                            clustered = curr_classes[key]
                            for value in clustered:
                                curr_clust_arrangement.append([key, value])
                else:
                    s_centroid = kMeans.min_centroid_distances(self)
                    for attributes, i in zip(self.dataSet, range(len(self.dataSet))):
                        if upper_bound[i] > s_centroid:  ## need more logic in else
                            for key in self.centroids.keys():
                                point_clust = curr_clust_arrangement[i][0]
                                x = curr_clust_arrangement[i][1]
                                c_x = self.centroids[point_clust]
                                c = self.centroids[key]
                                d_c_x_c = kMeans.inter_dist(self, c_x, c)
                                l_x_c = lower_bound[i][key]
                                u_x = upper_bound[i]

                                if point_clust != key and u_x > l_x_c and u_x > 0.5 * d_c_x_c:
                                    if r_x:
                                        d_x_c_x = kMeans.inter_dist(self, x, c_x)
                                        r_x = False
                                    else:
                                        d_x_c_x = u_x
                                    if d_x_c_x > l_x_c or d_x_c_x > 0.5 * d_c_x_c:
                                        sse_itr += 1
                                        d_x_c = kMeans.inter_dist(self, x, c)
                                        if d_x_c < d_x_c_x:
                                            curr_clust_arrangement[i][0] = key
                                            self.classes[key].append(x)
                                else:
                                    self.classes[point_clust].append(x)
                        else:
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
                            cls = self.distances.index(min(self.distances))
                            self.classes[cls].append(attributes)

                    new_centroids_mc = {}  # new centroids
                    for key in self.centroids.keys():
                        new_centroids_mc[key] = np.average(self.classes[key], axis=0)

                    for attributes, i in zip(self.dataSet, range(len(self.dataSet))):
                        point_clust = curr_clust_arrangement[i][0]
                        c = self.centroids[point_clust]
                        m_c = new_centroids_mc[point_clust]
                        d_c_m_c = kMeans.inter_dist(self, c, m_c)
                        lower_bound[i][point_clust] = max((l_x_c - d_c_m_c), 0)

                        u_x = upper_bound[i]
                        m_c_x = np.average(self.classes[point_clust], axis=0)
                        c_x = self.centroids[point_clust]
                        d_m_c_x_c_x = kMeans.inter_dist(self, m_c_x, c_x)
                        upper_bound[i] = u_x + d_m_c_x_c_x
                    r_x = True
                    self.centroids = new_centroids_mc

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
        self.clust_assignment = curr_clust_arrangement
        self.dataframe = pd.DataFrame(self.clust_assignment,
                                      columns=['Cluster Assigned', 'Data point'])


# Uniform Random
def uniform_random_train_data():  # 1000 dimensions, 10000 data points
    train = list()
    for i in range(0, 10000):
        data_point = np.random.randint(100, size=1000)
        train.append([i, data_point])
    return train


# Cluster Fitting Program Starts
def main():

    file_name = sys.argv[1]  # "Page-block.csv"
    function = sys.argv[2]  # distance function

    # for naming the output file
    idx = file_name.index('.')
    output = file_name[:idx]

    df, class_labels = text_process(file_name)
    dataSet = df  # [1]

    km = kMeans(len(class_labels.keys()), dataSet, function)  # [2]
    # dataSet = uniform_random_train_data()  # [3]
    # km = kMeans(3, dataSet, function)  # [4]
    start = time.clock()

    km.elkans_accelerator()  # cluster fit using elkans acceleration

    print("Took %f secs" % (time.clock() - start))
    print("The number of iterations taken to converge is ", km.current_iteration)
    print("The number of distance calculations made is ", km.sse_itr)
    print("The total sum of squared errors is, ", round(km.sse, 3))
    print("Cluster assignment to data points array is saved to the csv file in the source location with " +
          "the name Elkans_Result-Q3-" + output + ".csv")
    km.dataframe.to_csv(".\\Elkans_Result-Q3-"+function+output+".csv", encoding='utf-8', index=False)



# Main Program
if __name__ == "__main__":
    main()