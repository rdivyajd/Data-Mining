#!/usr/bin/env python3
# Divya Rajendran 2018
# Data Sets taken from https://grouplens.org/datasets/movielens/ 100k and 10M data sets
#
# Executing the program:
# python ./Q3.py
# Currently the program is set to run for k = 10 and all 5 folds of test and train data sets for the
# three distance functions "cosine, euclidean and Minkowski"
# Manually update the file at the lines 239, 240 and 241 to have preferred values of k, N and distance function

import pandas as pd
import operator
import math


# read the file into a data frame
def text_process(file_name, columns):

    delim = "\t" if file_name == ".\\ml-100k\\u.data" else " " if file_name == ".\\ml-100k\\u.info" else "|" \
        if file_name in ["/nfs/nfs5/home/research/predrag/b565/ml-100k/u.genre", "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.item", "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.user"] else "	" \
        if file_name in ["/nfs/nfs5/home/research/predrag/b565/ml-100k/u1.base", "/nfs/nfs5/home/research/predrag/b565/ml-100k/u1.test",
                         "/nfs/nfs5/home/research/predrag/b565/ml-100k/u2.base", "/nfs/nfs5/home/research/predrag/b565/ml-100k/u2.test",
                         "/nfs/nfs5/home/research/predrag/b565/ml-100k/u3.base", "/nfs/nfs5/home/research/predrag/b565/ml-100k/u3.test",
                         "/nfs/nfs5/home/research/predrag/b565/ml-100k/u4.base", "/nfs/nfs5/home/research/predrag/b565/ml-100k/u4.test",
                         "/nfs/nfs5/home/research/predrag/b565/ml-100k/u5.base", "/nfs/nfs5/home/research/predrag/b565/ml-100k/u5.test"] else None

    data = pd.read_csv(file_name, delimiter=delim, header=None, names=columns, encoding='latin-1') \
        if file_name == "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.item" else pd.read_csv(file_name, delimiter=delim, header=None, names=columns)

    return(data)


# recommendation system
class recommendations:
    def __init__(self, k, function, dataset_name, n=None):  # k=43, knn=5
        # initialise variables
        self.k, self.n = k, n
        self.dataset = dataset_name
        self.function = function
        self.occupation_dic, self.users_dic, self.movies_dic, self.genre_dic = dict(), dict(), dict(), dict()
        self.inference_rating, self.average_rating = dict(), dict()
        self.top_k_neighbors = []
        self.test_dic = {}
        self.mad, self.avg_mad = 0, 0
        self.initialize_columns()
        self.read_static_info()
        self.read_datasets()
        self.total_movies = len(self.movies_df)
        self.df_to_dic()
        self.average_algorithm()

    def initialize_columns(self):
        # column names
        self.user_ratings_columns = ["user_id", "item_id", "rating", "timestamp"]
        self.data_info_columns = ["count", "type"]
        self.genre_column = ["genre", "genre_id"]
        self.movies_column = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
                        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                        "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        self.user_columns = ["user_id", "age", "gender", "occupation", "zip_code"]
        self.occupation_columns = ["occupation"]

    def read_static_info(self):
        # Read Static Data into variables
        genres = "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.genre"  # movie genres
        items = "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.item"  # movies
        users = "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.user"  # user details
        occupation = "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.occupation"  # occupation stand alone
        self.genre_df = text_process(genres, self.genre_column)
        self.movies_df = text_process(items, self.movies_column)
        self.users_df = text_process(users, self.user_columns)
        self.occupation_df = text_process(occupation, self.occupation_columns)

    def read_datasets(self):
        if self.dataset == "ml-100k":
            user_ratings = "/nfs/nfs5/home/research/predrag/b565/ml-100k/u.data"
            self.ratings_df = text_process(user_ratings, self.user_ratings_columns)
        elif self.dataset == "ml-100k_cross_fold":
            train_data = "/nfs/nfs5/home/research/predrag/b565/ml-100k/u"+str(self.n)+".base"
            self.ratings_df = text_process(train_data, self.user_ratings_columns)
            test_data = "/nfs/nfs5/home/research/predrag/b565/ml-100k/u"+str(self.n)+".test"
            self.test_df = text_process(test_data, self.user_ratings_columns)
            # test ratings dictionary
            for index, row in self.test_df.iterrows():
                self.test_dic[row["user_id"]] = {row["item_id"]: row["rating"]}
        elif self.dataset == "ml-10M100K":
            print("Not yet . . . ")

    # get the data frame data into dictionary
    def df_to_dic(self):
        # occupation dictionary
        for index, row in self.occupation_df.iterrows():
            self.occupation_dic[row["occupation"]] = index+1

        # genres dictionary
        for index, row in self.genre_df.iterrows():
            self.genre_dic[row["genre"]] = row["genre_id"]

        # print(self.genre_dic)

        # movie dictionary
        for index, row in self.movies_df.iterrows():
            self.movies_dic[row["movie_id"]] = {"genres": [set(), 0]}
        for index, row in self.movies_df.iterrows():
            for genre in ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                          "Romance", "Sci-Fi", "Thriller", "War", "Western"]:
                if row[genre] == 1:
                    self.movies_dic[row["movie_id"]]["genres"][0].add(genre)
                    self.movies_dic[row["movie_id"]]["genres"][1] += self.movies_dic[row["movie_id"]]["genres"][1] + \
                                                                     self.genre_dic[genre]

        sample_movies = {}
        # sample movies
        for movie in self.movies_dic.keys():
            sample_movies[movie] = [int(0), self.movies_dic[movie]["genres"][1]]


        # user dictionary
        for index, row in self.users_df.iterrows():
            self.users_dic[row["user_id"]] = {"age": 0, "gender": 0, "occupation": 0, "movie_rating": sample_movies}
            self.users_dic[row["user_id"]]["gender"] = 0 if row["gender"] == "M" else 1
            self.users_dic[row["user_id"]]["occupation"] = self.occupation_dic[row["occupation"]]

        # Adding Movie ratings by user
        for index, row in self.ratings_df.iterrows():
            self.users_dic[row["user_id"]]["movie_rating"][row["item_id"]][0] = int(row["rating"])

    def average_algorithm(self):
        for user in self.users_dic.keys():
            for movie in self.movies_dic.keys():
                if self.users_dic[user]["movie_rating"][movie][0] == 0:
                    average_rating = sum([self.users_dic[user2]["movie_rating"][movie][0]
                                          for user2 in self.users_dic.keys()
                                          if user != user2 and movie in self.users_dic[user2]["movie_rating"].keys()])
                    # inference calculation
                    self.inference_rating[user] = {}
                    rating = average_rating/(len(self.users_dic) - 1)
                    self.average_rating[user] = {movie: rating}

    # distance function
    def distance(self, point, dist_chk_point):
        if self.function == "euclidean":
            return math.sqrt(sum([abs(point[i] - dist_chk_point[i]) ** 2 for i in range(0, len(point))]))
        elif self.function == "cosine":
            num, point_dist, check_dist = 0, 0, 0
            for i in range(0, len(point)):
                point_dist += point[i] ** 2
                check_dist += dist_chk_point[i] ** 2
                num += point[i] * dist_chk_point[i]
            denom = (check_dist * point_dist) ** 0.5
            return 1 - float(num) / float(denom)
        elif self.function == "Minkowski":
            return sum([abs(point[i] - dist_chk_point[i]) ** 3 for i in range(0, len(point))]) ** (float(1) / float(3))
        elif self.function == "cityblock":
            return sum([abs(point[i] - dist_chk_point[i]) for i in range(0, len(point))])

    # nearest neighbor calculation
    def nearest_neighbor(self, point, dataset):

        distances = [(dataset[i], self.distance(point[1:], dataset[i][1:])) for i in range(len(dataset))
                     if point[0] != dataset[i][0]]
        distances.sort(key=operator.itemgetter(1))
        length = self.k if len(distances) > self.k else len(distances)
        self.top_k_neighbors = [distances[i][0] for i in range(length)]

    # inference on user rating for to be watched movies
    # for every user i and every movie they didn't watch j - find top k similar users
    def inference_user_rating(self):
        for user in self.users_dic.keys():
            self.inference_rating[user] = {}
            for movie in self.movies_dic.keys():
                dataset, average_rating, rating = [], 0, 0
                if self.users_dic[user]["movie_rating"][movie][0] == 0:
                    point = [user, self.users_dic[user]["age"], self.users_dic[user]["gender"],
                             self.users_dic[user]["occupation"], self.users_dic[user]["movie_rating"][movie][1]]
                    for user2 in self.users_dic.keys():
                        if user != user2 and movie in self.users_dic[user2]["movie_rating"].keys():
                            dataset.append([user2, self.users_dic[user2]["age"], self.users_dic[user2]["gender"],
                                            self.users_dic[user2]["occupation"],
                                            self.users_dic[user2]["movie_rating"][movie][1]])
                            average_rating += self.users_dic[user2]["movie_rating"][movie][0]
                    # inference calculation
                    self.nearest_neighbor(point, dataset)
                    rating = float(average_rating)/float(len(self.users_dic) - 1) if len(self.top_k_neighbors) == 0 \
                        else sum([self.users_dic[row[0]]["movie_rating"][movie][0] for row in self.top_k_neighbors])/self.k
                else:
                    rating = self.users_dic[user]["movie_rating"][movie][0]
                self.inference_rating[user][movie] = rating


    def measure_performance(self):
        self.inference_user_rating()
        sum_r_i_j, num = 0, 0
        if self.dataset == "ml-100k_cross_fold":
            count = 0
            for user in self.test_dic.keys():
                count += 1
                for movie in self.test_dic[user].keys():
                    r_i_j = 1 if movie in self.inference_rating[user].keys() else 0
                    num_i_j = abs(self.inference_rating[user][movie] - self.test_dic[user][movie]) if r_i_j == 1 else 0
                    sum_r_i_j += r_i_j
                    num += r_i_j * num_i_j
        else:
            for user in self.inference_rating.keys():
                # print("came here1")
                for movie in self.inference_rating[user].keys():
                    # print("came here2")
                    r_i_j = 1 if movie in self.test_dic[user].keys() else 0
                    num_i_j = 0 if r_i_j == 0 else abs(self.inference_rating[user][movie] - self.test_dic[user][movie])
                    sum_r_i_j += r_i_j
                    num += r_i_j * num_i_j

        self.mad = num/sum_r_i_j
        # print("Performance of recommendation system is: ", self.mad)

    def average_performace(self):
        sum_r_i_j, num = 0, 0
        if self.dataset == "ml-100k_cross_fold":
            for user in self.test_dic.keys():
                for movie in self.average_rating[user].keys():
                    r_i_j = 1 if movie in self.test_dic[user].keys() else 0
                    num_i_j = abs(self.average_rating[user][movie] - self.test_dic[user][movie]) if r_i_j == 1 else 0
                    sum_r_i_j += r_i_j
                    num += r_i_j * num_i_j
        self.avg_mad = num/sum_r_i_j

# Main Program Starts
if __name__ == "__main__":
    for k in [10]:  # [5, 10, 15]
        for n in [1, 2, 3, 4, 5]:
            for function in ["euclidean", "cosine", "Minkowski"]:
                reco = recommendations(k, function, "ml-100k_cross_fold", n)
                reco.measure_performance()
                reco.average_performace()
                print("The performance of our algorithm for distance function:", function, " and k: ", k,
                      " and cross fold N: ", n, " is: ", round(reco.mad, 4))
                print("\nThe average algorithm performance is ", reco.avg_mad, "\n")

    # reco = recommendations(k, function, "ml-100k_cross_fold", n)
    # reco.measure_performance()
    # reco.average_performace()
    # print("The performance of our algorithm for distance function:", function, " and k: ", k,
    #       " and cross fold N: ", n, " is: ", round(reco.mad, 4))
    # print("\nThe average algorithm performance is ", reco.avg_mad, "\n")
