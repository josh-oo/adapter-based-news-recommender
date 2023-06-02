from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances
from numpy import argmax
import configparser
import pathlib
import numpy as np

class ClusteringAlg:
    """
    Interface class that contains methods used  by all Clustering algorithms.
    """
    def __init__(self):
        self.model = None
        self.representants = None # the "user story" of each cluster
        self.labels = None
        self.get_cluster_config()

    def get_cluster_config(self):
        """
        Gets parameters from config file and sets them as class attributes
        """
        config = configparser.ConfigParser()
        file_path = pathlib.Path(__file__).parent.parent.parent.parent / 'config.ini'
        config.read(file_path)
        self.n_clusters = int(config['Clustering']['NoClusters'])
        self.config = config

    def train(self, data):
        """
        Train model on data "data".
        NEEDS TO BE IMPLEMENTED IN CHILD CLASS
        :param data: numpy array of datapoints
        :return: model
        """
        pass

    def interpret(self, comparison_label):
        """
        Get location of the represantant of cluster comparison_label.
        This is not a simple array access as long as we are not sure that every sklearn method return the labels
        in the correct order.
        """
        for label, location in zip(self.representants[0], self.representants[1]):
            if label == comparison_label:
                return location
        raise Exception("Matching label not found")

    def extract_representations(self, X, mode='centroid'):
        """
        Calculates the representats for each cluster using the model stored at self.model.
        This can occur in two ways:
        1. centroid: The mean of all points in the cluster. This is not a real user, and is not representative if
        the cluster is curved.
        2. medoids: The cluster point which is closest to the centroid.
        NEEDS TO BE IMPLEMENTED IN CHILD CLASS
        :return: List of representants
        """
        if mode == 'centroid':
            self.representants = self.centroids(X)
        elif mode == 'medoid':
            self.representants = self.medoids(X)
        else:
            raise Exception("Not a valid mode")


    def centroids(self, X):
        """
        The mean of all points in the cluster. This is not a real user, and is not representative if
        the cluster is curved.
        NEEDS TO BE IMPLEMENTED IN CHILD CLASS
        :return: list of points
        """
        pass

    def medoids(self, X):
        """
        The cluster point which is closest to the centroid.
        NEEDS TO BE IMPLEMENTED IN CHILD CLASS
        :return: list of points
        """
        pass

    def predict(self, user):
        """
        Given a user embedding, return the cluster label it belongs to
        NEEDS TO BE IMPLEMENTED IN CHILD CLASS
        :return: cluster label
        """
        pass

    def suggest(self, comparing_vector, metric='max'):
        """
        Given a vector, representing a user in space, suggest another user (a represantant) from another cluster.
        TODO give different metrics for choosing the cluster model
        :param comparing_vector: user embedding
        :return: another user embedding
        """
        # for simplicity for now take cluster that is furthest away
        labels, locations = self.representants
        dists = euclidean_distances(comparing_vector.reshape(1, -1), locations)
        return argmax(dists), locations[argmax(dists)]

    def visualize(self, data, labels, user, representant):
        #TODO: plot colour also
        fig = plt.figure()
        n_components = len(data[0])
        if n_components == 2:
            ax = fig.add_subplot(111)
            for cluster in range(max(labels)):
                cluster_data = data[labels == cluster]
                result = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=1, alpha=0.1)
                c = result.get_facecolor()[0]
                repr = self.representants[1][cluster]
                ax.scatter(repr[0], repr[1], s=5, color=c)
            if user is not None:
                ax.scatter(user[0], user[1], c='red', s=10, label="User")
                ax.legend()
            if representant is not None:
                ax.scatter(representant[0], representant[1], color='black', s=10, label="Suggestion")
                ax.legend()
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            for cluster in range(max(labels)):
                cluster_data = data[labels == cluster]
                result = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:,2], s=1, alpha=0.2)
                c = result.get_facecolor()[0]
                repr = self.representants[1][cluster]
                ax.scatter(repr[0], repr[1], repr[2], s=5, color=c, alpha=1)
            if user is not None:
                ax.scatter(user[0], user[1], user[2], c='red', s=10, label="User")
                ax.legend()
            if representant is not None:
                ax.scatter(representant[0], representant[1], representant[2], c='black', s=10, label="Representant")
                ax.legend()
        # plt.title(title, fontsize=18)

        def measure_performance(self, metric='chi'):
            """
            Measure the cluster quality. For details on metrics see: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation.
            Note that this is not suitable to decide between different clustering techniques, but very useful to choose
            hyperparameters for one clustering algorithm.
            :param self:
            :param metric: Options are 'chi'=Calinski-Harabasz Index or 'dbi'=Davies-Bouldin Index
            :return:
            """
            pass

