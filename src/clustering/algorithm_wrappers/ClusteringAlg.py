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

        self.get_cluster_config()

    def get_cluster_config(self):
        """
        Gets parameters from config file and sets them as class attributes
        """
        config = configparser.ConfigParser()
        file_path = pathlib.Path(__file__).parent.parent.parent.parent / 'config.ini'
        config.read(file_path)
        self.n_clusters = int(config['Clustering']['NoClusters'])

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

    def extract_representations(self):
        """
        Calculates the representats for each cluster using the model stored at self.model.
        This can occur in several ways, the simplest being the mean / cluster center.
        NEEDS TO BE IMPLEMENTED IN CHILD CLASS
        :return: List of representants
        """
        pass

    def predict(self, user):
        """
        Given a user embedding, return the cluster label it belongs to
        NEEDS TO BE IMPLEMENTED IN CHILD CLASS
        :return: cluster label
        """
        pass

    def suggest(self, comparing_vector):
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
        fig = plt.figure()
        n_components = len(data[0])
        if n_components == 2:
            ax = fig.add_subplot(111)
            for cluster in range(self.n_clusters):
                cluster_data = data[labels == cluster]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=1, alpha=0.5)
            if user is not None:
                ax.scatter(user[0], user[1], c='red', s=10, label="User")
                ax.legend()
            if representant is not None:
                ax.scatter(representant[0], representant[1], c='black', s=10, label="Suggestion")
                ax.legend()
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            for cluster in range(self.n_clusters):
                cluster_data = data[labels == cluster]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:,2], s=1, alpha=0.5)
            if user is not None:
                ax.scatter(user[0], user[1], user[2], c='red', s=10, label="User")
                ax.legend()
            if representant is not None:
                ax.scatter(representant[0], representant[1], representant[2], c='black', s=10, label="Representant")
                ax.legend()
        # plt.title(title, fontsize=18)
