from sklearn.metrics import euclidean_distances
from numpy import argmax
import configparser
import pathlib


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
        Train model on data "data"
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
        :return: List of representants
        """
        pass

    def predict(self, user):
        """
        Given a user embedding, return the cluster label it belongs to
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

        return argmax(dists)
