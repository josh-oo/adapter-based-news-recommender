import configparser
import pathlib
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import euclidean_distances
import plotly.graph_objects as go


class KMeansWrapper:
    def __init__(self):
        self.model = None
        self.representants = None  # the "user story" of each cluster
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

    def extract_representations(self, X, mode='medoid'):
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

        self.repr_indeces = [np.nonzero(np.all(X == repr, axis=1))[0][0] for repr in self.representants[1]]

    def train(self, X):
        model = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto").fit(X)
        self.model = model
        self.labels = model.predict(X)

    def centroids(self, X):
        centers = self.model.cluster_centers_
        return list(range(self.n_clusters)), centers

    def medoids(self, X):
        centers = self.model.cluster_centers_
        repr = np.zeros(shape=(self.n_clusters, len(X[0])))
        for label, center in enumerate(centers):
            dists = euclidean_distances(center.reshape(1, -1), X[self.labels == label])
            repr[label] = X[self.labels == label][np.argmin(dists[0])]
        return list(range(self.n_clusters)), repr

    def predict(self, user):
        return self.model.predict(user[np.newaxis, ...])[0]

    def visualize(self, data, repr, points):
        labels = self.model.labels_
        self.visualize_old(data, labels, repr, points)

    def measure_performance(self, X, metric='chi'):
        if metric == 'chi':
            return metrics.calinski_harabasz_score(X, self.model.labels_)
        elif metric == 'dbi':
            return metrics.davies_bouldin_score(X, self.model.labels_)
        else:
            raise Exception('Not a valid value for the parameter "metric"')

    def suggest(self, comparing_vector, metric='max'):
        """
        Given a vector, representing a user in space, suggest another user (a represantant) from another cluster.
        :param comparing_vector: user embedding
        :param metric: options 1) 'max' gives the cluster whos medoid is furthest away from comparing vector. 2) a float
        between 0 and 1 gives the medoid represented by that percentage of distance from highest of lowest, e.g. 0.66
        gives the medoid whos at 2/3 of the max distance
        :return: another user embedding
        """
        # for simplicity for now take cluster that is furthest away
        labels, locations = self.representants
        distsance_ind = np.argsort(euclidean_distances(comparing_vector.reshape(1, -1), locations))[0]
        if metric == 'max':
            return distsance_ind[-1], locations[distsance_ind[-1]]
        else:
            try:
                percentage = int(metric)
                if not 0 < percentage <= 100:  # check if in range
                    raise ValueError
                index = int(len(distsance_ind) * percentage / 100) - 1
                return distsance_ind[index], locations[distsance_ind[index]]
            except ValueError:  # checks if float
                print("Not a valid suggestion metric. Pass value 'max' or percentage in between 1 and 100")

    def get_cluster_representant(self, id):
        if id > len(self.representants[0]):
            raise ValueError
        labels, locations = self.representants
        return locations[id], self.repr_indeces[id]

    def visualize_old(self, data, labels, repr, points=None) -> go:
        # TODO: plot colour also
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2],
                                   mode='markers',
                                   marker=dict(size=1),
                                   text=labels,
                                   hoverinfo="text+name",
                                   marker_color=labels, opacity=1, name="Historic Users"))
        fig.add_trace(go.Scatter3d(x=repr[:, 0], y=repr[:, 1], z=repr[:, 2],
                                   mode='markers',
                                   marker=dict(
                                       size=2),
                                   text=list(range(len(repr))),
                                   hoverinfo="text+name",
                                   marker_color=list(range(len(repr))), name="Exemplars"))

        for (label, point) in points:
            fig.add_trace(
                go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                             marker_symbol=['diamond'],
                             marker=dict(
                                 size=3),
                             hoverinfo="name",
                             mode='markers', name=label)
                # marker_color=[self.predict(user)]) # todo
            )
        fig.update_layout(height=800)  # todo configure
        self.figure = fig
