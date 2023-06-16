from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances

from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import OPTICS
import numpy as np

class OpticsWrapper(ClusteringAlg):
    def train(self, data):
        min_sample = int(self.config['Clustering.Optics']['min_samples'])
        labels = OPTICS(min_samples=min_sample).fit_predict(data)
        self.labels = labels

    def centroids(self, X):
        centers = np.zeros(shape=(max(self.labels), len(X[0]))) # no clusters, and dimensionality
        for label in range(max(self.labels)):
            centers[label] = np.mean(X[self.labels == label], axis=0)
        return list(range(max(self.labels))), centers

    def medoids(self, X):
        min_sample = int(self.config['Clustering.Optics']['min_samples'])
        labels = OPTICS(min_samples=min_sample).fit_predict(X)
        self.labels = labels
        centers = np.zeros(shape=(max(labels), len(X[0])))
        for label in range(max(labels)):
            centroid = np.mean(X[labels == label], axis=0)
            dists = euclidean_distances(centroid.reshape(1, -1), X[labels == label])
            centers[label] = X[labels == label][np.argmin(dists[0])]
        return list(range(max(labels))), centers

    def predict(self, user):
        labels, locations = self.representants
        dists = euclidean_distances(user.reshape(1, -1), locations)
        return np.argmin(dists)

    def visualize(self, data, user=None, representant=None):
        super().visualize(data, self.labels, user, representant)
