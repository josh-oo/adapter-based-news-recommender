from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import euclidean_distances


class AgglomorativeWrapper(ClusteringAlg):

    def train(self, data):
        labels = AgglomerativeClustering(n_clusters = self.n_clusters).fit_predict(data)
        self.labels = labels

    def centroids(self, X):
        centers = np.zeros(shape=(self.n_clusters, len(X[0]))) # no clusters, and dimensionality
        for label in range(self.n_clusters):
            centers[label] = np.mean(X[self.labels == label], axis=0)
        return list(range(self.n_clusters)), centers

    def medoids(self, X):
        centers = np.zeros(shape=(self.n_clusters, len(X[0])))
        for label in range(self.n_clusters):
            centroid = np.mean(X[self.labels == label], axis=0)
            dists = euclidean_distances(centroid.reshape(1, -1), X[self.labels == label])
            centers[label] = X[self.labels == label][np.argmin(dists[0])]
        return list(range(self.n_clusters)), centers

    def predict(self, user):
        labels, locations = self.representants
        dists = euclidean_distances(user.reshape(1, -1), locations)
        return np.argmin(dists[0])

    def visualize(self, data, points=None):
        super().visualize(data, self.labels, points)
