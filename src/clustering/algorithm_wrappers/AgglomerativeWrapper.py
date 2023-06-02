from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import euclidean_distances


class AgglomorativeWrapper(ClusteringAlg):

    def centroids(self, X):
        labels = AgglomerativeClustering(n_clusters = self.n_clusters).fit_predict(X)
        self.labels = labels
        centers = []
        for label in range(self.n_clusters):
            centers.append(np.mean(X[labels == label], axis=0))
        return list(range(self.n_clusters)), centers

    def medoids(self, X):
        labels = AgglomerativeClustering(n_clusters = self.n_clusters).fit_predict(X)
        self.labels = labels
        repr = []
        for label in range(self.n_clusters):
            centroid = np.mean(X[labels == label], axis=0)
            dists = euclidean_distances(centroid.reshape(1, -1), X[labels == label])
            repr.append(X[self.labels == label][np.argmin(dists[0])])
        return list(range(self.n_clusters)), repr

    def predict(self, user):
        labels, locations = self.representants
        dists = euclidean_distances(user.reshape(1, -1), locations)
        return np.argmin(dists[0])

    def visualize(self, data, user=None, representant=None):
        super().visualize(data, self.labels, user, representant)
