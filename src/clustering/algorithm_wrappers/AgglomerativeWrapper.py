from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import euclidean_distances


class AgglomorativeWrapper(ClusteringAlg):

    def extract_representations(self, X):
        labels = AgglomerativeClustering(n_clusters = self.n_clusters).fit_predict(X)
        self.labels = labels
        centers = []
        for label in range(self.n_clusters):
            centers.append(np.mean(X[labels == label], axis=0))
        self.representants = (list(range(self.n_clusters)), centers)

    def predict(self, user):
        labels, locations = self.representants
        dists = euclidean_distances(user.reshape(1, -1), locations)
        return np.argmin(dists)

    def visualize(self, data, user=None, representant=None):
        super().visualize(data, self.labels, user, representant)
