from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import euclidean_distances


class AgglomorativeWrapper(ClusteringAlg):

    def extract_representations(self, X):
        labels = AgglomerativeClustering(n_clusters = self.n_clusters).fit_predict(X)
        centers = []
        for label in range(self.n_clusters):
            indeces = labels == label
            centers.append(np.mean(X[indeces]))
        self.representants = (list(range(self.n_clusters)), centers)

    def predict(self, user):
        labels, locations = self.representants
        dists = euclidean_distances(user.reshape(1, -1), locations)
        return np.argmin(dists)
