from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances

from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import OPTICS
import numpy as np

class OpticsWrapper(ClusteringAlg):
    def extract_representations(self, data):
        min_sample = int(self.config['Clustering.Optics']['min_samples'])
        labels = OPTICS(min_samples=min_sample).fit_predict(data)
        self.labels = labels
        centers = []
        for label in range(max(self.labels)):
            centers.append(np.mean(data[labels == label], axis=0))
        self.representants = (list(range(max(self.labels))), centers)

    def predict(self, user):
        labels, locations = self.representants
        dists = euclidean_distances(user.reshape(1, -1), locations)
        return np.argmin(dists)

    def visualize(self, data, user=None, representant=None):
        super().visualize(data, self.labels, user, representant)
