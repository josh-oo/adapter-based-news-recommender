from matplotlib import pyplot as plt

from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import KMeans
import numpy as np

class KMeansWrapper(ClusteringAlg):
    def train(self, X):
        model = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto").fit(X)
        self.model = model

    def extract_representations(self):
        centers = self.model.cluster_centers_
        labels = self.model.predict(centers)
        self.representants = (labels, centers)

    def predict(self, user):
        return self.model.predict(user[np.newaxis, ...])

    def visualize(self, data, user=None, representant=None):
        labels = self.model.predict(data)
        super().visualize(data, labels, user, representant)
