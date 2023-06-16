from matplotlib import pyplot as plt
from sklearn import metrics
from src.clustering.algorithm_wrappers.ClusteringAlg import ClusteringAlg
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import euclidean_distances


class KMeansWrapper(ClusteringAlg):

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

    def visualize(self, data, points):
        labels = self.model.labels_
        super().visualize(data, labels, points)

    def measure_performance(self, X, metric='chi'):
        if metric == 'chi':
            return metrics.calinski_harabasz_score(X, self.model.labels_)
        elif metric == 'dbi':
            return metrics.davies_bouldin_score(X, self.model.labels_)
        else:
            raise Exception('Not a valid value for the parameter "metric"')