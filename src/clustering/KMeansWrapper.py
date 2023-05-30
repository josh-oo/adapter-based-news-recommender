from ClusteringAlg import ClusteringAlg
from sklearn.cluster import KMeans


class KMeansWrapper(ClusteringAlg):
    def train(self, X):
        model = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)
        self.model = model

    def extract_representations(self):
        centers = self.model.cluster_centers_
        labels = self.model.predict(centers)
        self.representants = (labels, centers)

    def predict(self, user):
        return self.model.predict([user])
