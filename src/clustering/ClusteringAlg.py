from sklearn.metrics import euclidean_distances
from numpy import argmax


class ClusteringAlg:
    def __init__(self):
        self.model = None
        self.representants = None

    def train(self, X):
        pass

    @staticmethod
    def interpret(prediction, representations):
        for label, location in zip(representations[0], representations[1]):
            if label == prediction:
                return location
        raise Exception("Matching label not found")

    def extract_representations(self):
        pass

    def predict(self, user):
        pass

    @staticmethod
    def suggest(status, representations):
        # for simplicity take cluster that is furthest away
        labels, locations = representations
        dists = euclidean_distances(status.reshape(1, -1), locations)

        return argmax(dists)
