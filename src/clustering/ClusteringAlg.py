from sklearn.metrics import euclidean_distances
from numpy import argmax


class ClusteringAlg:
    def __init__(self):
        self.model = None
        self.representants = None

    def train(self, X):
        pass

    def interpret(self, prediction):
        for label, location in zip(self.representants[0], self.representants[1]):
            if label == prediction:
                return location
        raise Exception("Matching label not found")

    def extract_representations(self):
        pass

    def predict(self, user):
        pass

    def suggest(self, comparing_vector):
        # for simplicity take cluster that is furthest away
        labels, locations = self.representants
        dists = euclidean_distances(comparing_vector.reshape(1, -1), locations)

        return argmax(dists)
