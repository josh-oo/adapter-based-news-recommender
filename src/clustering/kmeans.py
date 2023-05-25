from numpy import argmax
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances


def train(X):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    return kmeans


def interpret(prediction, representations):
    for label, location in zip(representations[0], representations[1]):
        if label == prediction:
            return location
    raise Exception("Matching label not found")


def extract_representations(model):
    centers = model.cluster_centers_
    labels = model.predict(centers)
    return (labels, centers)


def predict(model, user):
    return model.predict([user])


def suggest(status, representations):
    #for simplicity take cluster that is furthest away
    labels, locations = representations
    dists = euclidean_distances(status.reshape(1, -1), locations)

    return argmax(dists)