import numpy as np

from src.clustering.kmeans import train, interpret, extract_representations, predict, suggest

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
model = train(X)


user = [12, 3]

representations = extract_representations(model)

prediction = predict(model, user)
print(f"You are labeled as cluster {prediction[0]}")

cluster_representant = interpret(prediction, representations)
print(f"Your representant lies at {cluster_representant}")

user_suggestion = suggest(cluster_representant, representations)
print(f"Would you like to see a user from {user_suggestion}")
