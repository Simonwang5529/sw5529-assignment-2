import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.centroids = None

    def initialize_centroids(self, X):
        if self.init == 'random':
            random_idx = np.random.permutation(X.shape[0])
            return X[random_idx[:self.n_clusters]]


    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_iter):
            labels = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def predict(self, X):
        return self.assign_clusters(X)