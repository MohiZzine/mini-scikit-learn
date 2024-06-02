import numpy as np
from collections import Counter

def pairwise_distances(X, Y=None, metric='euclidean'):
    if Y is None:
        Y = X
    if metric == 'euclidean':
        return euclidean_distances(X, Y)
    elif metric == 'manhattan':
        return manhattan_distances(X, Y)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def euclidean_distances(X, Y):
    XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
    YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    distances = np.sqrt(XX + YY - 2 * np.dot(X, Y.T))
    return distances

def manhattan_distances(X, Y):
    return np.sum(np.abs(X[:, np.newaxis] - Y[np.newaxis, :]), axis=2)

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        metric = 'euclidean' if self.metric == 'minkowski' and self.p == 2 else 'manhattan'
        distances = pairwise_distances(X, self.X_train, metric=metric)
        neighbors_indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        
        if return_distance:
            neighbors_distances = np.sort(distances, axis=1)[:, :n_neighbors]
            return neighbors_distances, neighbors_indices
        else:
            return neighbors_indices

    def predict(self, X):
        neighbors_indices = self.kneighbors(X, return_distance=False)
        neighbor_votes = self.y_train[neighbors_indices]
        y_pred = [Counter(neighbor_votes[i]).most_common(1)[0][0] for i in range(len(neighbor_votes))]
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'p': self.p,
            'metric': self.metric,
            'metric_params': self.metric_params,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
