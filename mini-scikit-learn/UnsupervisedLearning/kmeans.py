import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        best_inertia = np.inf
        best_centers = None
        best_labels = None

        for _ in range(self.n_init):
            centers = self._initialize_centers(X)
            for _ in range(self.max_iter):
                labels = self._assign_labels(X, centers)
                new_centers = self._update_centers(X, labels)
                inertia = self._compute_inertia(X, labels, new_centers)
                if np.sum((new_centers - centers)**2) <= self.tol:
                    break
                centers = new_centers
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def predict(self, X):
        return self._assign_labels(X, self.cluster_centers_)

    def transform(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)

    def score(self, X):
        return -self.inertia_

    def _initialize_centers(self, X):
        if self.init == 'k-means++':
            return self._kmeans_plus_plus_init(X)
        elif self.init == 'random':
            return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        else:
            raise ValueError("Unsupported init method.")

    def _kmeans_plus_plus_init(self, X):
        np.random.seed(self.random_state)
        centers = [X[np.random.choice(X.shape[0])]]
        while len(centers) < self.n_clusters:
            dist_sq = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2), axis=1)
            prob = dist_sq / np.sum(dist_sq)
            cumulative_prob = np.cumsum(prob)
            r = np.random.rand()
            for i, p in enumerate(cumulative_prob):
                if r < p:
                    centers.append(X[i])
                    break
        return np.array(centers)

    def _assign_labels(self, X, centers):
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

    def _update_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def _compute_inertia(self, X, labels, centers):
        return np.sum([np.linalg.norm(X[labels == i] - centers[i])**2 for i in range(self.n_clusters)])

# Testing the implementation
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SklearnKMeans

def test():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

    # Initialize and train our K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)

    # Predictions and evaluation
    labels = kmeans.predict(X)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    print(f"Custom K-Means Inertia: {inertia:.4f}")
    print(f"Custom K-Means Silhouette Score: {silhouette:.4f}")

    # Compare with Scikit-Learn's K-Means
    sklearn_kmeans = SklearnKMeans(n_clusters=5, random_state=42)
    sklearn_kmeans.fit(X)
    sklearn_labels = sklearn_kmeans.predict(X)
    sklearn_inertia = sklearn_kmeans.inertia_
    sklearn_silhouette = silhouette_score(X, sklearn_labels)
    print(f"Scikit-Learn K-Means Inertia: {sklearn_inertia:.4f}")
    print(f"Scikit-Learn K-Means Silhouette Score: {sklearn_silhouette:.4f}")

test()
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

def test2():
    # Generate the make_circles dataset
    X, _ = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and train our K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)

    # Predictions and evaluation
    labels = kmeans.predict(X_scaled)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    print(f"Custom K-Means Inertia: {inertia:.4f}")
    print(f"Custom K-Means Silhouette Score: {silhouette:.4f}")

    # Compare with Scikit-Learn's K-Means
    sklearn_kmeans = SklearnKMeans(n_clusters=2, random_state=42)
    sklearn_kmeans.fit(X_scaled)
    sklearn_labels = sklearn_kmeans.predict(X_scaled)
    sklearn_inertia = sklearn_kmeans.inertia_
    sklearn_silhouette = silhouette_score(X_scaled, sklearn_labels)
    print(f"Scikit-Learn K-Means Inertia: {sklearn_inertia:.4f}")
    print(f"Scikit-Learn K-Means Silhouette Score: {sklearn_silhouette:.4f}")

test2()
