import numpy as np

class KMeans:
    """
    KMeans clustering algorithm implementation.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids
        to generate.
    init : {'k-means++', 'random'}, optional, default: 'k-means++'
        Method for initialization, defaults to 'k-means++'.
    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds.
    max_iter : int, optional, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol : float, optional, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int, optional
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic.

    Attributes
    ----------
    cluster_centers_ : array, shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : array, shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, random_state=None):
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
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training instances to cluster.
        """
        best_inertia = np.inf
        best_centers = None
        best_labels = None

        for _ in range(self.n_init):
            centers = self._initialize_centers(X)
            for _ in range(self.max_iter):
                labels = self._assign_labels(X, centers)
                new_centers = self._update_centers(X, labels)
                inertia = self._compute_inertia(X, labels, new_centers)
                if np.sum((new_centers - centers) ** 2) <= self.tol:
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
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_

    def fit_transform(self, X):
        """
        Compute clustering and transform X to cluster-distance space.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : array, shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        self.fit(X)
        return self.transform(X)

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self._assign_labels(X, self.cluster_centers_)

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : array, shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)

    def score(self, X):
        """
        Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data.

        Returns
        -------
        score : float
            Opposite of the inertia.
        """
        return -self.inertia_

    def _initialize_centers(self, X):
        """
        Initialize the cluster centers.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        centers : array, shape (n_clusters, n_features)
            Initial cluster centers.
        """
        if self.init == 'k-means++':
            return self._kmeans_plus_plus_init(X)
        elif self.init == 'random':
            return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        else:
            raise ValueError("Unsupported init method.")

    def _kmeans_plus_plus_init(self, X):
        """
        Initialize centers using the k-means++ method.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        centers : array, shape (n_clusters, n_features)
            Initial cluster centers.
        """
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
        """
        Assign labels to each sample in X based on the nearest center.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data to predict.
        centers : array, shape (n_clusters, n_features)
            Cluster centers.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

    def _update_centers(self, X, labels):
        """
        Update cluster centers.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data.
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.

        Returns
        -------
        centers : array, shape (n_clusters, n_features)
            Updated cluster centers.
        """
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def _compute_inertia(self, X, labels, centers):
        """
        Compute the inertia.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data.
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        centers : array, shape (n_clusters, n_features)
            Cluster centers.

        Returns
        -------
        inertia : float
            Sum of squared distances of samples to their closest cluster center.
        """
        return np.sum([np.linalg.norm(X[labels == i] - centers[i]) ** 2 for i in range(self.n_clusters)])
