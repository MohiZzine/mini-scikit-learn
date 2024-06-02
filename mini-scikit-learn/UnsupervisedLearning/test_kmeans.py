import unittest
import numpy as np
from kmeans import KMeans
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SklearnKMeans

class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.X_blobs, _ = make_blobs(n_samples=1000, centers=5, random_state=42)
        self.X_circles, _ = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.X_circles_scaled = self.scaler.fit_transform(self.X_circles)

    def test_kmeans_blobs(self):
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(self.X_blobs)
        labels = kmeans.predict(self.X_blobs)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(self.X_blobs, labels)
        print(f"Custom K-Means Inertia: {inertia:.4f}")
        print(f"Custom K-Means Silhouette Score: {silhouette:.4f}")

        sklearn_kmeans = SklearnKMeans(n_clusters=5, random_state=42, n_init=10)
        sklearn_kmeans.fit(self.X_blobs)
        sklearn_labels = sklearn_kmeans.predict(self.X_blobs)
        sklearn_inertia = sklearn_kmeans.inertia_
        sklearn_silhouette = silhouette_score(self.X_blobs, sklearn_labels)
        print(f"Scikit-Learn K-Means Inertia: {sklearn_inertia:.4f}")
        print(f"Scikit-Learn K-Means Silhouette Score: {sklearn_silhouette:.4f}")

        self.assertAlmostEqual(inertia, sklearn_inertia, places=0)
        self.assertAlmostEqual(silhouette, sklearn_silhouette, places=1)

    def test_kmeans_circles(self):
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(self.X_circles_scaled)
        labels = kmeans.predict(self.X_circles_scaled)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(self.X_circles_scaled, labels)
        print(f"Custom K-Means Inertia: {inertia:.4f}")
        print(f"Custom K-Means Silhouette Score: {silhouette:.4f}")

        sklearn_kmeans = SklearnKMeans(n_clusters=2, random_state=42, n_init=10)
        sklearn_kmeans.fit(self.X_circles_scaled)
        sklearn_labels = sklearn_kmeans.predict(self.X_circles_scaled)
        sklearn_inertia = sklearn_kmeans.inertia_
        sklearn_silhouette = silhouette_score(self.X_circles_scaled, sklearn_labels)
        print(f"Scikit-Learn K-Means Inertia: {sklearn_inertia:.4f}")
        print(f"Scikit-Learn K-Means Silhouette Score: {sklearn_silhouette:.4f}")

        self.assertAlmostEqual(inertia, sklearn_inertia, places=0)
        self.assertAlmostEqual(silhouette, sklearn_silhouette, places=1)

if __name__ == "__main__":
    unittest.main()
