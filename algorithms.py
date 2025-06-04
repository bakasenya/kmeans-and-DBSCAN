import numpy as np
import pandas as pd
from sklearn.utils import check_array
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2), axis=1)

def _euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=1)

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2), axis=1)


class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
        the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
        and/or encapsulate the necessary mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')
        n_samples, __ = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]
        
        for _ in range(self.max_iter):
            distances = np.array([euclidean_distance(X, center) for center in self.cluster_centers_])
            self.labels_ = np.argmin(distances, axis=0)
            
            new_updated_centers = np.array([X[self.labels_ == k].mean(axis=0) 
                                for k in range(self.n_clusters)])
            
            if np.all(self.cluster_centers_ == new_updated_centers):
                break
                
            self.cluster_centers_ = new_updated_centers
            
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances ('euclidean', 'manhattan', 'chebyshev').
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        
        # Constants for cluster labels
        self.UNCLASSIFIED = -1
        self.NOISE = -2

    def _get_neighbors(self, X, point_idx):
        """
        Find all points within eps distance of the point at point_idx using vectorized operations.
        """
        if self.metric == 'euclidean':
            distances = euclidean_distance(X, X[point_idx:point_idx+1])
        elif self.metric == 'manhattan':
            distances = manhattan_distance(X, X[point_idx:point_idx+1])
        elif self.metric == 'chebyshev':
            distances = chebyshev_distance(X, X[point_idx:point_idx+1])
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
            
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, cluster_id, neighbors=None):
        """
        Expand cluster from a core point.
        Returns True if cluster was expanded, False if point is not a core point.
        """
        if neighbors is None:
            neighbors = self._get_neighbors(X, point_idx)
        
        if len(neighbors) < self.min_samples:
            self.labels_[point_idx] = self.NOISE
            return False
        
        # Mark points as part of the cluster
        self.labels_[neighbors] = cluster_id
        
        # Process neighbors
        seeds = set(neighbors) - {point_idx}
        while seeds:
            current_point = seeds.pop()
            if self.labels_[current_point] == self.NOISE:
                self.labels_[current_point] = cluster_id
                continue
                
            current_neighbors = self._get_neighbors(X, current_point)
            
            if len(current_neighbors) >= self.min_samples:
                for neighbor_idx in current_neighbors:
                    if self.labels_[neighbor_idx] in [self.UNCLASSIFIED, self.NOISE]:
                        if self.labels_[neighbor_idx] == self.UNCLASSIFIED:
                            seeds.add(neighbor_idx)
                        self.labels_[neighbor_idx] = cluster_id
        
        return True

    def fit(self, X: np.ndarray, y=None):
        """
        Perform DBSCAN clustering.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation
        X = check_array(X, accept_sparse='csr')
        n_samples = X.shape[0]
        
        # Initialize labels
        self.labels_ = np.full(n_samples, self.UNCLASSIFIED)
        
        # Current cluster ID
        cluster_id = 0
        
        # Pre-compute neighbors for all points to avoid redundant calculations
        neighbors_dict = {}
        
        # Process each unclassified point
        for point_idx in range(n_samples):
            if self.labels_[point_idx] != self.UNCLASSIFIED:
                continue
                
            # Get or compute neighbors
            if point_idx not in neighbors_dict:
                neighbors_dict[point_idx] = self._get_neighbors(X, point_idx)
                
            if self._expand_cluster(X, point_idx, cluster_id, neighbors_dict[point_idx]):
                cluster_id += 1
                    
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels.
        """
        self.fit(X)
        return self.labels_

# if __name__ == "__main__":
#     # X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

#     # # Test CustomKMeans
#     # custom_kmeans = CustomKMeans(n_clusters=4, max_iter=300, random_state=42)
#     # custom_kmeans.fit(X)
#     # custom_labels = custom_kmeans.labels_

#     # # Test Sklearn's KMeans for comparison
#     # sklearn_kmeans = KMeans(n_clusters=4, random_state=42)
#     # sklearn_labels = sklearn_kmeans.fit_predict(X)

#     # # Plot results
#     # fig, ax = plt.subplots(1, 2, figsize=(12, 6))

#     # # CustomKMeans result
#     # ax[0].scatter(X[:, 0], X[:, 1], c=custom_labels, cmap='viridis', s=30)
#     # ax[0].scatter(custom_kmeans.cluster_centers_[:, 0], custom_kmeans.cluster_centers_[:, 1],
#     #             c='red', s=200, alpha=0.75, marker='X')
#     # ax[0].set_title("Custom KMeans Clustering")

#     # # Sklearn KMeans result
#     # ax[1].scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', s=30)
#     # ax[1].scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1],
#     #             c='red', s=200, alpha=0.75, marker='X')
#     # ax[1].set_title("Sklearn KMeans Clustering")

#     # plt.show()
    
#     # Generate synthetic data
#     X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

#     # Create an instance of CustomDBSCAN (using our implementation)
#     custom_dbscan = CustomDBSCAN(eps=0.5, min_samples=5, metric='euclidean')
#     custom_dbscan.fit(X)

#     # Use sklearn's DBSCAN
#     sklearn_dbscan = SklearnDBSCAN(eps=0.5, min_samples=5)
#     sklearn_labels = sklearn_dbscan.fit_predict(X)

#     # Compare labels from both implementations (CustomDBSCAN vs. sklearn DBSCAN)
#     print(f"CustomDBSCAN labels: {custom_dbscan.labels_}")
#     print(f"Sklearn DBSCAN labels: {sklearn_labels}")

#     # Calculate Adjusted Rand Index to measure the similarity between the two label sets
#     ari = adjusted_rand_score(custom_dbscan.labels_, sklearn_labels)
#     print(f"Adjusted Rand Index (ARI): {ari}")

#     # Plot the results for visual comparison
#     plt.figure(figsize=(8, 6))

#     # Plot CustomDBSCAN results
#     plt.subplot(1, 2, 1)
#     plt.scatter(X[:, 0], X[:, 1], c=custom_dbscan.labels_, cmap='viridis', marker='o')
#     plt.title('CustomDBSCAN Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')

#     # Plot Sklearn DBSCAN results
#     plt.subplot(1, 2, 2)
#     plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', marker='o')
#     plt.title('Sklearn DBSCAN Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')

#     plt.tight_layout()
#     plt.show()

