import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generate synthetic data
np.random.seed(0)
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]  # flip axes for better plotting

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=4)
gmm.fit(X)

# Predict clusters
labels = gmm.predict(X)

# Plotting
plt.figure(figsize=(10, 6))

# Plot original data
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=40, cmap='viridis', alpha=0.6, label='Original Data')

# Plot clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', alpha=0.6, marker='s', label='Clustered Data')

# Plot centroids of clusters
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X', label='Cluster Centers')

plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.colorbar()
plt.show()