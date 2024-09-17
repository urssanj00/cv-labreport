import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# Generate synthetic data
X, y = make_classification(n_samples=20, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Create an instance of Neighbours Classifier and fit the data.
k = 5  # Number of neighbors to consider
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# Plot decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)

# Define a new test point
test_point = np.array([[-0.3, 1]])  # Coordinates of the test point
plt.scatter(test_point[:, 0], test_point[:, 1], color='blue', marker='x', label='Test Point', s=100)

# Find nearest neighbors of the test point
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
distances, indices = nn.kneighbors(test_point)

# Highlight nearest neighbors and add legend
for i in indices[0]:
    plt.scatter(X[i, 0], X[i, 1], color='blue', marker='o', s=100, facecolors='none', label='Neighbor' if i == indices[0][0] else '')

# Predict the class of the test point
pred = clf.predict(test_point)
plt.text(test_point[0, 0], test_point[0, 1], f'Class {pred[0]}', color='blue', fontsize=12, ha='center', va='bottom')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"2-Class classification (k = {k})")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()