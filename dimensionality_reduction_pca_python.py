import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Step 1 Data Preparation
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
images, labels = lfw_people.images, lfw_people.target
target_names = lfw_people.target_names

n_samples, h, w = images.shape
X = images.reshape(n_samples, -1)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2 Subtract the mean image from the training data
mean_image = np.mean(X_train, axis=0)
X_train_centered = X_train - mean_image

# Step 3 Compute the covariance matrix
C = np.dot(X_train_centered, X_train_centered.T) / X_train_centered.shape[0]

# Step 4 Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Step 5 Sort the eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Compute the eigenfaces from the eigenvectors of the covariance matrix
eigenfaces = np.dot(X_train_centered.T, eigenvectors).T

# Normalize the eigenfaces
eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=1, keepdims=True)

# Select the top n_components eigenfaces (principal components)
n_components = 50
principal_components = eigenfaces[:n_components]

# Project the centered training data into the PCA space
X_train_pca = np.dot(X_train_centered, principal_components.T)

# Center the test data and project it into the PCA space
X_test_centered = X_test - mean_image
X_test_pca = np.dot(X_test_centered, principal_components.T)

# Function to find the closest training image using Euclidean distance
def find_closest_image(test_image_pca, X_train_pca):
    distances = np.linalg.norm(X_train_pca - test_image_pca, axis=1)
    closest_image_idx = np.argmin(distances)
    return closest_image_idx

# Predict a test image
test_image = X_test[0]
test_image_pca = np.dot(test_image - mean_image, principal_components.T)
closest_image_idx = find_closest_image(test_image_pca, X_train_pca)
predicted_label = y_train[closest_image_idx]

# Plot the mean image
plt.imshow(mean_image.reshape(h, w), cmap='gray')
plt.title('Mean Image')
plt.show()

# Plot the first few eigenfaces
fig, axes = plt.subplots(2, 5, figsize=(15, 6), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(principal_components[i].reshape(h, w), cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
plt.show()

# Show the test image and its predicted match
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(test_image.reshape(h, w), cmap='gray')
plt.title('Test Image')

plt.subplot(1, 2, 2)
match_image = X_train[closest_image_idx].reshape(h, w)
plt.imshow(match_image, cmap='gray')
plt.title('Matched Image')

plt.show()

print(f'Predicted label: {predicted_label}')
print(f'Predicted person: {target_names[predicted_label]}')