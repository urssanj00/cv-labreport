import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    """
    Initialize centroids randomly from the data points.

    Parameters:
        data (numpy.ndarray): Input data points.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Initial centroids.
    """
    centroids_idx = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[centroids_idx]
    return centroids

def assign_clusters(data, centroids):
    """
    Assign each data point to the closest centroid.

    Parameters:
        data (numpy.ndarray): Input data points.
        centroids (numpy.ndarray): Current centroids.

    Returns:
        numpy.ndarray: Indices of the closest centroid for each data point.
    """
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(data, clusters, k):
    """
    Update centroids based on the mean of data points in each cluster.

    Args:
        data (numpy.ndarray): Input data array.
        clusters (numpy.ndarray): Cluster assignments for each data point.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Updated centroids.
    """
    new_centroids = np.zeros((k, data.shape[1]))  # Remove third dimension
    for i in range(k):
        points = data[clusters == i]
        if len(points) > 0:
            new_centroids[i] = np.mean(points, axis=0)
    return new_centroids

def kmeans(data, k, max_iters=100):
    """
    Perform K-means clustering.

    Parameters:
        data (numpy.ndarray): Input data points.
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.

    Returns:
        numpy.ndarray: Final centroids.
        numpy.ndarray: Indices of the closest centroid for each data point.
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Load the image
image_path = '/content/drive/MyDrive/Intellipat/Computer_Vision/Data/ct.tif'
image = cv2.imread(image_path)

# Reshape the image into a 2D array of pixels
pixels = image.reshape((-1, 3))

# Define the number of clusters (i.e., segments)
num_clusters = 2

# Apply k-means clustering
centroids, clusters = kmeans(pixels, num_clusters)

# Get the segmented image by assigning each pixel to its nearest centroid
segmented_image = centroids[clusters].reshape(image.shape).astype(np.uint8)

# Display the original and segmented images using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image')
plt.axis('off')

plt.show()
