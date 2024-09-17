import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
image_path = '/content/drive/MyDrive/Intellipat/Computer_Vision/Data/AI-headCT.jpg'
image = cv2.imread(image_path)

# Reshape the image into a 2D array of pixels
pixels = image.reshape((-1, 3))

# Define the number of clusters (i.e., segments)
num_clusters = 3

# Apply k-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# Get the labels and centroids of the clusters
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Assign each pixel in the image to its corresponding centroid color
segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)

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
