# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image from Google Drive
path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Eiffel_Tower_20051010.jpg'

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Display the edges
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()

# Function to estimate depth
def estimate_depth(image, known_object_height=2):
    height, width = image.shape[:2]
    depth_map = np.zeros((height, width))

    for y in range(height):
        depth = (height - y) / height  # Normalize depth between 0 and 1
        depth_map[y, :] = depth * known_object_height

    return depth_map

# Apply depth estimation
depth_map = estimate_depth(image)

# Display the depth map
plt.figure(figsize=(10, 5))
plt.title('Estimated Depth Map')
plt.imshow(depth_map, cmap='jet')
plt.colorbar(label='Depth')
plt.axis('off')
plt.show()