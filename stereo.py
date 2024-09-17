
# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the stereo images from Google Drive
l_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Glass-IMG_4220.jpg'
r_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Glass-IMG_4222.jpg'

left_image = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)

# StereoBM matcher
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

# Normalize disparity map for visualization
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                     norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Display stereo images and disparity map
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Left Image')
plt.imshow(left_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Right Image')
plt.imshow(right_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Disparity Map')
plt.imshow(disparity_normalized, cmap='jet')
plt.colorbar(label='Disparity')
plt.axis('off')

plt.show()