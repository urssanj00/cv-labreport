#Import required library
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Load the image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Tower_Bridge_from_Shad_Thames.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Detect corners
dst = cv2.cornerHarris(image, 2, 3, 0.04)
#Assign thresholding
thresh = 0.004 * dst.max()
# Create an image copy to draw corners on
corner_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for j in range(dst.shape[0]):
  for i in range(dst.shape[1]):
    if dst[j, i] > thresh:
    # Draw red circles on corners
      cv2.circle(corner_image, (i, j), 3, (255, 0, 0),-1)
# Display the original image and corners
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(corner_image)
plt.title('Detected Corners')
plt.axis('off')
plt.show()