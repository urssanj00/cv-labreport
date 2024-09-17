import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/tommy.jpg'
color_image = cv2.imread(image_path)

# Convert the color image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with two different standard deviations for DoG
sigma1 = 1.0
sigma2 = 1.1

gaussian1 = cv2.GaussianBlur(gray_image, (0, 0), sigma1)
gaussian2 = cv2.GaussianBlur(gray_image, (0, 0), sigma2)

# Compute the Difference of Gaussian (DoG) by subtracting the two Gaussian-filtered images
dog = gaussian1 - gaussian2

# Apply Laplacian of Gaussian (LoG) filter
log_filtered = cv2.Laplacian(gaussian1, cv2.CV_64F)

# Plot the Difference of Gaussian (DoG) and Laplacian of Gaussian (LoG) filtered images
plt.figure(figsize=(12, 6))

# Original color image
plt.subplot(2, 2, 1)  # First row, first column
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title('Original Color Image')
plt.axis('off')

# Grayscale image
plt.subplot(2, 2, 2)  # First row, second column
plt.imshow(gray_image, cmap='gray')
plt.title('Original image converted to Grayscale')
plt.axis('off')

# DoG image
plt.subplot(2, 2, 3)  # Second row, first column
plt.imshow(dog, cmap='gray')
plt.title('Difference of Gaussian (DoG)')
plt.axis('off')

# LoG image
plt.subplot(2, 2, 4)  # Second row, second column
plt.imshow(log_filtered, cmap='gray')
plt.title('Laplacian of Gaussian (LoG)')
plt.axis('off')

plt.tight_layout()

plt.show()