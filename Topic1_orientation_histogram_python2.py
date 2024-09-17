import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the low contrast image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Tower_Bridge_from_Shad_Thames.jpg'
low_contrast_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to perform histogram equalization manually
def manual_histogram_equalization(image):
    # Calculate histogram
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')

    # Apply equalization
    equalized_image = cdf_normalized[image]

    return equalized_image

# Perform histogram equalization
equalized_image_manual = manual_histogram_equalization(low_contrast_image)
equalized_image_builtin = cv2.equalizeHist(low_contrast_image)

# Plot original and equalized images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(low_contrast_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(equalized_image_manual, cmap='gray')
plt.title('Equalized (Manual)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(equalized_image_builtin, cmap='gray')
plt.title('Equalized (Using cv2.equalizeHist)')
plt.axis('off')

plt.tight_layout()
plt.show()