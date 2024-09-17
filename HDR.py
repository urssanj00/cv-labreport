import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load the color image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Vivekananda.webp'

image = cv2.imread(image_path)

# Convert the color image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(gray_image)

# Simulate HDR by merging multiple exposures (here we just duplicate the
# original image for demonstration purposes)
# In a real scenario, you would take multiple images with different exposures.
exposures = [0.5, 1.0, 2.0]  # Different exposure weights
hdr_images = [cv2.convertScaleAbs(gray_image, alpha=e) for e in exposures]

# Merge exposures to create an HDR image
hdr = cv2.createMergeMertens().process(hdr_images)
hdr = np.clip(hdr*255, 0, 255).astype('uint8')  # Convert to 8-bit image for display

# Display the original, equalized, and HDR images using matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Histogram Equalization")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("HDR Imaging")
plt.imshow(hdr, cmap='gray')
plt.axis('off')

plt.show()
