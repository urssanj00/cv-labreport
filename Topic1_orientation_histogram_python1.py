import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Tower_Bridge_from_Shad_Thames.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Initialize histogram bins
hist_bins_manual = np.zeros(256)

# Calculate histogram manually
for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        pixel_value = gray_image[i, j]
        hist_bins_manual[pixel_value] += 1

# Calculate histogram using inbuilt function
hist_bins_inbuilt = cv2.calcHist([gray_image], [0], None, [256], [0,256])

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Orignal')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.plot(hist_bins_manual, color='black')
plt.title('Histogram  (Manually Calculated)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(hist_bins_inbuilt, color='black')
plt.title('Histogram  (Using cv2.calcHist)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()