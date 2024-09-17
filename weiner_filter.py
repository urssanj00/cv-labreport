import cv2
import numpy as np
import matplotlib.pyplot as plt


def wiener_filter(img, kernel, K):
    """
    Apply Wiener filter to an image.

    :param img: Input image (grayscale).
    :param kernel: Blurring kernel.
    :param K: Constant for Wiener filter (noise-to-signal ratio).
    :return: Restored image.
    """
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel

    # Convert image to frequency domain
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel, s=img.shape)

    # Wiener filter formula: H*(f) / (|H|^2 + K)
    wiener_filter = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)

    # Apply Wiener filter
    img_filtered_fft = wiener_filter * img_fft

    # Convert back to spatial domain
    img_filtered = np.fft.ifft2(img_filtered_fft)
    img_filtered = np.abs(img_filtered)  # Get the real part

    return img_filtered


# Load an image (grayscale)
#img = cv2.imread('noisy_image.jpg', 0)
# Load the image
image_path = "/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Oggy.webp"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Example of a simple blur kernel
kernel = np.ones((5, 5)) / 25

# Apply Wiener filter
restored_img = wiener_filter(img, kernel, K=0.01)

# Plot the results using matplotlib
plt.figure(figsize=(10, 5))

# Display original image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Noisy Image')
plt.axis('off')

# Display restored image
plt.subplot(1, 2, 2)
plt.imshow(restored_img, cmap='gray')
plt.title('Restored Image (Wiener Filter)')
plt.axis('off')

plt.show()
