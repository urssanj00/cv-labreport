import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to apply log transformation
def apply_log_transform(image):
    # Convert the image to float32 to prevent data loss
    img_float32 = np.float32(image)

    # Apply the log transformation
    log_image = cv2.log(1 + img_float32)

    # Normalize the image to the range [0, 255]
    cv2.normalize(log_image, log_image, 0, 255, cv2.NORM_MINMAX)

    # Convert back to 8-bit image
    log_image = np.uint8(log_image)

    return log_image


# Load an image from file
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Tower_Bridge_from_Shad_Thames.jpg'

input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if input_image is None:
    print("Error: Could not open or find the image.")
else:
    # Apply log transformation
    log_transformed_image = apply_log_transform(input_image)

    # Display the original and transformed images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.title('Original Image')
    plt.imshow(input_image, cmap='gray')

    plt.subplot(2, 1, 2)
    plt.title('Log Transformed Image')
    plt.imshow(log_transformed_image, cmap='gray')

    # Adjust the spacing between the two subplots
    plt.subplots_adjust(hspace=0.5)  # Increase hspace for vertical space

    plt.show()
