import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed_point, threshold):
    # Create a binary mask initialized with zeros
    mask = np.zeros_like(image, dtype=np.uint8)

    # Create a list to keep track of the points to be processed
    points_to_process = [seed_point, seed_point1]

    # Get the seed point intensity value
    seed_value = image[seed_point[1], seed_point[0]]

    # Iterate until there are no more points to process
    while len(points_to_process) > 0:
        # Get the next point to process
        current_point = points_to_process.pop(0)

        # Check if the current point is within the image boundaries
        if (current_point[0] >= 0 and current_point[0] < image.shape[1] and
            current_point[1] >= 0 and current_point[1] < image.shape[0]):
            # Check if the current point is already part of the region
            if mask[current_point[1], current_point[0]] == 0:
                # Get the intensity value of the current point
                current_value = image[current_point[1], current_point[0]]

                # Check if the intensity difference is within the threshold
                if abs(current_value - seed_value) < threshold:
                    # Add the current point to the region
                    mask[current_point[1], current_point[0]] = 255

                    # Add neighboring points to the list of points to process
                    points_to_process.append((current_point[0] + 1, current_point[1]))
                    points_to_process.append((current_point[0] - 1, current_point[1]))
                    points_to_process.append((current_point[0], current_point[1] + 1))
                    points_to_process.append((current_point[0], current_point[1] - 1))

    return mask

# Load the image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/tommy.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the seed point (choose any point within the region of interest)
seed_point = (image.shape[1]//2, image.shape[0]//2)
seed_point1 = (image.shape[1]//4, image.shape[0]//4)

# Set the threshold for region growing
threshold = 100  # Adjust this value as needed

# Perform region growing segmentation
segmented_image = region_growing(image, seed_point, threshold)

# Mark the seed point in red on the original image
image_with_seed_point = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.circle(image_with_seed_point, seed_point, 10, (255, 0, 0), -1)

# Display the original and segmented images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_with_seed_point)
axes[0].set_title('Original Image with Seed Point')
axes[0].axis('off')
axes[1].imshow(segmented_image, cmap='gray')
axes[1].set_title('Segmented Image')
axes[1].axis('off')
plt.show()