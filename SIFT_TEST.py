import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read the original image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/butterfly.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw the keypoints on a copy of the original image
img_with_keypoints = cv2.drawKeypoints(image, keypoints, None,  (255, 0, 0), 4)

# Combine original and modified images side by side
combined_image = np.hstack((image, img_with_keypoints))

# Show combined image
cv2_imshow(combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
