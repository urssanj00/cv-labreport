import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Tower_Bridge_from_Shad_Thames.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#initialize shift method to plot key points.
sift = cv2.SIFT_create()

#detect key points and descriptor

keyspoints, descriptors = sift.detectAndCompute(image, None)

image_with_keypoints = cv2.drawKeypoints(image, keyspoints, None)
#plot original images

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image with shft kyes points')
plt.axis('off')

plt.show()