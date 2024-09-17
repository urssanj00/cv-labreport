from scipy.datasets import face
from scipy.signal import wiener
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the image
image_path = "/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Oggy.jpg"
#img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#Adding noise to the image
gauss_noise=np.zeros((437,434),dtype=np.uint8)
cv2.randn(gauss_noise,0,255)
gauss_noise=(gauss_noise*0.5).astype(np.uint8)

gn_img=cv2.add(img,gauss_noise)


filtered_img = wiener(gn_img, (1, 1))  #Filter the image

plt.subplot(1,3, 1)
plt.title("Original Image")
plt.imshow(img)
plt.subplot(1,3, 2)
plt.title("Noisy Image")
plt.imshow(gn_img)
plt.subplot(1,3, 3)
plt.title("Filtered Image")
plt.imshow(filtered_img)
plt.show()