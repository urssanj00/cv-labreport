import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the low-resolution image
image_path = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/butterfly.jpg'
image = cv2.imread(image_path)

# Perform bicubic interpolation
scale_percent = 200, 400  # Scale image by 200%, 400%

for i in scale_percent:
  scale_percent_cur = i

  width = int(image.shape[1] * scale_percent_cur / 100)
  height = int(image.shape[0] * scale_percent_cur / 100)
  dim = (width, height)

  # Resize the image
  upscaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

  # Save the upscaled image
  cv2.imwrite('upscaled_image_bicubic.jpg', upscaled_image)

  plt.figure(figsize=(20, 10))
  plt.subplot(1, 2, 1)
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
  plt.title(f'Original Image Size: {image.shape}/{scale_percent_cur}')
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
  plt.title(f'Supper Resolution: size >> {upscaled_image.shape}')
  plt.axis('off')
  plt.show()
