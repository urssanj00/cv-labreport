import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path1 = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/Tower_Bridge_from_Shad_Thames.jpg'
image_path2 = '/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/tower-bridge-cover-picture.webp'

image1 = cv2.imread(image_path1, cv2.COLOR_BGR2GRAY)
image2 = cv2.imread(image_path2, cv2.COLOR_BGR2GRAY)

sift  = cv2.SIFT_create()

#detect key points and descriptor
keyspoints1, descriptors1 = sift.detectAndCompute(image1, None)
keyspoints2, descriptors2 = sift.detectAndCompute(image2, None)

# iminitalize bf matcher
bf = cv2.BFMatcher();

#match decripter between two images.
matches  = bf.match(descriptors1, descriptors2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

#draw matches between two images.
matched_image  = cv2.drawMatches(image1, keyspoints1, image2, keyspoints2, matches[:100],  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS )
print(matched_image)
image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

# Plot the matched image
plt.figure(figsize=(20, 10))
plt.imshow(image_rgb)
plt.title('SIFT Feature Matching')
plt.axis('off')
plt.show()
