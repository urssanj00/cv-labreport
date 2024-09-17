import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# Create a 101x101 image with all pixels set to 0
img = np.zeros((101, 101), dtype=np.uint8)

# Set the corner and middle pixels to 255
img[0, 0] = img[0, -1] = img[-1, 0] = img[-1, -1] = 255
img[50, 50] = 255  # Middle pixel

plt.imshow(img, aspect='auto',cmap='gray')

# Define the Hough transform for lines function
thetas = np.deg2rad(np.arange(-90, 90, 1))
width, height = img.shape
diag_len = int(round(math.sqrt(width * width + height * height)))
rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)
num_thetas = len(thetas)
accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
are_edges = img > 5
y_idxs, x_idxs = np.nonzero(are_edges)
for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]
    for t_idx in range(num_thetas):
        rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
        accumulator[rho, t_idx] += 1

# Define a function to display the Hough transform for lines
plt.figure()
plt.imshow(accumulator, aspect='auto', cmap='gist_ncar', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
plt.savefig('output.png', bbox_inches='tight')
plt.show()