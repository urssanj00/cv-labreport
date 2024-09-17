import numpy as np
import matplotlib.pyplot as plt

# Function to compute homography matrix from correspondences
def compute_homography(src, dst):
    A = []
    for i in range(src.shape[0]):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[-1, -1]

# Generate synthetic data: four corner points of a square and their transformed points
src_pts = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])
# Apply a transformation (e.g., rotation and translation)
theta = np.radians(30)  # 30 degrees rotation
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
translation = np.array([0.5, 0.5])
dst_pts = np.dot(src_pts, rotation_matrix) + translation

# Add some noise and outliers to the destination points
np.random.seed(0)
noise = np.random.normal(0, 0.05, dst_pts.shape)
dst_pts_noisy = dst_pts + noise
outliers = np.array([[1.5, 1.5], [2, 2]])
dst_pts_noisy_with_outliers = np.vstack([dst_pts_noisy, outliers])

# RANSAC to estimate homography
n_iterations = 1000
threshold = 0.1
best_homography = None
max_inliers = 0

for i in range(n_iterations):
    # Randomly select 4 points
    indices = np.random.choice(len(src_pts), 4, replace=False)
    src_subset = src_pts[indices]
    dst_subset = dst_pts_noisy[indices]

    # Compute homography
    H = compute_homography(src_subset, dst_subset)

    # Project all src points using the homography
    src_pts_homogeneous = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
    projected_pts = src_pts_homogeneous @ H.T
    projected_pts /= projected_pts[:, 2][:, np.newaxis]
    projected_pts = projected_pts[:, :2]

    # Compute distances to the actual dst points (only valid points, without outliers)
    distances = np.linalg.norm(projected_pts - dst_pts_noisy, axis=1)
    inliers = distances < threshold

    # Update the best homography if we found more inliers
    n_inliers = np.sum(inliers)
    if n_inliers > max_inliers:
        max_inliers = n_inliers
        best_homography = H

# Visualize the result
plt.figure(figsize=(8, 8))
plt.scatter(src_pts[:, 0], src_pts[:, 1], c='blue', label='Source Points')
plt.scatter(dst_pts_noisy_with_outliers[:, 0], dst_pts_noisy_with_outliers[:, 1],
            c='red', label='Destination Points with Noise and Outliers')
# Draw vectors from source points to corresponding noisy destination points
# (excluding outliers)
plt.quiver(src_pts[:, 0], src_pts[:, 1], dst_pts_noisy[:, 0] - src_pts[:, 0],
           dst_pts_noisy[:, 1] - src_pts[:, 1], angles='xy', scale_units='xy', scale=1,
           color='gray', alpha=0.5)

# Project all src points using the best homography
projected_pts = src_pts_homogeneous @ best_homography.T
projected_pts /= projected_pts[:, 2][:, np.newaxis]
projected_pts = projected_pts[:, :2]
plt.scatter(projected_pts[:, 0], projected_pts[:, 1], c='green', label='Projected Points')

plt.legend()
plt.title("RANSAC: Homography Estimation")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()
