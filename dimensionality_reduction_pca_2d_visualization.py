import numpy as np
import matplotlib.pyplot as plt

# Generate a 2D dataset
np.random.seed(0)
data = np.random.randn(2000, 2) @ np.array([[2, 0.5], [0.5, 1]]) + np.array([5, 10])
# np.array([[2, 0.5], [0.5, 1]]) - This matrix will be used to scale and rotate the data.
# np.array([5, 10]) - This effectively shifts the data such that the mean of the
# transformed data is centered around [5, 10].

# Step 1: Mean center the data
mean_data = np.mean(data, axis=0)
centered_data = data - mean_data

# Step 2: Calculate the covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)
#rowvar=True each row treated as variable instead of observation

# Step 3: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 4: Sort eigenvectors by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Transform data (Y = X_centered*W)
transformed_data = centered_data @ eigenvectors

# Plotting the data and principal components
plt.figure(figsize=(8, 6))
plt.scatter(centered_data[:, 0], centered_data[:, 1], alpha=0.5, label='Centered Data')

# Plot eigenvectors
for i in range(len(eigenvalues)):
    plt.quiver(0, 0, eigenvectors[0, i] * eigenvalues[i], eigenvectors[1, i] * eigenvalues[i],
               scale=1, scale_units='xy', angles='xy', color='red' if i == 0 else 'blue',
               label=f'Principal Component {i+1}' if i == 0 else f'Component {i+1}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot with Principal Components')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
