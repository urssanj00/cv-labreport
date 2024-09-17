import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

# Loading the Olivetti Faces Dataset (compressed into synthetic_faces.npz contains
# 400 images, each of size 64x64 pixels)
data = np.load('/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/orl.npz')

images = data['images']
labels = data['labels']

# Reshape images to 2D array (samples, features)
n_samples, h, w = images.shape
X = images.reshape(n_samples, -1)
y = labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LDA
lda = LDA()

# Fit LDA
X_train_lda = lda.fit_transform(X_train, y_train)


# Project test data into LDA space
X_test_lda = lda.transform(X_test)

# Predict using LDA
y_pred = lda.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Plot the first few LDA components (discriminants)
fig, axes = plt.subplots(2, 5, figsize=(15, 6), subplot_kw={'xticks': [], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(lda.scalings_[:, i].reshape(h, w), cmap='gray')
    ax.set_title(f"LDA Component {i+1}")
    
plt.tight_layout()
plt.show()

# Plot test image prediction and reconstructed image using LDA
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot test image and predicted label
test_idx = 0 # Adjust this index as needed
axes[0].imshow(X_test[test_idx].reshape(h, w), cmap='gray')
axes[0].set_title(f"Test Image\nTrue Label: {y_test[test_idx]}\nPredicted Label: {y_pred[test_idx]}")

# Reconstruct test image using LDA components
reconstructed_image = np.dot(X_test_lda[test_idx], lda.scalings_.T) + lda.xbar_
axes[1].imshow(reconstructed_image.reshape(h, w), cmap='gray')
axes[1].set_title("Reconstructed Image using LDA")
plt.tight_layout()
plt.show()

# Print shapes of variables
print(f"Shape of X_train_lda: {X_train_lda.shape}")
print(f"Shape of X_test_lda: {X_test_lda.shape}")
print(f"Shape of lda.scalings_: {lda.scalings_.shape}")
print(f"Shape of lda.xbar_: {lda.xbar_.shape}")