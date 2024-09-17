import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert the target labels to integers using the built-in int type
y = y.astype(int)  # Use int instead of np.int

# Normalize the images
X = X / 255.0

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
k = 3  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Plot some of the test images with their predictions
def plot_predictions(images, labels, predictions, n_images=10):
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        plt.subplot(1, n_images, i + 1)
        # Convert the DataFrame row to a NumPy array and reshape
        plt.imshow(images.iloc[i].values.reshape(28, 28), cmap='gray')
        plt.title(f'True: {labels.iloc[i]}\nPred: {predictions[i]}')
        plt.axis('off')
    plt.show()

# Visualize predictions for the first 10 test images
plot_predictions(X_test, y_test, y_pred, n_images=10)