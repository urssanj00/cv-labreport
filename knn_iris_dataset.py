# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Consider only the first two features for simplicity
y = iris.target

# Filter for only two classes (setosa and versicolor)
X = X[y < 2]
y = y[y < 2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
class_report = classification_report(y_test, y_pred, target_names=iris.target_names[:2])
print('Classification Report:')
print(class_report)

# Separate data for each class
X_setosa = X[y == 0]
X_versicolor = X[y == 1]

# Plot the scatter plot of input data
plt.figure(figsize=(12, 6))

# Plot setosa class
plt.scatter(X_setosa[:, 0], X_setosa[:, 1], color='blue', label='setosa', edgecolor='k')

# Plot versicolor class
plt.scatter(X_versicolor[:, 0], X_versicolor[:, 1], color='red', label='versicolor', edgecolor='k')

plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Scatter Plot of Iris Data (Setosa vs Versicolor)')
plt.legend()

# Plot the confusion matrix with annotations
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='white')

plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], iris.target_names[:2])
plt.yticks([0, 1], iris.target_names[:2])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()