import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

# Load the ORL dataset (assuming it is pre-saved in NumPy format)
data = np.load('/content/drive/MyDrive/Intellipaat/Computer_Vision/Data/orl.npz')
images = data['images']
labels = data['labels']

# Reshape images to 2D array (samples, features)
n_samples, h, w = images.shape
X = images.reshape(n_samples, -1)
y = labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA
n_components = 50
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pca_components = pca.components_

# Perform LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
lda_components = lda.scalings_.T

# Perform ICA
ica = FastICA(n_components=n_components, random_state=42)
X_train_ica = ica.fit_transform(X_train)
X_test_ica = ica.transform(X_test)
ica_components = ica.components_

# Plot the components
def plot_components(components, title, h, w, n_row=2, n_col=5):
    fig, axes = plt.subplots(n_row, n_col, figsize=(15, 6), subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(components[i].reshape(h, w), cmap='gray')
        ax.set_title(f"{title} {i+1}")
    plt.show()

plot_components(pca_components, 'PCA Component', h, w)
plot_components(lda_components, 'LDA Component', h, w, n_row=1, n_col=2)
plot_components(ica_components, 'ICA Component', h, w)

# Visualize data in 2D space for LDA and PCA
def plot_2d_projection(X, y, title):
    plt.figure()
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title(title)
    plt.show()

plot_2d_projection(X_train_pca[:, :2], y_train, 'PCA Projection')
plot_2d_projection(X_train_lda, y_train, 'LDA Projection')
plot_2d_projection(X_train_ica[:, :2], y_train, 'ICA Projection')

# Shapes of the transformed data
print("Shapes of transformed data:")
print(f"X_train_pca: {X_train_pca.shape}")
print(f"X_train_lda: {X_train_lda.shape}")
print(f"X_train_ica: {X_train_ica.shape}")
