# Install the pymetalog package
!pip install pymetalog

# Import necessary modules
from pymetalog import metalog
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neighbors import KernelDensity

# === Load and preprocess the MNIST dataset ===
# Load MNIST dataset (handwritten digit data)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Output the shapes of training and testing datasets
print("Training set shape:", x_train.shape, y_train.shape)
print("Testing set shape:", x_test.shape, y_test.shape)

# === Visualize the first 10 training images ===
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()

# === Define a function to filter out insignificant pixels ===
def filter_pixels(x, threshold=0.1):
    # Filter out pixels where the average value across all samples is below a threshold
    mask = np.mean(x, axis=0) > threshold
    return mask

# Apply the filter to identify significant pixels
pixel_mask = filter_pixels(x_train)

# Flatten training and testing datasets to retain only significant pixels
x_train_flat = x_train[:, pixel_mask]
x_test_flat = x_test[:, pixel_mask]

# === Train a Naive Bayes classifier and evaluate it ===
# Create and train a Gaussian Naive Bayes model using the filtered data
model = GaussianNB()
model.fit(x_train_flat, y_train)

# Predict labels for the test set
y_pred = model.predict(x_test_flat)

# Generate a confusion matrix to evaluate classification performance
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted Labels")  # Annotate X-axis: Predicted labels by the model
plt.ylabel("True Labels")       # Annotate Y-axis: Actual labels from the dataset
plt.title("Confusion Matrix")   # Add a title for clarity
plt.show()


# === Fit Gaussian distributions for each class ===
mean_values = np.zeros((10, np.sum(pixel_mask)))  # Mean for each significant pixel per class
std_values = np.zeros((10, np.sum(pixel_mask)))   # Std dev for each significant pixel per class

for i in range(10):
    # Extract all images of the current class and apply the pixel filter
    class_images = x_train[y_train == i][:, pixel_mask]

    # Compute mean and standard deviation for each significant pixel in this class
    mean_values[i] = np.mean(class_images, axis=0)
    std_values[i] = np.std(class_images, axis=0)

print("Gaussian distribution fitting completed:")
print("Mean shape:", mean_values.shape)
print("Standard deviation shape:", std_values.shape)

# === Fit Kernel Density Estimation (KDE) models for each class ===
kde_models = {}

for i in range(10):
    # Extract all images of the current class and flatten them for KDE fitting
    class_images = x_train[y_train == i][:, pixel_mask]

    # Create a KDE model with Gaussian kernel and fit it to the data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05)
    kde.fit(class_images)

    # Store the KDE model for this class
    kde_models[i] = kde

print("Kernel Density Estimation (KDE) fitting completed")

# Initialize dictionary for storing MetaLog models
metalog_models = {}

for i in range(10):
    # Extract all images of current class and flatten them into a single array
    class_images = x_train[y_train == i][:, pixel_mask].flatten()

    # Clean input data
    class_images = class_images[np.isfinite(class_images)]                # Remove NaN and inf values
    class_images[class_images < np.finfo(float).eps] = np.finfo(float).eps # Replace near-zero values with small positive constant
    class_images = np.clip(class_images, 0, 1)                           # Ensure all values are within [0,1]

    # Fit MetaLog distribution to cleaned data
    try:
        ml_model = metalog(
            x=class_images.tolist(),      # Convert numpy array to list format required by pymetalog
            bounds=[0, 1],                # Set bounds of distribution to [0,1]
            boundedness='b',              # Specify bounded distribution on both sides
            term_limit=10                 # Limit expansion terms to improve numerical stability
        )
        metalog_models[i] = ml_model      # Store model for current class

    except ValueError as e:
        print(f"Error fitting MetaLog model for class {i}: {e}")
