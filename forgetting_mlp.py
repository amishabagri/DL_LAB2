import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# ---------------------
# Load and Preprocess MNIST Dataset
# ---------------------
(num_tasks, img_size) = (10, 784)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to [0,1] range
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Reshape into 1D vectors of size 784
train_images = train_images.reshape(-1, img_size)
test_images = test_images.reshape(-1, img_size)

# ---------------------
# Generate Permuted MNIST Tasks
# ---------------------
# Create a dictionary to store permuted datasets
task_train_images, task_test_images = {}, {}

# Generate different random permutations for each task
for task_index in range(num_tasks):
    permutation = np.random.permutation(img_size)  # Random shuffle of pixels
    task_train_images[task_index] = train_images[:, permutation]  # Apply permutation
    task_test_images[task_index] = test_images[:, permutation]
