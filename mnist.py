import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

# Data exploration
'''
The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255.
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

'''
# What type of data are we dealing with?
print(type(train_images)) # <class 'numpy.ndarray'>

# Let's see the images and labels
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f'Label: {train_labels[i]}')
    plt.axis('off')
plt.show()
'''

# Data preprocessing
'''
The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255.
We need to normalize the pixel values to be between 0 and 1.
'''
train_images = train_images.astype('float32') / 255 # Normalization
test_images = test_images.astype('float32') / 255

train_images = train_images.reshape(-1, 28 * 28) # Flatten the images
test_images = test_images.reshape(-1, 28 * 28)
