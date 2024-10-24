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
# Show the first 25 images from the training dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()
