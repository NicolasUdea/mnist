import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
train_images = train_images.astype('float32') / 255  # Normalization
test_images = test_images.astype('float32') / 255

train_images = train_images.reshape(-1, 28 * 28)  # Flatten the images
test_images = test_images.reshape(-1, 28 * 28)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)  # One-hot encoding
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Model definition with ReLu activation function
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Model compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model training
history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# Model evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Test the model with a sample image
image = test_images[0].reshape(1, 28 * 28)
prediction = model.predict(image)
digit_predicted = np.argmax(prediction)
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f'Prediction: {digit_predicted}')
plt.show()

# Function to predict a custom image
def predict_custom_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 28 * 28).astype('float32') / 255
    prediction = model.predict(img_array)
    digit_predicted = np.argmax(prediction)
    print(f'Prediction for the custom image: {digit_predicted}')
    plt.imshow(img, cmap='gray')
    plt.title(f'Prediction: {digit_predicted}')
    plt.show()

# Test with a custom image
predict_custom_image('imagen.png')

# Save the model
model.save('mnist_model.h5')
print('Model saved successfully')
