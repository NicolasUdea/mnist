import os

# Set the environment variable to allow duplicate OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize the first few images
for i in range(5):
    plt.figure(figsize=(2, 2))
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis('off')
    plt.show()

# Preprocess the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the MLP model
model = models.Sequential()
model.add(layers.Dense(512, activation='sigmoid', input_shape=(28 * 28,)))
model.add(layers.Dense(256, activation='sigmoid'))
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# Test the model with a sample image
imagen = x_test[0].reshape(1, 28 * 28)
prediccion = model.predict(imagen)
digit_predicho = np.argmax(prediccion)
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f'Predicción: {digit_predicho}')
plt.show()

# Function to predict a custom image
def predecir_imagen_personalizada(ruta_imagen):
    img = Image.open(ruta_imagen).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 28 * 28).astype('float32') / 255
    prediccion = model.predict(img_array)
    digit_predicho = np.argmax(prediccion)
    print(f'Predicción para la imagen personalizada: {digit_predicho}')
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicción: {digit_predicho}')
    plt.show()

# Test with a custom image
predecir_imagen_personalizada('imagen.png')

# Save the model to an .h5 file
model.save(r'C:\Users\ahoga\Desktop\Talento_Tech\Talento_Tech\mlp_model.h5')
print("Model saved successfully.")
