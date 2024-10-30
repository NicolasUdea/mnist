import os
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Provide the full path to the model file
model_path = r'C:\Users\ahoga\Desktop\Talento_Tech\Talento_Tech/mlp_model.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f'Model file not found at {model_path}')
else:
    model = tf.keras.models.load_model(model_path)

    # Function to preprocess the image
    def preprocess_image(image):
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28 pixels
        image = np.array(image)  # Convert to numpy array
        image = image / 255.0  # Normalize values between 0 and 1
        image = np.reshape(image, (1, 28 * 28))  # Reshape for model input
        return image

    # Title of the application
    st.title('Handwritten Digit Classification')

    # Upload the image
    uploaded_file = st.file_uploader('Upload an image of a digit (0-9)', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make the prediction
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)

        # Display the prediction
        st.write(f'Prediction: **{predicted_digit}**')

        # Display probabilities
        for i in range(10):
            st.write(f'Digit {i}: {prediction[0][i]:.4f}')

        # Display processed image to see preprocessing
        plt.imshow(processed_image.reshape(28, 28), cmap='gray')  # Convert to 28x28 for visualization
        plt.axis('off')
        st.pyplot(plt)