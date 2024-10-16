import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

# Load the trained model
model_path = 'new_train.keras'
model = tf.keras.models.load_model(model_path)

# Function to load and preprocess the image
def load_and_preprocess_image(img, target_size=(224, 224)):
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to load class labels
def load_class_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        class_labels = file.readlines()
    class_labels = [label.strip() for label in class_labels]  # Remove whitespace
    return class_labels

# Load class labels
class_labels_path = 'class_labels.txt'
class_labels = load_class_labels(class_labels_path)

# Streamlit app layout
st.title("Plant Species Classification")
st.write("Upload an image of a plant or take a photo to get its species prediction.")

# Option to take a photo or upload an image
camera_input = st.camera_input("Take a photo")
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

if camera_input is not None:
    # If the user takes a photo
    img = image.load_img(camera_input, target_size=(224, 224))
    img_array = load_and_preprocess_image(img)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Get the predicted class name
    predicted_class_name = class_labels[predicted_class_index[0]]

    # Display the captured image and prediction
    st.image(img, caption='Captured Image', use_column_width=True)
    st.write(f'Predicted Class: {predicted_class_name}')  # Display predicted class name

elif uploaded_file is not None:
    # If the user uploads an image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = load_and_preprocess_image(img)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Get the predicted class name
    predicted_class_name = class_labels[predicted_class_index[0]]

    # Display the uploaded image and prediction
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Class: {predicted_class_name}')  # Display predicted class name
