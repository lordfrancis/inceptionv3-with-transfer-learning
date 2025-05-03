import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import keras
from keras import ops

from tensorflow.keras.models import load_model

model = keras.models.load_model('inception_transfer_learning.keras')

print(model)

st.title("Animal Classification App")
st.write("Upload an image of an animal to classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    label_map = {0: 'zebra', 1: 'woodpecker', 2: 'wombat'}

    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Preprocess the image
    image_resized = cv2.resize(image, (180, 180))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_map[predicted_index]

    # Display the image and prediction
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted: {predicted_label}")