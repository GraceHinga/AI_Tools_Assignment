import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

st.title("🧠 MNIST Digit Classifier")

# Load model (adjust filename if different)
model = tf.keras.models.load_model("mnist_cnn_model.h5")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file:
    import PIL.Image as Image
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    st.image(img, caption="Uploaded Image", use_container_width=True)
    prediction = np.argmax(model.predict(img_array))
    st.success(f"Predicted Digit: {prediction}")
else:
    st.info("Please upload an image to get started.")
