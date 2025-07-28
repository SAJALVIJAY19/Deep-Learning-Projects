import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="🥔 Potato Disease Classifier", layout="centered")

st.title("🥔 Potato Disease Classifier")
st.caption("Detects Early Blight, Late Blight, or Healthy leaves from potato plant images.")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/image_classifier_model.h5')
    return model

model = load_model()

# ✅ Use your actual class names here
CLASS_NAMES = ['Early_Blight', 'Healthy', 'Late_Blight']

# --- Prediction Function ---
def predict_image(img, model):
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Shape: (1, 256, 256, 3)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload an image of a potato leaf", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    st.write("🧠 Predicting...")
    predicted_class, confidence = predict_image(image, model)

    st.success(f"🎯 **Prediction:** {predicted_class}")
    st.info(f"🔒 **Confidence:** {confidence}%")