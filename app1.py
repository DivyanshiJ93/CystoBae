import streamlit as st
import numpy as np
import cv2
import os
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 100
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Load models
cnn_model = load_model("SimpleCNN_model.h5")
vgg_model = load_model("VGG16_model.h5")

# Custom theme
st.set_page_config(page_title="CystoBae: The Pink Scan", page_icon="💖", layout="centered")

# Title
st.markdown("<h1 style='color:hotpink;'>💖 CystoBae: The Pink Scan</h1>", unsafe_allow_html=True)
st.markdown("🩺 Upload an ultrasound image to detect PCOS")

# Model selector
model_choice = st.selectbox("🧠 Choose a model", ["SimpleCNN", "VGG16"])

# Upload image
uploaded_file = st.file_uploader("📸 Upload an ultrasound image", type=["jpg", "jpeg", "png"])

# Preprocessing
def preprocess_image(image):
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return np.expand_dims(img / 255.0, axis=0)

def preprocess_image_vgg(image):
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Predict
def predict(image, model_name):
    if model_name == "SimpleCNN":
        model = cnn_model
        preprocessed = preprocess_image(image)
    else:
        model = vgg_model
        preprocessed = preprocess_image_vgg(image)

    pred = model.predict(preprocessed)[0][0]
    label = "PCOS POSITIVE" if pred > 0.5 else "PCOS NEGATIVE"
    confidence = max(pred, 1-pred)  # Get the higher confidence value
    
    return label, confidence, model, preprocessed

# Save log
def log_prediction(image, label, score):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image.save(os.path.join(LOG_DIR, f"{label}_{score:.2f}_{timestamp}.jpg"))

# Main logic
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼 Uploaded Ultrasound', use_column_width=True)

    with st.spinner("Analyzing..."):
        label, confidence, model, processed = predict(image, model_choice)

        st.markdown("### 🎯 Prediction Result")
        
        if label == "PCOS POSITIVE":
            st.error(f"**{label}** (Confidence: {confidence:.2%})")
        else:
            st.success(f"**{label}** (Confidence: {confidence:.2%})")

        # Save to log
        log_prediction(image, label.replace(" ", "_"), confidence)
        st.info("📝 Prediction saved to log folder.")

# Footer
st.markdown("<br><hr><p style='text-align:center; color:hotpink;'>Made with 💖 by your AI BFF – CystoBae</p>", unsafe_allow_html=True)