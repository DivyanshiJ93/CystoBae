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
st.markdown("🩺 Upload an ultrasound image to detect PCOS with **AI-powered diagnosis & visual insights.**")

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

# Grad-CAM
def generate_gradcam(model, img_array, layer_name="conv2d"):
    import tensorflow as tf
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]
    weights = np.mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

# Predict
def predict(image, model_name):
    if model_name == "SimpleCNN":
        model = cnn_model
        preprocessed = preprocess_image(image)
    else:
        model = vgg_model
        preprocessed = preprocess_image_vgg(image)

    pred = model.predict(preprocessed)[0][0]
    label = "PCOS Positive" if pred > 0.5 else "Not Infected"

    return label, pred, model, preprocessed

# Save log
def log_prediction(image, label, score):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image.save(os.path.join(LOG_DIR, f"{label}_{score:.2f}_{timestamp}.jpg"))

# Main logic
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼 Uploaded Ultrasound', use_column_width=True)

    with st.spinner("Analyzing..."):
        label, score, model, processed = predict(image, model_choice)

        st.markdown("### 🎯 Prediction Result")
        st.success(f"**{label}** (Confidence: {score:.2f})")

        # Grad-CAM
        try:
            cam = generate_gradcam(model, processed, layer_name="conv2d")
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
            orig = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
            if orig.shape[-1] == 4: orig = orig[:, :, :3]
            superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

            st.markdown("### 🔍 Model Focus (Grad-CAM)")
            st.image(superimposed, caption="Grad-CAM Heatmap", use_column_width=True)
        except Exception as e:
            st.warning("⚠️ Grad-CAM visualization failed.")

        # Save to log
        log_prediction(image, label.replace(" ", "_"), score)
        st.info("📝 Prediction saved to log folder.")

# Footer
st.markdown("<br><hr><p style='text-align:center; color:hotpink;'>Made with 💖 by your AI BFF – CystoBae</p>", unsafe_allow_html=True)
