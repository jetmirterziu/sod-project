import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import os

# Import your model architecture
from sod_model import SODModel

# --- CONSTANTS ---
IMG_SIZE = 128
WEIGHTS_PATH = "results/best_model.weights.h5"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Salient Object Detection",
    layout="centered"
)

# --- TITLE & DESCRIPTION ---
st.title("Salient Object Detection")
st.markdown("""
**Xponian Cohort IV --- End-to-End ML Project --- JETMIR TERZIU**
""")

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    """
    Loads the model and weights once, then caches it 
    to prevent reloading on every user interaction.
    """
    try:
        # 1. Build Architecture
        model = SODModel()
        
        # 2. Dummy pass to initialize variables
        _ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))
        
        # 3. Load Weights
        if os.path.exists(WEIGHTS_PATH):
            model.load_weights(WEIGHTS_PATH)
            print("Weights loaded successfully.")
        else:
            st.error(f"Weights file not found at {WEIGHTS_PATH}. Please run train.py first.")
            return None
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner("Loading Model..."):
    model = load_model()

# --- PREPROCESSING & INFERENCE ---
def process_image(image_pil):
    """
    Resizes image for the model, runs prediction, and returns results.
    """
    # 1. Prepare Input (Resize to 128x128)
    img_original = np.array(image_pil)
    img_resized = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0) # (1, 128, 128, 3)

    # 2. Inference
    start_time = time.time()
    pred_mask = model.predict(img_batch, verbose=0)
    inference_time = time.time() - start_time

    # 3. Post-processing (Remove batch dim)
    pred_mask = np.squeeze(pred_mask) # (128, 128)
    
    # 4. Create Overlay (Resize mask back to ORIGINAL image size for better view)
    # We resize the 128x128 mask up to the original image dimensions
    orig_h, orig_w = img_original.shape[:2]
    mask_upscaled = cv2.resize(pred_mask, (orig_w, orig_h))
    
    # Create Red Heatmap
    heatmap = np.zeros_like(img_original)
    heatmap[:, :, 0] = (mask_upscaled * 255).astype(np.uint8) # Red channel
    
    # Blend: 70% Original + 30% Red
    overlay = cv2.addWeighted(img_original, 0.7, heatmap, 0.3, 0)

    return img_resized, pred_mask, overlay, inference_time

# --- USER INTERFACE ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display Uploaded Image
    image_pil = Image.open(uploaded_file).convert('RGB')
    
    # Run Processing
    img_input, mask_output, overlay_output, infer_time = process_image(image_pil)

    # --- DISPLAY RESULTS ---
    st.success(f"Detection Complete! Inference Time: {infer_time:.4f} seconds")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # Updated parameter here:
        st.image(img_input, caption="Model Input (128x128)", use_container_width=True)

    with col2:
        # Updated parameter here:
        st.image(mask_output, caption="Predicted Saliency Mask", clamp=True, use_container_width=True)

    with col3:
        # Updated parameter here:
        st.image(overlay_output, caption="Overlay (Original Size)", use_container_width=True)

    # Optional: Show metrics/confidence if needed
    st.markdown("---")
    st.markdown("### Explanation")
    st.info("The **Mask** shows the probability of each pixel being salient (White = High, Black = Low). The **Overlay** projects this prediction back onto your original image.")