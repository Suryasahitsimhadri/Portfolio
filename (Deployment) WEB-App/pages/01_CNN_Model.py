import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import io
import pickle

from model_handlers.cnn_handler import CNNModelHandler
from utils import set_page_config, save_uploaded_file

# Set page configuration
set_page_config("CNN Model Deployment")

st.title("Convolutional Neural Network (CNN) Model")
st.write("""
This page allows you to upload an image and use a pre-trained CNN model to make predictions.
CNNs are particularly effective for image classification tasks.
""")

# Initialize model handler
cnn_handler = CNNModelHandler()

# Create a sidebar for model settings
st.sidebar.header("Model Settings")

# Model selection
model_path = st.sidebar.text_input("Model Path", placeholder="Enter path to your CNN model file")

# Option to upload a model file
uploaded_model = st.sidebar.file_uploader("Or upload a model file", type=['h5', 'keras', 'pkl])
if uploaded_model is not None:
    # Save the uploaded model to a temporary file
    model_path = save_uploaded_file(uploaded_model)

# Class names input
class_names_input = st.sidebar.text_area("Class Names (one per line)", 
                                        placeholder="Enter class names (one per line)")
class_names = [name.strip() for name in class_names_input.split('\n') if name.strip()]

# Load the model when the user clicks the button
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model..."):
        if cnn_handler.load(model_path):
            if class_names:
                cnn_handler.set_class_names(class_names)
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model. Please check the path or file.")

# Main content area
st.header("Image Classification")

# Image upload
st.write("Upload an image for classification:")
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

# Check if model is loaded and file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)
    
    # Make prediction when button is clicked
    if st.button("Predict"):
        if cnn_handler.model is None:
            st.error("Please load a model first!")
        else:
            with st.spinner("Processing..."):
                try:
                    # Preprocess the image
                    preprocessed_image, original_image = cnn_handler.preprocess_image(uploaded_file)
                    
                    # Make prediction
                    result = cnn_handler.predict(preprocessed_image)
                    
                    # Display prediction results
                    cnn_handler.display_prediction(result, original_image)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Additional information
st.markdown("---")
st.header("About CNN Models")
st.write("""
Convolutional Neural Networks (CNNs) consist of multiple layers specialized for processing image data:

1. **Convolutional Layers**: Apply filters to detect features
2. **Pooling Layers**: Reduce spatial dimensions
3. **Fully Connected Layers**: Perform classification based on extracted features

CNNs excel at:
- Image classification
- Object detection
- Face recognition
- Image segmentation
""")

# Example architecture visualization
st.subheader("Example CNN Architecture")

# Create a simple visualization of a CNN architecture using matplotlib
fig, ax = plt.subplots(figsize=(10, 4))
layers = ['Input', 'Conv1', 'Pool1', 'Conv2', 'Pool2', 'Flatten', 'FC', 'Output']
layer_sizes = [64, 48, 36, 24, 16, 8, 6, 4]
layer_colors = ['#3498db', '#2ecc71', '#9b59b6', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#e67e22']

for i, (layer, size, color) in enumerate(zip(layers, layer_sizes, layer_colors)):
    rect = plt.Rectangle((i, 0), 0.8, size, color=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(i + 0.4, size/2, layer, ha='center', va='center', color='white', fontweight='bold')

ax.set_xlim(0, len(layers))
ax.set_ylim(0, max(layer_sizes) + 10)
ax.axis('off')
plt.title("Simplified CNN Architecture")

st.pyplot(fig)

# Instructions for model preparation
st.markdown("---")
st.header("Model Preparation Tips")
st.write("""
For best results:
1. Use a CNN model trained specifically for your image classification task
2. The model should be saved in h5 or SavedModel format
3. Provide the correct class names in the same order as used during training
4. Make sure your image is in a format compatible with the model (e.g., RGB, specific dimensions)
""")
