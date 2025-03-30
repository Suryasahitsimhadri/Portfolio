import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import io

from model_handlers.transfer_learning_handler import TransferLearningModelHandler
from utils import set_page_config, save_uploaded_file

# Set page configuration
set_page_config("Transfer Learning Model Deployment")

st.title("Transfer Learning Model")
st.write("""
This page allows you to use a pre-trained transfer learning model for image classification.
Transfer learning leverages knowledge from models trained on large datasets and applies it to specific tasks.
""")

# Initialize model handler
transfer_model_handler = TransferLearningModelHandler()

# Create a sidebar for model settings
st.sidebar.header("Model Settings")

# Model selection
model_path = st.sidebar.text_input("Model Path", placeholder="Enter path to your model file (.h5)")

# Option to upload a model file
uploaded_model = st.sidebar.file_uploader("Or upload a model file", type=['h5', 'keras'])
if uploaded_model is not None:
    # Save the uploaded model to a temporary file
    model_path = save_uploaded_file(uploaded_model)

# Base model selection
base_model_options = [
    "Custom", "VGG16", "VGG19", "ResNet50", "InceptionV3", 
    "MobileNetV2", "EfficientNetB0", "DenseNet121"
]
base_model = st.sidebar.selectbox("Base Model", base_model_options)

# Class names input
class_names_input = st.sidebar.text_area("Class Names (one per line)", 
                                        placeholder="Enter class names (one per line)")
class_names = [name.strip() for name in class_names_input.split('\n') if name.strip()]

# Load the model when the user clicks the button
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model..."):
        # Set preprocessing function based on base model
        if base_model != "Custom":
            try:
                if base_model == "VGG16":
                    from tensorflow.keras.applications.vgg16 import preprocess_input
                elif base_model == "VGG19":
                    from tensorflow.keras.applications.vgg19 import preprocess_input
                elif base_model == "ResNet50":
                    from tensorflow.keras.applications.resnet50 import preprocess_input
                elif base_model == "InceptionV3":
                    from tensorflow.keras.applications.inception_v3 import preprocess_input
                elif base_model == "MobileNetV2":
                    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                elif base_model == "EfficientNetB0":
                    from tensorflow.keras.applications.efficientnet import preprocess_input
                elif base_model == "DenseNet121":
                    from tensorflow.keras.applications.densenet import preprocess_input
                
                transfer_model_handler.set_preprocessing_function(preprocess_input)
                st.sidebar.info(f"Using {base_model} preprocessing function")
            except ImportError:
                st.sidebar.warning(f"Could not import preprocessing function for {base_model}. Using default preprocessing.")
        
        # Load the model
        if transfer_model_handler.load(model_path):
            if class_names:
                transfer_model_handler.set_class_names(class_names)
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model. Please check the path or file.")

# Main content area
st.header("Image Classification with Transfer Learning")

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
        if transfer_model_handler.model is None:
            st.error("Please load a model first!")
        else:
            with st.spinner("Processing..."):
                try:
                    # Preprocess the image
                    preprocessed_image, original_image = transfer_model_handler.preprocess_image(uploaded_file)
                    
                    # Make prediction
                    result = transfer_model_handler.predict(preprocessed_image)
                    
                    # Display prediction results
                    transfer_model_handler.display_prediction(result, original_image)
                    
                    # Option to visualize intermediate activations
                    st.subheader("Model Interpretability")
                    if st.checkbox("Visualize Model Activations"):
                        with st.spinner("Generating activation visualizations..."):
                            transfer_model_handler.visualize_intermediate_activations(preprocessed_image)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Additional information
st.markdown("---")
st.header("About Transfer Learning")
st.write("""
Transfer Learning is a machine learning technique where a pre-trained model on a large dataset is repurposed for a different but related task.

Key benefits:
1. **Reduced Training Time**: Leverages pre-learned features
2. **Less Data Required**: Can achieve good performance with smaller datasets
3. **Better Generalization**: Often performs better on new tasks

Common approaches:
- **Feature Extraction**: Use pre-trained model as a feature extractor
- **Fine-Tuning**: Adapt the pre-trained model by training it further on a new dataset
""")

# Visualization of transfer learning concept
st.subheader("Transfer Learning Concept")

# Create a visualization
fig, ax = plt.subplots(figsize=(10, 5))

# Draw boxes representing models and datasets
def draw_box(x, y, width, height, color, text, alpha=0.7):
    rect = plt.Rectangle((x, y), width, height, color=color, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontweight='bold')

# Source domain
draw_box(1, 4, 3, 2, '#3498db', 'Source Dataset\n(e.g., ImageNet)\n(Millions of images)', alpha=0.4)
draw_box(1, 1, 3, 2, '#e74c3c', 'Pre-trained Model\n(e.g., VGG16, ResNet)')

# Target domain
draw_box(6, 4, 3, 2, '#2ecc71', 'Target Dataset\n(Small, specific dataset)', alpha=0.4)
draw_box(6, 1, 3, 2, '#9b59b6', 'Fine-tuned Model\n(Your specific task)')

# Arrows
ax.arrow(2.5, 3, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.arrow(4, 2, 2, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.arrow(7.5, 4, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')

# Add labels
ax.text(2.5, 3.5, "Train", ha='center')
ax.text(5, 2.2, "Transfer Weights", ha='center')
ax.text(7.5, 3.5, "Fine-tune", ha='center')

ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
plt.title("Transfer Learning Process")

st.pyplot(fig)

# Common base models
st.subheader("Popular Pre-trained Models")

models_info = {
    "VGG16": "16-layer CNN by Oxford. Simple architecture with good feature extraction.",
    "ResNet50": "50-layer CNN by Microsoft. Introduced residual connections to train deep networks.",
    "InceptionV3": "48-layer CNN by Google. Uses inception modules with multiple filter sizes.",
    "MobileNetV2": "Lightweight CNN by Google. Designed for mobile and edge devices.",
    "EfficientNet": "CNN family by Google. Optimized for accuracy and efficiency trade-off.",
    "DenseNet121": "121-layer CNN with dense connections between layers."
}

for model, desc in models_info.items():
    st.markdown(f"**{model}**: {desc}")

# Instructions for model preparation
st.markdown("---")
st.header("Model Preparation Tips")
st.write("""
For best results:
1. Start with a pre-trained model like VGG16, ResNet50, or MobileNetV2
2. Fine-tune the model on your specific dataset
3. Save the entire model in h5 or SavedModel format
4. Provide the correct class names in the same order as used during training
5. Select the appropriate base model in the sidebar to ensure correct preprocessing
""")
