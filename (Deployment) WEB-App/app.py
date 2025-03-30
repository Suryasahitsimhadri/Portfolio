import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from utils import set_page_config

# Set page configuration
set_page_config("Deep Learning Models Dashboard")

st.title("Deep Learning Models Deployment")
st.write("""
This application allows you to use pre-trained deep learning models for inference.
Choose a model from the sidebar to get started.
""")

# Add information about each model
st.header("Available Models")

st.subheader("1. Convolutional Neural Network (CNN)")
st.write("""
CNNs are specialized neural networks for processing grid-like data, such as images.
They are designed to automatically and adaptively learn spatial hierarchies of features.
Use this model for image classification tasks.
""")

st.subheader("2. Recurrent Neural Network (RNN)")
st.write("""
RNNs are designed to work with sequence data by maintaining a state (memory) as they process each element.
They are suitable for tasks like time series prediction and text generation.
""")

st.subheader("3. Long Short-Term Memory (LSTM)")
st.write("""
LSTMs are a special kind of RNN designed to avoid the long-term dependency problem.
They excel at learning from data with long time lags between important events.
Use this model for sequence tasks where long-term dependencies are important.
""")

st.subheader("4. Transfer Learning")
st.write("""
Transfer learning involves using pre-trained models (like VGG16, ResNet, etc.) and
fine-tuning them on specific tasks. This approach often yields better results with 
less data and training time.
""")

st.subheader("5. Artificial Neural Network (ANN)")
st.write("""
Basic feed-forward neural networks for general-purpose machine learning tasks.
These models work well with tabular data and can solve classification or regression problems.
""")

# Instructions
st.header("How to Use")
st.write("""
1. Select a model from the sidebar or navigation
2. Upload your test data in the appropriate format
3. View the model's predictions and performance metrics
""")

# Model requirements
st.header("Data Requirements")
models_info = {
    "CNN": "Images (.jpg, .png, .jpeg)",
    "RNN": "Sequence data (CSV, text files)",
    "LSTM": "Sequence data (CSV, text files) with temporal dependencies",
    "Transfer Learning": "Images (.jpg, .png, .jpeg)",
    "ANN": "Tabular data (CSV)"
}

info_df = pd.DataFrame({
    "Model": models_info.keys(),
    "Data Type": models_info.values()
})

st.table(info_df)

# Footer
st.markdown("---")
st.markdown("© 2023 Deep Learning Models Deployment App")
