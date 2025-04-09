import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import io
import pickle

from model_handlers.rnn_handler import RNNModelHandler
from utils import set_page_config, save_uploaded_file

# Set page configuration
set_page_config("RNN Model Deployment")

st.title("Recurrent Neural Network (RNN) Model")
st.write("""
This page allows you to use a pre-trained RNN model for sequence data prediction or classification.
RNNs are effective for processing sequential data like text or time series.
""")

# Initialize model handler
rnn_handler = RNNModelHandler()

# Create a sidebar for model settings
st.sidebar.header("Model Settings")

# Model selection
model_path = st.sidebar.text_input("Model Path", placeholder="Enter path to your RNN model file")

# Option to upload a model file
uploaded_model = st.sidebar.file_uploader("Or upload a model file", type=['h5', 'keras', 'pkl'])
if uploaded_model is not None:
    # Save the uploaded model to a temporary file
    model_path = save_uploaded_file(uploaded_model)

# Input type selection
input_type = st.sidebar.selectbox(
    "Input Type",
    ["Text", "Time Series"],
    help="Select the type of input data for your RNN model"
)

# Class names input (for classification models)
class_names_input = st.sidebar.text_area("Class Names (one per line)", 
                                        placeholder="Enter class names (one per line)")
class_names = [name.strip() for name in class_names_input.split('\n') if name.strip()]

# Sequence length input
sequence_length = st.sidebar.number_input("Sequence Length", min_value=1, max_value=1000, value=50)

# Load the model when the user clicks the button
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model..."):
        if rnn_handler.load(model_path):
            rnn_handler.sequence_length = sequence_length
            if class_names:
                rnn_handler.set_class_names(class_names)
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model. Please check the path or file.")

# Main content area
st.header(f"RNN Model for {input_type} Processing")

# Different input methods based on input type
if input_type == "Text":
    st.subheader("Text Input")
    
    # Option to upload a text file
    uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
    
    if uploaded_file is not None:
        text_content = uploaded_file.getvalue().decode("utf-8")
        st.text_area("Text content", text_content, height=150)
        
        text_to_process = text_content
    else:
        # Or enter text directly
        text_to_process = st.text_area("Or enter text directly", height=150,
                                       placeholder="Enter text for processing...")
    
    # Make prediction when button is clicked
    if st.button("Process Text"):
        if not text_to_process:
            st.error("Please enter some text or upload a text file.")
        elif rnn_handler.model is None:
            st.error("Please load a model first!")
        else:
            with st.spinner("Processing..."):
                try:
                    # Preprocess the text
                    preprocessed_text = rnn_handler.preprocess_text(text_to_process)
                    
                    # Make prediction
                    result = rnn_handler.predict(preprocessed_text)
                    
                    # Display prediction results
                    rnn_handler.display_prediction(result)
                    
                except Exception as e:
                    st.error(f"Error during processing: {e}")
    
    # Text generation option (if model supports it)
    st.subheader("Text Generation")
    st.write("If your RNN model is trained for text generation, you can use it to generate new text.")
    
    seed_text = st.text_input("Seed Text", placeholder="Enter some seed text...")
    next_words = st.slider("Number of words to generate", min_value=10, max_value=500, value=50)
    
    if st.button("Generate Text"):
        if not seed_text:
            st.error("Please enter seed text.")
        elif rnn_handler.model is None:
            st.error("Please load a model first!")
        else:
            with st.spinner("Generating..."):
                try:
                    # Generate text
                    generated_text = rnn_handler.generate_text(seed_text, next_words)
                    
                    if generated_text:
                        st.subheader("Generated Text")
                        st.write(generated_text)
                    else:
                        st.warning("Text generation not supported with the current model configuration.")
                except Exception as e:
                    st.error(f"Error during text generation: {e}")

else:  # Time Series
    st.subheader("Time Series Input")
    
    # Option to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Column selection
            st.subheader("Select Columns")
            
            # Let the user select the time column and value columns
            time_col = st.selectbox("Select time/index column (optional)", 
                                   ["None"] + list(df.columns))
            
            value_cols = st.multiselect("Select value column(s) to process", 
                                        list(df.columns),
                                        default=list(df.columns)[0] if len(df.columns) > 0 else [])
            
            if not value_cols:
                st.error("Please select at least one value column.")
            else:
                # Prepare the data
                if time_col != "None":
                    st.line_chart(df.set_index(time_col)[value_cols])
                else:
                    st.line_chart(df[value_cols])
                
                # Extract values for processing
                data_to_process = df[value_cols].values
                
                # Make prediction when button is clicked
                if st.button("Process Time Series"):
                    if rnn_handler.model is None:
                        st.error("Please load a model first!")
                    else:
                        with st.spinner("Processing..."):
                            try:
                                # Preprocess the data
                                preprocessed_data = rnn_handler.preprocess_sequence_data(data_to_process)
                                
                                if preprocessed_data is not None:
                                    # Make prediction
                                    result = rnn_handler.predict(preprocessed_data)
                                    
                                    # Display prediction results
                                    rnn_handler.display_prediction(result, data_to_process)
                                    
                                    # Option to forecast future values
                                    st.subheader("Time Series Forecast")
                                    forecast_steps = st.slider("Number of steps to forecast", 
                                                             min_value=1, max_value=100, value=10)
                                    
                                    if st.button("Generate Forecast"):
                                        with st.spinner("Forecasting..."):
                                            forecast = rnn_handler.forecast_timeseries(
                                                preprocessed_data, forecast_steps)
                                            
                                            if forecast is not None:
                                                # Plot the forecast
                                                st.subheader("Forecast Results")
                                                
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                # Plot original data
                                                time_steps = range(len(data_to_process))
                                                ax.plot(time_steps, data_to_process[:, 0], 
                                                        label='Original Data')
                                                
                                                # Plot forecast
                                                forecast_steps = range(
                                                    len(data_to_process), 
                                                    len(data_to_process) + len(forecast)
                                                )
                                                ax.plot(forecast_steps, forecast[:, 0], 
                                                        label='Forecast', color='red')
                                                
                                                ax.set_xlabel('Time Step')
                                                ax.set_ylabel('Value')
                                                ax.set_title('Time Series Forecast')
                                                ax.legend()
                                                
                                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error during processing: {e}")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    
    else:
        st.info("Please upload a CSV file containing time series data.")

# Additional information
st.markdown("---")
st.header("About RNN Models")
st.write("""
Recurrent Neural Networks (RNNs) are designed to work with sequential data by maintaining an internal state (memory):

1. **Input Layer**: Receives sequential data
2. **Recurrent Layers**: Process each element while considering previous elements 
3. **Output Layer**: Produces predictions

RNNs excel at:
- Text classification
- Time series forecasting
- Speech recognition
- Natural language processing
""")

# Example architecture visualization
st.subheader("RNN Architecture")

# Create a simple visualization of RNN architecture
fig, ax = plt.subplots(figsize=(10, 4))

# Draw the unfolded RNN
def draw_cell(x, y, width, height, color, text):
    rect = plt.Rectangle((x, y), width, height, color=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', color='white', fontweight='bold')

# Draw nodes and connections
time_steps = 4
for t in range(time_steps):
    # Input node
    draw_cell(t*3, 0, 1, 1, '#3498db', f'x{t}')
    
    # Hidden state
    draw_cell(t*3, 2, 1, 1, '#e74c3c', f'h{t}')
    
    # Output node
    draw_cell(t*3, 4, 1, 1, '#2ecc71', f'y{t}')
    
    # Vertical connections
    ax.arrow(t*3 + 0.5, 1, 0, 0.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(t*3 + 0.5, 3, 0, 0.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Recurrent connection (except for the first cell)
    if t > 0:
        ax.arrow((t-1)*3 + 1, 2.5, t*3 - (t-1)*3 - 1, 0, head_width=0.1, head_length=0.1, 
                fc='black', ec='black', linestyle='dashed')

ax.axis('off')
ax.set_xlim(-0.5, time_steps*3 + 0.5)
ax.set_ylim(-0.5, 5.5)
plt.title("Unfolded RNN Architecture")

st.pyplot(fig)

# Instructions for model preparation
st.markdown("---")
st.header("Model Preparation Tips")
st.write("""
For best results:
1. Use an RNN model trained specifically for your sequence task
2. The model should be saved in h5 or SavedModel format
3. Provide the correct sequence length used during training
4. For text models, it's best to use the same tokenizer used during training
""")
