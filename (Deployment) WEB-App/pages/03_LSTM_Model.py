import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import io

from model_handlers.lstm_handler import LSTMModelHandler
from utils import set_page_config, save_uploaded_file

# Set page configuration
set_page_config("LSTM Model Deployment")

st.title("Long Short-Term Memory (LSTM) Model")
st.write("""
This page allows you to use a pre-trained LSTM model for sequence data with long-term dependencies.
LSTMs are specially designed RNNs that can remember information for long periods.
""")

# Initialize model handler
lstm_handler = LSTMModelHandler()

# Create a sidebar for model settings
st.sidebar.header("Model Settings")

# Model selection
model_path = st.sidebar.text_input("Model Path", placeholder="Enter path to your LSTM model file (.h5)")

# Option to upload a model file
uploaded_model = st.sidebar.file_uploader("Or upload a model file", type=['h5', 'keras'])
if uploaded_model is not None:
    # Save the uploaded model to a temporary file
    model_path = save_uploaded_file(uploaded_model)

# Input type selection
input_type = st.sidebar.selectbox(
    "Input Type",
    ["Text", "Time Series"],
    help="Select the type of input data for your LSTM model"
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
        if lstm_handler.load(model_path):
            lstm_handler.sequence_length = sequence_length
            if class_names:
                lstm_handler.set_class_names(class_names)
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model. Please check the path or file.")

# Main content area
st.header(f"LSTM Model for {input_type} Processing")

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
        elif lstm_handler.model is None:
            st.error("Please load a model first!")
        else:
            with st.spinner("Processing..."):
                try:
                    # Preprocess the text
                    preprocessed_text = lstm_handler.preprocess_text(text_to_process)
                    
                    # Make prediction
                    result = lstm_handler.predict(preprocessed_text)
                    
                    # Display prediction results
                    lstm_handler.display_prediction(result)
                    
                except Exception as e:
                    st.error(f"Error during processing: {e}")
    
    # Text generation option (if model supports it)
    st.subheader("Text Generation")
    st.write("If your LSTM model is trained for text generation, you can use it to generate new text.")
    
    seed_text = st.text_input("Seed Text", placeholder="Enter some seed text...")
    next_words = st.slider("Number of words to generate", min_value=10, max_value=500, value=50)
    
    if st.button("Generate Text"):
        if not seed_text:
            st.error("Please enter seed text.")
        elif lstm_handler.model is None:
            st.error("Please load a model first!")
        else:
            with st.spinner("Generating..."):
                try:
                    # Generate text
                    generated_text = lstm_handler.generate_text(seed_text, next_words)
                    
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
                    if lstm_handler.model is None:
                        st.error("Please load a model first!")
                    else:
                        with st.spinner("Processing..."):
                            try:
                                # Preprocess the data
                                preprocessed_data = lstm_handler.preprocess_sequence_data(data_to_process)
                                
                                if preprocessed_data is not None:
                                    # Make prediction
                                    result = lstm_handler.predict(preprocessed_data)
                                    
                                    # Display prediction results
                                    lstm_handler.display_prediction(result, data_to_process)
                                    
                                    # Option to forecast future values
                                    st.subheader("Time Series Forecast")
                                    forecast_steps = st.slider("Number of steps to forecast", 
                                                             min_value=1, max_value=100, value=10)
                                    
                                    if st.button("Generate Forecast"):
                                        with st.spinner("Forecasting..."):
                                            forecast = lstm_handler.forecast_timeseries(
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
st.header("About LSTM Models")
st.write("""
Long Short-Term Memory (LSTM) networks are a special kind of RNN designed to handle the vanishing gradient problem in traditional RNNs.

Key components of an LSTM cell:
1. **Forget Gate**: Decides what information to discard from the cell state
2. **Input Gate**: Updates the cell state with new information
3. **Output Gate**: Determines what to output based on the cell state

LSTMs excel at:
- Long-term sequence modeling
- Speech recognition
- Machine translation
- Time series prediction with long dependencies
""")

# Example architecture visualization
st.subheader("LSTM Cell Architecture")

# Create a visualization of LSTM cell
fig, ax = plt.subplots(figsize=(10, 6))

# Draw a simplified LSTM cell diagram
def draw_box(x, y, width, height, color, text, alpha=0.7):
    rect = plt.Rectangle((x, y), width, height, color=color, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontweight='bold')

def draw_circle(x, y, radius, color, text, textcolor='black', alpha=0.7):
    circle = plt.Circle((x, y), radius, color=color, alpha=alpha)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', color=textcolor, fontweight='bold')

# Cell state (horizontal line at top)
ax.plot([1, 9], [8, 8], 'k-', linewidth=2)
ax.arrow(9, 8, 0.5, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.text(5, 8.5, "Cell State", ha='center')

# Operations
draw_circle(2, 5, 0.5, '#3498db', '×', 'white')  # Forget gate
draw_circle(4, 5, 0.5, '#3498db', '+', 'black')  # Input gate
draw_circle(8, 5, 0.5, '#3498db', '×', 'white')  # Output gate

# Gates
draw_box(1.5, 3, 1, 1, '#e74c3c', 'σ')  # Forget gate
draw_box(3.5, 3, 1, 1, '#e74c3c', 'σ')  # Input gate
draw_box(5.5, 3, 1, 1, '#2ecc71', 'tanh')  # Cell update
draw_box(7.5, 3, 1, 1, '#e74c3c', 'σ')  # Output gate

# Hidden state (horizontal line at bottom)
ax.plot([1, 9], [1, 1], 'k-', linewidth=2)
ax.arrow(9, 1, 0.5, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.text(5, 0.5, "Hidden State", ha='center')

# Connect the components with arrows
# Vertical connections
ax.arrow(2, 1, 0, 1.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(4, 1, 0, 1.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(6, 1, 0, 1.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(8, 1, 0, 1.9, head_width=0.1, head_length=0.1, fc='black', ec='black')

# From gates to operations
ax.arrow(2, 4, 0, 0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(4, 4, 0, 0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(6, 4, 0, 0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(8, 4, 0, 0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Cell state interactions
ax.arrow(2, 5, 0, 3, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(4, 5, 0, 3, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(8, 5, 0, 3, head_width=0.1, head_length=0.1, fc='black', ec='black')

ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.title("Simplified LSTM Cell")

st.pyplot(fig)

# Instructions for model preparation
st.markdown("---")
st.header("Model Preparation Tips")
st.write("""
For best results:
1. Use an LSTM model trained specifically for your sequence task
2. The model should be saved in h5 or SavedModel format
3. Provide the correct sequence length used during training
4. LSTMs are particularly useful for tasks with long-term dependencies
5. For text models, it's best to use the same tokenizer used during training
""")
