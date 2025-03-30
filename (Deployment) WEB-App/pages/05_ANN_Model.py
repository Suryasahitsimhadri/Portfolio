import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import io

from model_handlers.ann_handler import ANNModelHandler
from utils import set_page_config, save_uploaded_file

# Set page configuration
set_page_config("ANN Model Deployment")

st.title("Artificial Neural Network (ANN) Model")
st.write("""
This page allows you to use a pre-trained ANN model for tabular data. 
ANNs are versatile models suitable for various classification and regression tasks.
""")

# Initialize model handler
ann_handler = ANNModelHandler()

# Create a sidebar for model settings
st.sidebar.header("Model Settings")

# Model selection
model_path = st.sidebar.text_input("Model Path", placeholder="Enter path to your ANN model file (.h5)")

# Option to upload a model file
uploaded_model = st.sidebar.file_uploader("Or upload a model file", type=['h5', 'keras'])
if uploaded_model is not None:
    # Save the uploaded model to a temporary file
    model_path = save_uploaded_file(uploaded_model)

# Task type selection
task_type = st.sidebar.selectbox(
    "Task Type",
    ["Classification", "Regression"],
    help="Select the type of task for your ANN model"
)

# Class names input (for classification models)
if task_type == "Classification":
    class_names_input = st.sidebar.text_area("Class Names (one per line)", 
                                            placeholder="Enter class names (one per line)")
    class_names = [name.strip() for name in class_names_input.split('\n') if name.strip()]

# Feature names input
feature_names_input = st.sidebar.text_area("Feature Names (one per line)", 
                                         placeholder="Enter feature names (one per line)")
feature_names = [name.strip() for name in feature_names_input.split('\n') if name.strip()]

# Load the model when the user clicks the button
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model..."):
        if ann_handler.load(model_path):
            # Set model properties
            ann_handler.set_task_type(task_type == "Classification")
            
            if task_type == "Classification" and class_names:
                ann_handler.set_class_names(class_names)
            
            if feature_names:
                ann_handler.set_feature_names(feature_names)
                
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model. Please check the path or file.")

# Main content area
st.header(f"ANN Model for {task_type}")

# Tabular data input
st.subheader("Input Data")
st.write("Upload a CSV file containing the data for prediction:")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Feature selection
        st.subheader("Feature Selection")
        
        if feature_names:
            # Check if all feature names exist in the dataframe
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                st.warning(f"The following features are not in the uploaded data: {', '.join(missing_features)}")
                st.info("Please select features from the uploaded data:")
                selected_features = st.multiselect("Select features for prediction", 
                                                  list(df.columns))
            else:
                st.info(f"Using pre-defined features: {', '.join(feature_names)}")
                selected_features = feature_names
        else:
            selected_features = st.multiselect("Select features for prediction", 
                                              list(df.columns),
                                              default=list(df.columns))
        
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            # Check if features exist in the data
            missing_cols = [col for col in selected_features if col not in df.columns]
            if missing_cols:
                st.error(f"The following selected features are not in the data: {', '.join(missing_cols)}")
            else:
                # Extract features for prediction
                X = df[selected_features]
                
                # Make prediction when button is clicked
                if st.button("Predict"):
                    if ann_handler.model is None:
                        st.error("Please load a model first!")
                    else:
                        with st.spinner("Processing..."):
                            try:
                                # Preprocess the data
                                preprocessed_data = ann_handler.preprocess_data(X)
                                
                                # Make prediction
                                result = ann_handler.predict(preprocessed_data)
                                
                                # Display prediction results
                                ann_handler.display_prediction(result)
                                
                                # Save predictions
                                st.subheader("Download Predictions")
                                
                                # Create a dataframe with predictions
                                if ann_handler.is_classification:
                                    # For classification
                                    if "predicted_class" in result:
                                        if ann_handler.class_names and max(result["predicted_class"]) < len(ann_handler.class_names):
                                            pred_df = pd.DataFrame({
                                                "Predicted Class": result["predicted_class"],
                                                "Class Name": [ann_handler.class_names[c] for c in result["predicted_class"]],
                                                "Confidence": result["confidence"]
                                            })
                                        else:
                                            pred_df = pd.DataFrame({
                                                "Predicted Class": result["predicted_class"],
                                                "Confidence": result["confidence"]
                                            })
                                else:
                                    # For regression
                                    pred_df = pd.DataFrame({
                                        "Predicted Value": result["predicted_value"]
                                    })
                                
                                # Add original data
                                for col in X.columns:
                                    pred_df[col] = X[col].values
                                
                                # Convert to CSV for download
                                csv = pred_df.to_csv(index=False)
                                
                                st.download_button(
                                    label="Download Predictions as CSV",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
    
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

else:
    # Sample data input form
    st.subheader("Or enter sample data:")
    
    if feature_names:
        # Create input fields for each feature
        sample_data = {}
        for feature in feature_names:
            sample_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
        
        if st.button("Predict Sample"):
            if ann_handler.model is None:
                st.error("Please load a model first!")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Convert to dataframe
                        sample_df = pd.DataFrame([sample_data])
                        
                        # Preprocess the data
                        preprocessed_data = ann_handler.preprocess_data(sample_df)
                        
                        # Make prediction
                        result = ann_handler.predict(preprocessed_data)
                        
                        # Display prediction results
                        ann_handler.display_prediction(result)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
    else:
        st.info("Please load a model and provide feature names to enable sample data entry.")

# Additional information
st.markdown("---")
st.header("About ANN Models")
st.write("""
Artificial Neural Networks (ANNs) are the foundation of deep learning, consisting of layers of interconnected nodes:

1. **Input Layer**: Receives feature data
2. **Hidden Layers**: Process and transform the data through weighted connections
3. **Output Layer**: Produces the final prediction

ANNs excel at:
- Classification tasks
- Regression problems
- Pattern recognition
- Feature learning
""")

# Example architecture visualization
st.subheader("ANN Architecture")

# Create a visualization of a simple ANN
fig, ax = plt.subplots(figsize=(10, 6))

# Network parameters
n_input = 4
n_hidden_1 = 5
n_hidden_2 = 5
n_output = 3

# Node coordinates
node_coords = {
    'input': [(1, i+1) for i in range(n_input)],
    'hidden1': [(3, i+0.5) for i in range(n_hidden_1)],
    'hidden2': [(5, i+0.5) for i in range(n_hidden_2)],
    'output': [(7, i+1.5) for i in range(n_output)]
}

# Draw nodes
node_colors = {
    'input': '#3498db',
    'hidden1': '#e74c3c',
    'hidden2': '#e74c3c',
    'output': '#2ecc71'
}

# Draw edges between layers
for layer1, layer2 in [('input', 'hidden1'), ('hidden1', 'hidden2'), ('hidden2', 'output')]:
    for coord1 in node_coords[layer1]:
        for coord2 in node_coords[layer2]:
            ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 'k-', alpha=0.1)

# Draw nodes
for layer, coords in node_coords.items():
    for x, y in coords:
        circle = plt.Circle((x, y), 0.3, color=node_colors[layer], alpha=0.7)
        ax.add_patch(circle)

# Add layer labels
ax.text(1, n_input+1.5, "Input Layer", ha='center', fontsize=12)
ax.text(3, n_hidden_1+1.5, "Hidden Layer 1", ha='center', fontsize=12)
ax.text(5, n_hidden_2+1.5, "Hidden Layer 2", ha='center', fontsize=12)
ax.text(7, n_output+1.5, "Output Layer", ha='center', fontsize=12)

ax.axis('off')
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
plt.title("Artificial Neural Network Architecture")

st.pyplot(fig)

# Instructions for model preparation
st.markdown("---")
st.header("Model Preparation Tips")
st.write("""
For best results:
1. Use an ANN model trained specifically for your classification or regression task
2. The model should be saved in h5 or SavedModel format
3. Provide the correct feature names in the same order as used during training
4. For classification, include class names to make predictions more interpretable
5. Ensure your test data matches the format used during training
""")
