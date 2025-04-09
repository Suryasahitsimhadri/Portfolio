import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import io

def set_page_config(title):
    """Set page configuration with consistent settings"""
    st.set_page_config(
        page_title=title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_model(model_path):
    """Load a TensorFlow/Keras model from path"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
def load_pkl_model(file_path):
    """Load a model saved as a .pkl file"""
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading .pkl model: {e}")
        return None


def plot_model_history(history):
    """Plot training history of a model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    if class_names:
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig

def load_and_preprocess_image(uploaded_file, target_size=(224, 224)):
    """Load and preprocess an image for CNN/Transfer Learning models"""
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    return np.expand_dims(image_array, axis=0), image  # Return preprocessed array and original image

def display_metrics(y_true, y_pred, model_type="classification"):
    """Display relevant metrics based on model type"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
    
    metrics = {}
    
    if model_type == "classification":
        # For classification models
        if len(np.unique(y_true)) > 2:  # multiclass
            avg = 'weighted'
        else:
            avg = 'binary'
            
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics["F1 Score"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    else:
        # For regression models
        metrics["Mean Squared Error"] = mean_squared_error(y_true, y_pred)
        metrics["Root Mean Squared Error"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["R² Score"] = r2_score(y_true, y_pred)
    
    return metrics

def plot_prediction_results(y_true, y_pred, model_type="classification"):
    """Plot prediction results based on model type"""
    if model_type == "classification":
        # For classification, show distribution of actual vs predicted classes
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        class_labels = sorted(list(set(np.concatenate([y_true, y_pred]))))
        
        return plot_confusion_matrix(cm, class_labels)
    else:
        # For regression, plot actual vs predicted values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_true, y_pred)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        
        return fig

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def save_uploaded_file(uploaded_file):
    # Code to save the file
    with open('uploads/' + uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.read())
    return 'uploads/' + uploaded_file.name

import os

def save_uploaded_file(uploaded_file):
    # Create the 'uploads' directory if it doesn't exist
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    # Save the uploaded file in the 'uploads' folder
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

import pickle

def load_pkl_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model
