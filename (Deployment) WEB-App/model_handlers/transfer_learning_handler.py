import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from utils import load_model, load_and_preprocess_image, display_metrics
import pickle

class TransferLearningModelHandler:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.input_shape = (224, 224, 3)  # Default input shape for many pre-trained models
        self.preprocessing_function = None
    
    def load(self, model_path=None):
        """Load the transfer learning model from the specified path"""
        if model_path:
            self.model_path = model_path
        
        if self.model_path:
            self.model = load_model(self.model_path)
            # Try to get input shape from model
            if self.model:
                try:
                    self.input_shape = self.model.input_shape[1:4]
                except:
                    st.warning("Couldn't determine input shape, using default (224, 224, 3)")
            return self.model is not None
        return False
    
    def set_class_names(self, class_names):
        """Set class names for prediction output"""
        self.class_names = class_names
    
    def set_preprocessing_function(self, preprocessing_function):
        """Set preprocessing function specific to the base model (e.g., VGG16, ResNet)"""
        self.preprocessing_function = preprocessing_function
    
    def preprocess_image(self, image_file):
        """Preprocess a single image for prediction"""
        target_size = (self.input_shape[0], self.input_shape[1])
        
        # Open image
        image = Image.open(image_file).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image)
        
        # Apply model-specific preprocessing if available
        if self.preprocessing_function:
            image_array = self.preprocessing_function(image_array)
        else:
            # Default preprocessing (rescale to [0,1])
            image_array = image_array / 255.0
        
        return np.expand_dims(image_array, axis=0), image
    
    def predict(self, image_array):
        """Make prediction on the preprocessed image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        predictions = self.model.predict(image_array)
        
        # Handle different model output formats
        if predictions.shape[-1] == 1:  # Binary classification
            predicted_class = int(predictions[0][0] > 0.5)
            confidence = float(predictions[0][0]) if predicted_class == 1 else 1 - float(predictions[0][0])
        else:  # Multi-class classification
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
        
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predictions": predictions[0]
        }
        return result
    
    def display_prediction(self, result, original_image):
        """Display prediction results"""
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        
        # Create columns for image and prediction
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(original_image, caption="Uploaded Image", width=300)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Display class name if available, otherwise just the index
            if self.class_names and predicted_class < len(self.class_names):
                class_name = self.class_names[predicted_class]
                st.success(f"Predicted Class: {class_name}")
            else:
                st.success(f"Predicted Class: {predicted_class}")
            
            st.info(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # If we have multiple classes, show the distribution
            if len(result["predictions"]) > 1:
                self.plot_prediction_distribution(result["predictions"])
    
    def plot_prediction_distribution(self, predictions):
        """Plot distribution of class probabilities"""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        x = range(len(predictions))
        labels = self.class_names if (self.class_names and len(self.class_names) == len(predictions)) else [f"Class {i}" for i in x]
        
        # Create bar chart
        bars = ax.bar(x, predictions)
        
        # Customize chart
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title('Class Probability Distribution')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def visualize_intermediate_activations(self, image_array, layer_name=None):
        """Visualize intermediate layer activations for the given image"""
        if self.model is None:
            st.error("Model not loaded. Cannot visualize activations.")
            return
        
        # If layer name not specified, use the last convolutional layer
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
            
            if layer_name is None:
                st.error("No convolutional layer found in the model.")
                return
        
        # Create a model that will output the activations
        try:
            layer_output = self.model.get_layer(layer_name).output
            activation_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_output)
            
            # Get activations
            activations = activation_model.predict(image_array)
            
            # Display a subset of the activations
            st.subheader(f"Activations from layer: {layer_name}")
            
            # Determine number of channels to show
            num_channels = min(16, activations.shape[-1])
            
            # Create a grid of activation maps
            fig, axs = plt.subplots(4, 4, figsize=(12, 12))
            axs = axs.flatten()
            
            for i in range(num_channels):
                axs[i].imshow(activations[0, :, :, i], cmap='viridis')
                axs[i].set_title(f'Channel {i}')
                axs[i].axis('off')
            
            # Hide any unused subplots
            for i in range(num_channels, len(axs)):
                axs[i].axis('off')
                
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error visualizing activations: {e}")
