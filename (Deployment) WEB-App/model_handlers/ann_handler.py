import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_model, display_metrics, plot_prediction_results
import pickle

class ANNModelHandler:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.is_classification = True  # Default to classification
        self.scaler = None
        self.feature_names = []
    
    def load(self, model_path=None):
        """Load the ANN model from the specified path"""
        if model_path:
            self.model_path = model_path
        
        if self.model_path:
            self.model = load_model(self.model_path)
            return self.model is not None
        return False
    
    def set_class_names(self, class_names):
        """Set class names for prediction output"""
        self.class_names = class_names
    
    def set_feature_names(self, feature_names):
        """Set feature names for input data"""
        self.feature_names = feature_names
    
    def set_task_type(self, is_classification=True):
        """Set whether the model is for classification or regression"""
        self.is_classification = is_classification
    
    def set_scaler(self, scaler):
        """Set scaler for data preprocessing"""
        self.scaler = scaler
    
    def preprocess_data(self, data):
        """Preprocess input data for prediction"""
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, np.ndarray):
                if self.feature_names and len(self.feature_names) == data.shape[1]:
                    data = pd.DataFrame(data, columns=self.feature_names)
                else:
                    data = pd.DataFrame(data)
            else:
                st.error("Input data must be a DataFrame or NumPy array")
                return None
        
        # Apply scaling if scaler is available
        if self.scaler:
            try:
                data_scaled = self.scaler.transform(data)
            except:
                # Create a new scaler if transform fails
                st.warning("Using a new scaler for preprocessing.")
                if isinstance(self.scaler, StandardScaler):
                    new_scaler = StandardScaler()
                else:
                    new_scaler = MinMaxScaler()
                data_scaled = new_scaler.fit_transform(data)
        else:
            # Create a default scaler
            st.info("No scaler provided. Using MinMaxScaler by default.")
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
        
        return data_scaled
    
    def predict(self, preprocessed_data):
        """Make prediction on the preprocessed data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        predictions = self.model.predict(preprocessed_data)
        
        # Handle different model output formats
        if self.is_classification:
            if predictions.shape[-1] == 1:  # Binary classification
                predicted_class = (predictions > 0.5).astype(int).flatten()
                confidences = np.where(predicted_class == 1, predictions, 1 - predictions).flatten()
            else:  # Multi-class classification
                predicted_class = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)
            
            result = {
                "predicted_class": predicted_class,
                "confidence": confidences,
                "predictions": predictions
            }
        else:  # Regression
            result = {
                "predicted_value": predictions.flatten(),
                "predictions": predictions
            }
        
        return result
    
    def display_prediction(self, result, original_data=None):
        """Display prediction results"""
        # Check task type and display accordingly
        if self.is_classification:
            # For classification models
            st.subheader("Classification Prediction Results")
            
            if len(result["predicted_class"]) == 1:
                # Single prediction
                predicted_class = result["predicted_class"][0]
                confidence = result["confidence"][0]
                
                # Display class name if available, otherwise just the index
                if self.class_names and predicted_class < len(self.class_names):
                    class_name = self.class_names[predicted_class]
                    st.success(f"Predicted Class: {class_name}")
                else:
                    st.success(f"Predicted Class: {predicted_class}")
                
                st.info(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
                
                # If we have multiple classes, show the distribution for the first prediction
                if result["predictions"].shape[1] > 1:
                    self.plot_prediction_distribution(result["predictions"][0])
            else:
                # Multiple predictions
                st.write("Multiple predictions:")
                
                # Create a DataFrame for predictions
                pred_df = pd.DataFrame({
                    "Predicted Class": result["predicted_class"],
                    "Confidence": result["confidence"]
                })
                
                # Add class names if available
                if self.class_names and max(result["predicted_class"]) < len(self.class_names):
                    pred_df["Class Name"] = [self.class_names[c] for c in result["predicted_class"]]
                
                st.dataframe(pred_df)
                
                # Plot class distribution
                self.plot_class_distribution(result["predicted_class"])
        
        else:
            # For regression models
            st.subheader("Regression Prediction Results")
            
            if len(result["predicted_value"]) == 1:
                # Single prediction
                st.success(f"Predicted Value: {result['predicted_value'][0]:.4f}")
            else:
                # Multiple predictions
                st.write("Prediction Results:")
                
                # Create a DataFrame for predictions
                pred_df = pd.DataFrame({
                    "Predicted Value": result["predicted_value"]
                })
                
                st.dataframe(pred_df)
                
                # Plot predicted values
                self.plot_regression_predictions(result["predicted_value"])
    
    def plot_prediction_distribution(self, predictions):
        """Plot distribution of class probabilities for a single prediction"""
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
    
    def plot_class_distribution(self, predicted_classes):
        """Plot distribution of predicted classes for multiple predictions"""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Count occurrences of each class
        unique_classes, counts = np.unique(predicted_classes, return_counts=True)
        
        # Get class labels
        if self.class_names and max(unique_classes) < len(self.class_names):
            labels = [self.class_names[c] for c in unique_classes]
        else:
            labels = [f"Class {c}" for c in unique_classes]
        
        # Create bar chart
        bars = ax.bar(labels, counts)
        
        # Customize chart
        ax.set_ylabel('Count')
        ax.set_title('Predicted Class Distribution')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    def plot_regression_predictions(self, predicted_values):
        """Plot distribution of predicted values for regression"""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create histogram
        ax.hist(predicted_values, bins=20, alpha=0.7)
        
        # Add mean and median lines
        mean_val = np.mean(predicted_values)
        median_val = np.median(predicted_values)
        
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')
        
        # Customize chart
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predicted Values')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
