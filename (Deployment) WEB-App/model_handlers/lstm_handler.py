import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from utils import load_model, display_metrics, plot_prediction_results

class LSTMModelHandler:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.sequence_length = 50  # Default sequence length
        self.vocab = None
        self.tokenizer = None
        self.max_features = 10000  # Default vocabulary size
        self.class_names = []
    
    def load(self, model_path=None):
        """Load the LSTM model from the specified path"""
        if model_path:
            self.model_path = model_path
        
        if self.model_path:
            self.model = load_model(self.model_path)
            # Try to get sequence length from model
            if self.model:
                try:
                    self.sequence_length = self.model.input_shape[1]
                except:
                    st.warning("Couldn't determine sequence length, using default (50)")
            return self.model is not None
        return False
    
    def set_class_names(self, class_names):
        """Set class names for prediction output"""
        self.class_names = class_names
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for text preprocessing"""
        self.tokenizer = tokenizer
    
    def preprocess_text(self, text):
        """Preprocess text for LSTM model"""
        if self.tokenizer:
            # Use the loaded tokenizer
            sequences = self.tokenizer.texts_to_sequences([text])
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, 
                maxlen=self.sequence_length, 
                padding='post'
            )
            return padded_sequences
        else:
            # Simple tokenization if no tokenizer provided
            # Clean the text
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            
            # Tokenize by words
            tokens = text.split()
            
            # Convert to sequence of integers (simple implementation)
            # In real scenario, you'd use a proper vocabulary
            token_ids = []
            for token in tokens:
                # Use hash function to convert tokens to integers within max_features range
                token_id = hash(token) % self.max_features
                token_ids.append(token_id)
            
            # Pad or truncate to sequence_length
            if len(token_ids) > self.sequence_length:
                token_ids = token_ids[:self.sequence_length]
            else:
                token_ids = token_ids + [0] * (self.sequence_length - len(token_ids))
            
            return np.array([token_ids])
    
    def preprocess_sequence_data(self, data, time_steps=None):
        """Preprocess sequence data (not text)"""
        if time_steps is None:
            time_steps = self.sequence_length
        
        # Convert to numpy if it's pandas
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        
        # Ensure data is 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Create sequences
        sequences = []
        for i in range(len(data) - time_steps + 1):
            sequences.append(data[i:i+time_steps])
        
        if len(sequences) == 0:
            st.error(f"Data is too short for the time steps ({time_steps}). Please provide more data.")
            return None
        
        return np.array(sequences)
    
    def predict(self, preprocessed_data):
        """Make prediction on the preprocessed data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        predictions = self.model.predict(preprocessed_data)
        
        # Handle different model output formats
        if len(predictions.shape) == 3:  # Sequence-to-sequence output
            # For seq2seq, we might want to return the entire sequence
            result = {
                "sequence_output": predictions[0],
            }
        elif predictions.shape[-1] == 1:  # Binary classification or regression
            if self.model.output_shape[-1] == 1 and 'sigmoid' in str(self.model.layers[-1].activation).lower():
                # Binary classification
                predicted_class = int(predictions[0][0] > 0.5)
                confidence = float(predictions[0][0]) if predicted_class == 1 else 1 - float(predictions[0][0])
                result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "predictions": predictions[0]
                }
            else:
                # Regression
                result = {
                    "predicted_value": float(predictions[0][0]),
                    "predictions": predictions[0]
                }
        else:  # Multi-class classification
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "predictions": predictions[0]
            }
        
        return result
    
    def display_prediction(self, result, original_data=None):
        """Display prediction results"""
        # Check result type and display accordingly
        if "sequence_output" in result:
            # For sequence-to-sequence models
            st.subheader("Sequence Output Prediction")
            
            # Plot the sequence output
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(result["sequence_output"])
            ax.set_title("Predicted Sequence")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            st.pyplot(fig)
            
        elif "predicted_class" in result:
            # For classification models
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            
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
        
        else:
            # For regression models
            st.subheader("Prediction Results")
            st.success(f"Predicted Value: {result['predicted_value']:.4f}")
            
            # If original data is provided, we can plot it with prediction
            if original_data is not None:
                self.plot_regression_result(original_data, result['predicted_value'])
    
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
    
    def plot_regression_result(self, original_data, prediction):
        """Plot regression prediction against original data"""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot original data
        if len(original_data.shape) > 1:
            # Multi-feature data, take the first feature only for plotting
            ax.plot(original_data[:, 0], label='Original Data (Feature 1)')
        else:
            ax.plot(original_data, label='Original Data')
        
        # Add prediction
        ax.axhline(y=prediction, color='r', linestyle='-', label='Prediction')
        
        ax.set_title('Data with Prediction')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def generate_text(self, seed_text, next_words=50):
        """Generate text using the LSTM model"""
        if not self.tokenizer:
            st.error("Tokenizer is not available. Cannot generate text.")
            return None
        
        generated_text = seed_text
        for _ in range(next_words):
            # Tokenize and pad the current text
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
            token_list = token_list[-self.sequence_length:]
            token_list = tf.keras.preprocessing.sequence.pad_sequences(
                [token_list], maxlen=self.sequence_length, padding='pre'
            )
            
            # Predict next token
            predicted_probs = self.model.predict(token_list)[0]
            predicted_token = np.argmax(predicted_probs)
            
            # Convert token to word and add to generated text
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_token:
                    generated_text += " " + word
                    break
        
        return generated_text
    
    def forecast_timeseries(self, input_sequence, forecast_steps=10):
        """Generate time series forecast using the LSTM model"""
        if self.model is None:
            st.error("Model is not loaded. Cannot generate forecast.")
            return None
        
        # Create a copy of the last sequence
        curr_sequence = input_sequence[0].copy()
        forecast = []
        
        for _ in range(forecast_steps):
            # Reshape for prediction and predict
            curr_sequence_reshaped = curr_sequence.reshape(1, self.sequence_length, -1)
            predicted = self.model.predict(curr_sequence_reshaped)[0]
            
            # Add prediction to forecast list
            forecast.append(predicted)
            
            # Update sequence for next iteration
            # Remove first and add prediction at the end
            curr_sequence = np.roll(curr_sequence, -1, axis=0)
            curr_sequence[-1] = predicted
        
        return np.array(forecast)
