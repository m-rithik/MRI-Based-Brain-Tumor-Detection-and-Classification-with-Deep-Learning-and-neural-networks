"""
Brain Tumor Detection Model
This module defines the CNN architecture and provides utilities for model creation and prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from PIL import Image

# Model configuration
IMG_SIZE = 224
NUM_CLASSES = 2  # Binary: tumor vs no tumor


def create_model():
    """
    Creates a CNN model for brain tumor detection.
    Architecture: Conv layers -> Pooling -> Dense layers -> Output
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def preprocess_image(image_path):
    """
    Preprocesses the input image for model prediction.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Preprocessed image array ready for prediction
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def get_class_labels():
    """Returns the class labels for predictions."""
    return ['No Tumor', 'Tumor Detected']


def interpret_prediction(prediction):
    """
    Interprets model prediction and returns human-readable results.
    
    Args:
        prediction: Model prediction array
        
    Returns:
        Dictionary with classification results and confidence
    """
    class_labels = get_class_labels()
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class]) * 100
    
    result = {
        'classification': class_labels[predicted_class],
        'confidence': round(confidence, 2),
        'probabilities': {
            'no_tumor': round(float(prediction[0][0]) * 100, 2),
            'tumor': round(float(prediction[0][1]) * 100, 2)
        },
        'tumor_detected': bool(predicted_class == 1)
    }
    
    return result
