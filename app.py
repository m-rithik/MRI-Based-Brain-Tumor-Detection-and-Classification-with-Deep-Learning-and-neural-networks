"""
Flask Backend for Brain Tumor Detection Web Application
Handles image uploads, model predictions, and serves the web interface.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from model import create_model, preprocess_image, interpret_prediction

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load or create the model."""
    global model
    
    # Check if a trained model exists
    model_path = 'models/brain_tumor_model.h5'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating new model (demo mode - random predictions)")
        model = create_model()
        # Note: In production, you would load actual trained weights here
        print("⚠️  Warning: Using untrained model for demonstration purposes")
        print("⚠️  Train the model with your dataset for accurate predictions")


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('.', path)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    Expects a file upload with key 'image'.
    """
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_image = preprocess_image(filepath)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Interpret results
        result = interpret_prediction(prediction)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Brain Tumor Detection System - Starting Server")
    print("=" * 60)
    
    # Load the model
    load_model()
    
    print("\n✓ Server initialized successfully")
    print("✓ Navigate to http://localhost:8000 in your browser")
    print("\n" + "=" * 60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8000)
