# Running the Brain Tumor Detection Web Application

## Prerequisites

Make sure you have Python 3.8+ installed on your system.

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web server)
- TensorFlow (deep learning)
- OpenCV (image processing)
- Other required packages

### 2. Start the Flask Server

```bash
python app.py
```

You should see output like:
```
============================================================
Brain Tumor Detection System - Starting Server
============================================================

Creating new model (demo mode - random predictions)
⚠️  Warning: Using untrained model for demonstration purposes
⚠️  Train the model with your dataset for accurate predictions

✓ Server initialized successfully
✓ Navigate to http://localhost:5000 in your browser

============================================================
```

### 3. Open the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## Using the Application

1. **Upload an MRI Image**
   - Click the upload area or drag & drop an MRI brain scan image
   - Supported formats: PNG, JPG, JPEG
   - Maximum file size: 16MB

2. **Analyze the Scan**
   - Click the "Analyze Scan" button
   - Wait for the AI to process the image (a few seconds)

3. **View Results**
   - Classification: Tumor Detected or No Tumor
   - Confidence percentage
   - Probability breakdown

4. **New Analysis**
   - Click "New Analysis" to upload another image

## Important Notes

### Demo Mode
The current implementation uses a **placeholder model** for demonstration purposes. To use it with real predictions:

1. Train a model on your MRI dataset
2. Save the trained model as `models/brain_tumor_model.h5`
3. Restart the Flask server

### Training a Model (Optional)

To train your own model:

1. Prepare your MRI dataset with labeled images
2. Organize into training/validation/test folders
3. Use the model architecture from `model.py`
4. Train and save the model:

```python
from model import create_model
import tensorflow as tf

# Create model
model = create_model()

# Train on your dataset (pseudocode)
# model.fit(train_data, epochs=50, validation_data=val_data)

# Save the trained model
os.makedirs('models', exist_ok=True)
model.save('models/brain_tumor_model.h5')
```

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, you can change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8000)  # Change 5000 to 8000
```

### CORS Errors
The application uses Flask-CORS to handle cross-origin requests. If you encounter CORS issues, make sure Flask-CORS is installed.

### TensorFlow Installation Issues
If TensorFlow installation fails, try:
```bash
pip install tensorflow --upgrade
```

## Project Structure

```
MRI-Brain-Tumor-Detection/
├── app.py                  # Flask backend server
├── model.py                # CNN model architecture
├── requirements.txt        # Python dependencies
├── index.html             # Frontend UI
├── styles.css             # Premium styling
├── script.js              # Frontend logic
├── uploads/               # Temporary upload storage (auto-created)
└── models/                # Model storage (create manually)
    └── brain_tumor_model.h5  # Trained model (optional)
```

## API Endpoints

- `GET /` - Serve the web interface
- `POST /api/predict` - Process image and return prediction
- `GET /api/health` - Check server health

## Credits

Built with:
- **Backend**: Flask, TensorFlow, Keras
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Design**: Medical-grade UI with glassmorphism and smooth animations
