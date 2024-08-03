from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import io
from PIL import Image
from math import tan, radians, ceil

app = Flask(__name__)

# Path to the model
model_path = 'Models/finalv2_model.keras'

# Load trained model
model = load_model(model_path)

@app.route('/')
def home():
    return "API is up and running!"

@app.route('/status')
def status():
    return jsonify({'status': 'API is up and running!'})

@app.route('/predict', methods=['POST'])
def predict():
    # Read image file from the request
    file = request.files['image']
    
    # Convert the image file to a numpy array
    in_memory_file = io.BytesIO(file.read())
    image = Image.open(in_memory_file)
    image = np.array(image)
    
    # Preprocess the image
    resized_image = cv2.resize(image, (128, 128))  # Resize image to match model input shape
    input_image = np.expand_dims(resized_image, axis=0) / 255.0  # Normalize image and add batch dimension

    # Predict bounding box
    predicted_box = model.predict(input_image)[0]  # Assuming batch size of 1
    x_min, y_min, width, height = predicted_box * 128  # Convert normalized box coordinates back to image scale
    x_max = x_min + width
    y_max = y_min + height
    
    # Jarak kamera dalam sentimeter (cm) dan tinggi kamera dalam sentimeter (cm)
    jarak_kamera_cm = 200  # Sesuaikan dengan jarak yang Anda gunakan
    tinggi_kamera_cm = 100  # Sesuaikan dengan tinggi kamera yang Anda gunakan
    fov_vertical = 45  # Sudut pandang vertikal kamera dalam derajat
    
    # Define threshold for detection
    min_detection_width = 5
    min_detection_height = 5

    # Check if object is detected based on width and height
    if width <= min_detection_width or height <= min_detection_height:
        return jsonify({
            'height_cm': 0,
            'status': 'Object not detected or too small'
        })
    
    # Hitung tinggi objek berdasarkan bounding box dalam sentimeter
    image_height, image_width, _ = image.shape
    angle_of_view = fov_vertical / 2
    bbox_height_cm = 2 * jarak_kamera_cm * tan(radians(angle_of_view)) * (height / image_height)
    tinggi_objek_cm = bbox_height_cm

    # Bulatkan tinggi_objek_cm ke atas dengan dua angka desimal
    tinggi_objek_rounded = ceil(tinggi_objek_cm * 100) / 100
    
    # Return the result as JSON
    result = {
        'height_cm': tinggi_objek_rounded,
        'status': 'Prediction complete'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
