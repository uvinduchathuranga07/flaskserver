from flask import Flask, request, jsonify
from shapely.geometry import Polygon
from pyproj import Transformer
import requests
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# API Key for Google Maps Static API
API_KEY = 'AIzaSyA5v7a8JQuQKpyrZRa70GFfl35FztYRF9A'

# Load the pre-trained U-Net model for rubber land segmentation
model = load_model('unet_rubber_land_segmentation.h5')

# Function to calculate land area based on coordinates
def calculate_land_area(latitude_longitude_coordinates):
    transformer = Transformer.from_crs('epsg:4326', 'epsg:3857')
    try:
        x_coords, y_coords = transformer.transform(*zip(*latitude_longitude_coordinates))
        polygon = Polygon(zip(x_coords, y_coords))
        area_meters = polygon.area
        area_perches = area_meters / 25.2929
        area_acres = area_meters * 0.000247105
        return area_meters, area_perches, area_acres
    except Exception as e:
        print("Error occurred during area calculation:", e)
        return None

# Function to generate and save a satellite image based on the provided coordinates
def generate_satellite_image(latitude_longitude_coordinates, output_file='satellite_image.png'):
    try:
        center_latitude = sum(coord[0] for coord in latitude_longitude_coordinates) / len(latitude_longitude_coordinates)
        center_longitude = sum(coord[1] for coord in latitude_longitude_coordinates) / len(latitude_longitude_coordinates)
        params = {
            'center': f'{center_latitude},{center_longitude}',
            'zoom': '17',
            'size': '640x640',
            'maptype': 'satellite',
            'key': API_KEY
        }
        url = 'https://maps.googleapis.com/maps/api/staticmap'
        response = requests.get(url, params=params)
        if response.status_code == 200:
            with open(output_file, 'wb') as file:
                file.write(response.content)
            return output_file
        else:
            print("Error fetching the satellite image:", response.status_code, response.text)
            return None
    except Exception as e:
        print("An error occurred while generating the satellite image:", e)
        return None

# Function to preprocess image for model prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return None
    image_resized = cv2.resize(image, (256, 256))
    image_resized = image_resized / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    return image_resized

# Function to predict rubber land area from the image
def predict_rubber_land_area(image_path):
    input_image = preprocess_image(image_path)
    if input_image is None:
        return None
    predicted_mask = model.predict(input_image)
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)
    total_pixels = predicted_mask_binary[0].size
    white_pixels = np.sum(predicted_mask_binary[0] == 1)
    percentage_white_space = (white_pixels / total_pixels) * 100
    return percentage_white_space

# Flask Route to calculate land area, generate satellite image, and predict rubber land area
@app.route('/land_analysis', methods=['POST'])
def land_analysis():
    data = request.json
    land_coordinates = data.get('land_coordinates', [])
    
    if not land_coordinates:
        return jsonify({"error": "No coordinates provided"}), 400
    
    # Step 1: Calculate land area
    area_meters, area_perches, area_acres = calculate_land_area(land_coordinates)
    if area_meters is None:
        return jsonify({"error": "Unable to calculate land area"}), 400
    
    # Step 2: Generate satellite image
    image_path = generate_satellite_image(land_coordinates)
    if not image_path:
        return jsonify({"error": "Unable to generate satellite image"}), 400
    
    # Step 3: Predict rubber land area
    percentage_rubber_land = predict_rubber_land_area(image_path)
    if percentage_rubber_land is None:
        return jsonify({"error": "Prediction failed"}), 400
    
    # Step 4: Calculate number of trees and latex yield
    trees_per_acre = 150  # Typical density of rubber trees per acre
    latex_per_tree = 3.1  # Latex yield per tree in liters per year

    # Number of trees and total latex yield
    if area_acres:
        num_trees = int(trees_per_acre * area_acres)
        total_latex_yield = num_trees * latex_per_tree
    else:
        num_trees = None
        total_latex_yield = None

    # Step 5: Prepare the result
    result = {
        'square_meters': round(area_meters, 2) if area_meters else None,
        'perches': round(area_perches, 2) if area_perches else None,
        'acres': round(area_acres, 2) if area_acres else None,
        'percentage_rubber_land_area': round(percentage_rubber_land, 2),
        'num_trees': num_trees,
        'latex_yield_per_year': round(total_latex_yield, 2) if total_latex_yield else None,
        'satellite_image_path': image_path
    }

    # Return the results as JSON
    return jsonify(result)

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
