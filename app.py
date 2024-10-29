from flask import Flask, request, jsonify
from shapely.geometry import Polygon
from pyproj import Transformer
import requests
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

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

# Function to convert lat/long to pixel coordinates using the Mercator projection for Google Maps
def lat_lon_to_pixel(lat, lon, zoom, tile_size=256):
    """Convert latitude and longitude to pixel coordinates."""
    siny = np.sin(np.radians(lat))
    siny = np.clip(siny, -0.9999, 0.9999)  # Avoid infinity values
    x = tile_size * (0.5 + lon / 360.0)
    y = tile_size * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    scale = 2 ** zoom
    return int(x * scale), int(y * scale)

# Function to crop the image to the bounding box of the provided coordinates
def crop_image_to_coordinates(image_path, coordinates, center_latitude, center_longitude, zoom=17, image_size=(640, 640)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load the image {image_path} for cropping.")
        return None

    # Image dimensions
    img_height, img_width = image.shape[:2]
    
    # Convert the center latitude/longitude to pixel coordinates (center of the image)
    center_x, center_y = lat_lon_to_pixel(center_latitude, center_longitude, zoom, tile_size=image_size[0])

    # Calculate the bounding box around the center pixel based on the geographic coordinates
    pixel_coords = [lat_lon_to_pixel(lat, lon, zoom, tile_size=image_size[0]) for lat, lon in coordinates]

    # Calculate min/max X and Y based on pixel coordinates of the bounding box
    min_x = min([p[0] for p in pixel_coords])
    min_y = min([p[1] for p in pixel_coords])
    max_x = max([p[0] for p in pixel_coords])
    max_y = max([p[1] for p in pixel_coords])

    # Translate bounding box relative to the center of the image (640x640) centered around center_x, center_y
    crop_min_x = max(0, min_x - center_x + img_width // 2)
    crop_min_y = max(0, min_y - center_y + img_height // 2)
    crop_max_x = min(img_width, max_x - center_x + img_width // 2)
    crop_max_y = min(img_height, max_y - center_y + img_height // 2)

    # Ensure the cropping coordinates are within the image dimensions
    if crop_min_x >= img_width or crop_min_y >= img_height or crop_max_x <= 0 or crop_max_y <= 0:
        print(f"Error: Cropping bounds are outside image dimensions. Image size: {image.shape}")
        return None

    # Crop the image using the calculated bounding box
    cropped_image = image[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    # Save the cropped image
    cropped_output_file = "cropped_" + image_path
    cv2.imwrite(cropped_output_file, cropped_image)
    print(f"Cropped image saved successfully to {cropped_output_file}")
    return cropped_output_file

# Function to generate and save a satellite image based on the provided coordinates and crop it
def generate_and_crop_satellite_image(latitude_longitude_coordinates, output_file='satellite_image.png'):
    try:
        center_latitude = sum(coord[0] for coord in latitude_longitude_coordinates) / len(latitude_longitude_coordinates)
        center_longitude = sum(coord[1] for coord in latitude_longitude_coordinates) / len(latitude_longitude_coordinates)

        # Generate the satellite image using Google Static Maps API
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
            print(f"Satellite image saved successfully to {output_file}")

            return crop_image_to_coordinates(output_file, latitude_longitude_coordinates, center_latitude, center_longitude)
        else:
            print(f"Error fetching the satellite image: {response.status_code}, {response.text}")
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

# Function to calculate percentage of target color (e.g. #1d5838) in the image
def calculate_color_percentage(image_path, target_color, tolerance=30):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the target color with tolerance
    lower_bound = np.array([max(0, c - tolerance) for c in target_color], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color], dtype=np.uint8)

    # Create a mask for pixels that fall within the target color range
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Calculate the number of matching pixels
    matching_pixels = np.sum(mask > 0)
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]

    # Calculate the percentage of matching pixels
    percentage_matching = (matching_pixels / total_pixels) * 100
    return round(percentage_matching, 2)

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
    
    # Step 2: Generate and crop satellite image
    image_path = generate_and_crop_satellite_image(land_coordinates)
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

    # Step 5: Prepare the result with color percentage
    percentage_target_color = calculate_color_percentage(image_path, (29, 88, 56))  # #1d5838 in RGB

    result = {
        'square_meters': round(area_meters, 2) if area_meters else None,
        'perches': round(area_perches, 2) if area_perches else None,
        'acres': round(area_acres, 2) if area_acres else None,
        'percentage_rubber_land_area': round(percentage_rubber_land, 2),
        'percentage_target_color': round(percentage_target_color, 2),
        'num_trees': num_trees,
        'latex_yield_per_year': round(total_latex_yield, 2) if total_latex_yield else None,
        'satellite_image_path': image_path
    }

    # Return the results as JSON
    return jsonify(result)

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
