from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained model
MODEL_PATH = 'food_calorie_model.h5'
model = load_model(MODEL_PATH)

# Define the food categories (must match the model's classes)
class_labels = [
    'apple_pie', 'pizza', 'sushi', 'fried_rice', 'burger', # Add all your class labels here
    # ...
]

# You can define a simple dictionary with calorie information for each class
calories_info = {
    'apple_pie': 300,
    'pizza': 266,
    'sushi': 200,
    'fried_rice': 300,
    'burger': 295,
    # Add all calorie information here
}

# Preprocess the image before passing it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize the image to fit the model input size
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Add a batch dimension
    img_tensor /= 255.  # Scale pixel values to [0, 1]
    return img_tensor

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the image part
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    
    # Save the image to a temporary location
    img_path = os.path.join('uploads', img_file.filename)
    img_file.save(img_path)

    # Preprocess the image and make predictions
    img_tensor = preprocess_image(img_path)
    predictions = model.predict(img_tensor)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_food = class_labels[predicted_class_idx]
    
    # Get calorie information based on the predicted food
    calories = calories_info.get(predicted_food, 'Unknown')

    # Clean up: Remove the uploaded file after processing
    os.remove(img_path)

    # Return the prediction and calorie information as JSON
    return jsonify({
        'food': predicted_food,
        'calories': calories
    })

if __name__ == '__main__':
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(host='0.0.0.0', port=5000)
