from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import requests
import json
import base64


app = Flask(__name__)

# Load Hair Disease Detection Model
disease_model = tf.keras.models.load_model('hair_model.h5')
class_names = ["Alopecia Areata","Contact Dermatitis","Folliculitis","Head Lice", "Lichen Planus", "Male Pattern Baldness","Psoriasis","Seborrheic Dermatitis","Telogen Effluvium","Tinea Capitis"]

# Gemini API Configuration

import os
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")


API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent"

print(app.static_folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hair', methods=['GET', 'POST'])
def hair():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = disease_model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            file_path = os.path.join('static', 'uploads', file.filename)
            file.save(file_path)
            return render_template('hair_result.html', result=predicted_class, confidence=confidence, image_file=file.filename)
    return render_template('hair.html')

@app.route('/hair_result')
def hair_result():
    return render_template('hair_result.html')

def get_gemini_response(image_bytes):
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + API_KEY
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }},
                {"text": "What does this image represent? Provide the output with the following categorized information only:- Other Names- Symptoms- Probable causes- Severity Levels (categorized as Mild, Moderate, or Severe)- Please consult a doctorEnsure the response is in plain text without any text decorations, highlights, or special formatting.Give each parameter in new line"}
            ]
        }]
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "Invalid response format from API."
    else:
        return f"API Error: {response.status_code}, {response.text}"

@app.route('/skin', methods=['GET', 'POST'])
def skin():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_bytes = file.read()
            result = get_gemini_response(img_bytes)
            print("API Response:", result)
            return render_template('skin.html', result=result)
    return render_template('skin.html')

@app.route('/wound', methods=['GET', 'POST'])
def wound():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_bytes = file.read()
            result = get_gemini_response(img_bytes)
            print("API Response:", result)
            return render_template('wound.html', result=result)
    return render_template('wound.html')

if __name__ == '__main__':
    app.run(debug=True)


