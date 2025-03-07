from flask import Flask, render_template, request, redirect, url_for
import os
import requests
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

hair_model = load_model('hair_model.h5')
API_KEY_SKIN = 'YOUR_SKIN_API_KEY'
API_KEY_WOUND = 'YOUR_WOUND_API_KEY'
API_URL_SKIN = 'https://your-skin-api-endpoint/predict'
API_URL_WOUND = 'https://your-wound-api-endpoint/predict'
GRAD_CAM_URL_SKIN = 'https://your-skin-api-endpoint/gradcam'
GRAD_CAM_URL_WOUND = 'https://your-wound-api-endpoint/gradcam'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hair')
def hair_page():
    return render_template('hair.html')

@app.route('/skin')
def skin_page():
    return render_template('skin.html')

@app.route('/wound')
def wound_page():
    return render_template('wound.html')

@app.route('/predict_hair', methods=['POST'])
def predict_hair():
    file = request.files['file']
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = load_and_preprocess_image(filepath)
        prediction = hair_model.predict(img)
        predicted_class_index = prediction.argmax()
        confidence = prediction[0][predicted_class_index]
        class_labels = ['class1', 'class2', 'class3'] # Replace with your class labels
        predicted_class = class_labels[predicted_class_index]

        return render_template('hair.html', prediction=predicted_class, confidence=confidence, image_path='uploads/' + filename)

    return redirect(url_for('hair_page'))

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    return predict_api('skin', API_URL_SKIN, API_KEY_SKIN, GRAD_CAM_URL_SKIN)

@app.route('/predict_wound', methods=['POST'])
def predict_wound():
    return predict_api('wound', API_URL_WOUND, API_KEY_WOUND, GRAD_CAM_URL_WOUND)

def predict_api(model_name, api_url, api_key, gradcam_url):
    file = request.files['file']
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        with open(filepath, 'rb') as image_file:
            response = requests.post(api_url, files={'file': image_file}, headers={'Authorization': f'Bearer {api_key}'})
            gradcam_response = requests.post(gradcam_url, files={'file': image_file}, headers={'Authorization': f'Bearer {api_key}'})

            if response.status_code == 200 and gradcam_response.status_code == 200:
                result = response.json()
                grad_cam_url = gradcam_response.json()['grad_cam_url']
                return render_template(f'{model_name}.html', prediction=result['class'], confidence=result['confidence'], image_path='uploads/' + filename, grad_cam=grad_cam_url)

    return redirect(url_for(f'{model_name}_page'))

def load_and_preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))  # Adjust size as needed for your model
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == '__main__':
    app.run(debug=True, port=8000)