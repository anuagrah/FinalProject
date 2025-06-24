from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

app = Flask(__name__)

# Load your trained model
model = load_model('fruit_quality_model.h5')  # Ensure this file exists

# Class labels (adjust based on your dataset)
CLASS_LABELS = ['Good', 'Bad', 'Overripe']

@app.route('/')
def home():
    return render_template('index.html')  # HTML file should be in a folder named "templates"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read and preprocess image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Predict
        prediction = model.predict(image)[0]
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
