import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

model = joblib.load('penguin_model.pkl')

@app.route('/')
def home():
    return "Penguin Classifier API. Available endpoints: /predict (POST)"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = [
            data['island'],
            data['culmen_length_mm'],
            data['culmen_depth_mm'],
            data['flipper_length_mm'],
            data['body_mass_g'],
            data['sex']
        ]

        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
    
        return jsonify({
            'species': prediction[0]
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)