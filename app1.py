from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.pkl")

# This is the route for the homepage (loads your HTML form)
@app.route('/')
def home():
    return render_template('index.html')  # Make sure templates/index.html exists

# This is the route that handles the prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data.get('year')
    prediction = model.predict(np.array([[year]]))[0]
    return jsonify({'predicted_temperature': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
