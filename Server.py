from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from waitress import serve
app = Flask(__name__)
model = joblib.load('model.pkl')  # Simplified loading, ensure 'model.pkl' is in your working directory

@app.route('/')
def home():
    return render_template('index.html')
app.route('/api', methods=['POST','GET'])
def predict():
    data = request.get_json(force=True)
    # Extract individual features based on the incoming JSON keys
    features = [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]
    prediction = model.predict([features])
    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    #serve(app, host="0.0.0.0", port=8000)
    app.run(port=5000, debug=True)

    
