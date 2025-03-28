import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained model
model_path = "model_top10.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Define the Flask app
app = Flask(__name__)

# Define feature names
top_features = ["ID", "D_AD_ORIT", "S_AD_ORIT", "K_SH_POST", "L_BLOOD", 
                "ANT_CA_S_n", "ZSN", "AGE", "TIME_B_S", "NITR_S"]

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html file for input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        data = [float(request.form[feature]) for feature in top_features]
        
        # Convert to numpy array and reshape for model input
        features_array = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][int(prediction)] * 100
        
        # Map prediction to risk level
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        
        return render_template('result.html', patient_id=data[0], risk_level=risk_level, probability=probability)
    except Exception as e:
        return render_template("result.html", error=f"Internal Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
