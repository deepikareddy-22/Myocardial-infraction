from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained XGBoost model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define column names
column_names = ["AGE", "SEX", "INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "IBS_NASL", "GB", "SIM_GIPERT", "DLIT_AG"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', columns=column_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        data = [float(x) for x in request.form.values()]
        features = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Pair column names with entered values
        input_data = dict(zip(column_names, data))
        
        return render_template("result.html", prediction=int(prediction), input_data=input_data)
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
