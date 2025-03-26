from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define top 10 important features based on feature importance analysis
top_features = ["ID", "D_AD_ORIT", "S_AD_ORIT", "K_SH_POST", "L_BLOOD", 
                "ANT_CA_S_n", "ZSN", "AGE", "TIME_B_S", "NITR_S"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', columns=top_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input data corresponding to top features
        data = [float(request.form.get(feature, 0)) for feature in top_features]
        features = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Pair column names with entered values
        input_data = dict(zip(top_features, data))
        
        return render_template("result.html", prediction=int(prediction), input_data=input_data)
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)

        
