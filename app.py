from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open("model_top10.pkl", "rb") as f:
    model = pickle.load(f)

# Define top 10 important features
top_features = ["ID", "D_AD_ORIT", "S_AD_ORIT", "K_SH_POST", "L_BLOOD", 
                "ANT_CA_S_n", "ZSN", "AGE", "TIME_B_S", "NITR_S"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', columns=top_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input data from form
        data = []
        for feature in top_features:
            value = request.form.get(feature, 0)  # Get input value (default 0 if missing)
            try:
                data.append(float(value))  # Convert to float
            except ValueError:
                return render_template("result.html", error=f"Invalid input for {feature}: {value}")

        # Convert input data to NumPy array
        features = np.array(data).reshape(1, -1)  

        # Ensure the model exists
        if model is None:
            return render_template("result.html", error="Model not loaded properly.")

        # Predict risk level
        prediction = model.predict(features)[0]  # Get predicted class (0 or 1)

        # Get probability (if available)
        try:
            probability = model.predict_proba(features)[0][int(prediction)] * 100
        except AttributeError:
            probability = "N/A"

        # Define risk level mapping
        risk_mapping = {0: "Low Risk", 1: "High Risk"}
        risk_level = risk_mapping.get(prediction, "Unknown")

        return render_template("result.html", 
                               patient_id=int(data[0]),  
                               risk_level=risk_level,
                               probability=probability)

    except Exception as e:
        return render_template("result.html", error=f"Internal Error: {str(e)}")

       

if __name__ == "__main__":
    app.run(debug=True)
