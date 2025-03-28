from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model correctly
with open("model_top10.pkl", "rb") as f:
    model = pickle.load(f)

# Check if the model is correctly loaded
if not hasattr(model, "predict"):
    raise ValueError("The loaded model is not a valid machine learning model.")

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
        # Extract input data
        data = [float(request.form.get(feature, 0)) for feature in top_features]
        features = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]  # 0 = Low Risk, 1 = High Risk

        # Get probability (if model supports it)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0][1] * 100  # Get probability for class 1 (high risk)
        else:
            probability = None  # If model doesn't support probability prediction

        # Risk Level Mapping
        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        return render_template(
            "result.html",
            patient_id=int(data[0]),
            risk_level=risk_level,
            probability=probability if probability else "N/A"
        )

    except Exception as e:
        return render_template("result.html", error=f"Internal Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
