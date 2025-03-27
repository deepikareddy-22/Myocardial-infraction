from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model_top10.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        full_name = request.form["full_name"]  # Get full name from form

        # Extract all feature inputs (excluding full_name)
        features = [float(request.form[key]) for key in request.form if key != "full_name"]
        
        # Debugging: Print the number of features received
        print(f"Received {len(features)} features: {features}")

        # Check if we have the expected 20 features
        expected_features = 20
        if len(features) != expected_features:
            return f"Error: Expected {expected_features} features, but got {len(features)}."

        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert numeric prediction to text
        if prediction == 1:
            result_text = "High Risk of Myocardial Infarction"
            result_color = "#dc3545"  # Red
            bg_color = "#ffebeb"
        else:
            result_text = "Low Risk of Myocardial Infarction"
            result_color = "#28a745"  # Green
            bg_color = "#e9ffe9"

        return render_template(
            "result.html",
            full_name=full_name,
            prediction=result_text,
            result_color=result_color,
            bg_color=bg_color,
            input_data=request.form
        )

if __name__ == "__main__":
    app.run(debug=True)
