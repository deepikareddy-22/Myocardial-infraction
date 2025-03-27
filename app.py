from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Ensure this matches the feature count used during training
EXPECTED_FEATURE_COUNT = 122

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        input_data = [float(request.form[key]) for key in request.form.keys()]
        
        # Ensure the correct number of features are passed
        if len(input_data) != EXPECTED_FEATURE_COUNT:
            return render_template('result.html', prediction_text=f"Error: Feature shape mismatch, expected: {EXPECTED_FEATURE_COUNT}, got {len(input_data)}", error=True)

        # Convert list to numpy array and reshape for model input
        final_features = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)[0]

        # Interpret prediction result
        result_text = "High Risk of Myocardial Infarction" if prediction == 1 else "Low Risk of Myocardial Infarction"

        return render_template('result.html', prediction_text=result_text, error=False)

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}", error=True)

if __name__ == "__main__":
    app.run(debug=True)
