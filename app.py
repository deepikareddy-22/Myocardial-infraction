from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from the form
        input_data = {key: float(request.form[key]) for key in request.form if key != "ID"}
        features_array = np.array(list(input_data.values())).reshape(1, -1)
        
        # Predict the outcome
        prediction = model.predict(features_array)[0]  # Assuming binary classification (0 or 1)

        # Define colors based on risk level
        if prediction == 0:
            result_text = "Low Risk of Myocardial Infarction"
            result_color = "#28a745"  # Green for low risk
            bg_color = "#e9ffe9"  # Light green background
        else:
            result_text = "High Risk of Myocardial Infarction"
            result_color = "#dc3545"  # Red for high risk
            bg_color = "#ffe9e9"  # Light red background

        return render_template(
            'result.html',
            prediction=result_text,
            input_data=input_data,
            result_color=result_color,
            bg_color=bg_color
        )

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
