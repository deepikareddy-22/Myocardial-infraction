from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model (ensure 'model.pkl' exists)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        features = [float(request.form[key]) for key in request.form if key != "ID"]
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]  # Assuming binary classification (0 or 1)
        
        # Interpret results
        if prediction == 0:
            result_text = "Low Risk of Myocardial Infarction"
        else:
            result_text = "High Risk of Myocardial Infarction"

        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
