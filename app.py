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

        # Convert prediction result
        if prediction == 0:
            result_text = "Low Risk of Myocardial Infarction"
        else:
            result_text = "High Risk of Myocardial Infarction"

        return render_template(
            'result.html',
            prediction=result_text,
            input_data=input_data
        )

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

           
