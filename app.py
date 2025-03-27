from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        features = [float(request.form[feature]) for feature in request.form]
        
        # Convert features to numpy array and reshape
        final_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)[0]
        
        # Map prediction to meaningful output
        result = "High Risk of Myocardial Infarction" if prediction == 1 else "Low Risk of Myocardial Infarction"
        
        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

        
            
