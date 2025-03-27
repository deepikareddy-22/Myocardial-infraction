from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from form
        features = [float(request.form.get(feature)) for feature in request.form]
        
        # Make prediction
        prediction = model.predict([np.array(features)])[0]
        
        # Map numeric prediction to meaningful labels
        result_text = "High Risk of Myocardial Infarction" if prediction == 1 else "Low Risk of Myocardial Infarction"
        
        return render_template("result.html", prediction=result_text)
    
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

            
           
        

            
