from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
    
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]  # Expecting a list of feature values
        prediction = model.predict([np.array(data)])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
