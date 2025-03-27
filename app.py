from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        features = [
            float(request.form["ID"]),
            float(request.form["D_AD_ORIT"]),
            float(request.form["S_AD_ORIT"]),
            float(request.form["K_SH_POST"]),
            float(request.form["L_BLOOD"]),
            float(request.form["ANT_CA_S_n"]),
            float(request.form["ZSN"]),
            float(request.form["AGE"]),
            float(request.form["TIME_B_S"]),
            float(request.form["NITR_S"])
        ]

        # Convert to NumPy array and reshape
        input_data = np.array([features]).reshape(1, -1)

        # Get model prediction (0 or 1)
        prediction = model.predict(input_data)[0]

        # Assign risk category and colors
        if prediction == 0:
            result_text = "Low Risk of Myocardial Infarction"
            result_color = "#28a745"  # Green
            bg_color = "#e9ffe9"      # Light Green
        else:
            result_text = "High Risk of Myocardial Infarction"
            result_color = "#dc3545"  # Red
            bg_color = "#ffe9e9"      # Light Red

        # Prepare data for rendering
        input_dict = {
            "ID": features[0],
            "D_AD_ORIT": features[1],
            "S_AD_ORIT": features[2],
            "K_SH_POST": features[3],
            "L_BLOOD": features[4],
            "ANT_CA_S_n": features[5],
            "ZSN": features[6],
            "AGE": features[7],
            "TIME_B_S": features[8],
            "NITR_S": features[9]
        }

        return render_template(
            "result.html",
            prediction=result_text,
            result_color=result_color,
            bg_color=bg_color,
            input_data=input_dict
        )

    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)

            
