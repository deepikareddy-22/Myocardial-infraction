from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model_top10.pkl", "rb") as f:
    model_data = pickle.load(f)

xgb_model = model_data["model"]  
top_10_features = model_data["top_10_features"]  

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        full_name = request.form["Full_Name"]  # Get Full Name
        ID = int(request.form["ID"])
        D_AD_ORIT = float(request.form["D_AD_ORIT"])
        S_AD_ORIT = float(request.form["S_AD_ORIT"])
        K_SH_POST = float(request.form["K_SH_POST"])
        L_BLOOD = float(request.form["L_BLOOD"])
        ANT_CA_S_n = int(request.form["ANT_CA_S_n"])
        ZSN = float(request.form["ZSN"])
        AGE = int(request.form["AGE"])
        TIME_B_S = float(request.form["TIME_B_S"])
        NITR_S = int(request.form["NITR_S"])

        # Create input array
        input_data = np.array([[D_AD_ORIT, S_AD_ORIT, K_SH_POST, L_BLOOD, ANT_CA_S_n, ZSN, AGE, TIME_B_S, NITR_S]])
        
        # Make Prediction
        prediction = xgb_model.predict(input_data)[0]
        
        # Assign risk level
        if prediction == 1:
            risk_level = "High Risk of Myocardial Infarction"
            result_color = "#dc3545"  # Red
            bg_color = "#f8d7da"
        else:
            risk_level = "Low Risk of Myocardial Infarction"
            result_color = "#28a745"  # Green
            bg_color = "#e9ffe9"

        # Pass data to result.html
        return render_template(
            "result.html",
            prediction=risk_level,
            result_color=result_color,
            bg_color=bg_color,
            full_name=full_name,  # Pass Full Name
            input_data=request.form  # Pass all input data
        )

    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
