from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("diabetes_scaler.joblib")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        # Get input values from the form
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabetes_pedigree = float(request.form["diabetes_pedigree"])
        age = float(request.form["age"])

        # Arrange inputs in the same order used during training
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        # Apply the trained scaler
        input_scaled = scaler.transform(input_data)

        # Predict
        result = model.predict(input_scaled)[0]

        # Convert 0/1 into readable output
        if result == 1:
            prediction = "Diabetes"
        else:
            prediction = "No Diabetes"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)