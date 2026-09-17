# Diabetes Prediction using Machine Learning

A machine learning web application that predicts whether a person is likely to have diabetes based on medical input features.

## Project Overview

This project uses a Logistic Regression machine learning model to predict diabetes based on the following 8 features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The trained model is integrated with a Flask web application. Users can enter the required values through a web form and receive a prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS
- Gunicorn

## Machine Learning

### Algorithm

Logistic Regression

### Preprocessing

- Invalid zero values in selected medical features were treated as missing values.
- Missing values were handled using median imputation.
- Features were standardized using `StandardScaler`.

### Prediction

The model produces two classes:

- `0` → No Diabetes
- `1` → Diabetes

> This application is a machine learning demonstration and should not be considered a medical diagnosis.

## Project Structure

```text
diabetes-flask-predictor/
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── diabetes_model.joblib
├── diabetes_scaler.joblib
├── requirements.txt
├── .gitignore
└── README.md
