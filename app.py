import numpy as np
import pandas as pd
import joblib
from jinja2 import Environment, Template
from flask import Flask, render_template, request


skin_thickness_df = pd.read_parquet("skinthickness.parquet")
scaler = joblib.load("scaler.pkl")
model = joblib.load("Diabetes-Prediction.pkl")

def bmi_calc(weight, height):
    return weight / (height ** 2)

def diabetes_pedigree(first_degree_diabetes, second_degree_diabetes, age):
    base = 0.1
    if first_degree_diabetes == 'yes' and second_degree_diabetes == 'yes':
        base = 0.7 + 0.4
    elif first_degree_diabetes == 'yes':
        base = 0.7
    elif second_degree_diabetes == 'yes':
        base = 0.4
    
    if age is not None:
        if age < 40:
            base *= 1.2
        elif age < 60:
            base *= 1.1
    return min(base, 1)

# Flask App
app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            age = request.form.get('age')
            pregnancies = request.form.get('pregnancies')
            glucose = request.form.get('glucose_level')
            bp = request.form.get('bp_level')
            insulin = request.form.get('insulin_level')
            weight = request.form.get('weight')
            height = request.form.get('height')
            f_d_y = request.form.get('first_degree_diabetes')
            s_d_y = request.form.get('second_degree_diabetes')

            if not all([age, pregnancies, glucose, bp, insulin, weight, height, f_d_y, s_d_y]):
                error_message = "Please fill all fields!"
                return render_template("index.html", prediction=error_message)

            age = int(age)
            pregnancies = int(pregnancies)
            glucose = float(glucose)
            bp = float(bp)
            insulin = float(insulin)
            weight = float(weight)
            height = float(height)

            bmi = bmi_calc(weight, height)

            dpf = diabetes_pedigree(f_d_y, s_d_y, age)

            skin_thickness = float(np.mean(skin_thickness_df))
            X_cols = {
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [bp],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [dpf],
                'Age': [age]
            }
            X = pd.DataFrame(X_cols)

            scaled_X = scaler.transform(X)
            prediction_bin= model.predict(scaled_X)[0]
            if prediction_bin == 1:
                prediction = "Diabetic"
            elif prediction_bin == 0:
                prediction = "Non Diabetic"
            else:
                prediction = None
        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render_template("index.html", prediction=error_message)

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
