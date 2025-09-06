import numpy as np
import pandas as pd
import joblib
import numpy.core._multiarray_umath
import numpy.core._dtype_ctypes
import joblib


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
    return base

skin_thickness_df = pd.read_parquet("skin_thickness.parquet")
scaler = joblib.load('scaler.pkl')
transformer=joblib.load('transformer.pkl')
model = joblib.load("Diabetes-Prediction.pkl")

def processing_inputs(age,pregnancies,glucose,bp,insulin,weight,height,f_d_y,s_d_y):
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
    error_message=None
    x=[age,glucose,bp,insulin,bmi]
    for value in x:
        if value<=0:
            error_message="Value(s) Can't be Zero or Negative"
            return None,error_message           
    if pregnancies<0:
        error_message="No. of Pregnancies Can't be Negative"
        return None,error_message
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
    trans_cols=['Pregnancies','SkinThickness','Insulin','DiabetesPedigreeFunction','Age']
    X[trans_cols]=transformer.transform(X[trans_cols])
    scaled_X = scaler.transform(X)
    prediction= model.predict_proba(scaled_X)[0][1]
    return prediction,error_message