import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request , jsonify
from util import processing_inputs


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
            glucose_level = request.form.get('glucose_level')
            bp_level = request.form.get('bp_level')
            insulin_level = request.form.get('insulin_level')
            weight = request.form.get('weight')
            height = request.form.get('height')
            first_degree_diabetes = request.form.get('first_degree_diabetes')
            second_degree_diabetes = request.form.get('second_degree_diabetes')

            prediction_prob,error_message = processing_inputs(age, pregnancies, glucose_level, bp_level, insulin_level, weight, height, first_degree_diabetes, second_degree_diabetes)            
            if error_message:
                prediction=None
                error_message=error_message
            else:
                if prediction_prob >=0.6:
                    prediction = "Diabetic"
                else:
                    prediction = "Non Diabetic"
        except Exception as e:
            error_message = "Something Went Wrong!"
            return render_template("index.html", prediction=None,error_message=error_message,age=age,pregnancies=pregnancies,glucose_level=glucose_level,bp_level=bp_level,insulin_level=insulin_level,weight=weight,height=height,first_degree_diabetes=first_degree_diabetes,second_degree_diabetes=second_degree_diabetes)

    return render_template("index.html",prediction=prediction,error_message=error_message,age=age,pregnancies=pregnancies,glucose_level=glucose_level,bp_level=bp_level,insulin_level=insulin_level,weight=weight,height=height,first_degree_diabetes=first_degree_diabetes,second_degree_diabetes=second_degree_diabetes)

@app.route("/api/predict",methods=['POST'])
def api_predict():
    prediction = None
    if request.method == 'POST':
        try:
             data=request.get_json(force=True)
             age = data.get('age')
             pregnancies = data.get('pregnancies')
             glucose = data.get('glucose_level')
             bp = data.get('bp_level')
             insulin = data.get('insulin_level')
             weight = data.get('weight')
             height = data.get('height')
             f_d_y = data.get('first_degree_diabetes')
             s_d_y = data.get('second_degree_diabetes')
             prediction,error_message=processing_inputs(age,pregnancies,glucose,bp,insulin,weight,height,f_d_y,s_d_y)
             if error_message:
                 return jsonify({'error':error_message})
        except Exception as e:
            error_message=f'Error: {str(e)}'
            return jsonify({'error':error_message})
    return jsonify({'prediction':prediction})

if __name__ == '__main__':
    app.run(debug=True)
