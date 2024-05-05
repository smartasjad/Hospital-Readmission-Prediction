from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the LabelEncoder object
ct = joblib.load("churnct")
sc = pickle.load(open("xgb_model.pkl", "rb"))
# Load the XGBoost model
model = joblib.load("xgb_model.pkl")
#model = load_model("xgb_model.h5")

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/y_predict', methods=["POST"])
def prediction():
    Race = request.form["Race"]
    Gender = request.form["Gender"]
    Age_Group = request.form["Age Group"]
    Weight = request.form["Weight"]
    Admission_Type = request.form["Admission Type"]
    Discharge_Type = request.form["Discharge Type"]
    Time_Spend_In_Hospital = int(request.form["Time Spend Int Hospital"])
    Number_Of_Lab_Procedure = int(request.form["Number Of Lab Procedure"])
    Number_Of_Procedure = int(request.form["Number Of Procedure"])
    Number_Of_Medication = int(request.form["Number Of Medication"])
    Number_Of_Diagnosis1 = int(request.form["Number Of Diagnosis1"])
    Number_Of_Diagnosis2 = int(request.form["Number Of Diagnosis2"])
    Number_Of_Diagnosis3 = int(request.form["Number Of Diagnosis3"])
    Number_Of_Diagnosis =int( request.form["Number Of Diagnosis"])
    Glucose_Serum = request.form["Glucose Serum"]
    A1Cresult = request.form["A1Cresult"]
    Metformin = request.form["Metformin"]
    Repaglinide = request.form["Repaglinide"]
    Nateglinide = request.form["Nateglinide"]
    Chlorpropamide = request.form["Chlorpropamide"]
    Glimepiride = request.form["Glimepiride"]
    Glipizide = request.form["Glipizide"]
    Glyburide = request.form["Glyburide"]
    Tolbutamide = request.form["Tolbutamide"]
    Pioglitazone = request.form["Pioglitazone"]
    Rosiglitazone = request.form["Rosiglitazone"]
    Acarbose = request.form["Acarbose"]
    Miglitol = request.form["Miglitol"]
    troglitazone = request.form["troglitazone"]
    Tolazamide = request.form["Tolazamide"]
    Insulin = request.form["Insulin"]
    Glyburide_metformin = request.form["Glyburide_metformin"]
    Glipizide_metformin = request.form["Glipizide_metformin"]
    DiabetesMed = request.form["DiabetesMed"]

    # Collecting input data into a list
    x_test = [Race, Gender, Age_Group, Weight, Admission_Type, Discharge_Type,Time_Spend_In_Hospital
              ,Number_Of_Lab_Procedure,Number_Of_Procedure,Number_Of_Medication,Number_Of_Diagnosis1,
              Number_Of_Diagnosis2,Number_Of_Diagnosis3,Number_Of_Diagnosis, Glucose_Serum, A1Cresult, 
              Metformin, Repaglinide, Nateglinide, Chlorpropamide,Glimepiride, Glipizide, Glyburide, 
              Tolbutamide, Pioglitazone, Rosiglitazone, Acarbose, Miglitol,
              troglitazone, Tolazamide, Insulin, Glyburide_metformin, Glipizide_metformin,
              DiabetesMed]

    # Transforming categorical variables using LabelEncoder
    '''
    x_test_encoded = le.transform(x_test)

    # Scaling the input data
    x_test_scaled = sc.transform([x_test_encoded])

    # Making prediction
    prediction = model.predict(x_test)
    '''
    #transformed_values = [le.transform([value]).tolist() if isinstance(value, str) else le.transform([str(value)]) for value in x_test]

    transformed_data = ct.transform([x_test])

    prediction=model.predict(transformed_data)
    
    # Interpreting prediction
    if prediction == [[2]]:
        text = "he/she will not admit again"
    elif prediction == [[1]]:
        text = "he/she will admit after 30 days"
    else:
        text = "he/she will admit within 30 days"

    return render_template("index.html", prediction_text=text)

if __name__ == "__main__":
    app.run(debug=True)
