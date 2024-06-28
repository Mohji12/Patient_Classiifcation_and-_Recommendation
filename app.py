import pandas as pd
import numpy as np
import streamlit as st
import joblib

rf_model = joblib.load("random.pkl")

# User Input 

st.title("Patient Classification")
st_Haematocrit = st.number_input("Enter HAEMATOCRIT Values: ")
st_Haemologbin = st.number_input("Enter HAEMOGLOBINS Values: ")
st_Erythrocyte = st.number_input("Enter ERYTHROCYTE Values: ")
st_leucocyte = st.number_input("Enter LEUCOCYTE Values: ")
st_thrombocyte = st.number_input("Enter THROMBOCYTE Values: ")
st_MCH = st.number_input("Enter MCH Values: ")
st_MCHC = st.number_input("Enter MCHC Values: ")
st_MCV = st.number_input("Enter MCV Values: ")
st_AGE = st.number_input("Enter AGE Values: ")
st_Sex = st.number_input("Enter SEX (Enter 1 for Male and 0 for Female) values: ")

user_data = [
    [
        st_Haematocrit,
        st_Haemologbin,
        st_Erythrocyte,
        st_leucocyte,
        st_thrombocyte,
        st_MCH,
        st_MCHC,
        st_MCV,
        st_AGE,
        st_Sex,
    ]
]
cols = [
    [
        "HAEMATOCRIT",
        "HAEMOGLOBINS",
        "ERYTHROCYTE",
        "LEUCOCYTE",
        "THROMBOCYTE",
        "MCH",
        "MCHC",
        "MCV",
        "AGE",
        "SEX",
    ]
]

pd_test_df = pd.DataFrame(user_data,columns=cols)
st.write(pd_test_df)

prediction  = rf_model.predict(pd_test_df)

if prediction == 0:
    care = "Out Care (Home Care) Required"
else:
    care = "In Care (Hospitalization) Required"

st.subheader("Action to be taken")
st.write(care)
