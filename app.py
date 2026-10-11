
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
st.title("Wine Quality Prediction System")
st.write("Enter the wine's chemical properties to predict its quality.")

model = joblib.load("wine_quality_model.pkl")

features = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

defaults = [7.0, 0.30, 0.30, 2.0, 0.05, 15.0, 46.0,
            0.996, 3.3, 0.65, 10.0]

values = {}
for i, feature in enumerate(features):
    values[feature] = st.number_input(
        feature.title(), value=float(defaults[i]), format="%.4f"
    )

if st.button("Predict Wine Quality"):
    sample = pd.DataFrame([values], columns=features)
    prediction = model.predict(sample)[0]
    st.success(f"Predicted wine quality: {prediction}")
