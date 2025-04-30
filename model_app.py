import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student G3 Predictor", layout="centered")
st.title("🎓 Student Final Grade (G3) Predictor")

# Load dataset
file_path = "modified.xls"
try:
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        df = pd.read_excel(file_path, engine='xlrd')

    if "G3" not in df.columns:
        st.error("❌ 'G3' column not found in the dataset.")
    else:
        # Preprocessing
        X = df.drop(columns=["G3"])
        y = df["G3"]
        X = pd.get_dummies(X)

        # Train model on all data
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)

        # Choose 10 most common features (you can manually adjust this)
        # For simplicity, here we hardcode 10 most relevant or easy-to-enter features
        top_features = [
            'age', 'Medu', 'Fedu', 'studytime', 'failures',
            'absences', 'goout', 'health', 'Walc', 'Dalc'
        ]

        st.subheader("📝 Enter your information:")

        user_input = {}
        for feature in top_features:
            user_input[feature] = st.number_input(f"{feature}:", min_value=0, value=0)

        # Prepare input
        if st.button("🎯 Predict G3"):
            input_df = pd.DataFrame([user_input])

            # Fill in missing columns
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X.columns]  # match order

            prediction = model.predict(input_df)[0]

            st.subheader("📊 Prediction Result")
            st.success(f"🎯 Predicted G3 grade: **{round(prediction, 2)}**")

except FileNotFoundError:
    st.error("⚠️ File 'modified.xls' not found.")
except Exception as e:
    st.error(f"🚫 An error occurred: {e}")
