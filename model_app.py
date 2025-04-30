import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student G3 Predictor", layout="centered")
st.title("🎓 Student Final Grade (G3) Predictor")

# 1. Load dataset
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

        # Train model on full features
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)

        # Get top 10 important features
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        top_features = importance_df["Feature"].head(10).tolist()
        st.success("Top 10 important features selected for prediction:")
        #st.write(top_features)

        # User input for those features
        st.subheader("📝 Enter values for the following features")

        user_input = {}
        for feature in top_features:
            if feature not in df.columns or df[feature].dtype == 'object':
                user_input[feature] = st.selectbox(f"{feature}:", [0, 1])
            else:
                user_input[feature] = st.number_input(f"{feature}:", value=0.0)

        # Predict
        if st.button("🎯 Predict G3"):
            input_df = pd.DataFrame([user_input])

            # Add missing columns
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X.columns]  # Ensure column order matches

            prediction = model.predict(input_df)[0]

            st.subheader("📊 Prediction Result")
            st.write(f"✅ Predicted G3 grade: **{round(prediction, 2)}**")

except FileNotFoundError:
    st.error("⚠️ File 'modified.xls' not found.")
except Exception as e:
    st.error(f"🚫 An error occurred: {e}")
