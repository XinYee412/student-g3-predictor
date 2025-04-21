import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student G3 Predictor", layout="centered")

st.title("📚 Student G3 Grade Predictor")
st.markdown("Upload your dataset (e.g., `modified.xls`) to predict student final grades and see model evaluation.")

# File uploader
uploaded_file = st.file_uploader("Upload Excel File", type=["xls", "xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    except:
        df = pd.read_excel(uploaded_file, engine='xlrd')

    st.success("✅ File uploaded successfully!")
    st.write("Preview of dataset:", df.head())

    if "G3" not in df.columns:
        st.error("❌ The dataset must contain a 'G3' column.")
    else:
        X = df.drop(columns=["G3"])
        y = df["G3"]
        X = pd.get_dummies(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initial model training
        model = RandomForestRegressor(n_estimators=200, random_state=50)
        model.fit(X_train, y_train)

        # Feature importance
        importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        top_n = st.slider("Select number of top features to use", min_value=5, max_value=len(feature_names), value=10)
        top_features = importance_df['Feature'].head(top_n).tolist()

        st.subheader("🔍 Top Important Features")
        st.write(top_features)

        # Retrain model with top features
        X_top = X[top_features]
        X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        success_rate = np.mean(np.abs(y_test - y_pred) <= 2.0) * 100

        st.subheader("📊 Model Evaluation")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared Score (R²):** {r2:.4f}")
        st.write(f"**Success Rate (±2):** {success_rate:.2f}%")

        # Optional: feature importance plot
        st.subheader("📈 Feature Importance")
        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'][:top_n], importance_df['Importance'][:top_n])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()
        st.pyplot(fig)

else:
    st.info("📥 Please upload an Excel file to begin.")
