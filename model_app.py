import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student G3 Predictor", layout="centered")
st.title("📚 Student G3 Grade Predictor")

file_path = "modified.xls"

try:
    # 自动尝试不同 engine
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        df = pd.read_excel(file_path, engine='xlrd')

    st.success("✅ File 'modified.xls' loaded successfully!")
    st.write("📄 Data preview:", df.head())

    if "G3" not in df.columns:
        st.error("❌ The dataset must contain a 'G3' column.")
    else:
        X = df.drop(columns=["G3"])
        y = df["G3"]
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Full model
        model_full = RandomForestRegressor(n_estimators=200, random_state=50)
        model_full.fit(X_train, y_train)

        importances = model_full.feature_importances_
        feature_names = X.columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        top_n = st.slider("Select number of top features to use", min_value=5, max_value=len(feature_names), value=10)
        top_features = importance_df['Feature'].head(top_n).tolist()

        st.subheader("🔍 Top Important Features")
        st.write(top_features)

        # Train model with top features
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
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Success Rate (±2):** {success_rate:.2f}%")

        # Plot feature importance
        st.subheader("📈 Feature Importance")
        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'][:top_n], importance_df['Importance'][:top_n])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()
        st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ The file 'modified.xls' was not found in the current directory.")
except Exception as e:
    st.error(f"🚫 Unexpected error occurred: {str(e)}")
