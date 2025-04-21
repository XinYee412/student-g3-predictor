import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_excel("modified.xls", engine='openpyxl')

# 2. Initial features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# 3. Train full model to get feature importance
model_full = RandomForestRegressor(n_estimators=100, random_state=42)
model_full.fit(X, y)

# 4. Get top N important features
importances = model_full.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_n = 5  # Choose top 5 important features
top_features = importance_df['Feature'].head(top_n).tolist()

print(f"\nTop {top_n} features used for training:")
print(top_features)

# 5. Use only top features for training
X_top = X[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

# 6. Train new model with selected features
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Prediction and Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
success_rate = np.mean(np.abs(y_test - y_pred) <= 2.0) * 100

print("\n🔍 Evaluation using top features:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
print(f"Success Rate (±2): {success_rate:.2f}%")
