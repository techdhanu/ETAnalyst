# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
#from sklearn.preprocessing import OneHotEncoder # No need for OneHotEncoder, pd.get_dummies is used

# Load dataset
df = pd.read_csv("travel_time_data1.csv")

# Check column names
print("Dataset Columns:", df.columns)

# Handle categorical variable - TrafficLevel
# First check the data type
print("TrafficLevel dtype:", df['TrafficLevel'].dtype)
print("TrafficLevel unique values:", df['TrafficLevel'].unique())

# **Calculate and add 'HourSin', 'HourCos' columns to df**
df['HourSin'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
df['HourCos'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)

# Convert categorical TrafficLevel to numeric using one-hot encoding
# Create a copy to avoid modifying the original dataframe
X = df[['HomeLat', 'HomeLon', 'OfficeLat', 'OfficeLon', 'DistanceKM',
       'HourOfDay', 'IsWeekday', 'HourSin', 'HourCos']].copy()  # Exclude TrafficLevel for now

# One-hot encode TrafficLevel
traffic_dummies = pd.get_dummies(df['TrafficLevel'], prefix='Traffic')
X = pd.concat([X, traffic_dummies], axis=1)  # Add the encoded columns to X

y = df["ETA_Minutes"]  # Target column

# Now handle missing values (should work without error as all columns are numeric)
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with optimized hyperparameters
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=200, learning_rate=0.05, depth=5, random_state=42, verbose=0)
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Train and evaluate models with cross-validation
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = np.mean(cross_val_score(model, X_test, y_test, cv=5))
    print(f"{name} Accuracy (R²): {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Print best model
print(f"\nBest Model: {best_model_name} with Accuracy (R²): {best_accuracy:.4f}")

import joblib

# Save the trained model
joblib.dump(model, 'catboost_eta_model.pkl')

# If using Google Colab, download the file
from google.colab import files
files.download('catboost_eta_model.pkl')