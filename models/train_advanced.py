import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Load Data
try:
    df = pd.read_csv('data/manit_placement_dataset.csv')
except:
    print("Error: Ensure 'data/manit_placement_dataset.csv' exists.")
    exit()

# 2. Data Cleaning: Match Web UI Scale
if df['dsa'].max() > 10: df['dsa'] = df['dsa'] / 10.0
if df['comm'].max() > 10: df['comm'] = df['comm'] / 10.0

# 3. Define Features and Target
X = df[['branch', 'cgpa', 'dsa', 'projects', 'internship', 'comm']]
y = df['status'] # 1 for Placed, 0 for Not Placed

numeric_features = ['cgpa', 'dsa', 'projects', 'internship', 'comm']
categorical_features = ['branch']

# 4. Standard Preprocessor (For Linear and Logistic)
standard_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Polynomial Preprocessor (FIXED: Applies Poly only to numbers)
poly_num_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

poly_preprocessor = ColumnTransformer(
    transformers=[
        ('num', poly_num_pipeline, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 6. Training all 3 Models
print("🚀 Initiating Advanced Training Pipeline...")

# --- Train Logistic ---
logistic_pipe = Pipeline(steps=[
    ('preprocessor', standard_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, C=1.0))
])
logistic_pipe.fit(X, y)
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic_pipe, f)
print("✅ Logistic Model Trained and Saved.")

# --- Train Linear ---
linear_pipe = Pipeline(steps=[
    ('preprocessor', standard_preprocessor),
    ('regressor', LinearRegression())
])
linear_pipe.fit(X, y)
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(linear_pipe, f)
print("✅ Linear Model Trained and Saved.")

# --- Train Polynomial ---
poly_pipe = Pipeline(steps=[
    ('preprocessor', poly_preprocessor),
    ('regressor', LinearRegression())
])
poly_pipe.fit(X, y)
with open('poly_model.pkl', 'wb') as f:
    pickle.dump(poly_pipe, f)
print("✅ Polynomial Model Trained and Saved.")

print("🎉 All models successfully compiled!")