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
# If your CSV has 100-scale scores, we convert them to 10-scale for the web sliders
if df['dsa'].max() > 10: df['dsa'] = df['dsa'] / 10.0
if df['comm'].max() > 10: df['comm'] = df['comm'] / 10.0

# 3. Define Features and Target
X = df[['branch', 'cgpa', 'dsa', 'projects', 'internship', 'comm']]
y = df['status'] # 1 for Placed, 0 for Not Placed

# 4. Preprocessing Pipeline
# We scale numbers and "One-Hot Encode" the branch names (CSE, ECE, etc.)
numeric_features = ['cgpa', 'dsa', 'projects', 'internship', 'comm']
categorical_features = ['branch']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Training all 3 Models with Best Practices
def train_and_save(model, name):
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipe.fit(X, y)
    with open(f'{name}_model.pkl', 'wb') as f:
        pickle.dump(pipe, f)
    print(f"✅ {name.capitalize()} Model Trained and Saved.")

# --- Train Logistic (Best for True Probability) ---
train_and_save(LogisticRegression(max_iter=1000, C=1.0), 'logistic')

# --- Train Linear ---
train_and_save(LinearRegression(), 'linear')

# --- Train Polynomial (Degree 2 for better pattern matching) ---
poly_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])
poly_pipe.fit(X, y)
with open('poly_model.pkl', 'wb') as f:
    pickle.dump(poly_pipe, f)
print("✅ Polynomial Model Trained and Saved.")

print("\n🚀 All models updated with Advanced Scaling and Branch Encoding!")
