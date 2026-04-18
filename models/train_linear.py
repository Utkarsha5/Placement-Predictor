# train_linear.py - Team Member 1
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

print("Member 1: Loading new MANIT placement data...")
df = pd.read_csv('data/manit_placement_dataset.csv')

# DATA ENGINEERING: Convert DSA and Comm from 100-scale to 10-scale to match the Web UI
df['dsa'] = df['dsa'] / 10.0
df['comm'] = df['comm'] / 10.0

# Select only the features we need (ignoring student_id)
X = df[['branch', 'cgpa', 'dsa', 'projects', 'internship', 'comm']]
y = df['status']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['cgpa', 'dsa', 'projects', 'internship', 'comm']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['branch'])
    ])

# Linear Model Pipeline
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("Member 1: Training Linear Regression baseline...")
linear_pipeline.fit(X, y)

with open('linear_model.pkl', 'wb') as f:
    pickle.dump(linear_pipeline, f)

print("✅ Member 1: linear_model.pkl saved successfully.")