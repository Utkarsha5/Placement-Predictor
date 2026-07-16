"""
Model Training Pipeline

Trains 3 models:
  - logistic_model.pkl  → Logistic Regression (classification)
  - linear_model.pkl    → Linear Regression (continuous baseline)
  - poly_model.pkl      → Polynomial Regression (non-linear patterns)

Usage:
    python models/train.py
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# 1. Load Data
df = pd.read_csv('data/manit_placement_dataset.csv')

# 2. Scale DSA and Comm from 0-100 to 0-10 (matches the web UI sliders)
if df['dsa'].max() > 10:
    df['dsa'] = df['dsa'] / 10.0
if df['comm'].max() > 10:
    df['comm'] = df['comm'] / 10.0

# 3. Define Features and Target
X = df[['branch', 'cgpa', 'dsa', 'projects', 'internship', 'comm']]
y = df['status']

numeric_features = ['cgpa', 'dsa', 'projects', 'internship', 'comm']
categorical_features = ['branch']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Preprocessors
standard_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

poly_num_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

poly_preprocessor = ColumnTransformer(
    transformers=[
        ('num', poly_num_pipeline, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# =============================================
# MODEL 1: Logistic Regression
# =============================================
print("Training Logistic Regression...")

logistic_pipe = Pipeline(steps=[
    ('preprocessor', standard_preprocessor),
    ('classifier', LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced'))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logistic_pipe, X, y, cv=cv, scoring='accuracy')
print(f"  CV Accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

logistic_pipe.fit(X_train, y_train)
print(f"  Test Accuracy: {accuracy_score(y_test, logistic_pipe.predict(X_test))*100:.1f}%")

logistic_pipe.fit(X, y)
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic_pipe, f)
print("  Saved logistic_model.pkl\n")

# =============================================
# MODEL 2: Linear Regression
# =============================================
print("Training Linear Regression...")

linear_pipe = Pipeline(steps=[
    ('preprocessor', standard_preprocessor),
    ('regressor', LinearRegression())
])
linear_pipe.fit(X_train, y_train)

lin_r2 = r2_score(y_test, linear_pipe.predict(X_test))
print(f"  Test R²: {lin_r2:.4f}")

linear_pipe.fit(X, y)
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(linear_pipe, f)
print("  Saved linear_model.pkl\n")

# =============================================
# MODEL 3: Polynomial Regression (degree 2)
# =============================================
print("Training Polynomial Regression...")

poly_pipe = Pipeline(steps=[
    ('preprocessor', poly_preprocessor),
    ('regressor', LinearRegression())
])
poly_pipe.fit(X_train, y_train)

poly_r2 = r2_score(y_test, poly_pipe.predict(X_test))
print(f"  Test R²: {poly_r2:.4f}")

poly_pipe.fit(X, y)
with open('poly_model.pkl', 'wb') as f:
    pickle.dump(poly_pipe, f)
print("  Saved poly_model.pkl\n")

# =============================================
# Summary
# =============================================
print("=" * 50)
print("  Training complete!")
print(f"  Logistic Regression:   {cv_scores.mean()*100:.1f}% accuracy (CV)")
print(f"  Linear Regression:     R² = {lin_r2:.4f}")
print(f"  Polynomial Regression: R² = {poly_r2:.4f}")
print("=" * 50)
