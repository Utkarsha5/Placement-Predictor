import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1. Generate Realistic Dataset
def generate_data(n_rows=500):
    np.random.seed(42)
    cgpa = np.random.uniform(5, 10, n_rows)
    dsa_score = np.random.uniform(1, 10, n_rows)
    projects = np.random.randint(0, 6, n_rows)
    internship = np.random.randint(0, 2, n_rows)
    comm_skills = np.random.uniform(1, 10, n_rows)

    # Realistic Placement Logic: Weighted probability
    # Base chance + weights from features
    placement_score = (
        (cgpa * 0.3) + 
        (dsa_score * 0.3) + 
        (projects * 0.1) + 
        (internship * 0.15) + 
        (comm_skills * 0.15)
    )
    # If score > threshold, placed = 1
    placed = (placement_score > 6.5).astype(int)

    df = pd.DataFrame({
        'CGPA': cgpa,
        'DSA_Score': dsa_score,
        'Projects': projects,
        'Internship': internship,
        'Comm_Skills': comm_skills,
        'Placed': placed
    })
    return df

# 2. Preparation
df = generate_data(500)
X = df.drop('Placed', axis=1)
y = df['Placed']

# Split dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

best_acc = 0
best_model_name = ""
results = {}

print("--- Model Performance ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.2f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

# 4. Evaluation of the Best Model
print(f"\n--- Best Model: {best_model_name} Evaluation ---")
y_pred_best = best_model.predict(X_test_scaled)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# 5. Visualizing Feature Importance (Using Random Forest)
plt.figure(figsize=(10, 5))
if best_model_name == "Random Forest":
    importances = best_model.feature_importances_
else:
    rf = RandomForestClassifier().fit(X_train_scaled, y_train)
    importances = rf.feature_importances_

feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.title('Which features impact placement most?')
plt.show()

# 6. Save Model and Scaler
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and Scaler saved successfully as model.pkl and scaler.pkl")