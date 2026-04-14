import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_data(n_rows=500):
    np.random.seed(42)
    cgpa = np.random.uniform(5, 10, n_rows)
    dsa_score = np.random.uniform(1, 10, n_rows)
    projects = np.random.randint(0, 6, n_rows)
    internship = np.random.randint(0, 2, n_rows)
    comm_skills = np.random.uniform(1, 10, n_rows)

    placement_score = (
        (cgpa * 0.3) + 
        (dsa_score * 0.3) + 
        (projects * 0.1) + 
        (internship * 0.15) + 
        (comm_skills * 0.15)
    )
    placed = (placement_score > 6.5).astype(int)

    return pd.DataFrame({
        'CGPA': cgpa,
        'DSA_Score': dsa_score,
        'Projects': projects,
        'Internship': internship,
        'Comm_Skills': comm_skills,
        'Placed': placed
    })

if __name__ == "__main__":
    # 1. Preparation
    df = generate_data(500)
    X = df.drop('Placed', axis=1)
    y = df['Placed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Model Training
    print("--- Training Logistic Regression Model ---")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # 3. Evaluation
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # 4. Save Model and Scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nSuccess! model.pkl and scaler.pkl updated.")