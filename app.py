from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data from HTML Form
    cgpa = float(request.form['cgpa'])
    dsa = float(request.form['dsa'])
    projects = int(request.form['projects'])
    internship = int(request.form['internship'])
    comm = float(request.form['comm'])

    # 2. Prepare data for model
    features = np.array([[cgpa, dsa, projects, internship, comm]])
    scaled_features = scaler.transform(features)
    
    # 3. Predict Probability
    prob = model.predict_proba(scaled_features)[0][1] * 100
    
    # 4. Hard-coded Suggestion Logic
    suggestions = []
    if dsa < 5: suggestions.append("Boost your DSA practice on LeetCode/GeeksforGeeks.")
    if projects < 2: suggestions.append("Try to complete at least 2 full-stack or ML projects.")
    if cgpa < 7: suggestions.append("Focus on improving your CGPA in upcoming semesters.")
    if internship == 0: suggestions.append("Look for 1-2 month virtual internships to gain experience.")

    result_text = "Likely to be Placed" if prob > 50 else "High Improvement Needed"
    
    return render_template('index.html', 
                           probability=f"{prob:.2f}%", 
                           result=result_text,
                           suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True)