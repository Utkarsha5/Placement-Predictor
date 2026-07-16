from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session
import pickle
import pandas as pd
import numpy as np
from fpdf import FPDF
import io

app = Flask(__name__)
app.secret_key = "campus_ml_secret_key"

# Load Models
try:
    with open('linear_model.pkl', 'rb') as f:
        linear_model = pickle.load(f)
    with open('poly_model.pkl', 'rb') as f:
        poly_model = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    print("All ML models loaded.")
except FileNotFoundError:
    print("WARNING: .pkl files not found. Run 'python models/train.py' first.")

# Helper Functions
def find_weakest_skill(stats):
    benchmarks = {'cgpa': 8.5, 'dsa': 8.5, 'projects': 4, 'internship': 2, 'comm': 8.5}
    deficits = {k: benchmarks[k] - stats[k] for k in benchmarks}

    if all(v <= 0 for v in deficits.values()):
        return "Profile Ready!"

    weakest_key = max(deficits, key=deficits.get)
    mapping = {'cgpa': 'CGPA', 'dsa': 'DSA', 'projects': 'Projects', 'internship': 'Internships', 'comm': 'Comm'}
    return mapping.get(weakest_key, 'DSA')

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('home'))

@app.route('/gym')
def gym():
    stats = session.get('last_stats', {'cgpa': 8.0, 'dsa': 7.0, 'projects': 2, 'internship': 1, 'comm': 7.0})

    benchmarks = {'cgpa': 8.5, 'dsa': 8.0, 'projects': 3, 'internship': 1, 'comm': 8.0}
    mapping = {'cgpa': 'CGPA', 'dsa': 'DSA', 'projects': 'Projects', 'internship': 'Internships', 'comm': 'Comm Skills'}

    strengths = [mapping[k] for k, b in benchmarks.items() if stats[k] >= b]
    weaknesses = [mapping[k] for k, b in benchmarks.items() if stats[k] < b]

    return render_template('gym.html', strengths=strengths, weaknesses=weaknesses)

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    model_choice = data.get('model_type', 'logistic')

    cgpa = float(data['cgpa'])
    dsa = float(data['dsa'])
    projects = int(data['projects'])
    internship = int(data['internship'])
    comm = float(data['comm'])

    session['last_stats'] = {
        'cgpa': cgpa, 'dsa': dsa, 'projects': projects,
        'internship': internship, 'comm': comm
    }

    input_df = pd.DataFrame({
        'branch': [data.get('branch', 'CSE')],
        'cgpa': [cgpa], 'dsa': [dsa], 'projects': [projects],
        'internship': [internship], 'comm': [comm]
    })

    try:
        if model_choice == 'linear':
            prob = np.clip(linear_model.predict(input_df)[0] * 100, 0, 100)
        elif model_choice == 'polynomial':
            prob = np.clip(poly_model.predict(input_df)[0] * 100, 0, 100)
        else:
            prob = logistic_model.predict_proba(input_df)[0][1] * 100

        prob = float(np.clip(prob, 0, 100))
    except Exception as e:
        print(f"Prediction Error: {e}")
        prob = 0

    benchmarks = {'cgpa': 7.0, 'dsa': 6.0, 'projects': 1, 'internship': 0, 'comm': 6.0}
    importance_list = [
        round((cgpa - benchmarks['cgpa']) * 12, 1),
        round((dsa - benchmarks['dsa']) * 10, 1),
        round((projects - benchmarks['projects']) * 8, 1),
        round((internship - benchmarks['internship']) * 15, 1),
        round((comm - benchmarks['comm']) * 5, 1)
    ]

    return jsonify({
        'probability': round(prob, 1),
        'importance': importance_list,
        'weakest': find_weakest_skill(session['last_stats'])
    })

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.form
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(79, 70, 229)
    pdf.cell(200, 20, "CAMPUS ML READINESS REPORT", ln=True, align='C')
    pdf.ln(10)

    pdf.set_fill_color(79, 70, 229)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 15, f"Placement Probability: {data['prob']}%", ln=True, align='C', fill=True)
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "STUDENT PROFILE ANALYSIS:", ln=True)
    pdf.set_font("Arial", size=11)

    for f in ['branch', 'cgpa', 'dsa', 'projects', 'internships', 'comm', 'weakest']:
        pdf.cell(200, 8, f"- {f.upper()}: {data.get(f, 'N/A')}", ln=True)

    pdf_output = pdf.output(dest='S').encode('latin-1')
    return send_file(io.BytesIO(pdf_output), as_attachment=True, download_name="CampusML_Report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
