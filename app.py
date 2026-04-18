from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pickle
import pandas as pd
import numpy as np
from fpdf import FPDF
import io

app = Flask(__name__)

# ==========================================
# 1. Load Predictive Models
# ==========================================
try:
    with open('linear_model.pkl', 'rb') as f:
        linear_model = pickle.load(f)
    with open('poly_model.pkl', 'rb') as f:
        poly_model = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    print("✅ All systems online. ML Models loaded.")
except FileNotFoundError:
    print("⚠️ WARNING: .pkl files not found. Run training scripts first.")

# ==========================================
# 2. Analytics Helper Functions
# ==========================================

def calculate_matches(stats):
    """Calculates how well the student fits different company tiers."""
    cgpa, dsa, comm = stats['cgpa'], stats['dsa'], stats['comm']
    proj = min(stats['projects'] * 2.5, 10)
    intern = min(stats['internship'] * 5, 10)

    return {
        'FAANG / Product': round((dsa * 0.5 + cgpa * 0.4 + proj * 0.1) * 10, 1),
        'Unicorn Startups': round((proj * 0.4 + intern * 0.3 + dsa * 0.2 + comm * 0.1) * 10, 1),
        'Service / Consulting': round((comm * 0.4 + cgpa * 0.3 + dsa * 0.3) * 10, 1)
    }

def find_weakest_skill(stats):
    """Identifies the primary gap for the Gym module."""
    benchmarks = {'cgpa': 8.5, 'dsa': 8.0, 'projects': 3, 'internship': 1, 'comm': 8.0}
    deficits = {k: benchmarks[k] - stats[k] for k in benchmarks}
    weakest_key = max(deficits, key=deficits.get)
    
    mapping = {
        'cgpa': 'CGPA', 
        'dsa': 'DSA', 
        'projects': 'Projects', 
        'internship': 'Internships', 
        'comm': 'Comm Skills'
    }
    # Default to DSA if they are perfect everywhere
    return mapping.get(weakest_key, 'DSA') if deficits[weakest_key] > 0 else 'DSA'

# ==========================================
# 3. Application Routes
# ==========================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gym')
def gym():
    weak_skill = request.args.get('weak_skill', 'DSA')
    advice_dict = {
        'DSA': "Master dynamic programming and graph traversals on LeetCode.",
        'Projects': "Build a full-stack project and deploy it live.",
        'Internships': "Real-world experience is missing. Apply for remote internships.",
        'Comm Skills': "Technical brilliance needs communication. Practice the STAR method.",
        'CGPA': "Focus intensely on core academic subjects to clear the screen rounds."
    }
    ai_advice = advice_dict.get(weak_skill, "Focus on building your foundational skills.")
    return render_template('gym.html', weak_skill=weak_skill, ai_advice=ai_advice)

@app.route('/predictor')
def predictor():
    # Renders the interactive slider page (GET request only)
    return render_template('predictor.html', submitted=False)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Real-time API for slider updates."""
    data = request.json
    model_choice = data.get('model_type', 'logistic')
    
    # Extract data for prediction
    raw_data = {
        'branch': [data.get('branch', 'CSE')],
        'cgpa': [float(data['cgpa'])],
        'dsa': [float(data['dsa'])],
        'projects': [int(data['projects'])],
        'internship': [int(data['internship'])],
        'comm': [float(data['comm'])]
    }
    input_df = pd.DataFrame(raw_data)
    
    # 1. Run Prediction
    try:
        if model_choice == 'linear': 
            prob = np.clip(linear_model.predict(input_df)[0] * 100, 0, 100)
        elif model_choice == 'polynomial': 
            prob = np.clip(poly_model.predict(input_df)[0] * 100, 0, 100)
        else: 
            prob = logistic_model.predict_proba(input_df)[0][1] * 100
    except: 
        prob = 0

    # 2. Calculate Feature Impact (EXPLICIT LIST ORDER TO PREVENT MIX-UPS)
    # Order must match Chart.js Labels: ['CGPA', 'DSA', 'Projects', 'Internships', 'Comm']
    benchmarks = {'cgpa': 7.0, 'dsa': 6.0, 'projects': 1, 'internship': 0, 'comm': 6.0}
    
    importance_list = [
        round((float(data['cgpa']) - benchmarks['cgpa']) * 12, 1),      # Index 0
        round((float(data['dsa']) - benchmarks['dsa']) * 10, 1),       # Index 1
        round((int(data['projects']) - benchmarks['projects']) * 8, 1), # Index 2
        round((int(data['internship']) - benchmarks['internship']) * 15, 1), # Index 3
        round((float(data['comm']) - benchmarks['comm']) * 5, 1)       # Index 4
    ]

    stats = {k: raw_data[k][0] for k in raw_data if k != 'branch'}
    
    return jsonify({
        'probability': round(prob, 1),
        'importance': importance_list,
        'matches': calculate_matches(stats),
        'weakest': find_weakest_skill(stats)
    })

@app.route('/download_report', methods=['POST'])
def download_report():
    """Generates a PDF report using request form data."""
    data = request.form
    pdf = FPDF()
    pdf.add_page()
    
    # Logo/Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(79, 70, 229) # Indigo
    pdf.cell(200, 20, "CAMPUS ML READINESS REPORT", ln=True, align='C')
    pdf.ln(10)
    
    # Probability Result
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, f"Predicted Probability: {data['prob']}%", ln=True)
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(200, 10, f"Identified Bottleneck: {data['weakest']}", ln=True)
    pdf.ln(10)
    
    # Input Stats Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "STUDENT PROFILE SUMMARY:", ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in data.items():
        if k not in ['prob', 'weakest']:
            pdf.cell(200, 8, f"- {k.upper()}: {v}", ln=True)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(200, 10, "This report is AI-generated for educational guidance purposes.", ln=True, align='C')
    
    # Fixed Output Logic for Latin-1 bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    
    return send_file(
        io.BytesIO(pdf_bytes),
        as_attachment=True,
        download_name="Placement_Analysis.pdf",
        mimetype='application/pdf'
    )

@app.route('/reset')
def reset():
    return redirect(url_for('predictor'))

if __name__ == '__main__':
    app.run(debug=True)
