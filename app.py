from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. Load the Team's Trained Models
# ==========================================
print("Booting up CampusML Placement Hub...")
try:
    with open('linear_model.pkl', 'rb') as f:
        linear_model = pickle.load(f)
    with open('poly_model.pkl', 'rb') as f:
        poly_model = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    print("✅ All three predictive models loaded successfully.")
except FileNotFoundError:
    print("⚠️ WARNING: Model .pkl files not found. Please run the training scripts first.")

# ==========================================
# 2. Helper Functions
# ==========================================
def find_weakest_skill(stats):
    """
    Diagnostic algorithm to find the weakest area for the Gym module.
    Compares student stats against competitive MANIT benchmarks.
    """
    # These are the ideal stats for a top-tier candidate
    benchmarks = {
        'CGPA': 8.5, 
        'DSA': 8.0, 
        'Projects': 3, 
        'Internships': 1, 
        'Comm Skills': 8.0
    }
    
    # Calculate how far below the benchmark the user is
    deficits = {
        'CGPA': benchmarks['CGPA'] - stats['cgpa'],
        'DSA': benchmarks['DSA'] - stats['dsa'],
        'Projects': benchmarks['Projects'] - stats['projects'],
        'Internships': benchmarks['Internships'] - stats['internship'],
        'Comm Skills': benchmarks['Comm Skills'] - stats['comm']
    }
    
    # Find the skill with the highest deficit
    weakest = max(deficits, key=deficits.get)
    
    # If they are above all benchmarks, suggest DSA as default advanced prep
    if deficits[weakest] <= 0:
        return 'DSA'
    return weakest

# ==========================================
# 3. Application Routes
# ==========================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gym')
def gym():
    weak_skill = request.args.get('weak_skill', None)
    
    # Dynamic AI Coaching Advice based on the weakest skill
    advice_dict = {
        'DSA': "Data Structures and Algorithms are the core of technical rounds. Focus on mastering dynamic programming and graph traversals on LeetCode.",
        'Projects': "Your theoretical knowledge is solid, but recruiters want to see it applied. Build a full-stack or ML project and deploy it live.",
        'Internships': "Real-world experience is missing from your resume. Apply for remote internships or contribute to open-source repositories.",
        'Comm Skills': "Technical brilliance needs to be communicated clearly. Practice the STAR method for your HR and behavioral rounds.",
        'CGPA': "Companies often use CGPA for initial shortlisting. Focus intensely on your core academic subjects this semester to clear the cutoff."
    }
    
    ai_advice = advice_dict.get(weak_skill, "Focus on building your foundational skills based on your predictor results.")
    
    return render_template('gym.html', weak_skill=weak_skill, ai_advice=ai_advice)

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        # 1. Extract Form Data
        data = {
            'branch': [request.form['branch']],
            'cgpa': [float(request.form['cgpa'])],
            'dsa': [float(request.form['dsa'])],
            'projects': [int(request.form['projects'])],
            'internship': [int(request.form['internship'])],
            'comm': [float(request.form['comm'])]
        }
        model_choice = request.form['model_type']
        
        # Convert dictionary to Pandas DataFrame for the models
        input_df = pd.DataFrame(data)
        
        # 2. Route to the Selected Model
        probability = 0
        
        try:
            if model_choice == 'linear':
                # Linear Regression predicts continuous values, we map it to percentage
                raw_pred = linear_model.predict(input_df)[0]
                probability = np.clip(raw_pred * 100, 0, 100) 
                
            elif model_choice == 'polynomial':
                # Polynomial Regression captures feature interactions
                raw_pred = poly_model.predict(input_df)[0]
                probability = np.clip(raw_pred * 100, 0, 100)
                
            elif model_choice == 'logistic':
                # Logistic Regression natively outputs class probabilities
                prob_array = logistic_model.predict_proba(input_df)
                probability = prob_array[0][1] * 100 # Index 1 is the 'Placed' class
        except Exception as e:
            print(f"Prediction Error: {e}")
            probability = 0

        # Format probability to 1 decimal place
        probability = round(probability, 1)

        # 3. Determine Presentation Styling
        if probability >= 75:
            result_color = "text-green-500"
            result_text = "Highly likely to be placed! Keep refining your skills."
        elif probability >= 50:
            result_color = "text-yellow-500"
            result_text = "Good chance, but room for improvement. Hit the gym!"
        else:
            result_color = "text-red-500"
            result_text = "High risk. Focus heavily on your weak areas."

        # 4. Diagnostics: Find weakest link
        weakest = find_weakest_skill({
            'cgpa': data['cgpa'][0], 
            'dsa': data['dsa'][0], 
            'projects': data['projects'][0], 
            'internship': data['internship'][0], 
            'comm': data['comm'][0]
        })

        return render_template('predictor.html', 
                               submitted=True, 
                               probability=probability,
                               result_color=result_color,
                               result_text=result_text,
                               weakest=weakest)

    # If it's a GET request, just show the empty form
    return render_template('predictor.html', submitted=False)

@app.route('/reset')
def reset():
    # Clears the form and redirects to a fresh predictor page
    return redirect(url_for('predictor'))

if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)