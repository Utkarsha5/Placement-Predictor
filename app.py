from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "super_secret_placement_key"

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'GET':
        return render_template('predictor.html', submitted=False)

    try:
        # 1. Capture inputs
        cgpa = float(request.form['cgpa'])
        dsa = float(request.form['dsa'])
        projects = int(request.form['projects'])
        internship = int(request.form['internship'])
        comm = float(request.form['comm'])

        # 2. Math & Logic
        features = np.array([[cgpa, dsa, projects, internship, comm]])
        scaled_features = scaler.transform(features)
        prob = model.predict_proba(scaled_features)[0][1] * 100
        
        user_stats = {"CGPA": cgpa / 10, "DSA": dsa / 10, "Projects": projects / 5, "Internships": internship / 2, "Comm Skills": comm / 10}
        weakest = min(user_stats, key=user_stats.get)

        session['weakest_skill'] = weakest
        session['placement_prob'] = round(prob, 1)

        # 3. Simulated AI Advice
        advice_map = {
            "DSA": f"With a score of {dsa}/10, you should prioritize String and Array patterns. Your probability is {round(prob,1)}%, but mastering Linked Lists could push this higher.",
            "Projects": "Your technical skills are good, but recruiters need to see proof. Build one Full-Stack application this month to showcase your end-to-end capability.",
            "Internships": "You lack industry exposure. Consider applying for remote 'Virtual Internships' on Forage to bridge the gap between college and corporate expectations.",
            "Comm Skills": f"Your technical score is strong, but your {comm}/10 communication score is a bottleneck. Practice explaining your projects out loud using the STAR method.",
            "CGPA": f"Focus on maintaining your current {cgpa} CGPA while maximizing your high {max(user_stats, key=user_stats.get)} score to stand out during the initial screening."
        }
        
        session['ai_coach'] = advice_map.get(weakest, "Keep practicing your core skills!")

        # 4. Results logic
        result_text = "Highly Likely to be Placed" if prob >= 50 else "High Improvement Needed"
        result_color = "text-green-600" if prob >= 50 else "text-red-600"

        return render_template('predictor.html', 
                               probability=round(prob, 1), 
                               result_text=result_text,
                               result_color=result_color,
                               weakest=weakest,
                               ai_advice=session['ai_coach'],
                               submitted=True)
    except Exception as e:
        return render_template('predictor.html', error=str(e), submitted=False)

@app.route('/gym')
def gym():
    weak_skill = session.get('weakest_skill')
    prob = session.get('placement_prob', 'N/A')
    ai_advice = session.get('ai_coach', 'Take the Predictor test to get a custom AI-generated training plan!')
    
    return render_template('gym.html', 
                           weak_skill=weak_skill, 
                           prob=prob,
                           ai_advice=ai_advice)

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)