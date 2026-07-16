# CampusML — Placement Predictor

A Flask web app that predicts campus placement probability using three ML models. Built with a real-time interactive dashboard, downloadable PDF reports, and a personalized training gym.

## Features

- **Multi-Model Prediction** — Switch between Logistic, Linear, and Polynomial Regression in real time
- **Interactive Dashboard** — Sliders for CGPA, DSA, Projects, Internships, and Communication skills with live Chart.js visualizations
- **Training Gym** — Identifies strengths/weaknesses and links to curated resources
- **PDF Reports** — Download a formatted placement readiness report
- **Dark Mode** — Full dark/light theme toggle

## Models

| Model | Type | Use Case |
|-------|------|----------|
| Logistic Regression | Classification | Primary — gives placement probability |
| Linear Regression | Regression | Interpretable baseline |
| Polynomial Regression (degree 2) | Regression | Captures non-linear feature interactions |

## Tech Stack

- **Backend:** Flask, scikit-learn, pandas, NumPy
- **Frontend:** Tailwind CSS, Chart.js, Font Awesome
- **PDF:** FPDF

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/Placement-Predictor.git
cd Placement-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the models (generates .pkl files)
python models/train.py

# Run the app
python app.py
```

The app will be available at `http://localhost:5001`.

## Project Structure

```
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── data/
│   └── manit_placement_dataset.csv
├── models/
│   └── train.py           # Trains all 3 models
├── templates/
│   ├── base.html          # Layout with navbar/footer
│   ├── home.html          # Landing page
│   ├── predictor.html     # Prediction dashboard
│   └── gym.html           # Training recommendations
```

## Retraining Models

```bash
python models/train.py
```

This generates `logistic_model.pkl`, `linear_model.pkl`, and `poly_model.pkl` in the project root.
