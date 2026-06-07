# CampusML: Student Placement Predictor 🎓🤖

CampusML is an end-to-end Machine Learning web application designed to evaluate a student's academic and behavioral metrics to predict their campus placement probability. 

Instead of relying on a single algorithm, this project features a multi-model architecture that compares Linear, Logistic, and Polynomial regression outputs to provide a highly accurate evaluation. It also features a dynamic "Training Gym" to identify skill gaps and provide targeted learning resources.

## 🚀 Key Features
* **Multi-Model ML Pipeline:** Implements Logistic Regression, Linear Regression, and Polynomial Regression to evaluate prediction variance.
* **Robust Data Preprocessing:** Utilizes `ColumnTransformer`, `StandardScaler` for continuous numerical data, and `OneHotEncoder` for categorical variables.
* **Interactive Flask Backend:** A fully functioning MVC-style backend that routes dynamic feature sets and processes HTTP requests.
* **Training Gym Dashboard:** Algorithmic logic that flags weaknesses against industry benchmarks and routes students to high-quality CS resources.
* **PDF Report Generation:** Dynamically generates downloadable performance reports utilizing FPDF streams.

## 💻 Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Backend Framework:** Flask
* **Data Visualization:** Matplotlib
* **Frontend:** HTML5, CSS3

## 🛠️ How to Run Locally

If you would like to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Utkarsha5/Placement-Predictor.git
   cd Placement-Predictor
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models and generate the `.pkl` files:**
   ```bash
   python train_advanced.py
   ```

4. **Start the Flask web server:**
   ```bash
   python app.py
   ```

5. **Open your browser and navigate to:**
   `http://127.0.0.1:5001`

---
*Built by Utkarsha Shrivastava and Nikhil Sharma*