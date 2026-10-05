# 🎓 Exam Marks Prediction • Academic Intelligence System

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2563eb?style=for-the-badge&logo=github)](https://rookiexcookie.github.io/exam-marks-prediction/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-Linear%20%7C%20Ridge%20%7C%20Random%20Forest-10b981?style=for-the-badge)](https://rookiexcookie.github.io/exam-marks-prediction/)

A modern, high-performance academic analytics dashboard and machine learning system that predicts student examination scores from study habits, attendance percentage, assignment performance, and historical academic assessments.

🔗 **Live Worldwide URL**: [https://rookiexcookie.github.io/exam-marks-prediction/](https://rookiexcookie.github.io/exam-marks-prediction/)

---

## 🌟 Key Features

1. **📊 Executive Overview & Analytics**
   - Cohort performance distribution charts, correlation heatmaps, and study hours vs. score scatter trends.
   - Real-time KPI summaries for cohort averages, median scores, and top predictors.

2. **🔮 Real-Time Exam Marks Predictor**
   - Dynamic prediction engine evaluating:
     - Daily Study Hours
     - Attendance Percentage
     - Assignment Scores (0-100)
     - Previous Exam Marks (0-100)
     - Internal Assessment Scores (0-50)
     - Preparation Days Before Exam
   - Interactive gauge showing expected score, percentage, letter grade, and 95% confidence interval bounds.

3. **🎯 Interactive Goal Planner ("What-If" Simulator)**
   - Enter your target grade or dream score (e.g., 90%).
   - Instantly calculates recommended adjustments in study hours, attendance, and prep days to reach your goal.

4. **⚙️ In-Browser Machine Learning Training & Model Comparison**
   - Live client-side training on synthetic cohorts (600+ student records).
   - Compare multiple regression algorithms:
     - **Linear Regression (Parametric OLS)**
     - **Ridge Regularized Regression (L2 Penalty)**
     - **Polynomial Regression (Degree 2 Interactions)**
     - **Random Forest Regressor (Ensemble Trees)**
   - Real-time metric comparison (MAE, MSE, RMSE, R² Score) and dynamic model switching.

5. **💡 Academic Insights & Feature Importance**
   - Normalized feature importance rankings displaying which variables exert the strongest impact on test outcomes.
   - Evidence-based academic recommendations.

6. **📁 Dataset Explorer & CSV Export**
   - Searchable, filterable student records table with one-click CSV export and dataset generation controls.

---

## 🏗️ Repository Architecture

```text
exam-marks-prediction/
├── index.html              # Main SaaS dashboard interface (6 interactive views)
├── style.css               # Clean SaaS design system (responsive, compact, light theme)
├── app.js                  # Frontend controller & Chart.js visualizations
├── ml_engine.js            # Client-side ML regression suite (Linear, Ridge, Poly, Forest)
├── data.js                 # Dataset generation & statistical distributions
├── backend/                # Python FastAPI Backend
│   ├── main.py             # REST API server
│   ├── predictor.py        # Prediction service
│   ├── schemas.py          # Pydantic schemas
│   ├── ml/                 # Python scikit-learn training & evaluation scripts
│   └── database/           # SQLite database persistence
├── data/                   # Historical CSV dataset
├── utils/                  # Python helper utilities
├── requirements.txt        # Python backend dependencies
└── .github/workflows/      # Automated GitHub Pages CI/CD workflow
```

---

## 🚀 Running Locally

### Option 1: Static Web App (Zero Setup)
Simply open `index.html` in any web browser, or host it locally:
```bash
# Python 3 built-in server
python -m http.server 3000
```
Then visit `http://localhost:3000`.

### Option 2: Python FastAPI Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🌐 Worldwide Access (GitHub Pages)

This project is automatically deployed and accessible worldwide at:
👉 **[https://rookiexcookie.github.io/exam-marks-prediction/](https://rookiexcookie.github.io/exam-marks-prediction/)**
