# 👥 HR Analytics & Attrition Prediction Intelligence

An end-to-end Data Science and Business Intelligence solution designed to analyze workforce dynamics, extract business-critical insights, and predict employee attrition risk in real-time. This project spans from raw data preprocessing to an interactive **Power BI Executive Dashboard** and a live **Streamlit Web Application** powered by a calibrated Support Vector Machine (SVM) model.

---

## 🚀 Project Architecture Pipeline

```text
Raw HR Data ──> EDA & Preprocessing (Pandas) ──> BI Dashboarding (Power BI)
                                                        │
Streamlit UI <── Joblib Artifacts <── Calibrated SVM <── Feature Engineering & Selection (SQL/Python)
```

---

## 🛠️ Tech Stack & Toolkit
* **Data Processing & Analytics:** Python, Pandas, NumPy, PostgreSQL
* **Machine Learning Frameworks:** Scikit-Learn (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM), Joblib
* **Business Intelligence:** Power BI (DAX, Interactive Matrix & KPI Layouts)
* **Web Deployment:** Streamlit Framework, HTML5/CSS3 Custom Styling

---
## 📂 Repository Breakdown

```text
├── HR_Analytics_Prediction/    # Directory for serialized ML artifacts
│   ├── best_model_hr.pkl       # Calibrated SVM Model Object
│   ├── scaler_hr.pkl           # Fitted StandardScaler Instance
│   ├── columns_hr.pkl          # Reference list of expected encoded columns
│   └── continuous_cols_hr.pkl  # Reference list of continuous variables
├── cleaning.ipynb              # Jupyter Notebook for EDA & Data Preprocessing Pipeline
├── app.py                      # Production Streamlit UI Application Script
└── README.md                   # Project Documentation Profile
```

---
## 📈 Stage-by-Stage Implementation Details

### 🧹 1. Exploratory Data Analysis (EDA) & Preprocessing
* Handled data cleanliness workflows, extreme outlier bounds, and null-value distributions natively using **Pandas** and **NumPy**.
* Identified structural correlations between core independent columns (e.g., Compensation metrics, Overtime triggers, and Manager relations) against employee attrition rates.

### 📊 2. Power BI Dashboarding & Data Insights
* Transformed preprocessed structural outputs into dynamic, executive-grade operational dashboards.
* Formulated advanced **DAX measures** to compute real-time attrition rates and workforce allocation parameters.
* Embedded slicers, functional custom matrices, and high-impact corporate KPI cards for cross-functional leadership reviews.

### ⚙️ 3. Feature Engineering, Scaling & Selection
* Conducted structured hot-encoding routines to transition categorical values into computational indicators matching relational structure guidelines.
* Applied **StandardScaler** to map continuous metrics across uniform distribution patterns, ensuring distance-based predictors operate safely without data leakages.
* Refined dimensional densities by keeping only high-impact demographic, economic, and operational features.

### 🧠 4. Machine Learning & Model Evaluation
The dataset was split using stratified techniques into Training and Testing subsets. We evaluated a spectrum of supervised classification algorithms to pinpoint peak performance thresholds:
1. **Logistic Regression:** Established standard linear baseline thresholds.
2. **K-Nearest Neighbors (KNN):** Modeled distance-centric neighborhood variances.
3. **Naive Bayes:** Calculated conditional distribution benchmarks.
4. **Decision Trees:** Evaluated rule-based node dependencies.
5. **Support Vector Machine (SVM):** Implemented high-dimensional margin boundary separations.

**The Winner:** The **Calibrated SVM Model** won the selection phase, securing the highest overall **F1-Score**, striking an optimized balance between Precision and Recall to minimize dangerous false-negative classification failures.

### 💾 5. Artifact Serialization & Deployment
* Serialized the optimized SVM engine and scaling constraints using **Joblib** to guarantee consistent testing transformations.
* Engineered a sleek, custom-designed dark-themed graphical user interface using **Streamlit** to collect real-time data inputs and generate immediate corporate retention assessments.

---
## 💻 How to Get Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com
cd HR_Analytics
```

### 2. Run the Predictive Application UI
Ensure your `HR_Analytics_Prediction/` folder contains your exported `.pkl` files, then trigger the engine:
```bash
streamlit run app.py
```
---
## 👥 Contributors & Contact
* **Developer:** [Ananya](https://github.com)
* **Target Focus:** Data Analytics / Business Intelligence / Machine Learning Applications
