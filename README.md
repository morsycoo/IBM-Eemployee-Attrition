# 🧠 IBM Employee Attrition Prediction using Deep Learning | PyTorch

An end-to-end deep learning project for predicting employee attrition using the IBM HR Analytics Employee Attrition Dataset.

The goal of this project is to identify employees who are likely to leave the company and help HR teams make proactive retention decisions before employee resignation occurs.

Built using **PyTorch**, this project follows a complete real-world machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Statistical Validation
* Deep Learning Modeling
* Hyperparameter Optimization
* Threshold Tuning
* Model Explainability (SHAP)
* Business Insights & HR Recommendations
* Production-Oriented Thinking

---

## 🚀 Project Overview

Employee attrition is one of the most costly challenges organizations face.

Losing experienced employees may lead to:

* Increased recruitment costs
* Productivity decline
* Knowledge loss
* Higher onboarding and training expenses
* Reduced organizational stability

Instead of reacting to resignations after they happen, organizations can leverage predictive analytics to proactively identify employees at risk of leaving.

This project develops an **Employee Attrition Prediction System** using deep learning to help HR teams improve employee retention strategies.

---

## 📌 Business Problem

Can we predict whether an employee is likely to leave the company based on workplace, behavioral, and satisfaction-related factors?

This project solves a **binary classification problem**.

### 🎯 Target Variable

`Attrition`

Classes:

* **0 → Employee Stays**
* **1 → Employee Leaves**

---

## 🎯 Project Objectives

This project focuses on:

* Understanding employee attrition drivers
* Building a predictive deep learning model
* Improving minority-class detection
* Reducing false negatives
* Explaining model decisions using SHAP
* Generating actionable HR recommendations
* Demonstrating production-oriented ML thinking

---

## 📂 Dataset

### IBM HR Analytics Employee Attrition & Performance Dataset

The dataset contains employee-related information such as:

* Employee demographics
* Salary & compensation
* Job role & department
* Overtime behavior
* Work-life balance
* Job satisfaction
* Environment satisfaction
* Career growth indicators
* Promotion history
* Travel frequency

### Dataset Summary

| Metric                  | Value |
| ----------------------- | ----- |
| Employees               | 1470  |
| Original Features       | 35    |
| Features After Encoding | 44    |

---

## 📚 Project Workflow

This project follows a complete **end-to-end machine learning workflow**.

### 1️⃣ Data Understanding & Exploration

* Dataset Inspection
* Missing Values Analysis
* Target Distribution Analysis
* Exploratory Data Analysis (EDA)
* Correlation Analysis
* Outlier Detection
* Distribution Analysis

---

### 2️⃣ Statistical Validation

Statistical tests were performed to validate relationships between employee features and attrition.

#### Numerical Features

Statistical testing included:

* T-Test
* P-value Analysis

Features tested:

* Age
* Monthly Income
* Years At Company

#### Categorical Features

Statistical testing included:

* Chi-Square Test
* Statistical Significance Analysis

Features tested:

* Overtime
* Job Role
* Marital Status

### Key Takeaway

Multiple employee-related factors showed statistically significant relationships with attrition risk.

---

### 3️⃣ Data Preprocessing

Data preprocessing included:

* Target Encoding
* One-Hot Encoding
* Feature Scaling
* Train / Validation / Test Split
* Tensor Preparation for PyTorch

---

### 4️⃣ Deep Learning Modeling

The project followed an iterative modeling strategy.

#### Baseline Model

A simple neural network was first implemented to establish a performance baseline.

Key observations:

* High bias toward majority class
* Weak minority-class detection
* Low recall for employee attrition

#### Model Improvement Strategy

The model was systematically improved using:

* Batch Normalization
* Dropout Regularization
* Weighted Loss Function
* Hyperparameter Tuning
* Threshold Optimization
* Early Stopping

---

## 🧠 Final Neural Network Architecture

```text
Input (44 Features)
        ↓
Linear (128)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Dropout (0.30)
        ↓
Linear (64)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Dropout (0.25)
        ↓
Linear (32)
        ↓
ReLU
        ↓
Linear (1)
        ↓
Sigmoid (during inference)
```

---

## ⚙️ Final Training Configuration

| Parameter        | Value             |
| ---------------- | ----------------- |
| Optimizer        | Adam              |
| Learning Rate    | 0.001             |
| Loss Function    | BCEWithLogitsLoss |
| Weighted Loss    | Yes               |
| Early Stopping   | Yes               |
| Threshold Tuning | Yes               |
| Final Threshold  | 0.45              |

---

## 📊 Final Model Performance

### Final Metrics

| Metric        | Score    |
| ------------- | -------- |
| Accuracy      | **85%**  |
| Precision     | **53%**  |
| Recall        | **67%**  |
| F1 Score      | **59%**  |
| ROC AUC Score | **0.83** |

### Why Recall Matters

In employee attrition prediction, missing employees likely to resign can be more costly than generating additional alerts.

Therefore, the model prioritizes:

```text
Higher Recall > Perfect Accuracy
```

This allows HR teams to proactively intervene earlier.

---

## 🎯 Threshold Tuning

Instead of relying on the default classification threshold (`0.50`), threshold tuning was performed to optimize business performance.

### Final Selected Threshold

```text
0.45
```

This improved employee attrition detection while maintaining acceptable overall performance.

---

## 🔍 SHAP Explainability

SHAP analysis was used to explain model predictions and identify the strongest employee attrition drivers.

### Top Features Influencing Attrition

* Overtime
* Job Role
* Number of Companies Worked
* Business Travel Frequency
* Job Satisfaction
* Environment Satisfaction
* Relationship Satisfaction
* Age
* Distance From Home
* Years Since Last Promotion

### Key Takeaway

Employee attrition is influenced by multiple organizational and behavioral factors rather than a single cause.

---

## 💼 Business Insights

Key findings generated from the project include:

✅ Overtime emerged as the strongest attrition signal

✅ Employees with lower satisfaction scores showed higher attrition risk

✅ Frequent business travel correlated with higher resignation probability

✅ Career stagnation indicators influenced attrition behavior

✅ Younger employees showed higher attrition tendency

---

## 👥 HR Recommendations

Based on model insights, HR teams may consider:

* Reducing excessive overtime
* Monitoring high-risk employee groups
* Improving employee satisfaction programs
* Supporting frequently traveling employees
* Strengthening retention for younger employees
* Improving promotion transparency
* Deploying proactive attrition monitoring systems

---

## 🏗️ Project Structure

```text
IBM-Employee-Attrition-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   └── employee_attrition_nn.pth
│
├── notebooks/
│   ├── archive/
│   │   ├── deep_learning_employee_attrition.ipynb
│   │   ├── Employee_Attrition_Prediction_PyTorch.ipynb
│   │   └── HR_Analytics_Employee_Attrition.ipynb
│   │
│   └── ibm_employee_attrition_prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* SHAP
* Jupyter Notebook

---

## 🚀 Installation

Clone the repository:

```bash
git clone <your-repository-link>
```

Navigate into the project:

```bash
cd IBM-Employee-Attrition-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## ▶️ How to Run

1. Open:

```text
notebooks/ibm_employee_attrition_prediction.ipynb
```

2. Run notebook cells sequentially from top to bottom.

The notebook includes:

* EDA
* Statistical Testing
* Deep Learning Training
* Threshold Tuning
* SHAP Explainability
* Business Insights
* Final Evaluation
* Save & Load Model Workflow

---

## 🔮 Future Improvements

Future work may include:

* Comparing multiple ML models (XGBoost, LightGBM, Random Forest)
* Automated hyperparameter optimization
* Cross-validation
* Real-world HR datasets
* Real-time HR dashboards
* Data drift monitoring
* Production deployment

---

## 🧠 Key Learning Outcomes

This project demonstrates practical experience in:

* Deep Learning with PyTorch
* Binary Classification
* Statistical Thinking
* Threshold Optimization
* Class Imbalance Handling
* Model Explainability
* SHAP Analysis
* Overfitting Detection
* Regularization Techniques
* Business-Oriented Machine Learning
* Production ML Thinking

---

## 🏁 Final Insight

> Better performance does not always come from more complex architectures.

Through systematic experimentation, a relatively simpler neural network achieved the strongest balance between:

* Accuracy
* Recall
* Generalization
* Stability
* Business usefulness

---

## 👨‍💻 Author

**Mahmoud Morsy**
AI Engineer | Machine Learning | Deep Learning | Data Science
