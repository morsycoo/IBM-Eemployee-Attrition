# 🧠 IBM HR Analytics Employee Attrition Prediction using Deep Learning | PyTorch

A deep learning project for predicting employee attrition using the IBM HR Analytics dataset.

The goal of this project is to identify employees who are likely to leave the company and help HR teams take proactive retention decisions.

Built using **PyTorch**, this project follows a complete end-to-end deep learning workflow including experimentation, tuning, optimization, model evaluation, and business interpretation.

---

# 📌 Problem Statement

Employee attrition is one of the biggest challenges organizations face.

Losing skilled employees can lead to:

* Increased hiring costs
* Productivity loss
* Reduced organizational efficiency
* Higher onboarding and training expenses

This project aims to predict whether an employee is likely to leave the company (**Attrition Prediction**) using a deep learning model trained on HR analytics data.

---

# 📂 Dataset

### IBM HR Analytics Employee Attrition & Performance Dataset

The dataset contains employee-related information such as:

* Employee demographics
* Salary & compensation
* Job role & department
* Work-life balance
* Job satisfaction
* Environment satisfaction
* Overtime behavior
* Career growth indicators

### 🎯 Target Variable

```text
Attrition
```

Classes:

```text
0 → Employee stays
1 → Employee leaves
```

---

# 📚 Project Roadmap

The project follows a complete deep learning workflow:

### 📂 Data Preparation

* Import Libraries
* Load Dataset
* Basic Data Exploration
* Target Analysis
* Data Cleaning
* Encoding Target & Categorical Features
* Train / Validation / Test Split
* Feature Scaling

### 🧠 Deep Learning Modeling

* Building the First Neural Network
* Baseline Model Training
* Baseline Evaluation
* Improving Generalization with Dropout
* Improved Model Training
* Improved Model Evaluation

### ⚙️ Optimization & Training

* Optimizer Comparison
* Learning Rate Tuning
* GPU Setup
* Professional Training Pipeline
* Early Stopping

### 📊 Model Evaluation

* Confusion Matrix
* ROC Curve & AUC
* Threshold Tuning
* Final Threshold Selection
* Business Decision Analysis

### ⚖️ Advanced Improvements

* Class Imbalance Handling
* Weighted Loss Function
* Model Stability Testing
* Architecture Tuning
* Hyperparameter Search

### 💾 Production Readiness

* Final Model Selection
* Save Final Model
* Load Saved Model

### 💼 Business Conclusion

* Final Business Insights
* Project Completion

---

# 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

# 🧠 Final Model Architecture

```text
Input (44 Features)
        ↓
Linear (64)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Dropout (0.3)
        ↓
Linear (32)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Dropout (0.3)
        ↓
Linear (1)
        ↓
Sigmoid (during inference)
```

### ⚙️ Final Configuration

| Parameter      | Value             |
| -------------- | ----------------- |
| Optimizer      | AdamW             |
| Learning Rate  | 0.001             |
| Dropout        | 0.3               |
| Weight Decay   | 1e-4              |
| Loss Function  | BCEWithLogitsLoss |
| Weighted Loss  | Yes               |
| Early Stopping | Yes               |

---

# 📊 Final Model Performance

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **86.88%** |
| Precision | **58.54%** |
| Recall    | **66.67%** |
| F1 Score  | **62.34%** |

---

# 🏆 Key Findings

### ✅ Weighted Loss Improved Recall

Handling class imbalance significantly improved the model's ability to detect employees likely to leave.

### ✅ Hyperparameter Search Improved Performance

Multiple architectures and configurations were tested to systematically identify the strongest model.

### ✅ Simpler Model Performed Better

One of the most important findings from this project:

> More complex models do not always perform better.

A simpler architecture achieved stronger generalization and better overall balance than deeper networks.

---

# 💼 Business Impact

This model can help HR teams:

✅ Detect employees likely to leave

✅ Improve retention strategies

✅ Reduce employee turnover

✅ Reduce hiring and training costs

✅ Support workforce planning

The model allows organizations to proactively identify at-risk employees before resignation occurs.

---

# 📁 Project Structure

```text
IBM_EMPLOYEE_ATTRITION/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   └── final_attrition_model.pth
│
├── notebooks/
│   └── deep_learning_employee_attrition.ipynb
│
├── src/
│   ├── model.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone <https://github.com/morsycoo/IBM-Eemployee-Attrition>
```

Navigate into the project:

```bash
cd IBM_EMPLOYEE_ATTRITION
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

# ▶️ How to Run

1. Open the notebook:

```text
notebooks/deep_learning_employee_attrition.ipynb
```

2. Run notebook cells sequentially from top to bottom.

The notebook includes:

* Data preprocessing
* Deep learning training
* Model optimization
* Threshold tuning
* Hyperparameter search
* Final model selection
* Save & load model workflow

---

# 🧠 Key Learning Outcomes

This project demonstrates practical experience in:

* Deep Learning with PyTorch
* Binary Classification
* Overfitting Detection
* Regularization Techniques
* Class Imbalance Handling
* Early Stopping
* Hyperparameter Tuning
* Model Evaluation
* Business-Oriented Machine Learning

---

# 🏁 Final Insight

> Better performance does not always come from more complex architectures.

Through systematic experimentation, a simpler neural network architecture achieved the best balance between:

* Accuracy
* Recall
* F1 Score
* Model Stability

---

# 👨‍💻 Author

**Mahmoud Morsy**  
AI Engineer | Data Science | Machine Learning | Deep Learning
