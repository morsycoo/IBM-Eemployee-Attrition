# 🧠 Employee Attrition Prediction using Deep Learning

A deep learning project for predicting employee attrition using the IBM HR Analytics dataset.

The goal of this project is to identify employees who are likely to leave the company and help HR teams make proactive retention decisions.

---

# 📌 Problem Statement

Employee attrition is one of the major challenges faced by organizations.

Losing skilled employees can lead to:

- Increased hiring costs
- Productivity loss
- Reduced team performance
- Higher training expenses

This project aims to predict whether an employee is likely to leave the company (**Attrition Prediction**) using a neural network built with **PyTorch**.

---

# 📂 Dataset

Dataset used:

**IBM HR Analytics Employee Attrition & Performance Dataset**

Target variable:

```text
Attrition
```

Classes:

```text
0 → Employee stays
1 → Employee leaves
```

Dataset contains:

- Employee demographics
- Job-related information
- Compensation
- Satisfaction metrics
- Work-life balance indicators

---

# 🛠️ Project Workflow

The project followed a complete deep learning pipeline:

### 1. Data Exploration
- Dataset inspection
- Missing value analysis
- Feature understanding
- Class imbalance analysis

### 2. Data Preprocessing
- Label encoding
- One-hot encoding
- Feature scaling
- Train / Validation / Test split

### 3. Deep Learning Modeling
- Neural Network implementation using PyTorch
- Overfitting detection
- Regularization using Dropout
- Weighted Loss Function
- Optimizer comparison
- Learning rate tuning
- Early stopping

### 4. Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve & AUC Score
- Threshold tuning
- Stability testing
- Hyperparameter search

---

# 🧠 Final Model Architecture

```text
Input (44 Features)
        ↓
Linear (64)
        ↓
ReLU
        ↓
Dropout (0.3)
        ↓
Linear (32)
        ↓
ReLU
        ↓
Dropout (0.3)
        ↓
Linear (1)
        ↓
BCEWithLogitsLoss
```

---

# ⚙️ Final Model Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Dropout | 0.3 |
| Weight Decay | 1e-4 |
| Loss Function | BCEWithLogitsLoss |
| Weighted Loss | Yes |

---

# 📊 Final Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **86.88%** |
| Precision | **58.54%** |
| Recall | **66.67%** |
| F1 Score | **62.34%** |

---

# 💼 Business Impact

This model can help HR teams:

✅ Detect employees likely to leave

✅ Reduce employee turnover

✅ Improve retention strategies

✅ Reduce recruitment costs

✅ Support workforce planning

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
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone <your-repo-link>
```

Install requirements:

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

2. Run all notebook cells from top to bottom.

3. The notebook includes:

- Data preprocessing
- Model training
- Evaluation
- Hyperparameter tuning
- Final model selection

---

# 🧠 Key Learning Outcomes

This project demonstrates:

- Deep Learning fundamentals
- Overfitting handling
- Regularization techniques
- Class imbalance handling
- Hyperparameter tuning
- Model evaluation
- Business-focused ML decision making

---

# 🏆 Final Insight

> More complex models do not always perform better.

After extensive experimentation, a simpler neural network architecture achieved the strongest overall performance.

---

# 👨‍💻 Author

**Mahmoud Morsy**  
AI Engineer | Data Science & Machine Learning