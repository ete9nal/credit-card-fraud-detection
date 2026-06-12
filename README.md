# 💳 Credit Card Fraud Detection

A machine learning project comparing 5 approaches to detect fraudulent credit card transactions under severe class imbalance (0.17% fraud).

---

## 📌 Project Overview

Credit card fraud detection is a classic imbalanced classification problem. A naive model predicting "legitimate" for every transaction achieves **99.83% accuracy** but catches **zero fraud**.

This project explores multiple approaches to handle class imbalance and finds the best balance between catching fraud and minimizing false alarms.

**Key metric:** PR-AUC — more honest than ROC-AUC for imbalanced datasets, as it focuses on Precision and Recall rather than True Negatives.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv              # Kaggle dataset
│
├── models/
│   └── fraud_model.pkl             # Saved final model (XGBoost + scale_pos_weight)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_modeling.ipynb           # Modeling, evaluation & comparison
│
├── .gitignore
├── README.md
└── pyproject.toml
```

---

## 📊 Dataset

> Download the dataset from Kaggle and place `creditcard.csv` in the `data/` folder:
> [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | 30 (V1–V28 PCA-transformed + Time + Amount) |
| Period | 2 days (September 2013) |

> Features V1–V28 are PCA-transformed for confidentiality. Only `Time`, `Amount` and `Class` remain in original form.

---

## 🔍 Exploratory Data Analysis

- Confirmed severe class imbalance — 99.83% legitimate vs 0.17% fraud
- `Amount` is right-skewed with values up to 25,000 → requires StandardScaler
- `Time` reflects daily transaction patterns (2-day period with clear activity cycles)
- No missing values (pre-processed Kaggle dataset)
- V1–V28 already scaled via PCA — only Amount and Time need scaling

---

## ⚙️ Preprocessing & Pipeline

Used `imblearn.Pipeline` with `ColumnTransformer`:

- **Amount & Time** → `StandardScaler`
- **V1–V28** → `passthrough` (already PCA-scaled)
- **Class imbalance** → SMOTE (synthetic oversampling) or `scale_pos_weight=578`

> SMOTE is applied **only inside the pipeline** — never touches the test set.

---

## 🤖 Models Compared

| Model | Technique | Precision | Recall | F1 | PR-AUC | CV PR-AUC |
|-------|-----------|-----------|--------|----|--------|-----------|
| Logistic Regression | SMOTE | 0.06 | 0.92 | 0.11 | 0.73 | 0.74 ± 0.02 |
| XGBoost | SMOTE | 0.69 | 0.87 | 0.77 | 0.87 | 0.85 ± 0.02 |
| XGBoost (tuned) | SMOTE + GridSearchCV | 0.67 | 0.86 | 0.75 | 0.86 | 0.86 ± 0.02 |
| XGBoost | scale_pos_weight=578 | **0.88** | 0.84 | **0.86** | **0.88** | 0.86 ± 0.03 |
| **XGBoost (tuned) ✅** | **scale_pos_weight + GridSearchCV** | **0.88** | **0.84** | **0.86** | **0.88** | **0.86 ± 0.02** |

---

## 🎯 Hyperparameter Tuning

Used `GridSearchCV` with 5-fold `StratifiedKFold` and `scoring='average_precision'` (PR-AUC):

```python
param_grid = {
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1],
    'model__n_estimators': [100, 200]
}
```

**Best parameters:** `learning_rate=0.1`, `max_depth=7`, `n_estimators=200`

> GridSearchCV confirmed baseline SPW parameters were already near-optimal.

---

## 💡 Key Decisions

**Why PR-AUC over ROC-AUC?**
ROC-AUC accounts for True Negatives (284K legitimate transactions) which inflates the score — a poor model can look great. PR-AUC focuses only on Precision and Recall, which is what actually matters for fraud.

**Why scale_pos_weight over SMOTE?**
- No synthetic data — model learns from real patterns only
- Precision: 0.69 → 0.88 — 3x fewer false alarms
- Works at the loss function level — more natural for tree-based models
- Faster training — no oversampling step needed

**Business impact:**
Per 100 fraud alerts — SMOTE: 31 false alarms, SPW: only **12 false alarms**.
Final model still catches **84% of all real fraud cases**.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — pipelines, preprocessing, evaluation
- **XGBoost** — gradient boosting classifier
- **imbalanced-learn** — SMOTE, imblearn Pipeline
- **Joblib** — model serialization
- **Poetry** — dependency management
- **Jupyter Notebook** — analysis & modeling

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ete9nal/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
poetry install

# Run notebooks
poetry run jupyter notebook
```

Download the dataset from Kaggle and place `creditcard.csv` in the `data/` folder.
