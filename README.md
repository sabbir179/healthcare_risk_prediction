# Healthcare Risk Prediction: Chronic Kidney Disease (CKD)

## Problem Statement

Chronic Kidney Disease (CKD) is a major public health concern, and early detection is critical for timely intervention.  
The goal of this project is to build a machine learning model that predicts CKD status using routine clinical and laboratory measurements.

---

## Dataset

- Source: Kaggle (Kidney Disease Risk Dataset)
- Size: 2,304 records × 9 features
- Target variable: `CKD_Status` (0 = No CKD, 1 = CKD)

### Key Features

- Age
- Creatinine_Level
- Blood Urea Nitrogen (BUN)
- Glomerular Filtration Rate (GFR)
- Urine_Output
- Diabetes
- Hypertension

---

## Approach

### 1. Exploratory Data Analysis (EDA)

- Inspected data types, distributions, and missing values
- Verified class balance for the target variable
- Explored feature distributions and correlations with CKD status

### 2. Baseline Model: Logistic Regression

- Used Logistic Regression as an interpretable baseline model
- Applied feature scaling using `StandardScaler`
- Achieved:
  - ROC-AUC ≈ 0.91
  - PR-AUC ≈ 0.93

### 3. Threshold Tuning for Medical Screening

- Evaluated default decision threshold (0.5)
- Lowered threshold to 0.4 to reduce false negatives
- Result:
  - Reduced missed CKD cases at the cost of more false positives
  - This trade-off is acceptable in a medical screening context

### 4. Model Comparison: Random Forest

- Trained a Random Forest classifier to capture non-linear relationships
- After removing downstream clinical variables to prevent data leakage, Random Forest achieved near-perfect performance

#### Important Note on Dataset Characteristics

The near-perfect performance of Random Forest suggests the dataset may be rule-based or synthetically generated, with CKD status deterministically derived from clinical thresholds (e.g., GFR, creatinine levels).  
Logistic Regression is therefore retained as a realistic and interpretable baseline for real-world clinical scenarios.

---

## Results Summary

| Model               | ROC-AUC | PR-AUC | Notes                                         |
| ------------------- | ------- | ------ | --------------------------------------------- |
| Logistic Regression | ~0.91   | ~0.93  | Interpretable baseline, realistic performance |
| Random Forest       | ~1.00   | ~1.00  | Likely rule-based dataset                     |

---

## Repository Structure

healthcare-risk-prediction/
│
├── notebooks/
│ └── 01_eda.ipynb
├── src/
│ ├── config.py
│ ├── data_load.py
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
├── requirements.txt
└── README.md
