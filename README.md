# HR Attrition — Intelligent Decision Support System

> Predict employee flight risk, explain the drivers, and give HR teams actionable recommendations — before anyone resigns.

---

## What This Project Does

This is a full end-to-end data science project built on the **IBM HR Analytics Employee Attrition dataset** (1,470 employees, 35 features). It follows the complete pipeline from raw data exploration through to a deployed interactive prototype.

The system answers one question: **which employees are most likely to leave, and why?**

---

## Project Structure

```
├── phase1_eda.ipynb                          # Data inspection & exploratory analysis
├── phase2_data_preparation.ipynb             # Cleaning, feature engineering, encoding, scaling
├── phase3_modelling.ipynb                    # Model training, evaluation, hyperparameter tuning
├── phase4_explainability.ipynb               # SHAP analysis, segmentation, business insights
├── app.py                                    # Streamlit IDSS prototype
├── best_model_logistic_regression_tuned.pkl  # Saved final model
├── X_train.csv                               # Processed training features
├── X_test.csv / y_train.csv / y_test.csv     # Train/test splits
├── data_dictionary.csv                       # Feature definitions
└── phase4_report.docx                        # Business insights report (Phase 4 deliverable)
```

---

## Pipeline Summary

| Phase | What was done |
|-------|---------------|
| **1 — EDA** | Dataset inspection, missing value analysis, bivariate plots, class imbalance identification (84% stayed / 16% left) |
| **2 — Preparation** | Dropped constant columns, capped outliers, engineered 4 new features, OHE + ordinal encoding, StandardScaler, 80/20 stratified split |
| **3 — Modelling** | Trained LR, Random Forest, XGBoost, LightGBM with 5-fold CV imbalance comparison (SMOTE vs class_weight vs baseline). Tuned best model with RandomizedSearchCV |
| **4 — Explainability** | SHAP LinearExplainer, feature importance vs coefficient comparison, 3-tier employee segmentation, 4 evidence-based business recommendations |
| **5 — Prototype** | Interactive Streamlit app with manual input and CSV batch prediction |

---

## Engineered Features

Four features were created during Phase 2, each grounded in an HR rationale:

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| `CompensationGap` | `MonthlyIncome − median(income for same JobLevel)` | Relative pay fairness predicts attrition better than absolute salary |
| `SatisfactionIndex` | `mean(Environment + Job + Relationship satisfaction)` | Composite wellbeing signal; reduces multicollinearity |
| `TenureToAgeRatio` | `YearsAtCompany / Age` | Career embeddedness — low ratio in young employees signals flight risk |
| `IncomePerYearAtCompany` | `MonthlyIncome / (YearsAtCompany + 1)` | Detects loyalty without proportional reward |

---

## Final Model

**Logistic Regression (Tuned)** — selected over tree-based models because attrition risk on this dataset is largely captured by linear feature combinations. Tree models achieved higher accuracy by predicting "Stayed" most of the time, missing the majority of actual attrition cases.

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8022 |
| Recall | 0.6809 |
| F1-Score | 0.5079 |
| Precision | 0.4051 |

**Best hyperparameters:** `C=0.5`, `solver=saga`, `penalty=l2`, `max_iter=2000`, `class_weight=balanced`

> Recall is the primary metric. A missed attrition case (False Negative) means no HR action is taken. A false alarm (False Positive) is just an unnecessary conversation — low cost.

---

## Top SHAP-Identified Attrition Drivers

1. **TenureToAgeRatio** — low ratio (young, short-tenured) is the strongest signal
2. **OverTime** — 2.9× attrition multiplier (30.5% vs 10.4%)
3. **YearsAtCompany** — captures stagnation when combined with promotion history
4. **MonthlyIncome** — underpaid relative to job-level peers
5. **BusinessTravel_Travel_Frequently** — compound workload-intensity signal

---

## Risk Segmentation

| Tier | Size | Actual Attrition Rate |
|------|------|----------------------|
| 🟢 Low Risk | 157 (53.4%) | 6.4% |
| 🟡 Medium Risk | 66 (22.4%) | 10.6% |
| 🔴 High Risk | 71 (24.1%) | **42.3%** |

---

## Running the Prototype

**Requirements:**
```
pip install streamlit shap scikit-learn xgboost lightgbm imbalanced-learn pandas numpy matplotlib seaborn
```

**Launch:**
```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

The app supports **manual employee input** (single prediction with SHAP explanation) and **CSV batch upload** (bulk predictions with per-employee SHAP analysis and HR recommendations).

---

## Dataset

[IBM HR Analytics Employee Attrition — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)  
1,470 employees · 35 features · Binary target: `Attrition` (Yes/No)
