# 🧠 Stroke Prediction Using Machine Learning

This project aims to predict the likelihood of stroke in patients using demographic and health-related attributes through various machine learning models. It also explores research gaps related to fairness, interpretability, and model performance, especially across gender-specific subgroups.

---

## 📌 Research Problem

Despite the increasing use of machine learning in healthcare, many stroke prediction models lack fairness, suffer from low recall for minority cases (stroke-positive patients), and are not interpretable enough for clinical adoption. This project addresses:

- Gender-specific prediction gaps
- Class imbalance issues
- Low interpretability of complex ML models

---

## 🎯 Objectives

- Predict stroke risk using demographic and clinical variables
- Compare baseline models (Random Forest, Logistic Regression, etc.)
- Enhance fairness, recall, and transparency through:
  - Gender-stratified modeling
  - SMOTE for class balancing
  - SHAP/LIME for interpretability

---

## 📊 Dataset

- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Records**: 5,110 patients
- **Features**:
  - Gender, Age, Hypertension, Heart Disease
  - Work Type, Residence Type, Smoking Status
  - Avg. Glucose Level, BMI, Stroke (target)

---

## 🧪 Methodology

1. **Preprocessing**:
   - Handled missing values (dropped rows with missing BMI)
   - Encoded categorical variables
   - Normalized features using `MinMaxScaler`

2. **Exploratory Data Analysis (EDA)**:
   - KDE plots, count plots, correlation analysis
   - Examined stroke distribution by gender, glucose, BMI, etc.

3. **Modeling**:
   - Applied multiple classifiers: Random Forest, SVC, Logistic Regression, Decision Tree, KNN
   - Hyperparameter tuning via `GridSearchCV`
   - Evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix

4. **Enhancements (Effort vs Impact Prioritized)**:
   - ✅ Gender-stratified modeling
   - ✅ SMOTE for class imbalance
   - ✅ SHAP/LIME for explainability
   - ✅ Ensemble techniques
   - ✅ Streamlit dashboard (planned)
   - ✅ Pipeline automation with `sklearn.pipeline` and `MLflow` (planned)

---

## 📈 Results Summary

| Model                  | Accuracy | Notes                                      |
|------------------------|----------|--------------------------------------------|
| Random Forest          | 96%      | High accuracy, poor stroke recall          |
| Logistic Regression    | 59.7%    | Better recall due to class weighting       |
| SVC                    | 96%      | Similar to RF                              |
| Decision Tree          | 96%      | Very fast, same issue with imbalance       |
| K-Nearest Neighbors    | 95.9%    | Best balance of performance & speed        |

---

## 📂 Folder Structure


---

## 📊 Visualizations

- 📍 Effort vs Impact Matrix
- 🕸️ Radar Chart: Research Gaps vs Proposed Solutions
- 📈 SHAP summary plots (planned)
- 📊 Confusion matrices & ROC curves

---

## 🔍 Research Questions

1. Does hypertension significantly increase the likelihood of stroke?
2. Is smoking status significantly associated with stroke occurrence?
3. Does average glucose level predict stroke risk?
4. Is BMI significantly associated with stroke likelihood?

---

## 📚 References

See full list in the [References Section](#references) of the report or thesis.

---

## 📌 Future Work

- Gender-specific SHAP analysis
- Real-time prediction dashboard (Streamlit)
- Integration with clinical interfaces
- Submission to healthcare ML workshops

---

## 🛠️ Requirements

```bash
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
