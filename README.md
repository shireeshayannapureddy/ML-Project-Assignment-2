# Heart Disease Prediction – Machine Learning Assignment
---
## a.Problem Statement

Heart disease is one of the leading causes of mortality worldwide, and early diagnosis plays a crucial role in reducing health risks and improving patient outcomes. Traditional diagnostic methods often rely on manual analysis of multiple medical parameters, which can be time-consuming and prone to human error.

The objective of this project is to develop a Machine Learning–based heart disease prediction system that can accurately classify whether a patient is likely to have heart disease based on clinical and demographic features. Multiple classification algorithms are implemented, evaluated, and compared to identify the most effective model.

Additionally, a Streamlit-based web application is developed to allow users to upload patient data, select different machine learning models, and visualize predictions along with evaluation metrics and confusion matrices.

## b. Dataset Description

The dataset used for this project is a Heart Disease dataset obtained from a public repository - Kaggle.
It contains medical and demographic attributes of patients that are used to predict the presence of heart disease.

- Number of instances: 1888
- Number of features: 13
- Target variable: target  
  - 0 → No Heart Disease  
  - 1 → Heart Disease

The dataset satisfies the minimum requirements of having more than 500 instances and more than 12 features.

---

## c. Machine Learning Models Implemented

The following six classification models were implemented using the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## d. Model Evaluation Results

All models were evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### e.Model Performance Summary

| ML Model Name        | Accuracy | AUC    | Precision | Recall | F1 Score | MCC   |
|---------------------|----------|--------|-----------|--------|----------|-------|
| Logistic Regression | 0.7275   | 0.8324 | 0.7067    | 0.8112 | 0.7553   | 0.4566 |
| Decision Tree       | 0.9762   | 0.9763 | 0.9795    | 0.9745 | 0.9770   | 0.9523 |
| KNN                 | 0.9127   | 0.9625 | 0.9095    | 0.9235 | 0.9165   | 0.8252 |
| Naive Bayes         | 0.7090   | 0.7898 | 0.6822    | 0.8214 | 0.7454   | 0.4223 |
| Random Forest       | 0.9788   | 0.9985 | 0.9747    | 0.9847 | 0.9797   | 0.9577 |
| XGBoost             | 0.9709   | 0.9982 | 0.9602    | 0.9847 | 0.9723   | 0.9420 |

---

## f.Observations on Model Performance

| ML Model Name        | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Performs reasonably well and provides good baseline results but struggles with complex patterns. |
| Decision Tree       | Achieves very high accuracy but may be prone to overfitting on unseen data. |
| KNN                 | Performs well with good balance between bias and variance but is sensitive to data scaling. |
| Naive Bayes         | Simple and fast model but shows lower accuracy due to strong independence assumptions. |
| Random Forest       | Best performing model with excellent accuracy, AUC, and MCC due to ensemble learning. |
| XGBoost             | Performs nearly as well as Random Forest with strong generalization and high AUC score. |

---

## g. Streamlit Application

A Streamlit web application was developed to deploy the trained models with the following features:
- CSV dataset upload (test data only)
- Model selection dropdown
- Display of evaluation metrics
- Classification report
- Confusion matrix visualization

---

## h. Conclusion

Among all the models implemented, Random Forest and XGBoost achieved the best overall performance.  
Ensemble methods proved to be more robust and accurate for heart disease prediction compared to individual classifiers.

This project demonstrates an end-to-end machine learning pipeline including data preprocessing, model training, evaluation, and deployment.
