# Diabetes Prediction Using Support Vector Machine (SVM)

## Project Overview
This project aims to build a machine learning model that can predict whether a person has diabetes based on various health parameters. Early prediction of diabetes can help in timely treatment and management of the disease, potentially reducing health risks and complications.

## Dataset Description
The dataset used is the **PIMA Indians Diabetes Database**, a well-known dataset in the machine learning community.

- **Source:** The dataset is publicly available and commonly used for diabetes prediction tasks.
- **Number of Samples:** 768
- **Features (Input Variables):**  
  - Pregnancies: Number of times pregnant  
  - Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance test  
  - BloodPressure: Diastolic blood pressure (mm Hg)  
  - SkinThickness: Triceps skin fold thickness (mm)  
  - Insulin: 2-Hour serum insulin (mu U/ml)  
  - BMI: Body mass index (weight in kg/(height in m)^2)  
  - DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)  
  - Age: Age in years  

- **Target Variable:**  
  - Outcome: Whether the patient has diabetes (1) or not (0)

## Problem Statement
Diabetes is a chronic condition affecting millions worldwide. The goal is to develop a reliable predictive model using health data that can classify individuals as diabetic or non-diabetic. This prediction task helps in early diagnosis, leading to better healthcare outcomes.

## Methods
### Data Preprocessing
- Loaded the dataset and performed initial exploratory data analysis.
- Handled missing or zero values in certain columns that are physiologically implausible (e.g., zero values for BloodPressure or BMI).
- Standardized the feature variables using `StandardScaler` to bring all values into a common range.

### Model Building
- Split the dataset into training and testing sets (commonly 80% training, 20% testing).
- Implemented a Support Vector Machine (SVM) classifier with a linear kernel.
- Trained the model on the training data and evaluated it on the testing data.

### Evaluation Metrics
- Used accuracy as the primary metric.
- Recommended also considering precision, recall, F1-score, and AUC-ROC for a comprehensive evaluation.

## Results
- The SVM classifier achieved an accuracy of approximately [Insert Accuracy Here] on the test set.
- The model successfully classified diabetic and non-diabetic cases but has room for improvement in sensitivity.
- Challenges included handling missing or skewed data distributions and potential class imbalance.

## Future Work and Improvements
- Robust imputation methods for missing data.
- Hyperparameter tuning (e.g., regularization parameter, kernel type).
- Exploration of other machine learning models for benchmarking (e.g., Random Forest, Logistic Regression).
- Use of explainability techniques like SHAP or LIME to interpret model predictions.
- Additional evaluation metrics to better understand model performance on imbalanced data.

## Tools and Libraries
- Python 3
- Pandas, NumPy for data handling
- Scikit-learn for modeling and evaluation
- Jupyter Notebook for development environment

## References
- [PIMA Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- Scikit-learn Documentation: https://scikit-learn.org
- Support Vector Machine tutorial: https://scikit-learn.org/stable/modules/svm.html

---

*This project is developed by Hrishav Sahu for educational purposes.*
