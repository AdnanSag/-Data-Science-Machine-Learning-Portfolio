# 🚀 Data Science & Machine Learning Portfolio

Welcome! This repository serves as a portfolio showcasing my competencies in Data Science and Machine Learning. It contains a collection of Python scripts demonstrating the end-to-end machine learning lifecycle: from data cleaning and feature engineering to model building and hyperparameter optimization.

All project files are located in the root directory for easy access.

## 🛠️ Tech Stack & Libraries
* **Languages:** Python, SQL
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost, LazyPredict
* **Algorithms:** Linear/Logistic Regression, SVM, KNN, Decision Trees, Random Forest, AdaBoost, Gradient Boosting

---

## 📂 Project Catalog

The projects are categorized below based on the problem type (Regression, Classification, etc.).

### 🔹 Regression Projects (Predicting Continuous Values)
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `1_Doğrusal_Regresyon_StudyHours_ExamScore).py`* | Predicting Student Exam Scores. | **Simple vs Multiple Linear Regression**, Feature Scaling, Prediction on new data. |
| `2_Polinom_Regresyon_CustomerSatisfaction.py`* | Analyzing Model Complexity. | **Polynomial Regression**, Visualizing **Overfitting**, Scikit-Learn **Pipelines**. |
| `4_Regularization_Lasso_vs_Linear.py`* | Regularization & Feature Selection. | **Lasso (L1) vs Linear Regression**, Importance of **Scaling** in regularization. |
| `5_FWI_Regression_AutoML_Analysis.py`* | Predicting Fire Weather Index (FWI). | Data Cleaning, Correlation Analysis, AutoML. |
| `11_SVM-Diamond-Price-Prediction-Regression.py`* | Estimating Diamond Prices. | **SVR vs Linear Regression** trade-off, Outlier Removal. |
| `18_Gym-Crowdedness-Prediction-Analysis.py`* | Predicting Gym Crowdedness. | **Random Forest** performance on non-linear time-series data. |
| `6_WWII_Weather_Regression_Analysis.py`* | WWII Weather Temperature Analysis. | **LassoCV** (Regularization), Date/Time feature extraction. |

### 🔹 Classification Projects (Predicting Categories)
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `21_Stellar-Classification-XGBoost.py`* | Classifying Stars, Galaxies, and Quasars. | **XGBoost**, Feature Importance (Redshift), Multiclass Classification. |
| `16_Diabetes-Prediction-ML-Pipeline.py`* | End-to-End Diabetes Diagnosis Pipeline. | **Preventing Data Leakage** (Median Imputation), Model Comparison. |
| `17_Diabetes-Ensemble-Learning-AdaBoost.py`* | Diabetes Prediction (Ensemble). | **AdaBoost** algorithm and manual parameter tuning. |
| `8_Logistic-Regression-Hyperparameter-Tuning.py`* | Bank Customer Churn Analysis. | Logistic Regression, Scaling, Confusion Matrix Analysis. |
| `12_Naive_bayes-Iris-Species-Classification-Comparison.py`* | Iris Species Classification. | **Naive Bayes vs SVM** (Kernel Trick comparison). |
| `14_Decision-Tree-Classification-Projects.py`* | Car Evaluation & Iris Analysis. | **Ordinal Encoding**, Decision Tree Visualization. |
| `20_Gradient-Boosting-Advanced-Analysis.py`* | Heart Disease & Concrete Strength. | **Gradient Boosting** (Reg & Clf), Correlation Filtering. |

### 🔹 Advanced Machine Learning Applications
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `19_Car_Price_Prediction_Adaboost.py`* | Used Car Price Prediction. | Handling High Cardinality with **Frequency Encoding**, AdaBoost. |
| `15_Ensemble-Techniques.py`* | Income Bracket Prediction (>50K). | Handling **Imbalanced Data**, **Target Encoding**, Robust Scaler. |
| `10_Advanced_Logistic_Regression.py`* | Fraud & Cyber Attack Detection. | **Class Weights** optimization, One-vs-Rest / One-vs-One strategies. |
| `13_KNN-Health-Energy-Analysis.py`* | Health Risk & Energy Consumption. | Finding optimal 'K' with **Elbow Method**, Feature Scaling. |
| `9_SVM-Multi-Domain-Analysis.py`* | Email Spam, Loan Risk & Seismic. | SVM Kernel Comparison (Linear, RBF, Poly). |

### 🔹 Data Science Fundamentals & Techniques
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `3_Data_Preprocessing_Manipulation.py`* | Comprehensive Toolkit for Data Science. | **SMOTE** (Imbalanced Data), **One-Hot/Ordinal Encoding**, Pandas Merging, Data Visualization. |
| `7_Python_SQL_Database_Basics.py`* | Student Database System with SQL. | SQLite, Table Creation (DDL), Data Manipulation (DML). |
---

## 📊 Methodology & Approach

In these projects, I followed the standard Data Science lifecycle:
1.  **EDA (Exploratory Data Analysis):** Understanding data distributions, outliers, and correlations.
2.  **Preprocessing:** Handling missing values (strictly preventing leakage), Scaling (Standard/Robust), and Encoding (One-Hot/Target/Frequency).
3.  **Model Selection:** Comparing Linear models against Tree-based Ensembles based on data linearity.
4.  **Optimization:** Using `GridSearchCV` and `RandomizedSearchCV` to fine-tune hyperparameters.
5.  **Evaluation:** Going beyond Accuracy; utilizing F1-Score, RMSE, R2 Score, and Confusion Matrices for robust evaluation.

---

## 💻 Installation & Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/AdnanSag/-Data-Science-Machine-Learning-Portfolio.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run a specific script:
    ```bash
    python 16_Diabetes-Prediction-ML-Pipeline.py
    ```
---
*Created by Adnan Sag*
