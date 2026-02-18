<div align="right">
  <a href="README.md">🇹🇷 Türkçe</a> | <a href="README_EN.md">🇬🇧 English</a>
</div>

# 🚀 <img src="https://flagcdn.com/w40/gb.png" width="32" alt="EN" style="vertical-align: middle;"> Data Science & Machine Learning Portfolio

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Visualization-purple?logo=plotly)
![Status](https://img.shields.io/badge/Status-Completed-success)

Welcome! This repository serves as a portfolio showcasing my competencies in Data Science and Machine Learning. It contains a collection of Python scripts demonstrating the end-to-end machine learning lifecycle: from data cleaning and feature engineering to model building and hyperparameter optimization.

All project files are located in the root directory for easy access.

## 🛠️ Tech Stack & Libraries
* **Languages:** Python, SQL
* **Data Analysis:** Pandas, NumPy, SciPy
* **Visualization:** Matplotlib, Seaborn, Plotly Express
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn, LazyPredict
* **Algorithms:** Linear/Logistic Regression, SVM, KNN, Decision Trees, Random Forest, AdaBoost, Gradient Boosting, Principal Component Analysis (PCA), K-Means, Hierarchical Clustering, DBSCAN
* **Optimization Tools:** Kneed (KneeLocator)

---

## 📸 Example Project Outputs & Visualizations

### 1. Model Performance & Feature Importance
![Feature Importance - LightGBM Titanic](image_3.png)
*Analysis of feature contributions in tree-based ensemble models (LightGBM) using the Titanic dataset.*

### 2. Exploratory Data Analysis (EDA) & Correlation
![Correlation Matrix](image_0.png)
*Investigating relationships between variables and checking for multicollinearity using a correlation heatmap.*

### 3. Geospatial Segmentation & Clustering Map
![World Map Clustering](image_1.png)
*Global socio-economic segmentation of countries mapped out interactively using K-Means and PCA.*

---

## 📂 Project Catalog

The projects are categorized below based on the problem type.

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
| `22_California_Housing_Price_Prediction_XGBoost.py`* | Predicting House Prices. | **Model Benchmarking** (Linear vs Trees), **XGBoost Hyperparameter Tuning**, Outlier Removal. |
| `24_Boston_Housing_Power_Transformation.py`* | Improving Linear Regression Performance. | **Yeo-Johnson & Box-Cox Transformations**, Handling Skewness. |
| `25_Medical_Cost_LGBM_BoxCox_Regression.py`* | Predicting Medical Insurance Costs. | **Box-Cox Target Transformation**, LightGBM, RandomizedSearchCV. |
| `26_California_Housing_Universal_Optimization.py`* | Automated Model Selection Pipeline. | **Universal Hyperparameter Tuning Function**, 9-Model Leaderboard, PowerTransformer. |

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
| `23_Titanic_LightGBM_XGBoost_Comparison.py`* | Titanic Survival Analysis. | **LightGBM vs XGBoost**, RandomizedSearchCV, Feature Importance Visualization. |

### 🔹 Advanced Machine Learning Applications
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `19_Car_Price_Prediction_Adaboost.py`* | Used Car Price Prediction. | Handling High Cardinality with **Frequency Encoding**, AdaBoost. |
| `15_Ensemble-Techniques.py`* | Income Bracket Prediction (>50K). | Handling **Imbalanced Data**, **Target Encoding**, Robust Scaler. |
| `10_Advanced_Logistic_Regression.py`* | Fraud & Cyber Attack Detection. | **Class Weights** optimization, One-vs-Rest / One-vs-One strategies. |
| `13_KNN-Health-Energy-Analysis.py`* | Health Risk & Energy Consumption. | Finding optimal 'K' with **Elbow Method**, Feature Scaling. |
| `9_SVM-Multi-Domain-Analysis.py`* | Email Spam, Loan Risk & Seismic. | SVM Kernel Comparison (Linear, RBF, Poly). |
| `27_Student_Performance_Dual_Pipeline.py`* | **Dual-Task Student Performance Analysis.** | **Dual-Task Modeling (Reg & Clf)**, RandomizedSearchCV, **Outlier Capping (IQR)**, Automated Model Leaderboard. |
| `28_PCA_Breast_Cancer_Analysis.py`* | PCA & Model Performance Analysis. | **Principal Component Analysis (PCA)**, Dimensionality Reduction Trade-offs, **StandardScaler**, Logistic vs Gradient Boosting. |

### 🔹 Data Science Fundamentals & Techniques
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `3_Data_Preprocessing_Manipulation.py`* | Comprehensive Toolkit for Data Science. | **SMOTE** (Imbalanced Data), **One-Hot/Ordinal Encoding**, Pandas Merging, Data Visualization. |
| `7_Python_SQL_Database_Basics.py`* | Student Database System with SQL. | SQLite, Table Creation (DDL), Data Manipulation (DML). |

### 🔹 Unsupervised Learning & Dimensionality Reduction
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `29_Clustering_Algorithms_Analysis.py` | Comparison of K-Means, Agglomerative, and DBSCAN. | Silhouette Analysis, Dendrograms, EPS Optimization. |
| `30_Country_Segmentation_PCA_KMeans.py` | Strategic Country Analysis for Resource Allocation. | PCA (3-Components), K-Means, Interative Geographic Visualization. |

---

## 📊 Methodology & Approach

In these projects, I followed the standard Data Science lifecycle:
1.  **EDA (Exploratory Data Analysis):** Understanding data distributions, outliers, and correlations.
2.  **Preprocessing:** IQR Outlier Capping, Dimensionality Reduction (PCA), Handling missing values, Scaling (Standard/Robust), and Encoding (One-Hot/Target/Frequency).
3.  **Model Selection:** Comparing Linear models vs. Tree-based Ensembles for supervised tasks, and evaluating K-Means vs. DBSCAN for clustering problems.
4.  **Optimization:** Using `GridSearchCV` and `RandomizedSearchCV` to fine-tune hyperparameters.
5.  **Evaluation:** Going beyond Accuracy; utilizing F1-Score, RMSE, R2 Score, and Confusion Matrices for robust evaluation.

---

## 💻 Installation & Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/AdnanSag/-Data-Science-Machine-Learning-Portfolio.git](https://github.com/AdnanSag/-Data-Science-Machine-Learning-Portfolio.git)
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

## 📬 Contact

If you'd like to talk about my projects or collaborate, feel free to reach out:
* **LinkedIn:** [Add Your Profile Here](https://www.linkedin.com/in/)
* **Kaggle:** [Add Your Profile Here](https://www.kaggle.com/)
* **Email:** yourname@email.com

*Created by Adnan Sag*
