import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def scaling(X_train, X_test, columns):
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)
    
    X_train_sca = pd.DataFrame(X_train_arr, columns=columns)
    X_test_sca = pd.DataFrame(X_test_arr, columns=columns)
    
    return X_train_sca, X_test_sca
    
def pca_func(X_train, X_test, n):
    pca = PCA(n_components=n)
    
    X_train_pca_arr = pca.fit_transform(X_train)
    X_test_pca_arr = pca.transform(X_test) 
    
    col_names = [f"PC {i+1}" for i in range(n)]
    
    X_train_pca = pd.DataFrame(X_train_pca_arr, columns=col_names)
    X_test_pca = pd.DataFrame(X_test_pca_arr, columns=col_names)
    
    return X_train_pca, X_test_pca

def model(X_train, y_train, X_test, y_test):
    logistic = LogisticRegression()
    gbc = GradientBoostingClassifier()
    
    logistic.fit(X_train, y_train)
    gbc.fit(X_train, y_train)
    
    print("\nLogistic Regression Results:")
    y_pred_log = logistic.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

    print("-" * 30)
    print("Gradient Boosting Results:")
    y_pred_gbc = gbc.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gbc):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gbc))

def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    X = df.drop('target', axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15, test_size=0.25)
    columns = data.feature_names
    
    # 1. Scaling İşlemi
    print("--- SCALED DATA RESULTS ---")
    X_train_sca, X_test_sca = scaling(X_train, X_test, columns)
    model(X_train_sca, y_train, X_test_sca, y_test)
    
    # 2. PCA İşlemi
    print("\n--- PCA RESULTS (n=4) ---")
    X_train_pca, X_test_pca = pca_func(X_train_sca, X_test_sca, 2) 
    model(X_train_pca, y_train, X_test_pca, y_test)

if __name__ == "__main__":
    main()

"""
 --> n=4

--- SCALED DATA RESULTS ---

Logistic Regression Results:
Accuracy: 0.9580
Confusion Matrix:
 [[49  4]
 [ 2 88]]
------------------------------
Gradient Boosting Results:
Accuracy: 0.9441
Confusion Matrix:
 [[46  7]
 [ 1 89]]

--- PCA RESULTS (n=4) ---

Logistic Regression Results:
Accuracy: 0.9371
Confusion Matrix:
 [[48  5]
 [ 4 86]]
------------------------------
Gradient Boosting Results:
Accuracy: 0.9161
Confusion Matrix:
 [[48  5]
 [ 7 83]]
 
 
 --> n=2
 --- SCALED DATA RESULTS ---

Logistic Regression Results:
Accuracy: 0.9580
Confusion Matrix:
 [[49  4]
 [ 2 88]]
------------------------------
Gradient Boosting Results:
Accuracy: 0.9510
Confusion Matrix:
 [[47  6]
 [ 1 89]]

--- PCA RESULTS (n=4) ---

Logistic Regression Results:
Accuracy: 0.9161
Confusion Matrix:
 [[47  6]
 [ 6 84]]
------------------------------
Gradient Boosting Results:
Accuracy: 0.8811
Confusion Matrix:
 [[45  8]
 [ 9 81]]
"""