import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def remove_highly_correlated_features(df, threshold=0.85):
    """
    Verilen eşik değerinin üzerinde korelasyona sahip sütunları tespit eder ve çıkarır.
    Multicollinearity problemini önlemek için kullanılır.
    """
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    
    if col_corr:
        print(f"Dropped Correlated Columns: {col_corr}")
        df = df.drop(col_corr, axis=1)
    else:
        print("No highly correlated features found.")
        
    return df

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """
    Gradient Boosting modelinin hangi özelliklere önem verdiğini görselleştirir.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- SENARYO 1: REGRESSION (CONCRETE DATA) ---
def run_concrete_regression(filepath):
    print("\n" + "="*50)
    print("SCENARIO 1: Concrete Strength Prediction (Regression)")
    print("="*50)
    
    try:
        df = pd.read_csv(filepath)
        X = df.drop("Strength", axis=1)
        y = df["Strength"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
        
        # Hyperparameter Tuning (RandomizedSearchCV)
        print("Running RandomizedSearchCV for Regressor...")
        params = {
            "n_estimators": [100, 150, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0]
        }
        
        gb_reg = GradientBoostingRegressor(random_state=42)
        rscv = RandomizedSearchCV(estimator=gb_reg, param_distributions=params, 
                                  n_iter=10, cv=3, n_jobs=-1, verbose=1)
        rscv.fit(X_train, y_train)
        
        print(f"Best Params: {rscv.best_params_}")
        
        # Tahmin ve Değerlendirme
        best_model = rscv.best_estimator_
        y_pred = best_model.predict(X_test)
        
        print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
        print(f"MAE      : {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"MSE      : {mean_squared_error(y_test, y_pred):.4f}")
        
        # Özellik Önem Düzeyi Grafiği
        plot_feature_importance(best_model, X.columns, "Concrete Feature Importance")

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Skipping regression task.")

# --- SENARYO 2: CLASSIFICATION (HEART DATA) ---
def run_heart_classification(filepath):
    print("\n" + "="*50)
    print("SCENARIO 2: Heart Disease Prediction (Classification)")
    print("="*50)
    
    try:
        df = pd.read_csv(filepath)
        
        # 1. Korelasyon Temizliği 
        df = remove_highly_correlated_features(df, threshold=0.85)
        
        X = df.drop("target", axis=1)
        y = df["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
        
        # 2. Hyperparameter Tuning (GridSearchCV)
        print("Running GridSearchCV for Classifier...")
        params = {
            "n_estimators": [100, 150],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 4],
            "subsample": [0.8, 1.0]
        }
        
        gb_clf = GradientBoostingClassifier(random_state=42)
        grid = GridSearchCV(estimator=gb_clf, param_grid=params, cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        print(f"Best Params: {grid.best_params_}")
        
        # 3. Değerlendirme
        best_clf = grid.best_estimator_
        y_pred = best_clf.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds', cbar=False)
        plt.title("Heart Disease Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Skipping classification task.")

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    # Regression Görevi
    run_concrete_regression("datasets/18-concrete_data.csv")
    
    # Classification Görevi
    run_heart_classification("datasets/19-heart.csv")