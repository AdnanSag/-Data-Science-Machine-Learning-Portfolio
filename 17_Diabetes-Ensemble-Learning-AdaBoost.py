import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

def load_and_clean_data(filepath):
    """
    Veriyi yükler, 0 olan (eksik) değerleri antrenman medyanı ile doldurur.
    Veri sızıntısını (Data Leakage) önlemek için medyan hesabı sadece train setinden yapılır.
    """
    try:
        df = pd.read_csv(filepath)
        X = df.drop("Outcome", axis=1)
        y = df['Outcome']
        
        # Eğitim ve Test setlerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
        
        # 0 Değerlerini (Eksik Veri) İşleme
        columns_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        
        for col in columns_to_check:
            # Sadece 0 olmayan değerlerin medyanını al
            median_val = X_train[X_train[col] != 0][col].median()
            
            # Train ve Test setlerini doldur
            X_train[col] = X_train[col].replace(0, median_val)
            X_test[col] = X_test[col].replace(0, median_val)
            
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None, None, None, None

def scale_data(X_train, X_test):
    """
    Verileri standartlaştırır (StandardScaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def evaluate_model(model, X_test, y_test, title="Model Evaluation"):
    """
    Model performansını raporlar ve Confusion Matrix görselleştirir.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"Accuracy Score: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Görselleştirme
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title(f'{title} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def run_hyperparameter_tuning(X_train, y_train):
    """
    GridSearchCV kullanarak AdaBoost için en iyi parametreleri bulur.
    """
    print("\nStarting Hyperparameter Tuning (GridSearchCV)...")
    
    param_grid = {
        "n_estimators": [50, 70, 100, 120, 150, 200],
        "learning_rate": [0.001, 0.01, 0.1, 1, 1.5]
    }
    
    grid = GridSearchCV(estimator=AdaBoostClassifier(random_state=42), 
                        param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_

# --- ANA UYGULAMA AKIŞI ---

if __name__ == "__main__":
    # 1. Veri Hazırlığı
    X_train, X_test, y_train, y_test = load_and_clean_data("datasets/16-diabetes.csv")
    
    if X_train is not None:
        # 2. Ölçeklendirme
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        
        # 3. Temel Model (Baseline AdaBoost)
        print("Training Baseline AdaBoost...")
        baseline_ada = AdaBoostClassifier(random_state=15)
        baseline_ada.fit(X_train_scaled, y_train)
        evaluate_model(baseline_ada, X_test_scaled, y_test, "Baseline AdaBoost")
        
        # 4. Optimizasyon (Tuning)
        best_ada = run_hyperparameter_tuning(X_train_scaled, y_train)
        
        # 5. Final Model Değerlendirmesi
        evaluate_model(best_ada, X_test_scaled, y_test, "Optimized AdaBoost")