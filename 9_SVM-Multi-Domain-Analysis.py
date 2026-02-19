import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Uyarıları filtrele
warnings.filterwarnings('ignore')

# Görselleştirme Ayarları
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

def evaluate_model(y_test, y_pred, title="Model Performance"):
    """
    Model sonuçlarını metrikler ve confusion matrix ile görselleştirir.
    """
    print(f"\n--- {title} ---")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def preprocess_data(filepath, target_col):
    """
    Veriyi yükler, hedef değişkeni ayırır, train/test split yapar ve ölçeklendirir (Scaling).
    Not: SVM mesafe tabanlı olduğu için Scaling kritik öneme sahiptir.
    """
    try:
        df = pd.read_csv(filepath)
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
        
        # SVM için Standardizasyon
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Skipping this case.")
        return None, None, None, None

# --- CASE 1: EMAIL CLASSIFICATION (Linear Kernel) ---
def run_email_classification():
    print("\n" + "="*40)
    print("CASE 1: Email Classification (Spam Detection)")
    print("="*40)
    
    X_train, X_test, y_train, y_test = preprocess_data("datasets/9-email_classification_svm.csv", "email_type")
    
    if X_train is not None:
        model = SVC(kernel="linear", random_state=15)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        evaluate_model(y_test, y_pred, title="SVM Linear Kernel (Emails)")

# --- CASE 2: LOAN RISK ANALYSIS (Kernel Comparison & Tuning) ---
def run_loan_risk_analysis():
    print("\n" + "="*40)
    print("CASE 2: Loan Risk Analysis (Kernel Comparison)")
    print("="*40)
    
    X_train, X_test, y_train, y_test = preprocess_data("datasets/9-loan_risk_svm.csv", "loan_risk")
    
    if X_train is not None:
        # Farklı çekirdek (kernel) fonksiyonlarının karşılaştırılması
        kernels = ["linear", "rbf", "sigmoid", "poly"]
        
        for k in kernels:
            model = SVC(kernel=k, random_state=15)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Kernel: {k.ljust(10)} | Accuracy: {acc:.4f}")

        # Hyperparameter Tuning (GridSearch)
        print("\nRunning GridSearchCV for RBF Kernel...")
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf"]
        }
        
        grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print(f"Best Params: {grid.best_params_}")
        y_pred_grid = grid.predict(X_test)
        evaluate_model(y_test, y_pred_grid, title="Optimized SVM (Loan Risk)")

# --- CASE 3: SEISMIC ACTIVITY (RBF Kernel) ---
def run_seismic_activity_analysis():
    print("\n" + "="*40)
    print("CASE 3: Seismic Activity Detection (Non-Linear)")
    print("="*40)
    
    X_train, X_test, y_train, y_test = preprocess_data("datasets/9-seismic_activity_svm.csv", "seismic_event_detected")
    
    if X_train is not None:
        # RBF Kernel karmaşık ve doğrusal olmayan düzlemleri ayırmak için idealdir
        model = SVC(kernel="rbf", random_state=15)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        evaluate_model(y_test, y_pred, title="SVM RBF Kernel (Seismic)")

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    # Fonksiyonları sırayla çalıştır
    run_email_classification()
    run_loan_risk_analysis()
    run_seismic_activity_analysis()