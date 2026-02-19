import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def evaluate_model(y_test, y_pred, title="Model Evaluation"):
    """
    Model performansını standart metriklerle raporlar.
    """
    print(f"\n--- {title} ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# --- SENARYO 1: ÇOK SINIFLI SINIFLANDIRMA (One-vs-One & One-vs-Rest) ---
def run_cyber_attack_analysis(filepath):
    """
    Çok sınıflı (Multiclass) siber saldırı verilerini OneVsOne ve OneVsRest
    stratejileri ile analiz eder.
    """
    print("\n" + "="*50)
    print("SCENARIO 1: Multiclass Cyber Attack Classification")
    print("="*50)
    
    try:
        df = pd.read_csv(filepath)
        X = df.drop("attack_type", axis=1)
        y = df["attack_type"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
        
        # Scaling (Lojistik Regresyon için önemlidir)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. One-vs-One (OVO) Strategy
        # Her sınıf çifti için bir model eğitir (Örn: A vs B, A vs C, B vs C...)
        print("Training One-vs-One Classifier...")
        ovo_model = OneVsOneClassifier(LogisticRegression(max_iter=1000))
        ovo_model.fit(X_train_scaled, y_train)
        y_pred_ovo = ovo_model.predict(X_test_scaled)
        evaluate_model(y_test, y_pred_ovo, title="One-vs-One Strategy")
        
        # 2. One-vs-Rest (OVR) Strategy
        # Her sınıf için "Bu sınıf vs Diğerleri" şeklinde model eğitir.
        print("Training One-vs-Rest Classifier...")
        ovr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        ovr_model.fit(X_train_scaled, y_train)
        y_pred_ovr = ovr_model.predict(X_test_scaled)
        evaluate_model(y_test, y_pred_ovr, title="One-vs-Rest Strategy")
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Skipping scenario.")

# --- SENARYO 2: DENGESİZ VERİ SETİ (Imbalanced Dataset & Class Weights) ---
def run_fraud_detection_analysis(filepath):
    """
    Dengesiz (Imbalanced) dolandırıcılık verisi için Class Weight optimizasyonu yapar.
    """
    print("\n" + "="*50)
    print("SCENARIO 2: Fraud Detection (Imbalanced Data Handling)")
    print("="*50)
    
    try:
        df = pd.read_csv(filepath)
        X = df.drop("is_fraud", axis=1)
        y = df["is_fraud"]
        
        # Veri dağılımını görselleştirme
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df["transaction_amount"], y=df["transaction_risk_score"], hue=df["is_fraud"], palette="coolwarm")
        plt.title("Transaction Amount vs Risk Score (Fraud Distribution)")
        plt.show()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Class Weight Kombinasyonları (Dengesiz veri için kritik adım)
        weights = [1, 10, 50, 100]
        class_weight_options = [{0: 1, 1: w} for w in weights] # 0 sınıfına 1, 1 sınıfına (Fraud) yüksek ağırlık
        
        # Hata almamak için Parametre Grid'ini Solver yeteneklerine göre ayırıyoruz
        param_grid = [
            # L1 destekleyenler
            {
                'solver': ['liblinear', 'saga'], 
                'penalty': ['l1'], 
                'C': [10, 1, 0.1], 
                'class_weight': class_weight_options
            },
            # L2 destekleyenler
            {
                'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag'], 
                'penalty': ['l2'], 
                'C': [10, 1, 0.1], 
                'class_weight': class_weight_options
            }
        ]
        
        print("Running GridSearchCV for Imbalanced Data...")
        cv = StratifiedKFold(n_splits=5)
        model = LogisticRegression(max_iter=1000)
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        
        print(f"Best Class Weights: {grid.best_params_['class_weight']}")
        print(f"Best Parameters: {grid.best_params_}")
        
        y_pred = grid.predict(X_test_scaled)
        evaluate_model(y_test, y_pred, title="Optimized Logistic Regression (Fraud)")

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Skipping scenario.")

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    # Senaryo 1: Çoklu Sınıflandırma
    run_cyber_attack_analysis("datasets/7-cyber_attack_data.csv")
    
    # Senaryo 2: Dengesiz Veri Analizi
    run_fraud_detection_analysis("datasets/8-fraud_detection.csv")