"""
NAIVE BAYES MODELLERİ HAKKINDA TEORİK NOTLAR:

1. Bernoulli Naive Bayes:
   - Özellikler ikili (binary) ise (Evet/Hayır, 0/1) kullanılır.
   - Örnek: Bir kelimenin metinde geçip geçmediği.
   - Standard Scaler genellikle gerekmez.

2. Multinomial Naive Bayes:
   - Özellikler ayrık sayılardan oluşuyorsa (Frekans, Kelime Sayısı) kullanılır.
   - Örnek: Bir kelimenin metinde kaç kez geçtiği.
   - Standard Scaler genellikle gerekmez.

3. Gaussian Naive Bayes:
   - Özellikler sürekli (continuous) değerlerse ve normal dağılıma yakınsa kullanılır.
   - Örnek: İris veri setindeki yaprak genişliği (cm cinsinden).
   - Standard Scaler performans artışı için kullanılabilir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Uyarıları filtrele
warnings.filterwarnings('ignore')

# Görselleştirme Ayarları
sns.set_style("whitegrid")

def load_and_preprocess_data(filepath):
    """
    Veriyi yükler, gereksiz sütunları atar, hedef değişkeni encode eder
    ve özellikler için standardizasyon (Scaling) uygular.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Gereksiz ID sütunu varsa temizle
        if "Id" in df.columns:
            df = df.drop("Id", axis=1)
            
        # Hedef değişkeni sayısal hale getir (Label Encoding)
        label_encoder = LabelEncoder()
        df["Species"] = label_encoder.fit_transform(df["Species"])
        
        X = df.drop("Species", axis=1)
        y = df["Species"]
        
        # Eğitim ve Test setlerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
        
        # Standardizasyon (SVM ve GaussianNB için önemlidir)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder
        
    except FileNotFoundError:
        print(f"Hata: '{filepath}' dosyası bulunamadı.")
        return None, None, None, None, None

def evaluate_model(model, X_test, y_test, model_name):
    """
    Modeli test eder ve sonuçları raporlar.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*40}")
    print(f"MODEL: {model_name}")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Görselleştirme
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def run_svm_comparison(X_train, X_test, y_train, y_test):
    """
    Farklı çekirdek (kernel) fonksiyonları ile SVM modellerini karşılaştırır.
    """
    kernels = ["linear", "rbf", "sigmoid", "poly"]
    
    for k in kernels:
        model = SVC(kernel=k, random_state=15)
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, f"SVM ({k.capitalize()} Kernel)")

def run_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """
    SVM için GridSearchCV kullanarak en iyi parametreleri bulur.
    """
    print(f"\n{'='*40}")
    print("OPTIMIZATION: Hyperparameter Tuning (GridSearchCV)")
    print(f"{'='*40}")
    
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"]
    }
    
    grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid.best_params_}")
    
    best_model = grid.best_estimator_
    evaluate_model(best_model, X_test, y_test, "Optimized SVM (Best Params)")

# --- ANA UYGULAMA AKIŞI ---

if __name__ == "__main__":
    # 1. Veri Hazırlığı
    X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data("datasets/11-iris.csv")
    
    if X_train is not None:
        # 2. Gaussian Naive Bayes Modeli
        # Iris verisi sürekli (continuous) olduğu için GaussianNB tercih edilir.
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        evaluate_model(gnb, X_test, y_test, "Gaussian Naive Bayes")
        
        # 3. SVM Modellerinin Karşılaştırılması
        run_svm_comparison(X_train, X_test, y_train, y_test)
        
        # 4. Hiperparametre Optimizasyonu
        run_hyperparameter_tuning(X_train, y_train, X_test, y_test)