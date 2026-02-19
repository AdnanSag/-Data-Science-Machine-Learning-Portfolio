import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Uyarıları filtrele (Bazı kombinasyonlar uyarı verebilir)
warnings.filterwarnings('ignore')

# Görselleştirme Ayarları
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_and_preprocess_data(filepath):
    """
    Veriyi yükler, kategorik değişkenleri (Dummy) dönüştürür 
    ve eğitim/test setlerine ayırır.
    """
    df = pd.read_csv(filepath)
    
    # Kategorik değişkenlerin sayısal hale getirilmesi (One-Hot Encoding)
    # drop_first=True, dummy variable tuzağını (multicollinearity) önler.
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop("subscribed", axis=1)
    y = df["subscribed"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Logistic Regression mesafe tabanlı çalıştığı için verileri standartlaştırır.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def evaluate_model(model, X_test, y_test, title="Model Evaluation"):
    """
    Model performansını metrikler ve confusion matrix ile görselleştirir.
    """
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    print(f"\n--- {title} ---")
    print(f"Accuracy Score: {score:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Görselleştirme
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{title} - Confusion Matrix')
    plt.show()

def run_hyperparameter_tuning(X_train, y_train):
    """
    GridSearchCV ve RandomizedSearchCV kullanarak en iyi hiperparametreleri bulur.
    """
    # Not: Her solver her penalty türünü desteklemez. 
    # Hata almamak için parametreleri gruplandırıyoruz.
    model = LogisticRegression(max_iter=1000)
    
    # Parametre Grid'i (Uyumluluk gözetilerek hazırlandı)
    param_grid = [
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [100, 10, 1, 0.1, 0.01]},
        {'solver': ['lbfgs', 'newton-cg'], 'penalty': ['l2'], 'C': [100, 10, 1, 0.1, 0.01]}
    ]
    
    cv = StratifiedKFold(n_splits=5) 
    
    # 1. Grid Search CV
    print("Running GridSearchCV...")
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best Grid Params: {grid.best_params_}")
    
    # 2. Random Search CV
    print("\nRunning RandomizedSearchCV...")
    # Random search için düz sözlük yapısı kullanılabilir (hata yönetimiyle)
    random_params = {
        'penalty': ['l2'], # lbfgs ve newton-cg ile uyumlu olması için l2 seçildi
        'C': [100, 10, 1, 0.1, 0.01],
        'solver': ['lbfgs', 'liblinear', 'newton-cg']
    }
    
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_params, 
                                  cv=5, n_iter=10, scoring="accuracy", n_jobs=-1, random_state=42)
    randomcv.fit(X_train, y_train)
    print(f"Best Random Params: {randomcv.best_params_}")
    
    return grid.best_estimator_

# --- ANA UYGULAMA AKIŞI ---

# 1. Veri Hazırlığı
# Not: Dosya yolunu kendi sistemine göre güncellemelisin
X_train, X_test, y_train, y_test = load_and_preprocess_data("datasets/6-bank_customers.csv")

# 2. Ölçeklendirme (Scaling)
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# 3. Temel Model (Baseline)
print("Training Baseline Model...")
base_model = LogisticRegression()
base_model.fit(X_train_scaled, y_train)
evaluate_model(base_model, X_test_scaled, y_test, title="Baseline Logistic Regression")

# 4. Hiperparametre Optimizasyonu
best_model = run_hyperparameter_tuning(X_train_scaled, y_train)

# 5. Optimize Edilmiş Modelin Değerlendirilmesi
evaluate_model(best_model, X_test_scaled, y_test, title="Tuned Logistic Regression")