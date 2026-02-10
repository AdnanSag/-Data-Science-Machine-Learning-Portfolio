"""
Student Performance ML Pipeline
--------------------------------------------------------------------------
Bu script, öğrenci verilerini analiz eder, temizler ve hem regresyon hem de 
sınıflandırma modelleri kullanarak sınav başarısını tahmin eder.
"""

# =========================
# 1. KÜTÜPHANELER
# =========================

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri Ön İşleme
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Regresyon Modelleri
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Sınıflandırma Modelleri
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrikler
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, confusion_matrix,
    classification_report, f1_score
)

# =========================
# 2. AYARLAR (CONFIG)
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.2
PASS_THRESHOLD = 60  # Geçme notu sınırı

# Aykırı değer baskılaması yapılacak sütunlar
OUTLIER_COLS = [
    "social_media_hours",
    "netflix_hours",
    "study_hours_per_day"
]

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# =========================
# 3. YARDIMCI FONKSİYONLAR
# =========================

def cap_outliers(df, col):
    """IQR yöntemi ile aykırı değerleri baskılar."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    
    # Alt ve üst sınırların dışındakileri sınırlara eşitle (Clipping)
    df[col] = np.clip(df[col], low, high)
    return df

def preprocess_data(df):
    """Veri temizleme, doldurma ve encoding işlemlerini yapar."""
    print(">> Veri ön işleme (Preprocessing) başlatıldı...")

    # ID sütunu varsa kaldır
    if "student_id" in df.columns:
        df = df.drop("student_id", axis=1)

    # 1. Aykırı Değer Baskılama
    for col in OUTLIER_COLS:
        if col in df.columns:
            df = cap_outliers(df, col)

    # 2. Eksik Veri Doldurma (Mode ile)
    if df["parental_education_level"].isnull().sum() > 0:
        df["parental_education_level"].fillna(
            df["parental_education_level"].mode()[0],
            inplace=True
        )

    # 3. Ordinal Encoding (Sıralı Kategorikler)
    diet_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
    internet_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2}

    # Haritalamayı uygula (Eğer sütunlar varsa)
    if "diet_quality" in df.columns: df["diet_quality"] = df["diet_quality"].map(diet_map)
    if "internet_quality" in df.columns: df["internet_quality"] = df["internet_quality"].map(internet_map)
    if "parental_education_level" in df.columns: df["parental_education_level"] = df["parental_education_level"].map(edu_map)

    # 4. Binary Encoding (Evet/Hayır)
    binary_cols = ["part_time_job", "extracurricular_participation"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1})

    # 5. One-Hot Encoding (Nominal Kategorikler - Cinsiyet gibi)
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)

    print(">> Ön işleme tamamlandı.")
    return df

def plot_distributions(df):
    """Sayısal değişkenlerin dağılımını çizer."""
    print(">> Dağılım grafikleri hazırlanıyor...")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    
    # Çok fazla sütun varsa hata vermemesi için
    cols_to_plot = [c for c in num_cols if c != "exam_score"]
    
    n_cols = 3
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], kde=True, color="skyblue")
        plt.title(f"{col} Dağılımı")
    
    plt.tight_layout()
    plt.show()
    print(">> Grafik kapatıldı, işleme devam ediliyor.")

# =========================
# 4. MODEL TUNING MOTORU
# =========================

def run_model_tuning(X_train, y_train, X_test, y_test, models, task="regression"):
    """
    Verilen modeller üzerinde RandomizedSearchCV ile hiperparametre araması yapar.
    """
    results = []
    best_model = None
    best_score = -np.inf
    best_name = ""
    
    # Görev tipine göre skorlama metriği seçimi
    scoring_metric = "r2" if task == "regression" else "accuracy"

    print(f"\n{'='*10} {task.upper()} MODELLERİ EĞİTİLİYOR {'='*10}")

    for entry in models:
        name = entry["name"]
        model = entry["model"]
        params = entry["params"]

        print(f" -> {name} optimize ediliyor...")

        # Hızlı sonuç için n_iter=10
        search = RandomizedSearchCV(
            model,
            params,
            n_iter=10, 
            cv=3,
            scoring=scoring_metric,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )

        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        preds = best_estimator.predict(X_test)

        if task == "regression":
            score = r2_score(y_test, preds)
            metric2 = np.sqrt(mean_squared_error(y_test, preds)) # RMSE
            results.append({"Model": name, "R2 Score": score, "RMSE": metric2})
        else:
            score = accuracy_score(y_test, preds)
            metric2 = f1_score(y_test, preds)
            results.append({"Model": name, "Accuracy": score, "F1 Score": metric2})

        # En iyi modeli kaydet
        if score > best_score:
            best_score = score
            best_model = best_estimator
            best_name = name

    # Sonuçları DataFrame'e çevir ve sırala
    df_results = pd.DataFrame(results).sort_values(
        by="R2 Score" if task == "regression" else "Accuracy", 
        ascending=False
    )

    print(f"\n🏆 En Başarılı {task.capitalize()} Modeli: {best_name} (Skor: {best_score:.4f})")
    return df_results, best_model, best_name

# =========================
# 5. MODEL TANIMLARI
# =========================

def get_regression_models():
    return [
        {"name": "Ridge", "model": Ridge(), "params": {"alpha": [0.1, 1, 10, 100]}},
        {"name": "Lasso", "model": Lasso(), "params": {"alpha": [0.001, 0.01, 0.1, 1]}},
        {"name": "SVR", "model": SVR(), "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}},
        {"name": "RandomForest", "model": RandomForestRegressor(random_state=RANDOM_STATE),
         "params": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}},
        {"name": "XGBoost", "model": XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE),
         "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}}
    ]

def get_classification_models():
    return [
        {"name": "LogisticRegression", "model": LogisticRegression(solver="liblinear"),
         "params": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]}},
        {"name": "SVC", "model": SVC(),
         "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}},
        {"name": "RandomForest", "model": RandomForestClassifier(random_state=RANDOM_STATE),
         "params": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}},
        {"name": "XGBoost", "model": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
         "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}}
    ]

# =========================
# 6. ANA PROGRAM (MAIN)
# =========================

def main():
    # 1. Dosya Okuma
    file_name = "student_habits_performance.csv"
    try:
        df = pd.read_csv(file_name)
        print(f">> '{file_name}' başarıyla yüklendi. Boyut: {df.shape}")
    except FileNotFoundError:
        print(f"HATA: '{file_name}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        sys.exit(1)

    # 2. EDA (İsteğe bağlı görselleştirme)
    plot_distributions(df)

    # 3. Ön İşleme
    df = preprocess_data(df)

    # 4. Veri Bölme (Train/Test)
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5. Ölçeklendirme (Scaling) - Data Leakage önlemek için split'ten sonra yapılır
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------
    # GÖREV A: REGRESYON (Puan Tahmini)
    # ---------------------------------------------------------
    reg_models = get_regression_models()
    df_reg_results, best_reg_model, best_reg_name = run_model_tuning(
        X_train, y_train, X_test, y_test, reg_models, task="regression"
    )

    print("\n--- Regresyon Sonuç Tablosu ---")
    print(df_reg_results)

    # Regresyon Görselleştirme
    y_pred_reg = best_reg_model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_reg, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Regresyon: {best_reg_name} (Gerçek vs Tahmin)")
    plt.xlabel("Gerçek Puan")
    plt.ylabel("Tahmin Edilen Puan")
    plt.show()

    # ---------------------------------------------------------
    # GÖREV B: SINIFLANDIRMA (Geçti/Kaldı)
    # ---------------------------------------------------------
    # Hedef değişkeni binary formata çevir
    y_train_cls = (y_train >= PASS_THRESHOLD).astype(int)
    y_test_cls = (y_test >= PASS_THRESHOLD).astype(int)

    clf_models = get_classification_models()
    df_clf_results, best_clf_model, best_clf_name = run_model_tuning(
        X_train, y_train_cls, X_test, y_test_cls, clf_models, task="classification"
    )

    print("\n--- Sınıflandırma Sonuç Tablosu ---")
    print(df_clf_results)

    # Sınıflandırma Görselleştirme (Confusion Matrix)
    y_pred_cls = best_clf_model.predict(X_test)
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Sınıflandırma: {best_clf_name} (Confusion Matrix)")
    plt.xlabel("Tahmin (0: Kaldı, 1: Geçti)")
    plt.ylabel("Gerçek (0: Kaldı, 1: Geçti)")
    plt.show()

    print(f"\n>>> {best_clf_name} Detaylı Rapor:")
    print(classification_report(y_test_cls, y_pred_cls, target_names=['Kaldı', 'Geçti']))

if __name__ == "__main__":
    main()