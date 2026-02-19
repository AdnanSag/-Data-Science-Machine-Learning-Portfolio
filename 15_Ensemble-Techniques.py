import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Uyarıları filtrele
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

def load_and_clean_data(filepath):
    """
    Veriyi yükler, sütun isimlerini düzenler ve '?' gibi bozuk verileri temizler.
    """
    col_names = ["age", "workclass", "finalweight", "education", "education_num", "marital_status",
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                 "hours_per_week", "native_country", "income"]
    
    try:
        df = pd.read_csv(filepath, names=col_names, header=0)
        
        # String sütunlardaki gereksiz boşlukları temizle
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
            
        # '?' değerlerini NaN ile değiştir
        df.replace("?", np.nan, inplace=True)
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def target_encode_country(X_train, y_train, X_test):
    """
    'native_country' sütunu çok fazla kategori içerdiği için One-Hot Encoding yerine
    Target Encoding uygular.
    """
    # Hedef değişkeni binary (0/1) formata çevir
    y_train_binary = y_train.apply(lambda x: 1 if x == ">50K" else 0)
    
    # Her ülke için ortalama gelir olasılığını hesapla
    target_means = y_train_binary.groupby(X_train["native_country"]).mean()
    
    # Train ve Test setlerine maple
    X_train["native_country_encoded"] = X_train['native_country'].map(target_means)
    X_test["native_country_encoded"] = X_test['native_country'].map(target_means)
    
    # Test setinde olup Train setinde olmayan ülkeler için genel ortalamayı kullan (Cold Start)
    global_mean = y_train_binary.mean()
    X_train["native_country_encoded"].fillna(global_mean, inplace=True)
    X_test["native_country_encoded"].fillna(global_mean, inplace=True)
    
    # Orijinal sütunu düşür
    X_train.drop("native_country", axis=1, inplace=True)
    X_test.drop("native_country", axis=1, inplace=True)
    
    return X_train, X_test

def preprocess_features(df):
    """
    Eksik verileri doldurur, encoding ve scaling işlemlerini uygular.
    """
    X = df.drop('income', axis=1)
    y = df["income"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # 1. Eksik Verilerin Doldurulması (Imputation)
    # SettingWithCopyWarning hatasını önlemek için .copy() kullanıyoruz
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    for col in ["workclass", "occupation", "native_country"]:
        mode_val = X_train[col].mode()[0]
        X_train[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)
        
    # 2. Target Encoding (Native Country için)
    X_train, X_test = target_encode_country(X_train, y_train, X_test)
    
    # 3. One-Hot Encoding & Scaling
    # RobustScaler, gelir verisindeki uç değerlere (outliers) karşı StandartScaler'dan daha dayanıklıdır.
    categorical_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    scaler = RobustScaler()
    X_train_final = scaler.fit_transform(X_train_processed)
    X_test_final = scaler.transform(X_test_processed)
    
    # Sütun isimlerini geri kazanmak (Feature Importance grafiği için gerekli)
    new_cols = preprocessor.get_feature_names_out()
    
    return X_train_final, X_test_final, y_train, y_test, new_cols

def evaluate_model(model, X_test, y_test, title="Model Performance"):
    y_pred = model.predict(X_test)
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples', cbar=False)
    plt.title(f"{title} Confusion Matrix")
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Random Forest'ın en güçlü yanı olan özellik önem düzeylerini görselleştirir.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # İlk 20 özelliği gösterelim
    top_n = 20
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    
    # Sütun isimlerini düzelt (cat__workclass_Private -> Private gibi)
    cleaned_names = [feature_names[i].split('__')[-1] for i in indices[:top_n]]
    
    plt.xticks(range(top_n), cleaned_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    df = load_and_clean_data("datasets/14-income_evaluation.csv")
    
    if df is not None:
        # Ön İşleme
        X_train, X_test, y_train, y_test, feature_names = preprocess_features(df)
        
        # 1. Baseline Random Forest
        print("Training Baseline Random Forest...")
        rfc = RandomForestClassifier(n_estimators=100, random_state=15)
        rfc.fit(X_train, y_train)
        evaluate_model(rfc, X_test, y_test, "Baseline RF")
        
        # Feature Importance Görselleştirme
        plot_feature_importance(rfc, feature_names)
        
        # 2. Hyperparameter Tuning (RandomizedSearchCV)
        print("\nRunning RandomizedSearchCV (This may take a moment)...")
        rf_params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        
        # CV=3 ve n_iter=10 yaparak süreci hızlandırıyoruz
        rscv = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42), 
                                  param_distributions=rf_params, 
                                  n_iter=10, cv=3, n_jobs=-1, verbose=1)
        
        rscv.fit(X_train, y_train)
        print(f"Best Parameters: {rscv.best_params_}")
        
        best_model = rscv.best_estimator_
        evaluate_model(best_model, X_test, y_test, "Tuned RF")