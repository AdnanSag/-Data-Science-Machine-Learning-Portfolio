import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Uyarıları sessize al
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==========================================
# PROJE: Titanic Survival Prediction (LightGBM vs XGBoost)
# Amaç: Gradient Boosting algoritmalarını hiperparametre optimizasyonu ile kıyaslamak
# ==========================================

def load_and_preprocess_data():
    """
    Veri setini yükler, eksik verileri doldurur ve ön işleme yapar.
    """
    print("--- Veri Yükleniyor ve İşleniyor ---")
    df = sns.load_dataset("titanic")
    
    # Gereksiz sütunları at
    drop_cols = ["deck", "embark_town", "alive"]
    df = df.drop(drop_cols, axis=1)
    
    # Eksik verileri doldur
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    
    # Boolean/Kategorik verileri sayısal tipe çevir
    df["adult_male"] = df["adult_male"].astype(int)
    df["alone"] = df["alone"].astype(int)
    
    return df

def feature_engineering(df):
    """
    One-Hot Encoding işlemlerini gerçekleştirir.
    """
    X = df.drop("survived", axis=1)
    y = df["survived"]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    
    # Kategorik değişkenler
    categorical_cols = ["sex", "class", "embarked", "who"]
    
    # Column Transformer (One-Hot Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    # Dönüşüm işlemi
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    
    # Sütun isimlerini geri kazanma (DataFrame formatı için)
    feature_names = preprocessor.get_feature_names_out()
    
    X_train_df = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_encoded, columns=feature_names)
    
    return X_train_df, X_test_df, y_train, y_test

def train_optimize_model(model, param_grid, X_train, y_train, model_name="Model"):
    """
    RandomizedSearchCV kullanarak modeli eğitir ve en iyi parametreleri bulur.
    """
    print(f"\n--- {model_name} Optimizasyonu Başlıyor ---")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # 20 farklı kombinasyon dene
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    print(f"En İyi Parametreler ({model_name}): {random_search.best_params_}")
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Model performansını değerlendirir ve raporlar.
    """
    y_pred = model.predict(X_test)
    print(f"\n>>> {model_name} Performans Raporu <<<")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.show()

def plot_feature_importance(model, feature_names, model_name="LightGBM"):
    """
    Modelin özellik önem düzeylerini görselleştirir.
    """
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_imp_df.head(10), palette="viridis")
    plt.title(f"{model_name} - Top 10 Feature Importance")
    plt.show()

# --- ANA UYGULAMA AKIŞI ---
if __name__ == "__main__":
    
    # 1. Veri Hazırlığı
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = feature_engineering(df)
    
    # 2. LightGBM Modeli ve Optimizasyonu
    lgb_params = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    
    best_lgb = train_optimize_model(lgb.LGBMClassifier(verbosity=-1), lgb_params, X_train, y_train, "LightGBM")
    evaluate_model(best_lgb, X_test, y_test, "LightGBM Best Model")
    plot_feature_importance(best_lgb, X_train.columns, "LightGBM")
    
    # 3. XGBoost Modeli ve Optimizasyonu
    xgb_params = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.6, 0.8, 1.0]
    }
    
    best_xgb = train_optimize_model(XGBClassifier(eval_metric='logloss'), xgb_params, X_train, y_train, "XGBoost")
    evaluate_model(best_xgb, X_test, y_test, "XGBoost Best Model")