import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier, plot_importance

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

def load_and_preprocess_data(filepath):
    """
    Veriyi yükler, gereksiz kimlik sütunlarını atar ve
    Label Encoding & Scaling işlemlerini yapar.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Analiz için gereksiz olan kimlik ve teknik cihaz bilgileri
        columns_to_drop = ["objid", "specobjid", "rerun", "field", "camcol", "run"]
        df.drop(columns_to_drop, axis=1, inplace=True)
        
        # Hedef Değişken Dönüşümü (STAR, GALAXY, QSO -> 0, 1, 2)
        le = LabelEncoder()
        df["class"] = le.fit_transform(df["class"])
        
        # Sınıf isimlerini saklayalım (Görselleştirme için gerekli olacak)
        class_names = le.classes_
        
        X = df.drop("class", axis=1)
        y = df["class"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)
        
        # Scaling (XGBoost için zorunlu değil ama bazen yakınsamayı hızlandırır)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature names (XGBoost feature importance için sütun isimlerini geri yükleyelim)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        return X_train_df, X_test_df, y_train, y_test, class_names
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None, None, None, None, None

def visualize_redshift_distribution(df, class_names):
    """
    Redshift (Kırmızıya Kayma) özelliğinin sınıflara göre dağılımını çizer.
    Bu özellik gök cisimlerini ayırt etmede kritiktir.
    """
    # Görselleştirme için veriyi tekrar birleştiriyoruz (sadece bu fonksiyon içinde)
    # Not: Bu fonksiyonu main içinde ham data ile çağırmak daha mantıklı olabilir.
    pass 

def evaluate_model(model, X_test, y_test, class_names, title="Model Performance"):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model):
    """
    XGBoost'un hangi özelliklere en çok önem verdiğini gösterir.
    """
    plt.figure(figsize=(10, 8))
    plot_importance(model, height=0.5, title="XGBoost Feature Importance", color='purple')
    plt.show()

# --- ANA UYGULAMA AKIŞI ---

if __name__ == "__main__":
    # 1. Veri Hazırlığı
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data("20-digitalskysurvey.csv")
    
    if X_train is not None:
        print(f"Classes detected: {class_names}")
        
        # 2. Baseline XGBoost Model
        print("Training Baseline XGBoost...")
        xgb = XGBClassifier(n_estimators=100, random_state=15, eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        
        evaluate_model(xgb, X_test, y_test, class_names, "Baseline XGBoost")
        
        # Feature Importance (Redshift'in önemini burada göreceğiz)
        # Not: plot_importance kendi plt.show()'unu çağırabilir, bazen inline gerekebilir.
        plot_importance(xgb)
        plt.title("Feature Importance (Baseline)")
        plt.show()

        # 3. Hyperparameter Tuning (GridSearchCV)
        print("\nRunning GridSearchCV (This might take time)...")
        
        params = {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200],
            "max_depth": [5, 8],
            "colsample_bytree": [0.8, 1]
        }
        
        # İşlem hızlansın diye n_jobs=-1 ve cv=3 kullanıyoruz
        grid_search = GridSearchCV(estimator=XGBClassifier(eval_metric='mlogloss', random_state=15), 
                                   param_grid=params, cv=3, n_jobs=-1, verbose=1)
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        
        # En iyi model değerlendirmesi
        best_xgb = grid_search.best_estimator_
        evaluate_model(best_xgb, X_test, y_test, class_names, "Tuned XGBoost")