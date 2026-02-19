import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

# Uyarıları kapat
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==========================================
# PROJE: Medical Cost Prediction (LightGBM + Box-Cox)
# Amaç: Sağlık harcamalarını tahmin etmek ve Box-Cox dönüşümü ile hedef değişkeni normalize etmek.
# ==========================================

def load_data():
    """Veri setini yükler ve temizler."""
    df = pd.read_csv("datasets/24-medical_cost.csv")
    
    # Gereksiz ID sütunu varsa sil
    if "Id" in df.columns:
        df.drop("Id", inplace=True, axis=1)
        
    # Binary Mapping (Manuel encoding)
    # Kadın/Erkek ve Sigara İçen/İçmeyen ayrımı 0-1 yapılır
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    
    return df

def plot_eda(df):
    """Keşifçi Veri Analizi grafiklerini çizer."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sigara İçenlerin Dağılımı
    sns.countplot(data=df, x="smoker", ax=axes[0, 0])
    axes[0, 0].set_title("Smoker Distribution (0: No, 1: Yes)")
    
    # 2. Bölge Dağılımı
    sns.countplot(data=df, x="region", ax=axes[0, 1])
    axes[0, 1].set_title("Region Distribution")
    
    # 3. Yaş ve Masraf İlişkisi (Sigara detaylı)
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", ax=axes[1, 0])
    axes[1, 0].set_title("Age vs Charges (Hue: Smoker)")
    
    # 4. Hedef Değişken (Charges) Dağılımı - Çarpıklığı görmek için
    sns.histplot(data=df, x="charges", kde=True, ax=axes[1, 1], color="red")
    axes[1, 1].set_title("Charges Distribution (Skewed)")
    
    plt.tight_layout()
    plt.show()

def preprocessing(df):
    """One-Hot Encoding ve Train-Test Split işlemlerini yapar."""
    X = df.drop("charges", axis=1)
    y = df["charges"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    
    # Sadece 'region' sütunu kategorik kaldı, onu encode edelim
    categorical_cols = ["region"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    
    return X_train_encoded, X_test_encoded, y_train, y_test

def train_lgbm(X_train, y_train, X_test, y_test, use_boxcox=False):
    """LightGBM modelini eğitir. İsteğe bağlı Box-Cox dönüşümü uygular."""
    
    lambda_y = None # Lambda değerini saklamak için
    
    # Eğer Box-Cox seçildiyse hedef değişkeni dönüştür
    if use_boxcox:
        print("\n--- Box-Cox Dönüşümü Uygulanıyor ---")
        # Hedef değişkenin dağılımını 'Normal Dağılım'a yaklaştırır
        y_train_transformed, lambda_y = boxcox(y_train)
        target_train = y_train_transformed
    else:
        target_train = y_train

    # Hiperparametre Izgarası
    param_grid = {
        "num_leaves": [31, 50],
        "max_depth": [-1, 10],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 300],
        "min_child_samples": [20, 30],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    # RandomizedSearchCV
    model = RandomizedSearchCV(
        estimator=LGBMRegressor(verbosity=-1),
        param_distributions=param_grid,
        cv=5,
        n_iter=10, # Hız için iterasyon düşürüldü
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=15
    )
    
    model.fit(X_train, target_train)
    print(f"Best Params (BoxCox={use_boxcox}): {model.best_params_}")
    
    # Tahmin
    y_pred = model.predict(X_test)
    
    # Eğer Box-Cox yapıldıysa, tahminleri geri (inverse) dönüştürmeliyiz
    if use_boxcox:
        y_pred = inv_boxcox(y_pred, lambda_y)
    
    # Değerlendirme
    r2 = r2_score(y_test, y_pred) # Önemli: Önce y_test, sonra y_pred
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    
    return r2

# --- ANA AKIŞ ---
if __name__ == "__main__":
    # 1. Yükle
    df = load_data()
    
    # 2. Görselleştir 
    plot_eda(df)
    
    # 3. Hazırla
    X_train, X_test, y_train, y_test = preprocessing(df)
    
    # 4. Model 1: Normal LightGBM
    print("--- Model 1: Standard LightGBM ---")
    r2_normal = train_lgbm(X_train, y_train, X_test, y_test, use_boxcox=False)
    
    # 5. Model 2: LightGBM + Box-Cox Transformation
    # Hedef değişken (charges) sağa çarpık olduğu için bu yöntem genelde skoru artırır.
    print("--- Model 2: LightGBM with Box-Cox Target Transformation ---")
    r2_boxcox = train_lgbm(X_train, y_train, X_test, y_test, use_boxcox=True)
    
    print(f"\nImprovement: {r2_boxcox - r2_normal:.4f}")