import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from scipy.stats import boxcox, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==========================================
# PROJE: Boston Housing Price Prediction (Power Transformation)
# Amaç: Veri dağılımındaki çarpıklığı (Skewness) gidererek Linear Regression performansını artırmak.
# Teknikler: Yeo-Johnson (Features için), Box-Cox (Target için)
# ==========================================

def load_data():
    """Veri setini yükler."""
    # Boston veri seti standart olarak 14 kolondan oluşur.
    # Eksik kolonlar eklenmezse veri kayması (misalignment) oluşur.
    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", 
        "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    
    try:
        # delimiter=r"\s+" boşlukla ayrılmış dosyaları okur
        df = pd.read_csv("datasets/23-boston.csv", header=None, delimiter=r"\s+", names=column_names)
        print("Veri Seti Yüklendi. Boyut:", df.shape)
        return df
    except FileNotFoundError:
        print("Hata: 'datasets/23-boston.csv' dosyası bulunamadı.")
        return None

def plot_all_histogram(df, title_prefix=""):
    """Tüm sayısal sütunların histogramını çizer."""
    plt.close('all')
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 4 * n_rows), constrained_layout=True)
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color="skyblue")
        axes[i].set_title(f"{title_prefix} {col}")
        axes[i].set_xlabel("")
    
    # Boş kalan grafikleri sil
    for i in range(len(num_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.show()

def check_skewness(df):
    """Verideki çarpıklık oranlarını gösterir."""
    print("\n--- Skewness (Çarpıklık) Değerleri ---")
    skew_vals = df.apply(skew).sort_values(ascending=False)
    print(skew_vals.head(5)) # En çarpık 5 özellik
    print("--------------------------------------")


def inverse_boxcox(y, lambda_):
    """Box-Cox dönüşümünü tersine çevirir (Tahminleri orijinal birime döndürmek için)."""
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_ + 1, 1 / lambda_)

# --- ANA AKIŞ ---

if __name__ == "__main__":
    df = load_data()
    
    if df is not None:
        # 1. Veri Hazırlığı
        check_skewness(df)
        
        # Orijinal Veri Dağılımını Görselleştir
        plot_all_histogram(df, title_prefix="Original")
    
        X = df.drop("MEDV", axis=1)
        y = df["MEDV"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

        # ==========================================
        # SENARYO 1: Dönüşüm Yapılmadan (Raw Data)
        # ==========================================
        print("\n--- Senaryo 1: Dönüşüm Yok (Baseline) ---")
        model_raw = LinearRegression()
        model_raw.fit(X_train, y_train)
        y_pred_raw = model_raw.predict(X_test)
        
        r2_raw = r2_score(y_test, y_pred_raw)
        mse_raw = mean_squared_error(y_test, y_pred_raw)
        print(f"R2 Score: {r2_raw:.4f}")
        print(f"MSE: {mse_raw:.4f}")

        # ==========================================
        # SENARYO 2: Power Transformation (Yeo-Johnson & Box-Cox)
        # ==========================================
        print("\n--- Senaryo 2: Power Transformation Uygulandı ---")
        
        # 1. Özellikler (X) için Yeo-Johnson Dönüşümü (Negatif değerleri de destekler)
        pt_X = PowerTransformer(method="yeo-johnson")
        X_train_transformed = pt_X.fit_transform(X_train)
        X_test_transformed = pt_X.transform(X_test)
        
        # Dönüşüm sonrası histogramları görmek istersen:
        X_train_trans_df = pd.DataFrame(X_train_transformed, columns=X.columns)
        plot_all_histogram(X_train_trans_df, title_prefix="Transformed")

        # 2. Hedef (y) için Box-Cox Dönüşümü (Sadece pozitif değerler için)
        y_train_transformed, lambda_y = boxcox(y_train)
        
        # Modeli Eğit
        model_trans = LinearRegression()
        model_trans.fit(X_train_transformed, y_train_transformed)
        
        # Tahmin Yap (Tahminler şu an Box-Cox formatında)
        y_pred_trans_raw = model_trans.predict(X_test_transformed)
        
        # Tahminleri Orijinal Birime Geri Çevir (Inverse Transform)
        y_pred_final = inverse_boxcox(y_pred_trans_raw, lambda_y)
        
        # Değerlendirme
        r2_trans = r2_score(y_test, y_pred_final)
        mse_trans = mean_squared_error(y_test, y_pred_final)
        
        print(f"R2 Score: {r2_trans:.4f}")
        print(f"MSE: {mse_trans:.4f}")
        
        print(f"\n>>> İyileştirme (R2 Farkı): {r2_trans - r2_raw:.4f}")