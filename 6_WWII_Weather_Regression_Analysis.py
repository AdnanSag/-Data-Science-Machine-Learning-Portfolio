import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Görselleştirme Ayarları
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def load_and_clean_data(filepath):
    """
    Veri setini yükler, gereksiz sütunları temizler, 
    eksik verileri doldurur ve özellik mühendisliği adımlarını uygular.
    """
    # low_memory=False, büyük dosyalarda tip uyarısını engeller
    df = pd.read_csv(filepath, low_memory=False)
    
    # Sütun isimlerindeki boşlukların temizlenmesi
    df.columns = df.columns.str.strip()
    
    # Tamamı boş olan sütunların atılması
    df.dropna(axis=1, how="all", inplace=True)

    # Analiz için gereksiz veya çok fazla eksik veriye sahip sütunların silinmesi
    cols_to_drop = [
        "WindGustSpd", "DR", "SPD", "SND", "FT", "FB", "FTI", "ITH", 
        "PGT", "SD3", "RHX", "RHN", "RVG", "WTE", "PoorWeather", "TSHDSBRSGF"
    ]
    df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    # Tarih formatının düzeltilmesi
    df['Date'] = pd.to_datetime(df['Date'])

    # Sayısal olması gereken sütunların dönüşümü ve eksik verilerin ortalama ile doldurulması
    numeric_cols_to_fix = ['Snowfall', 'Precip', 'PRCP']
    
    for col in numeric_cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    # Veri sızıntısını (Data Leakage) önlemek için hedef değişkenle 
    # doğrudan ilişkili (Max, Min temp gibi) sütunların çıkarılması
    cols_leakage = ['MAX', 'MIN', 'MEA', 'YR', 'MO', 'DA', 'PRCP', 'SNF']
    df.drop(columns=cols_leakage, errors='ignore', inplace=True)
    
    # Kalan eksik verilerin temizlenmesi
    df.dropna(inplace=True)

    # Tarihten yeni özellikler türetme
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    return df

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Linear ve Lasso regresyon modellerini eğitir ve sonuçları raporlar.
    """
    results = {}

    # 1. Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    
    results['Linear'] = {
        'model': lin_reg,
        'pred': y_pred_lin,
        'mae': mean_absolute_error(y_test, y_pred_lin),
        'mse': mean_squared_error(y_test, y_pred_lin),
        'r2': r2_score(y_test, y_pred_lin)
    }

    print("--- LINEAR REGRESSION METRICS ---")
    print(f"MAE: {results['Linear']['mae']:.4f}")
    print(f"MSE: {results['Linear']['mse']:.4f}")
    print(f"R2 Score: {results['Linear']['r2']:.4f}\n")

    # 2. LassoCV (Optimized L1 Regularization)
    lasso = LassoCV(cv=5, random_state=15, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    results['Lasso'] = {
        'model': lasso,
        'pred': y_pred_lasso,
        'mae': mean_absolute_error(y_test, y_pred_lasso),
        'mse': mean_squared_error(y_test, y_pred_lasso),
        'r2': r2_score(y_test, y_pred_lasso),
        'alpha': lasso.alpha_
    }

    print("--- OPTIMIZED LASSO (LassoCV) METRICS ---")
    print(f"Optimal Alpha: {lasso.alpha_:.4f}")
    print(f"MAE: {results['Lasso']['mae']:.4f}")
    print(f"MSE: {results['Lasso']['mse']:.4f}")
    print(f"R2 Score: {results['Lasso']['r2']:.4f}")
    
    return results

def plot_performance(y_test, results, feature_names):
    """
    Modellerin tahmin başarısını ve Lasso katsayılarını görselleştirir.
    """
    plt.figure(figsize=(14, 6))

    # Grafik 1: Linear Regression Tahminleri
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, results['Linear']['pred'], alpha=0.5, color='blue', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.xlabel('Actual MeanTemp')
    plt.ylabel('Predicted MeanTemp')
    plt.legend()

    # Grafik 2: Lasso Regression Tahminleri
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, results['Lasso']['pred'], alpha=0.5, color='green', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title(f"LassoCV (Alpha={results['Lasso']['alpha']:.4f}): Actual vs Predicted")
    plt.xlabel('Actual MeanTemp')
    plt.ylabel('Predicted MeanTemp')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Grafik 3: Lasso Özellik Önem Düzeyleri (Feature Importance)
    lasso_model = results['Lasso']['model']
    lasso_coefs = pd.Series(lasso_model.coef_, index=feature_names)
    non_zero_coefs = lasso_coefs[lasso_coefs.abs() > 1e-4].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=non_zero_coefs.values, y=non_zero_coefs.index, palette='viridis')
    plt.title('LassoCV Feature Importance (Coefficients)')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.show()

# --- ANA UYGULAMA AKIŞI ---

# 1. Veri Hazırlığı
df = load_and_clean_data("datasets/Summary of Weather.csv")

# Hedef değişken (MeanTemp) ve Özelliklerin ayrılması
# 'MeanTemp' hedef olduğu için X'ten çıkarılır, Date ise format gereği modelde kullanılmaz
X = df.drop(["MeanTemp", "Date"], axis=1)
y = df["MeanTemp"]

# Eğitim ve Test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Standardizasyon (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Modelleme ve Değerlendirme
model_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

# 3. Görselleştirme
plot_performance(y_test, model_results, X.columns)