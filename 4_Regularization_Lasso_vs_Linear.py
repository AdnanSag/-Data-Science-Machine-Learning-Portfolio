import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Görselleştirme ayarları
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# ==========================================
# PROJE 4: Regularization (Lasso) vs Linear Regression
# Amaç: Çok sayıda özellik içeren veri setlerinde Lasso'nun etkisi
# Veri Seti: Summary of Weather (WW2 Weather Data)
# ==========================================

# 1. Veri Yükleme ve Temizleme
# low_memory=False: Büyük dosyalarda tip çıkarımı uyarısını önler
df = pd.read_csv("Summary of Weather.csv", low_memory=False)

# Sütun isimlerindeki boşlukları temizle
df.columns = df.columns.str.strip()

# Tamamı boş olan sütunları kaldır
df.dropna(axis=1, inplace=True, how="all")

# Gereksiz veya çok fazla eksik veri içeren sütunların listesi
cols_to_drop = [
    "WindGustSpd", "DR", "SPD", "SND", "FT", "FB", "FTI", "ITH", 
    "PGT", "SD3", "RHX", "RHN", "RVG", "WTE", "PoorWeather", "TSHDSBRSGF",
    "MAX", "MIN", "MEA", "YR", "MO", "DA", "PRCP", "SNF" 
]
df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

# Tarih formatını düzeltme
df['Date'] = pd.to_datetime(df['Date'])

# 2. Sayısal Dönüşümler ve Eksik Veri Doldurma
# 'Snowfall' ve 'Precip' gibi sütunlarda metinler (örn: 'T' - Trace) olabilir, bunları sayıya çeviriyoruz.
numeric_cols = ['Snowfall', 'Precip']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') # Hatalı değerleri NaN yap
        df[col] = df[col].fillna(df[col].mean()) # NaN'ları ortalama ile doldur

# Kalan satırlardaki eksikleri temizle
df.dropna(inplace=True)

# 3. Model Hazırlığı (Train-Test Split & Scaling)
# Hedef: MeanTemp (Ortalama Sıcaklık)
X = df.drop(["MeanTemp", "Date"], axis=1)
y = df["MeanTemp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Standardizasyon (Lasso için çok önemlidir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Modellerin Eğitilmesi ve Karşılaştırılması

# --- Model 1: Linear Regression (Baseline) ---
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)

# --- Model 2: Lasso Regression (L1 Regularization) ---
# alpha: Ceza katsayısı (Arttıkça model daha çok sadeleşir)
lasso_model = Lasso(alpha=0.1) 
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

# 5. Performans Değerlendirmesi
def print_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} Results ---")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}\n")

print_metrics("Linear Regression", y_test, y_pred_linear)
print_metrics("Lasso Regression", y_test, y_pred_lasso)

# Lasso'nun hangi özellikleri sıfırladığını görmek için (Feature Selection)
# print("Lasso Coefficients:", lasso_model.coef_)

# 6. Görselleştirme (Yan Yana Karşılaştırma)
fig, axes = plt.subplots(1, 2)

# Linear Plot
axes[0].scatter(y_test, y_pred_linear, color='blue', alpha=0.5, label='Predictions')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Fit')
axes[0].set_title("Linear Regression: Actual vs Predicted")
axes[0].set_xlabel("Actual Temperature")
axes[0].set_ylabel("Predicted Temperature")
axes[0].legend()

# Lasso Plot
axes[1].scatter(y_test, y_pred_lasso, color='red', alpha=0.5, label='Predictions')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Fit')
axes[1].set_title("Lasso Regression: Actual vs Predicted")
axes[1].set_xlabel("Actual Temperature")
axes[1].legend()

plt.tight_layout()
plt.show()