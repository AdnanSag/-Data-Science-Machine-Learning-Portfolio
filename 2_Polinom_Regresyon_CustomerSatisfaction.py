import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

# ==========================================
# PROJE 3: Polinom Regresyon ve Model Karmaşıklığı Analizi
# Amaç: Doğrusal olmayan verilerde regresyon ve Overfitting (Aşırı Öğrenme) gözlemi
# ==========================================

# 1. Veri Yükleme ve Hazırlık
df = pd.read_csv("datasets/3-customersatisfaction.csv")

# Gereksiz index sütununu temizleme
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Değişkenlerin Ayrılması
X = df[["Customer Satisfaction"]]
y = df["Incentive"]

# Veri Setinin Bölünmesi (%80 Train, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Veriyi Görselleştirme (Doğrusallık Kontrolü)
plt.figure(figsize=(10, 6))
plt.scatter(df["Customer Satisfaction"], df["Incentive"], color='blue', alpha=0.6, label='Data Points')
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.title("Customer Satisfaction vs Incentive (Non-Linear Relationship)")
plt.legend()
plt.show()

# ==========================================
# BÖLÜM 1: Doğrusal Regresyon (Linear Regression - Baseline)
# ==========================================

# Veriyi Ölçeklendirme (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # DİKKAT: Test verisi sadece transform edilir, fit edilmez!

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

print(f"Linear Regression R2 Score: {r2_score(y_test, y_pred_lin):.4f}")
# Beklenen Sonuç: Düşük skor, çünkü veri doğrusal değil (Underfitting).

# ==========================================
# BÖLÜM 2: Polinom Regresyon (Polynomial Regression)
# ==========================================

# Pipeline Kurulumu: Ölçeklendirme -> Polinom Özellikleri -> Regresyon
# Degree 2 ile deneme (Parabolik ilişki)
poly_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2)),
    ("reg", LinearRegression())
])

poly_pipeline.fit(X_train, y_train) # Pipeline ham veriyi alır, içinde scale eder
y_pred_poly = poly_pipeline.predict(X_test)

print(f"Polynomial Regression (Degree=2) R2 Score: {r2_score(y_test, y_pred_poly):.4f}")
# Beklenen Sonuç: Daha yüksek skor, model veriye uyum sağladı.

# ==========================================
# BÖLÜM 3: Model Karmaşıklığı ve Overfitting Analizi
# Farklı polinom derecelerinin (Degrees) denenmesi
# ==========================================

# Yeni gelen test verilerini yükleyelim
new_df = pd.read_csv("datasets/3-newdatas.csv")
# Sütun isimlendirmesi
if "0" in new_df.columns:
    new_df.rename(columns={"0": "Customer Satisfaction"}, inplace=True)
X_new = new_df[["Customer Satisfaction"]]


def evaluate_poly_degree(degree, X_train, y_train, X_test, y_test, X_new):
    """
    Belirli bir polinom derecesi için modeli eğitir, grafiğini çizer ve skorunu hesaplar.
    """
    # Pipeline oluşturma
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree)),
        ("reg", LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Skor Hesaplama
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Degree: {degree} | Train R2: {train_score:.4f} | Test R2: {test_score:.4f}")
    
    # Görselleştirme için pürüzsüz çizgi oluşturma
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred = pipeline.predict(X_range)
    
    # Yeni veri tahmini (Görselleştirmede göstermek için)
    y_new_pred = pipeline.predict(X_new)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test Data')
    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label=f'Model (Degree={degree})')
    
    plt.title(f"Polynomial Regression (Degree={degree})")
    plt.xlabel("Customer Satisfaction")
    plt.ylabel("Incentive")
    plt.legend()
    plt.show()

# Döngü ile 1'den 10'a kadar dereceleri deneme
# Degree arttıkça modelin eğitim verisini ezberlediğini (Overfitting) göreceğiz.
degrees_to_test = [1, 2, 5, 10] # Örnek olarak seçilen dereceler

for d in degrees_to_test:
    evaluate_poly_degree(d, X_train, y_train, X_test, y_test, X_new)