import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lazypredict.Supervised import LazyRegressor

# Görselleştirme Ayarları
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_and_clean_data(filepath):
    """
    Veri setini yükler, yapısal bozuklukları giderir, bölge ayrımını yapar 
    ve gerekli tip dönüşümlerini gerçekleştirir.
    """
    df = pd.read_csv(filepath)
    
    # Sütun isimlerindeki gereksiz boşlukların temizlenmesi
    df.columns = df.columns.str.strip()
    
    # Bölge (Region) bilgisinin eklenmesi (0: Bejaia, 1: Sidi-Bel Abbes)
    # Veri seti yapısı gereği ilk 122 satır bir bölgeyi, kalanı diğer bölgeyi temsil eder.
    df.loc[:122, "Region"] = 0
    df.loc[122:, "Region"] = 1
    
    # Tekrar eden başlık satırlarının ve eksik verilerin temizlenmesi
    df = df[df['day'] != 'day']
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Veri tiplerinin uygun formatlara dönüştürülmesi
    cols_int = ["day", "month", "year", "Temperature", "RH", "Ws", "Region"]
    cols_float = ["Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    
    df[cols_int] = df[cols_int].astype(int)
    df[cols_float] = df[cols_float].astype(float)
    
    # Hedef sınıfın binary hale getirilmesi (0: not fire, 1: fire)
    df["Classes"] = np.where(df["Classes"].astype(str).str.contains('not fire'), 0, 1)
    
    # Model başarısını etkilemeyen tarih sütunlarının çıkarılması
    df.drop(["day", "month", "year"], axis=1, inplace=True)
    
    return df

def get_high_correlation_features(dataset, threshold=0.85):
    """
    Çoklu doğrusal bağlantı (Multicollinearity) problemini önlemek için 
    belirtilen eşik değerinin üzerindeki korelasyona sahip özellikleri tespit eder.
    """
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

def evaluate_model(model, X_test, y_test, model_name):
    """
    Eğitilen modelin performansını MAE, MSE ve R2 metrikleri ile raporlar.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2 Score: {r2:.4f}\n")
    return y_pred

# --- ANA UYGULAMA AKIŞI ---

# 1. Veri Hazırlığı
df = load_and_clean_data("4-Algerian_forest_fires_dataset.csv")

X = df.drop("FWI", axis=1)
y = df["FWI"]

# Eğitim ve Test setlerine ayırma (%25 Test, %75 Eğitim)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# 2. Özellik Seçimi ve Ölçeklendirme
# Yüksek korelasyonlu özelliklerin tespiti ve çıkarılması
corr_features = get_high_correlation_features(X_train, threshold=0.85)
print(f"Dropped High Correlation Features: {corr_features}\n")

X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)

# Verinin standartlaştırılması (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Modelleme (Regresyon Modelleri)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
evaluate_model(lin_reg, X_test_scaled, y_test, "Linear Regression")

# LassoCV (L1 Regularization - Cross Validation ile)
lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)
evaluate_model(lasso_cv, X_test_scaled, y_test, f"LassoCV (Alpha={lasso_cv.alpha_:.4f})")

# RidgeCV (L2 Regularization - Cross Validation ile)
ridge_cv = RidgeCV(cv=5).fit(X_train_scaled, y_train)
evaluate_model(ridge_cv, X_test_scaled, y_test, f"RidgeCV (Alpha={ridge_cv.alpha_:.4f})")

# ElasticNetCV (L1 + L2 Karma)
elastic_cv = ElasticNetCV(cv=5, random_state=42).fit(X_train_scaled, y_train)
evaluate_model(elastic_cv, X_test_scaled, y_test, f"ElasticNetCV (Alpha={elastic_cv.alpha_:.4f})")

# 4. AutoML İle Geniş Kapsamlı Model Kıyaslaması
print("--- Running LazyRegressor AutoML ---")
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

# 5. Sonuçların Görselleştirilmesi
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lin_reg.predict(X_test_scaled), color='blue', alpha=0.6, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title("Actual vs Predicted FWI (Linear Regression)")
plt.legend()
plt.show()