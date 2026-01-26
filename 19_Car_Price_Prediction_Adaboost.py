import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

def load_and_clean_data(filepath):
    """
    Veriyi yükler, gereksiz sütunları atar, kopyaları temizler 
    ve aykırı değerleri (Outlier) filtreler.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Gereksiz indeks sütunu temizliği
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
            
        # Tekrar eden satırların temizlenmesi
        df = df.drop_duplicates(keep="first", ignore_index=True)
        
        # Hatalı veri girişi düzeltme (0 koltuk -> 5 koltuk)
        df.loc[df['seats'] == 0, "seats"] = 5
        
        # Aykırı Değerlerin (Outliers) Filtrelenmesi
        # Fiyatı 10 Milyon altı ve KM'si 600.000 altı olanlar (Domain bilgisi)
        df = df[(df["selling_price"] < 10000000) & (df["km_driven"] < 600000)]
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def visualize_data(df):
    """
    Araç yaşı ile satış fiyatı arasındaki ilişkiyi yakıt türüne göre görselleştirir.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['vehicle_age'], y=df['selling_price'], hue=df['fuel_type'], alpha=0.6)
    plt.title("Vehicle Age vs Selling Price (by Fuel Type)")
    plt.xlabel("Vehicle Age")
    plt.ylabel("Selling Price")
    plt.show()

def frequency_encoding(X_train, X_test, columns):
    """
    Yüksek kardinaliteye sahip kategorik değişkenler (Marka, Model vb.) için
    Frequency Encoding uygular. Veri sızıntısını önlemek için frekanslar
    sadece X_train üzerinden hesaplanır.
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in columns:
        # Frekansları hesapla (Train seti üzerinden)
        freq = X_train[col].value_counts() / len(X_train)
        
        # Train ve Test setine maple
        X_train_encoded[col + '_freq'] = X_train[col].map(freq)
        X_test_encoded[col + '_freq'] = X_test[col].map(freq)
        
        # Test setinde olup Train'de olmayan yeni kategoriler için ortalama frekans ata
        mean_freq = freq.mean()
        X_test_encoded[col + '_freq'] = X_test_encoded[col + '_freq'].fillna(mean_freq)
        
    # Orijinal kategorik sütunları düşür
    X_train_encoded = X_train_encoded.drop(columns, axis=1)
    X_test_encoded = X_test_encoded.drop(columns, axis=1)
    
    return X_train_encoded, X_test_encoded

def preprocess_data(df):
    """
    Feature Engineering süreçlerini yönetir (Split, Freq Encoding, OneHot Encoding).
    """
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
    
    # 1. Frequency Encoding (Marka, Model gibi çok seçenekli sütunlar için)
    freq_columns = ['car_name', 'brand', 'model']
    X_train, X_test = frequency_encoding(X_train, X_test, freq_columns)
    
    # 2. One-Hot Encoding (Az seçenekli kategorik sütunlar için)
    onehot_columns = ['seller_type', 'fuel_type', 'transmission_type']
    
    transformer = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), onehot_columns)
        ],
        remainder="passthrough"
    )
    
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    
    return X_train_transformed, X_test_transformed, y_train, y_test

def evaluate_model(y_true, y_pred, title="Model Performance"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    print(f"\n--- {title} ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"MSE      : {mse:.2f}")

# --- ANA UYGULAMA ---

if __name__ == "__main__":
    # 1. Veri Yükleme ve Temizlik
    df = load_and_clean_data("17-cardekho.csv")
    
    if df is not None:
        # 2. Görselleştirme (Opsiyonel)
        # visualize_data(df)
        
        # 3. Ön İşleme (Encoding)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # 4. Baseline AdaBoost Model
        print("Training Baseline AdaBoost...")
        base_model = AdaBoostRegressor(random_state=15)
        base_model.fit(X_train, y_train)
        y_pred_base = base_model.predict(X_test)
        evaluate_model(y_test, y_pred_base, "Baseline AdaBoost")
        
        # 5. Hyperparameter Tuning (RandomizedSearchCV)
        print("\nRunning RandomizedSearchCV...")
        
        params = {
            "n_estimators": [50, 80, 100, 120],
            "learning_rate": [0.001, 0.01, 0.1, 1.0, 2.0],
            "loss": ["linear", "square", "exponential"]
        }
        
        # AdaBoostRegressor varsayılan olarak DecisionTreeRegressor(max_depth=3) kullanır.
        rcv = RandomizedSearchCV(estimator=AdaBoostRegressor(random_state=15), 
                                 param_distributions=params, 
                                 scoring='r2', cv=5, n_jobs=-1, verbose=1)
        
        rcv.fit(X_train, y_train)
        print(f"Best Parameters: {rcv.best_params_}")
        
        # Final Tahmin
        y_pred_tuned = rcv.predict(X_test)
        evaluate_model(y_test, y_pred_tuned, "Tuned AdaBoost")