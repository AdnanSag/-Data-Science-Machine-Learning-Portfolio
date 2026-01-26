import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
pd.options.display.max_columns = None

def load_and_clean_data(filepath):
    """
    Veriyi yükler, gereksiz indeks sütununu atar ve
    fiziksel olarak imkansız olan (0 boyutlu) verileri temizler.
    """
    # Unnamed sütunu genellikle gereksiz indekstir, okurken index_col=0 diyerek kurtulabiliriz
    df = pd.read_csv(filepath, index_col=0)
    
    # x, y, z boyutlarının 0 olması fiziksel olarak imkansızdır, bunları temizliyoruz
    # (Bu işlem yaklaşık 20-30 satırı siler)
    df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]
    
    return df

def remove_outliers(df):
    """
    Veri setindeki uç değerleri (outliers) belirli eşik değerlerine göre filtreler.
    Bu eşikler EDA (Keşifçi Veri Analizi) aşamasında belirlenmiştir.
    """
    # Depth ve Table için filtreleme
    df = df[(df["depth"] < 75) & (df["depth"] > 45)]
    df = df[(df["table"] < 75) & (df["table"] > 40)]
    
    # Boyutlar (x, y, z) için filtreleme
    df = df[(df["z"] < 30) & (df["z"] > 2)]
    df = df[(df["y"] < 75)]
    
    return df

def preprocess_features(df):
    """
    Kategorik verileri (Cut, Color, Clarity) sayısal hale getirir
    ve veriyi ölçeklendirir.
    """
    X = df.drop("price", axis=1)
    y = df["price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    
    # Kategorik Dönüşüm (Label Encoding)
    categorical_cols = ["cut", "color", "clarity"]
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
        
    # Standardizasyon (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(y_test, y_pred, model_name):
    """
    Model performansını metriklerle raporlar.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {model_name} Results ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}\n")

def train_linear_regression(X_train, X_test, y_train, y_test):
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Linear Regression")

def train_svr_grid_search(X_train, X_test, y_train, y_test):
    """
    SVR Modeli için Grid Search uygular.
    UYARI: SVR büyük veri setlerinde çok yavaş çalışır.
    """
    print("Starting Grid Search for SVR (This might take a while)...")
    

    param_grid = {
        "C": [0.1, 10, 100],  
        "kernel": ["rbf"],
        "gamma": ["scale"]
    }
    
    grid = GridSearchCV(estimator=SVR(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    
    print(f"Best SVR Params: {grid.best_params_}")
    
    y_pred = grid.predict(X_test)
    evaluate_model(y_test, y_pred, "Optimized SVR")

# --- ANA UYGULAMA AKIŞI ---

if __name__ == "__main__":
    # 1. Veri Yükleme ve Temizlik
    print("Loading data...")
    df = load_and_clean_data("10-diamonds.csv")
    
    # 2. Aykırı Değerlerin Temizlenmesi
    df = remove_outliers(df)
    
    # 3. Ön İşleme (Encoding & Scaling)
    X_train, X_test, y_train, y_test = preprocess_features(df)
    
    # 4. Modelleme
    # Linear Regression (Hızlı ve Baz Model)
    train_linear_regression(X_train, X_test, y_train, y_test)
    
    # SVR (Grid Search) - Yavaş çalışabilir
    train_svr_grid_search(X_train, X_test, y_train, y_test)