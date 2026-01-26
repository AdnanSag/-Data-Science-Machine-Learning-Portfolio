import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def plot_elbow_method(X_train, X_test, y_train, y_test, mode="classification"):
    """
    En iyi 'K' değerini bulmak için hata oranlarını görselleştirir (Elbow Method).
    Bu grafik, modelin neden o K değeriyle kurulduğunu açıklar.
    """
    error_rates = []
    k_range = range(1, 20)
    
    for i in k_range:
        if mode == "classification":
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            # Hata oranı = 1 - Doğruluk oranı
            error_rates.append(np.mean(pred_i != y_test))
        else:
            knn = KNeighborsRegressor(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            # Regresyon için MSE (Hata Kareler Ortalaması) kullanıyoruz
            error_rates.append(mean_squared_error(y_test, pred_i))

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, error_rates, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title(f'Error Rate vs. K Value ({mode.capitalize()})')
    plt.xlabel('K Value')
    plt.ylabel('Error Rate / MSE')
    plt.show()

# --- BÖLÜM 1: KNN CLASSIFICATION (SAĞLIK RİSKİ ANALİZİ) ---
def run_classification_task(filepath):
    print("\n" + "="*50)
    print("TASK 1: Health Risk Classification (KNN)")
    print("="*50)
    
    try:
        df = pd.read_csv(filepath)
        X = df.drop("high_risk_flag", axis=1)
        y = df["high_risk_flag"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # En iyi K değerini görmek için grafik çizdiriyoruz
        # plot_elbow_method(X_train_scaled, X_test_scaled, y_train, y_test, mode="classification")
        
        # Model Eğitimi (n_neighbors=5 varsayılan, Elbow metoduna göre değiştirilebilir)
        classifier = KNeighborsClassifier(n_neighbors=5, algorithm="auto", weights="uniform")
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Health Risk - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")

# --- BÖLÜM 2: KNN REGRESSION (ENERJİ TÜKETİMİ TAHMİNİ) ---
def run_regression_task(filepath):
    print("\n" + "="*50)
    print("TASK 2: House Energy Regression (KNN)")
    print("="*50)
    
    try:
        df_reg = pd.read_csv(filepath)
        X_reg = df_reg.drop("daily_energy_consumption_kwh", axis=1)
        y_reg = df_reg["daily_energy_consumption_kwh"]
        
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=15)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model Eğitimi
        regressor = KNeighborsRegressor(n_neighbors=5, algorithm="auto")
        regressor.fit(X_train_scaled, y_train)
        y_pred = regressor.predict(X_test_scaled)
        
        # Değerlendirme Metrikleri
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"R2 Score : {r2:.4f}")
        print(f"MAE      : {mae:.4f}")
        print(f"MSE      : {mse:.4f}")
        
        # Gerçek vs Tahmin Grafiği
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='orange')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('KNN Regression: Actual vs Predicted Energy')
        plt.xlabel('Actual Consumption')
        plt.ylabel('Predicted Consumption')
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    # Classification Görevi
    run_classification_task("12-health_risk_classification.csv")
    
    # Regression Görevi
    run_regression_task("12-house_energy_regression.csv")