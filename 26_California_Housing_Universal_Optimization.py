import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Modeller
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Metrikler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Ayarlar
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ==========================================
# PROJE: California Housing Price Prediction (XGBoost & Pipelines) universal tuning and transformed data
# ==========================================

def load_data(filepath):
    """Veriyi yükler, temizler ve One-Hot Encoding uygular."""
    try:
        df = pd.read_csv(filepath) 
        
        # 1. Eksik Veri Doldurma
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
        
        # 2. Outlier Temizliği (IQR Yöntemi)
        Q1 = df["median_house_value"].quantile(0.25)
        Q3 = df["median_house_value"].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df["median_house_value"] < (Q1 - 1.5 * IQR)) | (df["median_house_value"] > (Q3 + 1.5 * IQR)))]
        
        # 3. Encoding
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
        
        return df
    except FileNotFoundError:
        print(f"Hata: '{filepath}' dosyası bulunamadı.")
        return None

def compare_models(X_train, y_train, X_test, y_test):
    """Farklı regresyon modellerini Pipeline ile karşılaştırır (Baseline)."""
    
    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "K-Neighbors": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    
    results = []
    print("\n--- Model Karşılaştırma Sonuçları (Baseline) ---")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({"Model": name, "R2 Score": r2, "RMSE": rmse})
        print(f"{name:20} | R2: {r2:.4f} | RMSE: {rmse:.2f}")
        
    return pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

def optimize_any_model(model, param_grid, X_train, y_train, model_name="Model"):
    """
    Herhangi bir model ve parametre setini alıp RandomizedSearchCV uygular.
    """
    print(f"\n--- {model_name} Hiperparametre Optimizasyonu Başlıyor ---")
    
    # Eğer parametre grid boşsa (Linear Regression gibi), direkt eğit ve döndür
    if not param_grid:
        model.fit(X_train, y_train)
        print(f"{model_name} için optimize edilecek parametre yok, varsayılan eğitildi.")
        return model

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10, 
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        scoring='r2'
    )
    
    random_search.fit(X_train, y_train)
    print(f"En İyi Parametreler: {random_search.best_params_}")
    return random_search.best_estimator_
# ----------------------------------------------

def plot_results(model, X_test, y_test, model_name="Best Model"):
    """Gerçek vs Tahmin ve Feature Importance grafiklerini çizer."""
    y_pred = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Grafik 1: Gerçek vs Tahmin
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[0], color='blue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_title(f"{model_name}: Actual vs Predicted Prices")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    
    # Grafik 2: Feature Importance
    # Pipeline içindeki modeli bulmaya çalışalım
    if isinstance(model, Pipeline):
        if 'model' in model.named_steps:
            estimator = model.named_steps['model']
        else:
            estimator = model # Bulamazsa kendisi
    else:
        estimator = model

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        features = X_test.columns
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices][:10], y=features[indices][:10], ax=axes[1], palette="viridis")
        axes[1].set_title("Top 10 Feature Importances")
    else:
        axes[1].text(0.5, 0.5, "Bu model Feature Importance desteklemiyor.", ha='center')
    
    plt.tight_layout()
    plt.show()

def check_skewness(df):
    """Verideki çarpıklığı kontrol eder."""
    print("\n--- Skewness Değerleri ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skew_vals = df[numeric_cols].apply(skew).sort_values(ascending=False)
    print(skew_vals.head(5))

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    filepath = "21-housing.csv" 
    
    df = load_data(filepath)
    
    if df is not None:
        check_skewness(df)
        
        X = df.drop("median_house_value", axis=1)
        y = df["median_house_value"]
        
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Power Transformation
        numeric_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                        'total_bedrooms', 'population', 'households', 'median_income']
        
        pt = PowerTransformer(method='yeo-johnson')
        X_train[numeric_cols] = pt.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = pt.transform(X_test[numeric_cols])
        
        print("\nVeri dönüşümü ve ayrımı tamamlandı. Modellemeye geçiliyor...")

        # 1. Modelleri Karşılaştır (Base Models)
        # results_df = compare_models(X_train, y_train, X_test, y_test)
        
        # ---------------------------------------------------------
        # 2. TÜM MODELLER VE HİPERPARAMETRE IZGARALARI (GRIDS)
        # ---------------------------------------------------------
    
        models_and_params = [
            {
                "name": "Linear Regression",
                "model": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
                "params": {} # Optimize edilecek bir şey yok
            },
            {
                "name": "Lasso",
                "model": Pipeline([("scaler", StandardScaler()), ("model", Lasso())]),
                "params": {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}
            },
            {
                "name": "Ridge",
                "model": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
                "params": {"model__alpha": [0.01, 0.1, 1, 10, 100]}
            },
            {
                "name": "K-Neighbors",
                "model": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
                "params": {"model__n_neighbors": [3, 5, 7, 9, 11], "model__weights": ["uniform", "distance"]}
            },
            {
                "name": "Decision Tree",
                "model": DecisionTreeRegressor(random_state=42),
                "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
            },
            {
                "name": "Random Forest",
                "model": RandomForestRegressor(random_state=42),
                "params": {"n_estimators": [50, 100], "max_depth": [10, 20, None], "min_samples_split": [2, 5]}
            },
            {
                "name": "AdaBoost",
                "model": AdaBoostRegressor(random_state=42),
                "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1.0]}
            },
            {
                "name": "Gradient Boosting",
                "model": GradientBoostingRegressor(random_state=42),
                "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
            },
            {
                "name": "XGBoost",
                "model": XGBRegressor(random_state=42, objective='reg:squarederror'),
                "params": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "n_estimators": [100, 300],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                }
            }
        ]

        # ---------------------------------------------------------
        # 3. DÖNGÜ: MODELLERİ GEZ, OPTİMİZE ET VE KAYDET
        # ---------------------------------------------------------
        
        results = []
        best_overall_model = None
        best_overall_r2 = -np.inf
        best_model_name = ""
        
        print(f"\n{'='*20} MODEL OPTİMİZASYON SÜRECİ BAŞLIYOR {'='*20}")
        
        for entry in models_and_params:
            name = entry["name"]
            model = entry["model"]
            params = entry["params"]
            
            # Fonksiyonu çağır ve optimize edilmiş modeli al
            optimized_model = optimize_any_model(model, params, X_train, y_train, model_name=name)
            
            # Test verisiyle gerçek performans ölçümü
            y_pred = optimized_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Sonuçları kaydet
            results.append({"Model": name, "R2 Score": r2, "RMSE": rmse})
            print(f">>> {name} Test Sonucu -> R2: {r2:.4f} | RMSE: {rmse:.2f}")
            
            # Lider model kontrolü
            if r2 > best_overall_r2:
                best_overall_r2 = r2
                best_overall_model = optimized_model
                best_model_name = name

        # ---------------------------------------------------------
        # 4. FİNAL RAPORU
        # ---------------------------------------------------------
        
        results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
        print("\n" + "="*50)
        print("FİNAL LİDERLİK TABLOSU (LEADERBOARD)")
        print("="*50)
        print(results_df)
        
        print(f"\n🏆 KAZANAN MODEL: {best_model_name} (R2: {best_overall_r2:.4f})")
        
        # En iyi modelin detaylı grafiği
        plot_results(best_overall_model, X_test, y_test, model_name=best_model_name)