import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def load_and_preprocess_data(filepath):
    """
    Loads data, handles outliers, fills missing values, and applies One-Hot Encoding.
    """
    try:
        df = pd.read_csv(filepath)
        
        # 1. Outlier Removal (Using IQR Method on Target Variable)
        target_col = "median_house_value"
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        
        # 2. Missing Value Imputation
        # Filling 'total_bedrooms' with median to be robust against outliers
        df_clean["total_bedrooms"] = df_clean["total_bedrooms"].fillna(df_clean["total_bedrooms"].median())
        
        # 3. Encoding Categorical Variables
        df_clean = pd.get_dummies(df_clean, columns=["ocean_proximity"], drop_first=True)
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def compare_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple regression models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }
    
    results = []
    print("--- Model Comparison Results ---")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        _, rmse, r2 = evaluate_model(y_test, y_pred)
        results.append({"Model": name, "R2 Score": r2, "RMSE": rmse})
        print(f"{name:20} | R2: {r2:.4f} | RMSE: {rmse:.2f}")
        
    return pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

def optimize_xgboost(X_train, y_train):
    """
    Performs RandomizedSearchCV to find best hyperparameters for XGBoost.
    """
    print("\n--- Starting XGBoost Hyperparameter Optimization ---")
    
    params = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 8, 10],
        "n_estimators": [100, 200, 500],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "subsample": [0.5, 0.7, 1.0]
    }
    
    xgb = XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=params,
        n_iter=15,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        scoring='r2'
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best Params: {random_search.best_params_}")
    return random_search.best_estimator_

def plot_results(model, X_test, y_test):
    """
    Plots Actual vs Predicted values and Feature Importance.
    """
    y_pred = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[0], color='blue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_title("Actual vs Predicted Prices")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    
    # Plot 2: Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_test.columns
    
    sns.barplot(x=importances[indices][:10], y=features[indices][:10], ax=axes[1], palette="viridis")
    axes[1].set_title("Top 10 Feature Importances")
    axes[1].set_xlabel("Importance Score")
    
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    filepath = "21-housing.csv" 
    
    df = load_and_preprocess_data(filepath)
    
    if df is not None:
        X = df.drop("median_house_value", axis=1)
        y = df["median_house_value"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
        
        # 1. Compare all base models
        results_df = compare_models(X_train, y_train, X_test, y_test)
        
        # 2. Optimize the best performing model (Usually XGBoost)
        best_xgb = optimize_xgboost(X_train, y_train)
        
        # 3. Final Evaluation
        final_mae, final_rmse, final_r2 = evaluate_model(y_test, best_xgb.predict(X_test))
        print(f"\nFinal Optimized XGBoost -> R2: {final_r2:.4f} | RMSE: {final_rmse:.2f}")
        
        # 4. Visualization
        plot_results(best_xgb, X_test, y_test)