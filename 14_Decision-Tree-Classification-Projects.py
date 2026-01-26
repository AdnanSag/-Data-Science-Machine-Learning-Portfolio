import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

def evaluate_model(y_test, y_pred, title="Model Evaluation"):
    """
    Model performansını metriklerle raporlar.
    """
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title(f'{title} - Confusion Matrix')
    plt.show()

# --- PROJE 1: CAR EVALUATION DATASET (Kategorik Veri & Ordinal Encoding) ---
def run_car_evaluation_project(filepath):
    print("\n" + "="*50)
    print("PROJECT 1: Car Evaluation Analysis")
    print("="*50)
    
    try:
        # Veri Yükleme ve Başlıklandırma
        col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        df = pd.read_csv(filepath, names=col_names, header=0)
        
        # Veri Temizliği (String -> Int Dönüşümü)
        # '5more' -> 5 ve 'more' -> 5 dönüşümleri
        df["doors"] = df["doors"].replace("5more", "5").astype(int)
        df["persons"] = df["persons"].replace("more", "5").astype(int)
        
        X = df.drop("class", axis=1)
        y = df["class"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
        
        # Ön İşleme: Kategorik verilerde sıralama (Ranking) olduğu için OrdinalEncoder kullanıyoruz.
        categorical_cols = ["buying", "maint", "lug_boot", "safety"]
        numerical_cols = ["doors", "persons"]
        
        ordinal_encoder = OrdinalEncoder(categories=[
            ["low", "med", "high", "vhigh"], # buying
            ["low", "med", "high", "vhigh"], # maint
            ["small", "med", "big"],         # lug_boot
            ["low", "med", "high"]           # safety
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", ordinal_encoder, categorical_cols)
            ], 
            remainder="passthrough" # Sayısal sütunları (doors, persons) olduğu gibi bırak
        )
        
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Hyperparameter Tuning (GridSearchCV)
        print("Running GridSearchCV for Decision Tree...")
        param_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [3, 5, 10, 15, None],
            "max_features": ["sqrt", "log2", None]
        }
        
        grid = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), 
                            param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train_transformed, y_train)
        
        print(f"Best Parameters: {grid.best_params_}")
        
        # En iyi model ile tahmin
        best_tree = grid.best_estimator_
        y_pred = best_tree.predict(X_test_transformed)
        evaluate_model(y_test, y_pred, "Optimized Decision Tree (Car)")
        
        # Ağaç Görselleştirme
        # Not: ColumnTransformer sıralamayı değiştirdiği için feature_names listesini yeniden düzenliyoruz
        feature_names_ordered = categorical_cols + numerical_cols
        
        plt.figure(figsize=(16, 10))
        plot_tree(best_tree, feature_names=feature_names_ordered, 
                  class_names=best_tree.classes_, filled=True, rounded=True, fontsize=10)
        plt.title("Optimized Decision Tree Visualization")
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")

# --- PROJE 2: IRIS DATASET (Basit Karar Ağacı) ---
def run_iris_project(filepath):
    print("\n" + "="*50)
    print("PROJECT 2: Iris Species Classification")
    print("="*50)
    
    try:
        df = pd.read_csv(filepath)
        
        # Gereksiz ID sütunu varsa atalım
        if "Id" in df.columns:
            df = df.drop("Id", axis=1)
            
        X = df.drop("Species", axis=1)
        y = df["Species"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        
        # Model Eğitimi (Default Parametreler)
        tree_model = DecisionTreeClassifier(random_state=10)
        tree_model.fit(X_train, y_train)
        
        y_pred = tree_model.predict(X_test)
        evaluate_model(y_test, y_pred, "Decision Tree (Iris)")
        
        # Görselleştirme
        plt.figure(figsize=(12, 8))
        plot_tree(tree_model, feature_names=X.columns, 
                  class_names=tree_model.classes_, filled=True)
        plt.title("Iris Decision Tree")
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    # Car Evaluation Projesi
    run_car_evaluation_project("13-car_evaluation.csv")
    
    # Iris Projesi
    run_iris_project("11-iris.csv")