import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator

# --- Yardımcı Fonksiyonlar ---

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file '{file_path}' not found. Please check the file path.")
        return None
    
def plot_all_histograms(df, title_prefix=""):     
    if df is None: return
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return
        
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")
        
    plt.tight_layout()
    plt.show()

def scaling(df):

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

# --- Ana Kümeleme Fonksiyonları ---

def kmeancluster():
    print("\n--- K-Means Clustering Başlıyor ---")
    
    # 1. Veri Yükleme
    df = load_data("26-customer_data.csv")
    if df is None: return

    # Sütun isimlendirme kontrolü
    try:
        X_df = df[['Annual_Income', 'Spending_Score']]
    except KeyError:
        X_df = df[['Annual Income (k$)', 'Spending Score (1-100)']]
        X_df.columns = ['Annual_Income', 'Spending_Score']

    # 2. Ham Veri Görselleştirme
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=X_df, x="Annual_Income", y="Spending_Score")
    plt.title("Raw Data Distribution (K-Means)")
    plt.show()

    # 3. Scaling
    X_scaled = scaling(X_df)

    # 4. Elbow Method (WCSS)
    wcss = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Elbow Grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o')
    plt.xticks(k_range)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    # 5. Optimal K Bulma
    kl = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    print(f"The optimal number of clusters is: {optimal_k}")

    # Eğer KneeLocator bulamazsa varsayılan 5 yapalım
    final_k = optimal_k if optimal_k else 5 
    
    # 6. Final Model
    kmeans_final = KMeans(n_clusters=final_k, init="k-means++", random_state=42)
    y_pred = kmeans_final.fit_predict(X_scaled)
    
    # 7. METRİKLER
    print(f"\n--- Metrics for K-Means (k={final_k}) ---")
    s_score = silhouette_score(X_scaled, y_pred)
    ch_score = calinski_harabasz_score(X_scaled, y_pred)
    db_score = davies_bouldin_score(X_scaled, y_pred)
    
    print(f"Silhouette Score       : {s_score:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.4f}")
    print(f"Davies-Bouldin Score   : {db_score:.4f}")
    print("-" * 30)

    # 8. Sonuç Görselleştirme
    plot_df = pd.DataFrame(X_scaled, columns=X_df.columns)
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df, 
        x="Annual_Income", 
        y="Spending_Score", 
        hue=y_pred,     
        palette='viridis', 
        s=100                 
    )
    plt.title(f"K-Means Results (K={final_k})\nSilhouette: {s_score:.2f}")
    plt.legend(title="Cluster")
    plt.show()

def hierarchicalcluster():
    print("\n--- Hierarchical Clustering Başlıyor ---")
    # 1. Veri Yükleme
    df = load_data("27-mall_customers.csv")
    if df is None: return

    # 2. Histogramlar
    plot_all_histograms(df, title_prefix="Hist: ")

    # 3. Preprocessing
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
        
    # Dendrogram için tüm veriyi scale et
    df_scaled_matrix = scaling(df)
    df_scaled = pd.DataFrame(df_scaled_matrix, columns=df.columns)

    plt.figure(1, figsize=(10, 8))
    # Dendrogram çizimi
    dendogram = sch.dendrogram(sch.linkage(df_scaled, method="ward"))
    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    plt.show()

    # 4. Feature Kombinasyonları ve Skorlar
    features_2D = ["Annual Income (k$)", "Spending Score (1-100)"]
    features_3D = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    features_4D = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    
    for feats in [features_2D, features_3D, features_4D]:
        # İlgili featureları seç
        x_subset = df[feats]
        
        # Scale et
        x_scaled = scaling(x_subset)
        
        # Modeli kur
        xhc = AgglomerativeClustering(n_clusters=5)
        yhc = xhc.fit_predict(x_scaled)
        
        # Skorları yazdır
        print(f"\nFeatures: {feats}")
        print("Silhouette Score       : ", silhouette_score(x_scaled, yhc))
        print("Calinski-Harabasz Score: ", calinski_harabasz_score(x_scaled, yhc))
        print("Davies-Bouldin Score   : ", davies_bouldin_score(x_scaled, yhc))
        print("-" * 30)

        # Sadece 2D olanı görselleştirelim 
        if len(feats) == 2:
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=x_subset.iloc[:,0], y=x_subset.iloc[:,1], hue=yhc, palette="Set2")
            plt.title(f"Hierarchical Clusters (2D features)")
            plt.show()

def dbscancluster():
    print("\n--- DBScan Clustering Başlıyor ---")
    # 1. Veri Yükleme
    df = load_data("28-urban_pedestrian_locations_with_labels.csv")
    if df is None: return

    # 2. Histogramlar
    plot_all_histograms(df, title_prefix="Hist: ")

    # 3. Preprocessing
    if 'true_cluster' in df.columns:
        df = df.drop("true_cluster",axis=1)

    # 4. Scaling
    df_scaled_matrix = scaling(df)
    df_scaled=pd.DataFrame(df_scaled_matrix,columns=["x_position","y_position"])

    # model 
    eps_values= [0.1,0.2,0.3,0.4,0.5,0.6]
    min_samples_values = [4,5,6]
    results=[]
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps,min_samples=min_samples).fit(df_scaled)
            labels=dbscan.labels_
            if len(set(labels))<=1:
                continue
            
            silhouette=silhouette_score(df_scaled,labels)
            results.append(
                {
                    "eps":eps,
                    "min_samples":min_samples,
                    "silhoutte_score":silhouette,
                    "n_clusters":len(set(labels))-(1 if -1 in labels else 0)
                }
                )
    results_df=pd.DataFrame(results)
    results_df=results_df.sort_values(by="silhoutte_score",ascending=False)
    print (results_df)


def main():
    kmeancluster()
    hierarchicalcluster()
    dbscancluster()

if __name__ == "__main__":
    main()