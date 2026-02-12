import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from kneed import KneeLocator
import math
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
def main():
    #print("k_mean \n")
    #kmeancluster()
    print("hierarchical \n")
    hierarchiccluster()

def hierarchiccluster():
    try:
        df=pd.read_csv("27-mall_customers.csv")
    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
        return

    def plot_all_histograms(df,title_prefix=""):        
        num_cols=df.select_dtypes(include=[np.number]).columns
        n_cols=3
        n_rows=math.ceil(len(num_cols)/n_cols)
        plt.figure(figsize=(5*n_cols,4*n_rows))
        for i , col in enumerate(num_cols,1):
            plt.subplot(n_rows,n_cols,i)
            sns.histplot(df[col],kde=True,bins=30)
            plt.title(f"{title_prefix}{col}")
            plt.xlabel("")
            plt.ylabel("")
        plt.tight_layout()
        plt.show()

    plot_all_histograms(df)
    label_encoder=LabelEncoder()
    df['Gender']=label_encoder.fit_transform(df['Gender'])
    df=df.drop('CustomerID',axis=1)
    scaler=MinMaxScaler()
    df_scaled=scaler.fit_transform(df)
    df=pd.DataFrame(df_scaled,columns=df.columns)
    import scipy.cluster.hierarchy as sch
    plt.figure(1,figsize=(10,8))
    dendogram=sch.dendrogram(sch.linkage(df,method="ward"))
    plt.title("Dendogram")
    plt.xlabel("customers")
    plt.ylabel("distance")
    plt.show()
    from sklearn.cluster import AgglomerativeClustering
    X=df[["Annual Income (k$)","Spending Score (1-100)"]].copy()
    hc=AgglomerativeClustering(n_clusters=5)
    y_hc=hc.fit_predict(X)    
    print(silhouette_score(X,y_hc))
    df['cluster']= y_hc
    sns.scatterplot(data=df,x="Annual Income (k$)",y="Spending Score (1-100)",hue="cluster",palette="Set2")
    plt.title("Customer Clusters")
    plt.show()

    features_2D=["Annual Income (k$)","Spending Score (1-100)"]
    features_3D=["Age","Annual Income (k$)","Spending Score (1-100)"]
    features_4D=["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]
    for feats in[features_2D,features_3D,features_4D]:
        x=df[feats]
        x_scaled=MinMaxScaler().fit_transform(x)
        xhc=AgglomerativeClustering(n_clusters=5)
        yhc=xhc.fit_predict(x_scaled)
        print(f"\n features:{feats}")
        print("silhouette_score ",silhouette_score(x_scaled,yhc))
        print("calinski_harabasz_score ",calinski_harabasz_score(x_scaled,yhc))
        print("davies_bouldin_score ",davies_bouldin_score(x_scaled,yhc))
        print("-"*30)



def kmeancluster():
    try:
        df = pd.read_csv("26-customer_data.csv")
    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
        return

    X = df[['Annual_Income', 'Spending_Score']]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score")
    plt.title("Raw Data Distribution")
    plt.show()

    X_train, X_test = train_test_split(X, random_state=15, test_size=0.2)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    wcss = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmeans.fit(X_train_scaled)
        wcss.append(kmeans.inertia_)
    
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o')
    plt.xticks(k_range)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    kl = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    print(f"The optimal number of clusters is: {optimal_k}")

    final_k = optimal_k if optimal_k else 5 
    
    kmeans_final = KMeans(n_clusters=final_k, init="k-means++", random_state=42)
    kmeans_final.fit(X_train_scaled)
    
    y_pred = kmeans_final.predict(X_test_scaled)

    plot_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df, 
        x="Annual_Income", 
        y="Spending_Score", 
        hue=y_pred,     
        palette='viridis', 
        s=100                 
    )
    plt.title(f"Clustered Data (K={final_k})")
    plt.show()

    silhouette_scores = []
    for k in range(2, 11):
        kmeans_sil = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmeans_sil.fit(X_train_scaled) # <--- Fixed: Fitting the CORRECT instance
        score = silhouette_score(X_train_scaled, kmeans_sil.labels_)
        silhouette_scores.append(score)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o', color='red')
    plt.xticks(range(2, 11))
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Scores")
    plt.grid(True)
    plt.show()


    
if __name__ == "__main__":
    main()