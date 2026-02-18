import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import math
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file '{file_path}' not found. Please check the file path.")
        return None

def analyzing(df):
    pd.set_option('display.max_columns', None)
    print("-"*60)
    print(df.info())
    print("-"*60)
    print(df.describe())
    
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
    sns.heatmap(df.corr(numeric_only=True),annot=True)
    plt.show()

def scaling(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled_matrix = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled_matrix, columns=df.columns)
    return df_scaled

def PrincipalComponentAnalysis(df):
    from sklearn.decomposition import PCA
    pca=PCA()
    pca_df = pd.DataFrame(pca.fit_transform(df))
    plt.step(list(range(1,10)),np.cumsum(pca.explained_variance_ratio_))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylabel("Variance Covered")
    plt.title("Variance Covered")
    plt.show()
    return pca_df

def kmeancluster(df):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator

    print("\n--- K-Means Clustering Başlıyor ---")
    
    # Elbow Method (WCSS)
    wcss = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    # Optimal K Bulma
    kl = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    # Eğer KneeLocator bulamazsa varsayılan 5 yapalım
    final_k = optimal_k if optimal_k else 5 
    
    # Final Model
    model = KMeans(n_clusters=final_k, init="k-means++", random_state=42)
    model.fit(df)
    labels =model.labels_

    # METRİKLER
    print(f"\n--- Metrics for K-Means (k={final_k}) ---")
    s_score = silhouette_score(df, labels)
    df['Class']=labels
    return df

def world_visualization(df2):
    import plotly.express as px
    df2.loc[df2['Class']==0,'Class'] = "Budget Needed"
    df2.loc[df2['Class']==1,'Class'] = "In Between"
    df2.loc[df2['Class']==2,'Class'] = "No Budget Needed"
    fig = px.choropleth(
        df2[['country','Class']],
        locationmode="country names",
        locations= "country",
        title="Needed Budget by Country",
        color=df2["Class"],
        color_discrete_map={
            "Budget Needed":"Red",
            "In Between":"Yellow",
            "No Budged Needed":"Green"
        })
    fig.update_geos(fitbounds = "locations",visible=True)
    fig.show()

def main():
    df2 = load_data("29-country_data.csv")
    #analyzing(df2)
    #plot_all_histograms(df2)
        
    df = df2.drop("country",axis=1)

    df_scaled=scaling(df)

    pca_df=PrincipalComponentAnalysis(df_scaled)
    pca_df=pca_df.iloc[:, :3]

    kmean_df=kmeancluster(pca_df)
    df2['Class']=kmean_df ["Class"]
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
    plt.subplot(1,2,1)
    sns.boxplot(data=df2,x="Class",y="child_mort")
    plt.title("child_mort vs class")
    plt.subplot(1,2,2)
    sns.boxplot(data=df2,x="Class",y="income")
    plt.title("income vs class")
    plt.show()

    """
    2 --> no budget needed 
    1 --> in between
    0 --> budget needed
    """
    world_visualization(df2)
    
if __name__ == "__main__":
    main()