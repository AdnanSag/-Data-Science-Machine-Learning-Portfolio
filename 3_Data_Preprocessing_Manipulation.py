import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Görselleştirme ayarları
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================
# BÖLÜM 1: NUMPY & TEMEL DİZİ İŞLEMLERİ
# Amaç: Vektörel işlemlerin ve matris yapısının anlaşılması
# ==========================================

def numpy_basics():
    print("--- NumPy Basics ---")
    
    # Dizi Oluşturma
    my_list = [1, 2, 3, 4, 5]
    np_array = np.array(my_list)
    zeros = np.zeros(3)
    ones = np.ones((3, 4)) # 3x4'lük 1'ler matrisi
    
    # Vektörel İşlemler (Element-wise Operations)
    arr1 = np.array([1, 2])
    arr2 = np.array([2, 3])
    sum_arr = arr1 + arr2  # Çıktı: [3, 5]
    
    print(f"Array Sum: {sum_arr}")
    print(f"Ones Matrix Shape: {ones.shape}")

# ==========================================
# BÖLÜM 2: PANDAS İLE VERİ MANİPÜLASYONU
# Amaç: Dataframe oluşturma, temizleme ve birleştirme
# ==========================================

def pandas_operations():
    print("\n--- Pandas Operations ---")
    
    # 1. Series ve DataFrame Oluşturma
    series_dict = pd.Series({"Adnan": 1, "Mehmet": 2})
    
    # Random Data Frame
    data = np.random.randn(3, 4)
    df = pd.DataFrame(data, index=["Row1", "Row2", "Row3"], columns=["A", "B", "C", "D"])
    
    # Seçim ve Filtreleme
    print(f"Select Column A:\n{df['A']}")
    print(f"Select Row by Index (iloc):\n{df.iloc[1]}")
    
    # Sütun Ekleme/Çıkarma
    df["Extra"] = 10
    df.drop("Extra", axis=1, inplace=True)
    
    # 2. Dosya Okuma ve İstatistiksel Özet (Örnek Kod)
    # df_weather = pd.read_excel('weather.xlsx') 
    # print(df_weather.describe()) 
    # print(df_weather.isna().sum()) # Eksik veri kontrolü

# ==========================================
# BÖLÜM 3: VERİ BİRLEŞTİRME (MERGE & CONCAT)
# Amaç: İlişkisel veri tabanı mantığıyla tabloları birleştirme
# ==========================================

def merge_concat_examples():
    # Örnek Veri Setleri
    df1 = pd.DataFrame({'ID': [1, 2], 'Val1': ['A', 'B']})
    df2 = pd.DataFrame({'ID': [1, 2], 'Val1': ['C', 'D']})
    
    # Concat (Alt alta ekleme)
    df_concat = pd.concat([df1, df2], ignore_index=True)
    
    # Merge (Yan yana birleştirme - Join mantığı)
    df_left = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'A': ['A0', 'A1', 'A2']})
    df_right = pd.DataFrame({'key': ['K0', 'K1', 'K3'], 'B': ['B0', 'B1', 'B2']})
    
    # Inner Join (Kesişim)
    df_merge_inner = pd.merge(df_left, df_right, on='key', how='inner')
    
    print("\n--- Merge Operation (Inner) ---")
    print(df_merge_inner)

# ==========================================
# BÖLÜM 4: GÖRSELLEŞTİRME (MATPLOTLIB & SEABORN)
# Amaç: Veri dağılımını ve ilişkileri görselleştirme
# ==========================================

def visualization_demo():
    age = [20, 25, 30, 35, 40]
    weight = [60, 65, 75, 80, 85]
    sex = ['F', 'M', 'M', 'F', 'M']
    
    # Matplotlib Subplots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(age, weight, "r*-")
    plt.title("Matplotlib: Age vs Weight")
    plt.xlabel("Age")
    plt.ylabel("Weight")
    
    # Seaborn Scatterplot
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=age, y=weight, hue=sex, s=100)
    plt.title("Seaborn: Age vs Weight by Gender")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# BÖLÜM 5: DENGESİZ VERİ SETLERİ (IMBALANCED DATA)
# Amaç: Resampling ve SMOTE teknikleri ile sınıf dengesizliğini çözme
# ==========================================

def handling_imbalanced_data():
    print("\n--- Imbalanced Data Handling ---")
    
    # Sentetik Dengesiz Veri Oluşturma
    np.random.seed(42)
    n_majority = 900
    n_minority = 100
    
    df_majority = pd.DataFrame({'feature': np.random.randn(n_majority), 'target': 0})
    df_minority = pd.DataFrame({'feature': np.random.randn(n_minority), 'target': 1})
    
    df_imbalanced = pd.concat([df_majority, df_minority])
    print(f"Original Class Distribution:\n{df_imbalanced['target'].value_counts()}")
    
    # 1. Upsampling (Azınlık sınıfını çoğaltma)
    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=n_majority, 
                                     random_state=42)
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    print(f"Upsampled Class Distribution:\n{df_upsampled['target'].value_counts()}")
    
    # 2. SMOTE (Sentetik Veri Üretimi)
    # Not: SMOTE özellik uzayında interpolasyon yapar
    smote = SMOTE(random_state=42)
    X = df_imbalanced[['feature']]
    y = df_imbalanced['target']
    
    X_smote, y_smote = smote.fit_resample(X, y)
    print(f"SMOTE Resampled Shape: {X_smote.shape}")

# ==========================================
# BÖLÜM 6: ENCODING (KATEGORİK VERİ DÖNÜŞÜMÜ)
# Amaç: Kategorik verileri sayısal modellere hazırlama
# ==========================================

def encoding_techniques():
    print("\n--- Encoding Techniques ---")
    
    # Örnek Veri: Titanic benzeri
    df = pd.DataFrame({
        'Sex': ['male', 'female', 'female', 'male'],
        'Class': ['Third', 'First', 'Second', 'First'],
        'Town': ['S', 'C', 'S', 'Q']
    })
    
    # 1. Label Encoding (Binary sınıflar için ideal)
    le = LabelEncoder()
    df['Sex_Encoded'] = le.fit_transform(df['Sex'])
    
    # 2. Ordinal Encoding (Sıralı kategoriler için - Low < Medium < High)
    # Sıralamayı biz belirliyoruz: Third < Second < First
    class_order = [['Third', 'Second', 'First']]
    oe = OrdinalEncoder(categories=class_order)
    df['Class_Encoded'] = oe.fit_transform(df[['Class']])
    
    # 3. One-Hot Encoding (Sırasız nominal veriler için)
    # drop_first=True -> Dummy Variable Trap'ten kaçınmak için
    df_onehot = pd.get_dummies(df, columns=['Town'], drop_first=True)
    
    print(df_onehot[['Sex_Encoded', 'Class_Encoded', 'Town_Q', 'Town_S']].head())

if __name__ == "__main__":
    # Fonksiyonları sırayla çalıştır
    numpy_basics()
    pandas_operations()
    # merge_concat_examples()
    # visualization_demo()
    handling_imbalanced_data()
    encoding_techniques()