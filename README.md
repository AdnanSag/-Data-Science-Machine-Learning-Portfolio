#### (ENGLISH) <img src="https://flagcdn.com/w40/gb.png" width="32" alt="EN" style="vertical-align: middle;">
# 🚀 Data Science & Machine Learning Portfolio

Welcome! This repository serves as a portfolio showcasing my competencies in Data Science and Machine Learning. It contains a collection of Python scripts demonstrating the end-to-end machine learning lifecycle: from data cleaning and feature engineering to model building and hyperparameter optimization.

All project files are located in the root directory for easy access.

## 🛠️ Tech Stack & Libraries
* **Languages:** Python, SQL
* **Data Analysis:** Pandas, NumPy, SciPy
* **Visualization:** Matplotlib, Seaborn ,Plotly Express
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn, LazyPredict
* **Algorithms:** Linear/Logistic Regression, SVM, KNN, Decision Trees, Random Forest, AdaBoost, Gradient Boosting, LightGBM, Principal Component Analysis (PCA), K-Means, Hierarchical Clustering, DBSCAN
* **Optimization Tools:** Kneed (KneeLocator)

---

## 📂 Project Catalog

The projects are categorized below based on the problem type (Regression, Classification, etc.).

### 🔹 Regression Projects (Predicting Continuous Values)
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `1_Doğrusal_Regresyon_StudyHours_ExamScore).py`* | Predicting Student Exam Scores. | **Simple vs Multiple Linear Regression**, Feature Scaling, Prediction on new data. |
| `2_Polinom_Regresyon_CustomerSatisfaction.py`* | Analyzing Model Complexity. | **Polynomial Regression**, Visualizing **Overfitting**, Scikit-Learn **Pipelines**. |
| `4_Regularization_Lasso_vs_Linear.py`* | Regularization & Feature Selection. | **Lasso (L1) vs Linear Regression**, Importance of **Scaling** in regularization. |
| `5_FWI_Regression_AutoML_Analysis.py`* | Predicting Fire Weather Index (FWI). | Data Cleaning, Correlation Analysis, AutoML. |
| `11_SVM-Diamond-Price-Prediction-Regression.py`* | Estimating Diamond Prices. | **SVR vs Linear Regression** trade-off, Outlier Removal. |
| `18_Gym-Crowdedness-Prediction-Analysis.py`* | Predicting Gym Crowdedness. | **Random Forest** performance on non-linear time-series data. |
| `6_WWII_Weather_Regression_Analysis.py`* | WWII Weather Temperature Analysis. | **LassoCV** (Regularization), Date/Time feature extraction. |
| `22_California_Housing_Price_Prediction_XGBoost.py`* | Predicting House Prices. | **Model Benchmarking** (Linear vs Trees), **XGBoost Hyperparameter Tuning**, Outlier Removal. |
| `24_Boston_Housing_Power_Transformation.py`* | Improving Linear Regression Performance. | **Yeo-Johnson & Box-Cox Transformations**, Handling Skewness. |
| `25_Medical_Cost_LGBM_BoxCox_Regression.py`* | Predicting Medical Insurance Costs. | **Box-Cox Target Transformation**, LightGBM, RandomizedSearchCV. |
| `26_California_Housing_Universal_Optimization.py`* | Automated Model Selection Pipeline. | **Universal Hyperparameter Tuning Function**, 9-Model Leaderboard, PowerTransformer. |
### 🔹 Classification Projects (Predicting Categories)
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `21_Stellar-Classification-XGBoost.py`* | Classifying Stars, Galaxies, and Quasars. | **XGBoost**, Feature Importance (Redshift), Multiclass Classification. |
| `16_Diabetes-Prediction-ML-Pipeline.py`* | End-to-End Diabetes Diagnosis Pipeline. | **Preventing Data Leakage** (Median Imputation), Model Comparison. |
| `17_Diabetes-Ensemble-Learning-AdaBoost.py`* | Diabetes Prediction (Ensemble). | **AdaBoost** algorithm and manual parameter tuning. |
| `8_Logistic-Regression-Hyperparameter-Tuning.py`* | Bank Customer Churn Analysis. | Logistic Regression, Scaling, Confusion Matrix Analysis. |
| `12_Naive_bayes-Iris-Species-Classification-Comparison.py`* | Iris Species Classification. | **Naive Bayes vs SVM** (Kernel Trick comparison). |
| `14_Decision-Tree-Classification-Projects.py`* | Car Evaluation & Iris Analysis. | **Ordinal Encoding**, Decision Tree Visualization. |
| `20_Gradient-Boosting-Advanced-Analysis.py`* | Heart Disease & Concrete Strength. | **Gradient Boosting** (Reg & Clf), Correlation Filtering. |
| `23_Titanic_LightGBM_XGBoost_Comparison.py`* | Titanic Survival Analysis. | **LightGBM vs XGBoost**, RandomizedSearchCV, Feature Importance Visualization. |

### 🔹 Advanced Machine Learning Applications
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `19_Car_Price_Prediction_Adaboost.py`* | Used Car Price Prediction. | Handling High Cardinality with **Frequency Encoding**, AdaBoost. |
| `15_Ensemble-Techniques.py`* | Income Bracket Prediction (>50K). | Handling **Imbalanced Data**, **Target Encoding**, Robust Scaler. |
| `10_Advanced_Logistic_Regression.py`* | Fraud & Cyber Attack Detection. | **Class Weights** optimization, One-vs-Rest / One-vs-One strategies. |
| `13_KNN-Health-Energy-Analysis.py`* | Health Risk & Energy Consumption. | Finding optimal 'K' with **Elbow Method**, Feature Scaling. |
| `9_SVM-Multi-Domain-Analysis.py`* | Email Spam, Loan Risk & Seismic. | SVM Kernel Comparison (Linear, RBF, Poly). |
| `27_Student_Performance_Dual_Pipeline.py`* | **Dual-Task Student Performance Analysis.** Predicting exact scores (Regression) and Pass/Fail status (Classification). | **Dual-Task Modeling (Reg & Clf)**, RandomizedSearchCV, **Outlier Capping (IQR)**, Automated Model Leaderboard. |
| `28_PCA_Breast_Cancer_Analysis.py`* | PCA & Model Performance Analysis. | **Principal Component Analysis (PCA)**, Dimensionality Reduction Trade-offs, **StandardScaler**, Logistic vs Gradient Boosting Comparison. |

### 🔹 Data Science Fundamentals & Techniques
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `3_Data_Preprocessing_Manipulation.py`* | Comprehensive Toolkit for Data Science. | **SMOTE** (Imbalanced Data), **One-Hot/Ordinal Encoding**, Pandas Merging, Data Visualization. |
| `7_Python_SQL_Database_Basics.py`* | Student Database System with SQL. | SQLite, Table Creation (DDL), Data Manipulation (DML). |
---

### 🔹 Unsupervised Learning & Dimensionality Reduction
| File Name | Description | Key Techniques |
| :--- | :--- | :--- |
| `29_Clustering_Algorithms_Analysis.py` | Comparison of K-Means, Agglomerative, and DBSCAN. | Silhouette Analysis, Dendrograms, EPS Optimization. |
| `30_Country_Segmentation_PCA_KMeans.py` | Strategic Country Analysis for Resource Allocation. | PCA (3-Components), K-Means, Interative Geographic Visualization. |

## 📊 Methodology & Approach

In these projects, I followed the standard Data Science lifecycle:
1.  **EDA (Exploratory Data Analysis):** Understanding data distributions, outliers, and correlations.
2.  **Preprocessing:** Advanced Data Cleaning including IQR Outlier Capping, Dimensionality Reduction (PCA) for high-dimensional datasets , Handling missing values (strictly preventing leakage), Scaling (Standard/Robust), and Encoding (One-Hot/Target/Frequency).
3. **Model Selection:** Comparing Linear models vs. Tree-based Ensembles for supervised tasks, and evaluating K-Means vs. DBSCAN for clustering problems.
4.  **Optimization:** Using `GridSearchCV` and `RandomizedSearchCV` to fine-tune hyperparameters.
5.  **Evaluation:** Going beyond Accuracy; utilizing F1-Score, RMSE, R2 Score, and Confusion Matrices for robust evaluation.

---

## 💻 Installation & Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/AdnanSag/-Data-Science-Machine-Learning-Portfolio.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run a specific script:
    ```bash
    python 16_Diabetes-Prediction-ML-Pipeline.py
    ```
---




#### (TÜRKÇE) <img src="https://flagcdn.com/w40/tr.png" width="32" alt="TR" style="vertical-align: middle;">
# 🚀 Veri Bilimi ve Makine Öğrenmesi Portfolyosu

Hoş geldiniz! Bu depo, Veri Bilimi ve Makine Öğrenmesi alanındaki yetkinliklerimi sergileyen bir portfolyo niteliğindedir. Veri temizleme ve özellik mühendisliğinden (feature engineering) model kurma ve hiperparametre optimizasyonuna kadar uçtan uca makine öğrenmesi yaşam döngüsünü gösteren bir dizi Python betiği (script) içerir. 

Tüm proje dosyaları, kolay erişim için kök dizinde (root directory) bulunmaktadır.

## 🛠️ Teknoloji Yığını ve Kütüphaneler
* **Diller:** Python, SQL
* **Veri Analizi:** Pandas, NumPy, SciPy
* **Görselleştirme:** Matplotlib, Seaborn, Plotly Express
* **Makine Öğrenmesi:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn, LazyPredict
* **Algoritmalar:** Doğrusal/Lojistik Regresyon (Linear/Logistic Regression), SVM, KNN, Karar Ağaçları (Decision Trees), Rastgele Orman (Random Forest), AdaBoost, Gradient Boosting, LightGBM, Temel Bileşenler Analizi (PCA), K-Means, Hiyerarşik Kümeleme (Hierarchical Clustering), DBSCAN
* **Optimizasyon Araçları:** Kneed (KneeLocator)

---

## 📂 Proje Kataloğu

Projeler, problem türüne göre (Regresyon, Sınıflandırma vb.) aşağıda kategorize edilmiştir.

### 🔹 Regresyon Projeleri (Sürekli Değer Tahmini)
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `1_Doğrusal_Regresyon_StudyHours_ExamScore).py`* | Öğrenci Sınav Puanlarının Tahmini. | **Basit ve Çoklu Doğrusal Regresyon**, Özellik Ölçeklendirme (Feature Scaling), Yeni veri üzerinde tahmin. |
| `2_Polinom_Regresyon_CustomerSatisfaction.py`* | Model Karmaşıklığı Analizi. | **Polinom Regresyon**, **Aşırı Öğrenme (Overfitting)** Görselleştirmesi, Scikit-Learn **Boru Hatları (Pipelines)**. |
| `4_Regularization_Lasso_vs_Linear.py`* | Düzenlileştirme (Regularization) ve Özellik Seçimi. | **Lasso (L1) ve Doğrusal Regresyon Karşılaştırması**, Düzenlileştirmede **Ölçeklendirmenin (Scaling)** önemi. |
| `5_FWI_Regression_AutoML_Analysis.py`* | Orman Yangını Hava İndeksi (FWI) Tahmini. | Veri Temizleme, Korelasyon Analizi, AutoML. |
| `11_SVM-Diamond-Price-Prediction-Regression.py`* | Elmas Fiyat Tahmini. | **SVR ve Doğrusal Regresyon** ödünleşimi (trade-off), Aykırı Değer (Outlier) Temizleme. |
| `18_Gym-Crowdedness-Prediction-Analysis.py`* | Spor Salonu Yoğunluk Tahmini. | Doğrusal olmayan zaman serisi verilerinde **Rastgele Orman (Random Forest)** performansı. |
| `6_WWII_Weather_Regression_Analysis.py`* | 2. Dünya Savaşı Hava Sıcaklığı Analizi. | **LassoCV** (Düzenlileştirme), Tarih/Saat özellik çıkarımı. |
| `22_California_Housing_Price_Prediction_XGBoost.py`* | Ev Fiyatı Tahmini. | **Model Karşılaştırması (Benchmarking)** (Doğrusal vs. Ağaçlar), **XGBoost Hiperparametre Ayarlama**, Aykırı Değer Temizleme. |
| `24_Boston_Housing_Power_Transformation.py`* | Doğrusal Regresyon Performansını Artırma. | **Yeo-Johnson ve Box-Cox Dönüşümleri**, Çarpıklık (Skewness) Giderme. |
| `25_Medical_Cost_LGBM_BoxCox_Regression.py`* | Sağlık Sigortası Maliyet Tahmini. | **Box-Cox Hedef Dönüşümü**, LightGBM, RandomizedSearchCV. |
| `26_California_Housing_Universal_Optimization.py`* | Otomatikleştirilmiş Model Seçim Boru Hattı. | **Evrensel Hiperparametre Ayarlama Fonksiyonu**, 9-Modelli Liderlik Tablosu (Leaderboard), PowerTransformer. |

### 🔹 Sınıflandırma Projeleri (Kategori Tahmini)
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `21_Stellar-Classification-XGBoost.py`* | Yıldız, Galaksi ve Kuasar Sınıflandırması. | **XGBoost**, Özellik Önemi (Kırmızıya Kayma/Redshift), Çok Sınıflı Sınıflandırma (Multiclass). |
| `16_Diabetes-Prediction-ML-Pipeline.py`* | Uçtan Uca Diyabet Teşhis Boru Hattı. | **Veri Sızıntısını Önleme (Data Leakage)** (Medyan Atama), Model Karşılaştırması. |
| `17_Diabetes-Ensemble-Learning-AdaBoost.py`* | Diyabet Tahmini (Topluluk Öğrenmesi - Ensemble). | **AdaBoost** algoritması ve manuel parametre ayarlama. |
| `8_Logistic-Regression-Hyperparameter-Tuning.py`* | Banka Müşteri Kayıp (Churn) Analizi. | Lojistik Regresyon, Ölçeklendirme, Karmaşıklık Matrisi (Confusion Matrix) Analizi. |
| `12_Naive_bayes-Iris-Species-Classification-Comparison.py`* | İris Türleri Sınıflandırması. | **Naive Bayes ve SVM** (Çekirdek Hilesi - Kernel Trick karşılaştırması). |
| `14_Decision-Tree-Classification-Projects.py`* | Araç Değerlendirme ve İris Analizi. | **Sıralı Kodlama (Ordinal Encoding)**, Karar Ağacı Görselleştirmesi. |
| `20_Gradient-Boosting-Advanced-Analysis.py`* | Kalp Hastalığı ve Beton Dayanımı. | **Gradient Boosting** (Regresyon ve Sınıflandırma), Korelasyon Filtreleme. |
| `23_Titanic_LightGBM_XGBoost_Comparison.py`* | Titanic Hayatta Kalma Analizi. | **LightGBM ve XGBoost**, RandomizedSearchCV, Özellik Önemi Görselleştirmesi. |

### 🔹 İleri Düzey Makine Öğrenmesi Uygulamaları
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `19_Car_Price_Prediction_Adaboost.py`* | İkinci El Araç Fiyat Tahmini. | **Frekans Kodlama (Frequency Encoding)** ile Yüksek Kardinalite (High Cardinality) Yönetimi, AdaBoost. |
| `15_Ensemble-Techniques.py`* | Gelir Grubu Tahmini (>50B). | **Dengesiz Veri (Imbalanced Data)** Yönetimi, **Hedef Kodlama (Target Encoding)**, Robust Scaler. |
| `10_Advanced_Logistic_Regression.py`* | Dolandırıcılık ve Siber Saldırı Tespiti. | **Sınıf Ağırlıkları (Class Weights)** optimizasyonu, One-vs-Rest / One-vs-One stratejileri. |
| `13_KNN-Health-Energy-Analysis.py`* | Sağlık Riski ve Enerji Tüketimi. | **Dirsek Yöntemi (Elbow Method)** ile optimum 'K' değerini bulma, Özellik Ölçeklendirme. |
| `9_SVM-Multi-Domain-Analysis.py`* | E-posta Spam, Kredi Riski ve Sismik Analiz. | SVM Çekirdek (Kernel) Karşılaştırması (Doğrusal, RBF, Polinom). |
| `27_Student_Performance_Dual_Pipeline.py`* | **Çift Görevli Öğrenci Performans Analizi.** Kesin puan tahmini (Regresyon) ve Geçti/Kaldı durumu (Sınıflandırma). | **Çift Görevli Modelleme (Reg & Clf)**, RandomizedSearchCV, **Aykırı Değer Baskılama (IQR Capping)**, Otomatik Model Liderlik Tablosu. |
| `28_PCA_Breast_Cancer_Analysis.py`* | PCA ve Model Performans Analizi. | **Temel Bileşenler Analizi (PCA)**, Boyut İndirgeme Ödünleşimleri, **StandardScaler**, Lojistik ve Gradient Boosting Karşılaştırması. |

### 🔹 Veri Bilimi Temelleri ve Teknikleri
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `3_Data_Preprocessing_Manipulation.py`* | Veri Bilimi için Kapsamlı Araç Kiti. | **SMOTE** (Dengesiz Veri), **One-Hot/Sıralı Kodlama (Ordinal Encoding)**, Pandas Birleştirme (Merging), Veri Görselleştirme. |
| `7_Python_SQL_Database_Basics.py`* | SQL ile Öğrenci Veritabanı Sistemi. | SQLite, Tablo Oluşturma (DDL), Veri Manipülasyonu (DML). |

---

### 🔹 Gözetimsiz Öğrenme ve Boyut İndirgeme
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `29_Clustering_Algorithms_Analysis.py` | K-Means, Aglomeratif (Yığınsal) Kümeleme ve DBSCAN Karşılaştırması. | Silüet Analizi, Dendrogramlar, EPS Optimizasyonu. |
| `30_Country_Segmentation_PCA_KMeans.py` | Kaynak Dağıtımı için Stratejik Ülke Analizi. | PCA (3-Bileşen), K-Means, İnteraktif Coğrafi Görselleştirme. |

## 📊 Metodoloji ve Yaklaşım

Bu projelerde standart Veri Bilimi yaşam döngüsünü izledim:
1.  **EDA (Keşifçi Veri Analizi):** Veri dağılımlarını, aykırı değerleri ve korelasyonları anlama.
2.  **Ön İşleme (Preprocessing):** IQR Aykırı Değer Baskılama (Capping), yüksek boyutlu veri setleri için Boyut İndirgeme (PCA), eksik değer yönetimi (veri sızıntısını kesinlikle önleyerek), Ölçeklendirme (Standard/Robust) ve Kodlama (One-Hot/Target/Frequency) gibi ileri düzey veri temizleme işlemleri.
3.  **Model Seçimi:** Gözetimli (supervised) görevler için Doğrusal modellerle Ağaç tabanlı Toplulukların (Ensembles) karşılaştırılması ve kümeleme problemleri için K-Means ile DBSCAN'in değerlendirilmesi.
4.  **Optimizasyon:** Hiperparametrelerin ince ayarı için `GridSearchCV` ve `RandomizedSearchCV` kullanımı.
5.  **Değerlendirme:** Yalnızca Doğruluk (Accuracy) metriklerinin ötesine geçerek; sağlam bir değerlendirme için F1-Skoru, RMSE, R2 Skoru ve Karmaşıklık Matrislerinin (Confusion Matrices) kullanılması.

---

## 💻 Kurulum ve Kullanım

1.  Depoyu klonlayın:
    ```bash
    git clone [https://github.com/AdnanSag/-Data-Science-Machine-Learning-Portfolio.git](https://github.com/AdnanSag/-Data-Science-Machine-Learning-Portfolio.git)
    ```
2.  Bağımlılıkları yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
3.  Belirli bir betiği çalıştırın:
    ```bash
    python 16_Diabetes-Prediction-ML-Pipeline.py
    ```
---
*Created by Adnan Sag*


