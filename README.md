<div align="right">
  <a href="README.md">🇹🇷 Türkçe</a> | <a href="README_EN.md">🇬🇧 English</a>
</div>

# 🚀 <img src="https://flagcdn.com/w40/tr.png" width="32" alt="TR" style="vertical-align: middle;"> Veri Bilimi ve Makine Öğrenmesi Portfolyosu

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Visualization-purple?logo=plotly)
![Status](https://img.shields.io/badge/Status-Completed-success)

Hoş geldiniz! Bu depo, Veri Bilimi ve Makine Öğrenmesi alanındaki yetkinliklerimi sergileyen bir portfolyo niteliğindedir. Veri temizleme ve özellik mühendisliğinden (feature engineering) model kurma ve hiperparametre optimizasyonuna kadar uçtan uca makine öğrenmesi yaşam döngüsünü gösteren bir dizi Python betiği içerir. 

Tüm proje dosyaları, kolay erişim için kök dizinde (root directory) bulunmaktadır.

## 🛠️ Teknoloji Yığını ve Kütüphaneler
* **Diller:** Python, SQL
* **Veri Analizi:** Pandas, NumPy, SciPy
* **Görselleştirme:** Matplotlib, Seaborn, Plotly Express
* **Makine Öğrenmesi:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn, LazyPredict
* **Algoritmalar:** Doğrusal/Lojistik Regresyon, SVM, KNN, Karar Ağaçları, Rastgele Orman, AdaBoost, Gradient Boosting, Temel Bileşenler Analizi (PCA), K-Means, Hiyerarşik Kümeleme, DBSCAN
* **Optimizasyon Araçları:** Kneed (KneeLocator)

---

## 📸 Örnek Proje Çıktıları ve Görselleştirmeler

### 1. Model Performansı ve Özellik Önemi (Feature Importance)
![Feature Importance](https://via.placeholder.com/800x400.png?text=Buraya+XGBoost/LightGBM+Feature+Importance+Gorseli+Gelecek)
*Ağaç tabanlı modellerde (XGBoost/LightGBM) hangi özelliklerin tahmine en çok etki ettiğinin analizi.*

### 2. Keşifçi Veri Analizi (EDA) ve Korelasyon
![Korelasyon Matrisi](https://via.placeholder.com/800x400.png?text=Buraya+Korelasyon+Matrisi+Isi+Haritasi+Gelecek)
*Veri setindeki değişkenler arasındaki ilişkilerin ısı haritası (heatmap) ile incelenmesi ve çoklu doğrusal bağlantı (multicollinearity) kontrolü.*

### 3. Coğrafi Segmentasyon ve Kümeleme Haritası
![Dünya Haritası Kümeleme](https://via.placeholder.com/800x400.png?text=Buraya+Plotly+Dunya+Haritasi+Gorseli+Gelecek)
*K-Means ve PCA kullanılarak ülkelerin sosyo-ekonomik durumlarına göre harita üzerinde interaktif segmentasyonu.*

---

## 📂 Proje Kataloğu

Projeler, problem türüne göre aşağıda kategorize edilmiştir.

### 🔹 Regresyon Projeleri (Sürekli Değer Tahmini)
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `1_Doğrusal_Regresyon_StudyHours_ExamScore).py`* | Öğrenci Sınav Puanlarının Tahmini. | **Basit ve Çoklu Doğrusal Regresyon**, Özellik Ölçeklendirme (Feature Scaling), Yeni veri üzerinde tahmin. |
| `2_Polinom_Regresyon_CustomerSatisfaction.py`* | Model Karmaşıklığı Analizi. | **Polinom Regresyon**, **Aşırı Öğrenme (Overfitting)** Görselleştirmesi, Scikit-Learn **Boru Hatları (Pipelines)**. |
| `4_Regularization_Lasso_vs_Linear.py`* | Düzenlileştirme ve Özellik Seçimi. | **Lasso (L1) ve Doğrusal Regresyon**, Düzenlileştirmede **Ölçeklendirmenin (Scaling)** önemi. |
| `5_FWI_Regression_AutoML_Analysis.py`* | Orman Yangını Hava İndeksi (FWI) Tahmini. | Veri Temizleme, Korelasyon Analizi, AutoML. |
| `11_SVM-Diamond-Price-Prediction-Regression.py`* | Elmas Fiyat Tahmini. | **SVR ve Doğrusal Regresyon** ödünleşimi, Aykırı Değer (Outlier) Temizleme. |
| `18_Gym-Crowdedness-Prediction-Analysis.py`* | Spor Salonu Yoğunluk Tahmini. | Doğrusal olmayan zaman serisi verilerinde **Rastgele Orman (Random Forest)** performansı. |
| `6_WWII_Weather_Regression_Analysis.py`* | 2. Dünya Savaşı Hava Sıcaklığı Analizi. | **LassoCV** (Düzenlileştirme), Tarih/Saat özellik çıkarımı. |
| `22_California_Housing_Price_Prediction_XGBoost.py`* | Ev Fiyatı Tahmini. | **Model Karşılaştırması** (Doğrusal vs. Ağaçlar), **XGBoost Hiperparametre Ayarlama**, Aykırı Değer Temizleme. |
| `24_Boston_Housing_Power_Transformation.py`* | Doğrusal Regresyon Performansını Artırma. | **Yeo-Johnson ve Box-Cox Dönüşümleri**, Çarpıklık (Skewness) Giderme. |
| `25_Medical_Cost_LGBM_BoxCox_Regression.py`* | Sağlık Sigortası Maliyet Tahmini. | **Box-Cox Hedef Dönüşümü**, LightGBM, RandomizedSearchCV. |
| `26_California_Housing_Universal_Optimization.py`* | Otomatikleştirilmiş Model Seçim Boru Hattı. | **Evrensel Hiperparametre Ayarlama Fonksiyonu**, 9-Modelli Liderlik Tablosu, PowerTransformer. |

### 🔹 Sınıflandırma Projeleri (Kategori Tahmini)
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `21_Stellar-Classification-XGBoost.py`* | Yıldız, Galaksi ve Kuasar Sınıflandırması. | **XGBoost**, Özellik Önemi (Kırmızıya Kayma/Redshift), Çok Sınıflı Sınıflandırma. |
| `16_Diabetes-Prediction-ML-Pipeline.py`* | Uçtan Uca Diyabet Teşhis Boru Hattı. | **Veri Sızıntısını Önleme (Data Leakage)** (Medyan Atama), Model Karşılaştırması. |
| `17_Diabetes-Ensemble-Learning-AdaBoost.py`* | Diyabet Tahmini (Topluluk Öğrenmesi). | **AdaBoost** algoritması ve manuel parametre ayarlama. |
| `8_Logistic-Regression-Hyperparameter-Tuning.py`* | Banka Müşteri Kayıp (Churn) Analizi. | Lojistik Regresyon, Ölçeklendirme, Karmaşıklık Matrisi Analizi. |
| `12_Naive_bayes-Iris-Species-Classification-Comparison.py`* | İris Türleri Sınıflandırması. | **Naive Bayes ve SVM** (Çekirdek Hilesi - Kernel Trick karşılaştırması). |
| `14_Decision-Tree-Classification-Projects.py`* | Araç Değerlendirme ve İris Analizi. | **Sıralı Kodlama (Ordinal Encoding)**, Karar Ağacı Görselleştirmesi. |
| `20_Gradient-Boosting-Advanced-Analysis.py`* | Kalp Hastalığı ve Beton Dayanımı. | **Gradient Boosting** (Regresyon ve Sınıflandırma), Korelasyon Filtreleme. |
| `23_Titanic_LightGBM_XGBoost_Comparison.py`* | Titanic Hayatta Kalma Analizi. | **LightGBM ve XGBoost**, RandomizedSearchCV, Özellik Önemi Görselleştirmesi. |

### 🔹 İleri Düzey Makine Öğrenmesi Uygulamaları
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `19_Car_Price_Prediction_Adaboost.py`* | İkinci El Araç Fiyat Tahmini. | **Frekans Kodlama** ile Yüksek Kardinalite Yönetimi, AdaBoost. |
| `15_Ensemble-Techniques.py`* | Gelir Grubu Tahmini (>50B). | **Dengesiz Veri (Imbalanced Data)** Yönetimi, **Hedef Kodlama (Target Encoding)**, Robust Scaler. |
| `10_Advanced_Logistic_Regression.py`* | Dolandırıcılık ve Siber Saldırı Tespiti. | **Sınıf Ağırlıkları (Class Weights)** optimizasyonu, One-vs-Rest / One-vs-One stratejileri. |
| `13_KNN-Health-Energy-Analysis.py`* | Sağlık Riski ve Enerji Tüketimi. | **Dirsek Yöntemi (Elbow Method)** ile optimum 'K' değerini bulma, Özellik Ölçeklendirme. |
| `9_SVM-Multi-Domain-Analysis.py`* | E-posta Spam, Kredi Riski ve Sismik Analiz. | SVM Çekirdek (Kernel) Karşılaştırması (Doğrusal, RBF, Polinom). |
| `27_Student_Performance_Dual_Pipeline.py`* | **Çift Görevli Öğrenci Performans Analizi.** | **Çift Görevli Modelleme (Reg & Clf)**, RandomizedSearchCV, **Aykırı Değer Baskılama (IQR Capping)**, Liderlik Tablosu. |
| `28_PCA_Breast_Cancer_Analysis.py`* | PCA ve Model Performans Analizi. | **Temel Bileşenler Analizi (PCA)**, Boyut İndirgeme Ödünleşimleri, **StandardScaler**, Lojistik vs. Gradient Boosting. |

### 🔹 Veri Bilimi Temelleri ve Teknikleri
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `3_Data_Preprocessing_Manipulation.py`* | Veri Bilimi için Kapsamlı Araç Kiti. | **SMOTE** (Dengesiz Veri), **One-Hot/Sıralı Kodlama**, Pandas Birleştirme, Veri Görselleştirme. |
| `7_Python_SQL_Database_Basics.py`* | SQL ile Öğrenci Veritabanı Sistemi. | SQLite, Tablo Oluşturma (DDL), Veri Manipülasyonu (DML). |

### 🔹 Gözetimsiz Öğrenme ve Boyut İndirgeme
| Dosya Adı | Açıklama | Temel Teknikler |
| :--- | :--- | :--- |
| `29_Clustering_Algorithms_Analysis.py` | K-Means, Aglomeratif Kümeleme ve DBSCAN. | Silüet Analizi, Dendrogramlar, EPS Optimizasyonu. |
| `30_Country_Segmentation_PCA_KMeans.py` | Kaynak Dağıtımı için Ülke Analizi. | PCA (3-Bileşen), K-Means, İnteraktif Coğrafi Görselleştirme. |

---

## 📊 Metodoloji ve Yaklaşım

Bu projelerde standart Veri Bilimi yaşam döngüsünü izledim:
1.  **EDA (Keşifçi Veri Analizi):** Veri dağılımlarını, aykırı değerleri ve korelasyonları anlama.
2.  **Ön İşleme (Preprocessing):** IQR Aykırı Değer Baskılama, Boyut İndirgeme (PCA), eksik değer yönetimi, Ölçeklendirme (Standard/Robust) ve Kodlama (One-Hot/Target/Frequency).
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

## 📬 İletişim

Projelerim hakkında konuşmak veya işbirliği yapmak isterseniz bana ulaşabilirsiniz:
* **LinkedIn:** [Profilinizi Buraya Ekleyin](https://www.linkedin.com/in/)
* **Kaggle:** [Profilinizi Buraya Ekleyin](https://www.kaggle.com/)
* **E-posta:** adiniz@email.com

*Created by Adnan Sag*
