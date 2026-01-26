import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# PROJE 1: Basit Doğrusal Regresyon (Simple Linear Regression)
# Tahmin: Çalışma Saatine Göre Sınav Notu
# ==========================================

# 1. Veri Yükleme ve Görselleştirme
df = pd.read_csv("1-studyhours.csv")

plt.figure(figsize=(10, 6))
plt.scatter(df["Study Hours"], df["Exam Score"], color='blue', alpha=0.6)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

# 2. Değişkenlerin Belirlenmesi
X = df[["Study Hours"]]  # Bağımsız değişken (Feature Matrix)
y = df["Exam Score"]     # Bağımlı değişken (Target Vector)

# 3. Eğitim ve Test Bölünmesi (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# 4. Veri Standardizasyonu (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Eğitimi
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Simple Regression - Intercept: {model.intercept_:.2f}")
print(f"Simple Regression - Coefficients: {model.coef_[0]:.2f}")

# 6. Tahmin ve Performans Değerlendirmesi
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("Simple Linear Regression Performance:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.3f}")
print("-" * 30)


# ==========================================
# PROJE 2: Çoklu Doğrusal Regresyon (Multiple Linear Regression)
# Tahmin: Çoklu Faktörlere Göre Sınav Notu
# ==========================================

# 1. Veri Yükleme ve Temizleme
df_multi = pd.read_csv("2-multiplegradesdataset.csv")

# Eksik verileri temizleme
columns_to_check = ['Study Hours', 'Sleep Hours', 'Attendance Rate', 'Social Media Hours', 'Exam Score']
df_cleaned = df_multi.dropna(subset=columns_to_check)

# İlişkiyi görselleştirme (Örnek: Study Hours vs Exam Score)
plt.figure(figsize=(10, 6))
sns.regplot(x=df_cleaned['Study Hours'], y=df_cleaned['Exam Score'])
plt.title("Study Hours vs Exam Score (Regression Plot)")
plt.show()

# 2. Değişkenlerin Belirlenmesi
X_multi = df_cleaned[["Study Hours", "Sleep Hours", "Attendance Rate", "Social Media Hours"]]
y_multi = df_cleaned["Exam Score"]

# 3. Eğitim ve Test Bölünmesi
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.25, random_state=15)

# 4. Standardizasyon
scaler_multi = StandardScaler()
X_train_m_scaled = scaler_multi.fit_transform(X_train_m)
X_test_m_scaled = scaler_multi.transform(X_test_m)

# 5. Model Eğitimi
multi_model = LinearRegression()
multi_model.fit(X_train_m_scaled, y_train_m)

# 6. Yeni Bir Öğrenci İçin Tahmin Yapma Senaryosu
new_student_data = {
    "Study Hours": [5],
    "Sleep Hours": [7],
    "Attendance Rate": [90],
    "Social Media Hours": [4]
}

new_student_df = pd.DataFrame(new_student_data)
# Modeli eğitirken kullandığımız scaler ile yeni veriyi de dönüştürmeliyiz
new_student_scaled = scaler_multi.transform(new_student_df)

predicted_score = multi_model.predict(new_student_scaled)
print(f"Predicted Exam Score for New Student: {predicted_score[0]:.2f}")