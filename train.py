import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
import time  # Time modülünü ekledik
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
df = pd.read_csv('/mnt/c/Users/burak/OneDrive/Masaüstü/EV/istanbul_satilik_evler_2023.csv')

# Fiyat dağılımı
plt.figure(figsize=(50, 20))
sns.countplot(x='Price', data=df)
plt.title('Fiyat Dağılımı')
plt.xlabel('Fiyat')
plt.ylabel('Frekans')
plt.show()

# Oda sayısı dağılımı
plt.figure(figsize=(10, 6))
sns.countplot(x='Room', data=df)
plt.title('Oda Sayısı Dağılımı')
plt.xlabel('Oda Sayısı')
plt.ylabel('Frekans')
plt.show()

# Evin Genişlik Dağılımı
plt.figure(figsize=(40, 10))
sns.countplot(x='Area', data=df)
plt.title('Evin Genişlik Dağılımı')
plt.xlabel('Ev Genişliği')
plt.ylabel('Frekans')
plt.show()

# Konum Dağılımı
plt.figure(figsize=(50, 20))
sns.countplot(x='Location', data=df)
plt.title('Konum Dağılımı')
plt.xlabel('Konumlar')
plt.ylabel('Frekans')
plt.show()

# Fiyat ve Oda Sayısı Arasındaki Dağılım
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Room', y='Price', data=df)
plt.title('Oda Sayısı ve Fiyat Arasındaki İlişki')
plt.xlabel('Oda Sayısı')
plt.ylabel('Fiyat')
plt.show()

# Fiyat ve Konum Arasındaki İlişki
plt.figure(figsize=(40, 10))
sns.boxplot(x='Location', y='Price', data=df)
plt.title('Fiyat ve Konum Arasındaki İlişki')
plt.xlabel('Konum')
plt.ylabel('Fiyat')
plt.show()

# Fiyat ve Genişlik Arasındaki İlişki
plt.figure(figsize=(10, 6))
sns.regplot(x='Area', y='Price', data=df, scatter_kws={'s':10},)
plt.title('Fiyat ve Genişlik Arasındaki İlişki')
plt.xlabel('Alan (m²)')
plt.ylabel('Fiyat')
plt.show()

# Türkçe karakterleri İngilizce karakterlere dönüştürme fonksiyonu
def convert_turkish_characters(text):
    mapping = str.maketrans("çÇğĞıİöÖşŞüÜ", "cCgGiIoOsSuU")
    return text.translate(mapping)

# "Location" sütunundaki Türkçe karakterleri dönüştürme
df['Location'] = df['Location'].apply(convert_turkish_characters)

# Eksik değerleri kontrol etme
eksik_deger =df.isnull().sum()
print("Eksik değerler:\n", eksik_deger)

# "Room" kolonundaki verileri sayısal hale getirme
df['Room'] = df['Room'].apply(lambda x: eval(str(x)) if isinstance(x, str) else x)

# "Location" sütununu one-hot encode etme
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
location_encoded = encoder.fit_transform(df[['Location']])
location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['Location']))
df = pd.concat([df, location_df], axis=1)
df.drop('Location', axis=1, inplace=True)

# Korelasyon Matrisi
corr = df.corr()  # Veri setindeki sayısal sütunlar arasındaki korelasyonları hesaplar
plt.figure(figsize=(50, 20))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()

# "Width" sütununu "Area" olarak yeniden adlandırma
df.rename(columns={'Width': 'Area'}, inplace=True)

# Bağımsız ve bağımlı değişkenleri ayırma
X = df.drop('Price', axis=1)
Y = df['Price']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Modellerin listesi
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}


results = []

# Modelleri eğitme ve değerlendirme
for name, model in models.items():
    start_time = time.time()  # Eğitim süresini ölçmek için başlama zamanı
    model.fit(X_train, y_train)
    end_time = time.time()  # Eğitim süresinin bitiş zamanı
    training_time = end_time - start_time  # Eğitim süresi
    print(f"{name} için eğitim süresi: {training_time:.4f} saniye") # Eğitim süresini yazdırma

    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

# Sonuçları sıralama (R2 en yüksek olan en iyi modeldir)
print(results_df.sort_values(by="R2", ascending=False))

# R² Değerini karşılaştırmak için bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="R2", data=results_df)
plt.title('Model Performans Karşılaştırması (R2)')
plt.xticks(rotation=45)
plt.show()

# MAE karşılaştırması
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="MAE", data=results_df)
plt.title('Model Performans Karşılaştırması (MAE)')
plt.xticks(rotation=45)
plt.show()

# MSE karşılaştırması
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="MSE", data=results_df)
plt.title('Model Performans Karşılaştırması (MSE)')
plt.xticks(rotation=45)
plt.show()

# En iyi modeli seçme
best_model_info = max(results, key=lambda x: x['R2'])
best_model_name = best_model_info['Model']
best_model = models[best_model_name]

print(f"En başarılı model: {best_model_name} (R2: {best_model_info['R2']})")

# En iyi modeli ve encoder'ı kaydetme
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(encoder, 'encoder.pkl')