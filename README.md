# 🏡 Ev Fiyat Tahmini Uygulaması

Bu proje, İstanbul'daki satılık evlerin fiyatlarını tahmin etmek için bir makine öğrenmesi modeli kullanmaktadır. Model, farklı algoritmaların performanslarını değerlendirerek en iyi sonucu veren modeli seçer ve tahminlerde bulunur.

📌 Proje İçeriği

train.py: Makine öğrenmesi algoritmalarını kullanarak modeli eğiten ve en iyi modeli seçerek kaydeden dosya.

arayüz.py: Kullanıcıdan giriş alarak eğitilmiş modeli kullanarak fiyat tahmini yapan Tkinter tabanlı bir arayüz.

best_model.pkl: En iyi sonucu veren makine öğrenmesi modelini içeren dosya.

encoder.pkl: Konum bilgilerini sayısal forma dönüştüren One-Hot Encoder nesnesini içeren dosya.

🚀 Kurulum ve Kullanım

1️⃣ Gereksinimleri Yükleyin

Aşağıdaki komutları çalıştırarak gerekli kütüphaneleri yükleyin:

pip install pandas joblib scikit-learn matplotlib seaborn tkinter

2️⃣ Modeli Eğitme

train.py dosyasını çalıştırarak verileri işleyin ve en iyi modeli eğitin:

python train.py

Model eğitildikten sonra, en iyi model best_model.pkl ve konumları dönüştüren encoder encoder.pkl olarak kaydedilir.

3️⃣ Tahmin Arayüzünü Başlatma

Tkinter tabanlı GUI'yi başlatmak için:

python arayüz.py

Açılan pencerede semt, oda sayısı, genişlik (m²), yaş ve kat bilgilerini girerek ev fiyatı tahmini alabilirsiniz.

📊 Kullanılan Makine Öğrenmesi Modelleri

train.py dosyasında aşağıdaki modeller test edilmiştir:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

Support Vector Machine (SVR)

K-Nearest Neighbors (KNN)

En iyi R² skorunu alan model best_model.pkl olarak kaydedilir.

🔍 Veri Kümesi

Bu proje İstanbul'daki satılık evlerin özelliklerini içeren bir veri seti kullanmaktadır. Veri kümesi aşağıdaki sütunları içermektedir:

Price: Ev fiyatı (bağımlı değişken)

Room: Oda sayısı

Area: Ev genişliği (m²)

Age: Binanın yaşı

Floor: Kat bilgisi

Location: Semt bilgisi (One-Hot Encoding ile dönüştürülmüştür)

🌟 Olası Geliştirmeler

Daha fazla özellik eklenerek modelin doğruluğunun artırılması.

Derin öğrenme modelleri ile kıyaslama yapılması.

Daha büyük veri setleriyle modelin tekrar eğitilmesi.
