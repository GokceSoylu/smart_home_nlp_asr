
---

# 📄 `README_TR.md`

```md
# Akıllı Ev Sesli Komut Tanıma Sistemi  
### CSE 431 – Natural Language Processing with Machine Learning  
**Dönem Projesi – Aşama 2**

---

## 1. Proje Tanımı

Bu projede, konuşma sinyallerinden elde edilen Mel tabanlı akustik özellikler kullanılarak bir akıllı ev komut tanıma sistemi geliştirilmiştir. Metin tabanlı doğal dil işleme yaklaşımlarından farklı olarak, bu aşamada doğrudan ses verileri üzerinden sınıflandırma yapılmıştır.

Ciddi sınıf dengesizliği içeren bir veri kümesi üzerinde farklı makine öğrenmesi ve derin öğrenme modelleri karşılaştırılmıştır.

---

## 2. Veri Kümesi

- Dosya: `dataset_mel_01.xlsx`
- Toplam örnek sayısı: 27.471
- Özellik boyutu: 480 (Mel tabanlı akustik özellikler)
- Sınıf sayısı: 69
- Etiket sütunu: `target_label`

Her satır sabit uzunlukta bir ses segmentini temsil etmektedir.

---

## 3. Proje Klasör Yapısı

smart_home_asr_project2/
├── data/
│ ├── dataset_mel_01.xlsx
│ └── dataset_split.npz
├── src/
│ ├── 01_check_dataset_and_revise.py
│ ├── 07_dataset.py
│ ├── 08_train_test.py
│ └── 09_confusion_matrix.py
├── results/
│ ├── model_comparison_results.csv
│ └── confusion_matrix_mlp.png
├── README_EN.md
└── README_TR.md

---

## 4. Çalıştırma Sırası

Scriptler aşağıdaki sırayla çalıştırılmalıdır:

```bash
cd src
python 01_check_dataset_and_revise.py
python 07_dataset.py
python 08_train_test.py
python 09_confusion_matrix.py
```
## 5. Kullanılan Modeller
- Decision Tree
- Random Forest
- Linear SVM
- Çok Katmanlı Algılayıcı (MLP)
## 6. Değerlendirme Ölçütleri

Veri kümesindeki sınıf dengesizliği nedeniyle aşağıdaki macro ortalamalı metrikler kullanılmıştır:

- Precision
- Recall
- F1-score

En iyi model Macro F1-score ölçütüne göre seçilmiştir.
## 7. Sonuçlar
En iyi performans Çok Katmanlı Algılayıcı (MLP) modeli ile elde edilmiştir.
Detaylı sonuçlar:
results/model_comparison_results.csv

results/confusion_matrix_mlp.png

## 8. Gereksinimler
- Python 3.10
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

---

## Bonus – Meta Öğrenme (Roof Model)

Pencere tabanlı konuşma tanıma modeline ek olarak, proje kapsamında bonus olarak bir meta öğrenme (roof model) yaklaşımı uygulanmıştır.

Alt seviye model, ses kayıtlarını kısa zaman pencereleri üzerinden tahmin etmektedir. Ancak tek bir ses kaydı birden fazla pencere tahmini ve sessizlik içerebildiğinden, bu çıktıları birleştirerek tek bir nihai komut kararı veren bir üst seviye modele ihtiyaç duyulmuştur.

Bu amaçla, alt modelin pencere bazlı tahminleri log dosyaları olarak kaydedilmiştir. Her log dosyası, aşağıdaki özellikler çıkarılarak tek bir özellik vektörüne dönüştürülmüştür:
- Tahmin edilen komut metinlerinden elde edilen TF-IDF özellikleri
- İstatistiksel özellikler:
  - sessiz olmayan pencere sayısı
  - sessiz olmayan pencere oranı
  - ortalama ve maksimum güven skorları
  - en baskın komut oranı
  - tahmin geçiş sayısı

Bu özellikler kullanılarak oluşturulan `roof_dataset.csv` dosyasında her satır bir ses kaydını temsil etmektedir. Bu veri kümesi üzerinde eğitilen roof model, her kayıt için tek bir nihai komut etiketi üretmiştir.

Elde edilen sonuçlar, hiyerarşik (alt model + roof model) mimarinin ve meta öğrenme yaklaşımının konuşma komutu tanıma sisteminin kararlılığını ve doğruluğunu artırdığını göstermektedir.


## 9. Hazırlayan

Gökçe Soylu

Aydın Adnan Menderes Üniversitesi

Bilgisayar Mühendisliği Bölümü