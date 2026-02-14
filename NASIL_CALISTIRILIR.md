# 🚀 Projeyi Nasıl Çalıştırılır?

Bu kılavuz, araç altı tespit sistemini adım adım nasıl çalıştıracağınızı gösterir.

## 📋 İçindekiler
1. [Gereksinimler](#1-gereksinimler)
2. [Kurulum](#2-kurulum)
3. [Kullanım Senaryoları](#3-kullanım-senaryoları)
4. [Sorun Giderme](#4-sorun-giderme)

---

## 1. Gereksinimler

### Sistem Gereksinimleri
- **Python**: 3.8 veya üzeri
- **İşletim Sistemi**: Windows 10/11 (Linux ve macOS da çalışır)
- **RAM**: En az 4GB (8GB önerilir)
- **Disk**: En az 2GB boş alan (model ve veri için)

### Gerekli Kütüphaneler
- TensorFlow 2.8+
- OpenCV 4.5+
- NumPy, Pandas, Matplotlib, Seaborn
- PIL (Pillow)
- Diğer kütüphaneler (requirements.txt'de listelenmiştir)

---

## 2. Kurulum

### Adım 1: Proje Klasörüne Git
```bash
cd C:\Users\MONSTER\Desktop\pyhton\CNN
```

### Adım 2: Gerekli Kütüphaneleri Yükle
```bash
pip install -r requirements.txt
```

**Not**: Eğer hata alırsanız, kütüphaneleri tek tek yükleyebilirsiniz:
```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn pillow
```

### Adım 3: Model Dosyasını Kontrol Et
Eğitilmiş model dosyasının (`improved_vehicle_underneath_model.h5`) mevcut olduğundan emin olun.
- **Model yoksa**: Önce model eğitmeniz gerekir (Bkz: [Model Eğitimi](#senaryo-2-model-eğitimi))

---

## 3. Kullanım Senaryoları

### Senaryo 1: Hızlı Test (Model Test)
Mevcut modelin performansını test etmek için:

```bash
python quick_improve.py
```

**Ne yapar:**
- ✅ Mevcut modeli test görüntüleri üzerinde test eder
- ✅ Doğruluk oranını gösterir
- ✅ İyileştirme önerileri sunar
- ✅ Yeni konfigürasyon dosyası oluşturur

**Çıktı:**
```
=== MEVCUT MODEL TEST ===
📸 Test görüntüleri test ediliyor...
=== ÖZET ===
Toplam Test: 3
Doğru Tahmin: 2
Genel Doğruluk: 66.7%
```

---

### Senaryo 2: Model Eğitimi
Yeni bir model eğitmek veya mevcut modeli iyileştirmek için:

```bash
python improved_training.py
```

**Ne yapar:**
- ✅ Veri setini yükler ve hazırlar
- ✅ EfficientNetB0 tabanlı model oluşturur
- ✅ Modeli eğitir (150 epoch)
- ✅ Model performansını değerlendirir
- ✅ Modeli kaydeder (`improved_vehicle_underneath_model.h5`)
- ✅ Grafikleri oluşturur (training history, confusion matrix)

**Süre:** 1-3 saat (donanıma bağlı)

**Çıktı:**
- `improved_vehicle_underneath_model.h5` - Eğitilmiş model
- `improved_training_history.png` - Eğitim grafikleri
- `improved_confusion_matrix.png` - Performans matrisi
- `improved_class_names.json` - Sınıf isimleri

---

### Senaryo 3: Gerçek Zamanlı Tespit (Kamera ile)
Kameradan canlı görüntü alarak gerçek zamanlı tespit yapmak için:

```bash
python improved_realtime.py
```

**Ne yapar:**
- ✅ Kamerayı açar (varsayılan: kamera 0)
- ✅ Her kareyi analiz eder
- ✅ Araç altı tespit edildiğinde sınıflandırma yapar
- ✅ Ekranda sonuçları gösterir
- ✅ Tehlikeli madde tespit edildiğinde sesli uyarı verir

**Kontroller:**
- **Q tuşu**: Programdan çık
- **R tuşu**: Kamerayı yeniden başlat

**Çıktı:**
- Ekranda canlı görüntü ve tespit sonuçları
- Debug bilgileri (parlaklık, koyu oranı, vb.)
- FPS ve işlem süresi bilgisi

---

### Senaryo 4: Ensemble Model Eğitimi (İleri Seviye)
Birden fazla modeli birleştirerek daha güçlü bir model oluşturmak için:

```bash
python ensemble_training.py
```

**Ne yapar:**
- ✅ 3 farklı model eğitir (EfficientNetB0, ResNet50, VGG16)
- ✅ Modelleri birleştirir (ensemble)
- ✅ Daha yüksek doğruluk hedefler

**Süre:** 3-6 saat (çok daha uzun sürer)

---

## 4. Detaylı Kullanım

### Model Dosyası Kontrolü
Model dosyasının mevcut olup olmadığını kontrol edin:
```bash
dir improved_vehicle_underneath_model.h5
```

### Farklı Kamera Kullanımı
Eğer birden fazla kameranız varsa, `improved_realtime.py` dosyasında `start_camera()` fonksiyonunu düzenleyin:
```python
self.start_camera(camera_id=1)  # 1. kamerayı kullan
```

### Test Görüntüleri ile Test
Belirli bir görüntüyü test etmek için `quick_improve.py` dosyasındaki test görüntüleri listesini düzenleyebilirsiniz.

---

## 5. Sorun Giderme

### Sorun 1: "ModuleNotFoundError"
**Çözüm:** Gerekli kütüphaneleri yükleyin:
```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pillow
```

### Sorun 2: "Model dosyası bulunamadı"
**Çözüm:** Önce model eğitmeniz gerekir:
```bash
python improved_training.py
```

### Sorun 3: "Kamera açılamadı"
**Çözüm:** 
- Kameranın başka bir program tarafından kullanılmadığından emin olun
- Kamera ID'sini değiştirmeyi deneyin (0, 1, 2, vb.)
- USB kamerasını çıkarıp tekrar takın

### Sorun 4: "CUDA hatası" (GPU kullanıyorsanız)
**Çözüm:** TensorFlow GPU sürümünü yükleyin veya CPU kullanın:
```bash
pip install tensorflow-cpu  # CPU sürümü
```

### Sorun 5: "Memory Error"
**Çözüm:** 
- Batch size'ı küçültün (`improved_training.py` içinde `BATCH_SIZE = 2`)
- Daha az epoch kullanın
- Daha küçük görüntü boyutu kullanın

---

## 6. Örnek Kullanım Akışı

### İlk Kez Kullanım
1. **Kurulum:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Eğitimi:**
   ```bash
   python improved_training.py
   ```
   (1-3 saat sürebilir)

3. **Model Test:**
   ```bash
   python quick_improve.py
   ```

4. **Gerçek Zamanlı Kullanım:**
   ```bash
   python improved_realtime.py
   ```

### Günlük Kullanım
1. **Hızlı Test:**
   ```bash
   python quick_improve.py
   ```

2. **Gerçek Zamanlı Tespit:**
   ```bash
   python improved_realtime.py
   ```

---

## 7. Önemli Notlar

⚠️ **Dikkat:**
- Model eğitimi uzun sürebilir (1-3 saat)
- Gerçek zamanlı tespit için kameraya ihtiyaç vardır
- Model dosyası (`*.h5`) yaklaşık 50-100 MB olabilir
- Eğitim sırasında bilgisayarınız yavaşlayabilir

✅ **İpuçları:**
- İlk kullanımda mutlaka `quick_improve.py` ile test edin
- Model eğitimi sırasında bilgisayarınızı kapatmayın
- Gerçek zamanlı tespitte ışık koşullarına dikkat edin
- Debug bilgilerini takip ederek sistemin çalıştığını doğrulayın

---

## 8. Hızlı Başlangıç

En hızlı şekilde başlamak için:

```bash
# 1. Kütüphaneleri yükle
pip install -r requirements.txt

# 2. Modeli test et (model varsa)
python quick_improve.py

# 3. Gerçek zamanlı tespit başlat
python improved_realtime.py
```

---

## 9. Destek

Sorun yaşıyorsanız:
1. Hata mesajını okuyun
2. Yukarıdaki "Sorun Giderme" bölümüne bakın
3. Model dosyasının mevcut olduğundan emin olun
4. Gerekli kütüphanelerin yüklü olduğunu kontrol edin

---

**Başarılar! 🚀**

