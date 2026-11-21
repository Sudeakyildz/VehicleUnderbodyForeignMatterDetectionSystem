# 🚗 Araç Altı Yabancı Madde Tespit Sistemi

Araç altı görüntüsünde **yabancı madde / şüpheli nesne var mı yok mu** sorusunu yapay zekâ ile yanıtlayan bir karar-destek sistemidir. Güvenlik personeline yardımcı olmak için tasarlanmıştır; **tek başına karar verici değildir**.

- 🚨 `YabanciMaddeOlan`: araç altında yabancı madde var
- ✅ `YabanciMaddeOlmayan`: araç altı temiz

## Nasıl çalışır?

```
Kamera/Video → (ön filtre: araç altı görünüyor mu?) → EfficientNet sınıflandırıcı
             → son N tahminin çoğunluk kararı → alarm + ses + kayıt (detections/)
```

1. **Ön filtre** (klasik OpenCV): kadrajda koyu, geniş, yatay yapılı bir araç altı var mı? (`--no-gate` ile kapatılır)
2. **Model**: ImageNet ile ön eğitilmiş EfficientNet (transfer learning). Çıktı: `P(YabanciMaddeOlan)`.
3. **Karar**: olasılık, eğitimde doğrulama setinden seçilen **eşiğin** (varsayılan 0.40, yani 0.5'ten daha hassas) üstündeyse tehlikeli sayılır. Tek kareye değil, son birkaç kareye bakılır (yanlış alarm azalır).
4. **Kayıt**: alarmda görüntü `detections/` klasörüne, zaman ve olasılıkla `detections/log.csv` dosyasına yazılır.

## Kurulum

Python 3.9 – 3.12 önerilir (TensorFlow henüz çok yeni Python sürümlerini desteklemeyebilir).

```bash
pip install -r requirements.txt
```

## Kullanım

| Amaç | Komut |
|------|-------|
| Modeli eğit | `python improved_training.py` |
| Ensemble eğit (3 model) | `python ensemble_training.py` |
| Test setini değerlendir | `python quick_improve.py` |
| Tek görüntü test et | `python quick_improve.py --image yol/gorsel.jpg` |
| Canlı kamera | `python improved_realtime.py` |
| Video dosyasıyla test | `python improved_realtime.py --video test.mp4` |
| Ensemble ile canlı | `python improved_realtime.py --meta ensemble_meta.json` |

Canlı kontroller: **Q** çıkış, **R** kaynağı yeniden aç, **G** ön filtreyi aç/kapat.

## Proje yapısı

```
common.py                  Ortak ayarlar, ön işleme, Predictor (eğitim = test = canlı aynı ön işleme)
improved_training.py       2 aşamalı eğitim (başlık → ince ayar), eşik seçimi, model_meta.json
ensemble_training.py       EfficientNetB0/B1/B2 ensemble, ensemble_meta.json
quick_improve.py           Tüm test setini değerlendirir, eşik analizi, tek görüntü tahmini
improved_realtime.py       Canlı tespit (yumuşatma, alarm, kayıt)
AracAltiGorseller_Train/   Eğitim verisi (YabanciMaddeOlan / YabanciMaddeOlmayan)
AracAltiGorseller_Test/    Test verisi
```

## Güncel sonuçlar (EfficientNetB0, eşik 0.51, 16 test görüntüsü)

| | Değer |
|---|---|
| Tehlikeli yakalama (recall) | 8 / 8 |
| Yanlış alarm | 2 / 8 temiz görüntü |
| Doğruluk / ROC-AUC | %88 / 1.00 |

Bu sayılar **çok küçük ve yapay zekâ ile üretilmiş bir test setinden** gelir; gerçek sahada aynı performans beklenmemelidir. Eşik doğrulama setinden seçildi; test seti eşik analizi için `python quick_improve.py` çalıştırın.

## Önemli notlar (dürüst durum)

- **Veri azdır ve büyük kısmı yapay zekâ ile üretilmiştir** (75 eğitim + 16 test görüntüsü). Açık veri seti bulunmadığı için bu bir kısıt. Bu yüzden:
  - Test sonuçları çok güvenilir değildir; 1 görüntü yüzde 6 oynatır.
  - Model, üretilmiş görüntülerin stiline aşırı uyum sağlamış olabilir; **gerçek kamera görüntüsünde performans düşebilir**.
  - Gerçek sahadan (farklı araç, ışık, zemin) görüntü toplayıp `Train/Test` klasörlerine eklemek en büyük iyileştirmedir.
- Eski README'deki "%100 tespit / %0 yanlış negatif" iddiaları kaldırılmıştır; o sonuçlar 16 görüntülük testte ve tutarsız ön işleme ile elde edilmişti.
- Test görüntüleri eğitim görüntülerinin yakın kopyası olmamalıdır (özellikle aynı prompt'tan üretilenler), aksi halde skor şişer.
- Eski `improved_vehicle_underneath_model.h5` hatalı ön işlemeyle eğitilmişti ve yeni Keras'ta yüklenmiyor; artık kullanılmıyor (silinebilir). `improved_config.json` da kullanılmıyor (ayarlar kodda).
- Bu sistem yanlış negatif (kaçırma) üretebilir. Kritik kontrol noktalarında mutlaka manuel kontrol ile birlikte kullanın.

## Geliştirme fikirleri

1. Gerçek saha verisi toplamak; mümkünse tehlikeli nesne konumunu işaretleyip (YOLO gibi) **nesne tespitine** geçmek.
2. Grad-CAM ile modelin görüntüde nereye baktığını göstermek.
3. Çoklu kamera, web arayüzü, güvenlik sistemi entegrasyonu.
