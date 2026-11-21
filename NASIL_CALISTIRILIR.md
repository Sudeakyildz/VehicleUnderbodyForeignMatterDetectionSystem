# 🚀 Projeyi Nasıl Çalıştırırım?

PowerShell'de proje klasöründe çalıştırın (kendi yolunuzu kullanın):

```powershell
cd "C:\Users\HUAWE\OneDrive\Desktop\VehicleUnderbodyForeignMatterDetectionSystem-main"
```

## 1. Kurulum
Python 3.9–3.12 önerilir (TensorFlow çok yeni Python sürümlerinde kurulmayabilir).
```powershell
pip install -r requirements.txt
```

## 2. Modeli eğit (ilk kez veya veri değişince)
```powershell
python improved_training.py
```
Çıktılar: `improved_vehicle_underneath_model.keras`, `model_meta.json` (karar eşiği dahil), confusion matrix ve eğitim grafikleri.
Süre: CPU'da yaklaşık 15–40 dakika (veri az olduğu için).

İsteğe bağlı, daha sağlam sonuç için 3 modelli ensemble:
```powershell
python ensemble_training.py
```

## 3. Test et
```powershell
python quick_improve.py
python quick_improve.py --image AracAltiGorseller_Test/YabanciMaddeOlan/00.jpg
```

## 4. Canlı tespit
```powershell
python improved_realtime.py
python improved_realtime.py --video test.mp4
python improved_realtime.py --meta ensemble_meta.json
```
Tuşlar: **Q** çıkış, **R** kaynağı yeniden aç, **G** ön filtreyi aç/kapat.
Ön filtre ortamınıza uymuyorsa (araç altı hiç algılanmıyorsa) `--no-gate --debug` deneyin.
Alarmlar `detections/` klasörüne kaydedilir.

## Sorun giderme
| Sorun | Çözüm |
|-------|-------|
| `No module named tensorflow` | Python sürümünüz çok yeni olabilir; 3.11/3.12 kullanın |
| Model bulunamadı | Önce `python improved_training.py` |
| Kamera açılmıyor | Başka program kullanıyor olabilir; `--camera 1` deneyin |
| Bellek hatası | `python improved_training.py --batch-size 4` |
| Sürekli "araç altı bekleniyor" | `--no-gate` ile ön filtreyi kapatın |
