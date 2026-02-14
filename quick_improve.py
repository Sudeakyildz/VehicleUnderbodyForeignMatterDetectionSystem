# -*- coding: utf-8 -*-
import sys
import io
# Windows terminalinde encoding sorununu çöz
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np  # Sayısal hesaplamalar için numpy kütüphanesi
from tensorflow.keras.models import load_model  # Kaydedilmiş modeli yüklemek için
from tensorflow.keras.preprocessing.image import img_to_array  # Görüntüyü numpy array'e çevirmek için
from tensorflow.keras.applications.efficientnet import preprocess_input  # EfficientNet için ön işleme fonksiyonu
import cv2  # Görüntü okuma ve işleme için OpenCV
import json  # JSON dosya işlemleri için

def test_current_model():
    """Mevcut modeli test görüntüleri üzerinde test et"""
    print("=== MEVCUT MODEL TEST ===")
    
    # Model yükle - eğitilmiş modeli dosyadan yükle
    model = load_model('improved_vehicle_underneath_model.h5')
    
    # Test görüntüleri - test edilecek görüntüler ve beklenen sonuçları
    test_images = [
        ('AracAltiGorseller_Test/YabanciMaddeOlan/00.jpg', 'Tehlikeli Madde'),  # Tehlikeli madde içeren görüntü
        ('AracAltiGorseller_Test/YabanciMaddeOlan/07.jpg', 'Tehlikeli Madde'),  # Tehlikeli madde içeren görüntü
        ('AracAltiGorseller_Test/YabanciMaddeOlmayan/34.jpg', 'Temiz')  # Temiz görüntü
    ]
    
    results = []  # Test sonuçlarını saklamak için liste
    
    # Her test görüntüsü için döngü
    for img_path, expected in test_images:
        try:
            # Görüntüyü yükle ve işle
            img = cv2.imread(img_path)  # Görüntüyü dosyadan oku (BGR formatında)
            img_resized = cv2.resize(img, (224, 224))  # Görüntüyü model için gerekli boyuta yeniden boyutlandır (224x224)
            img_array = img_to_array(img_resized)  # Görüntüyü numpy array'e çevir
            img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle (1, 224, 224, 3)
            processed_img = preprocess_input(img_array)  # EfficientNet için ön işleme yap (normalizasyon)
            
            # Tahmin yap - modelden tahmin al
            prediction = model.predict(processed_img, verbose=0)  # Model tahmin yapar (verbose=0 sessiz mod)
            predicted_class = np.argmax(prediction[0])  # En yüksek olasılıklı sınıfın indeksini bul
            confidence = np.max(prediction[0]) * 100  # En yüksek olasılığı yüzde olarak hesapla
            
            # Sınıf isimleri - sınıf indekslerine karşılık gelen isimler
            class_names = ['YabanciMaddeOlan', 'YabanciMaddeOlmayan']  # Sınıf isimleri listesi
            predicted_name = class_names[predicted_class]  # Tahmin edilen sınıfın ismini al
            
            # Sonuç kontrolü - tahminin doğru olup olmadığını kontrol et
            is_correct = (predicted_name == 'YabanciMaddeOlan' and expected == 'Tehlikeli Madde') or \
                        (predicted_name == 'YabanciMaddeOlmayan' and expected == 'Temiz')  # Tahmin beklenenle eşleşiyor mu?
            
            # Sonuç sözlüğü oluştur
            result = {
                'image': img_path,  # Görüntü dosya yolu
                'expected': expected,  # Beklenen sonuç
                'predicted': predicted_name,  # Tahmin edilen sınıf
                'confidence': confidence,  # Güven skoru (yüzde)
                'correct': is_correct  # Tahmin doğru mu?
            }
            
            results.append(result)  # Sonucu listeye ekle
            
            # Sonuçları yazdır
            print(f"📸 {img_path}")  # Görüntü yolunu yazdır
            print(f"   Beklenen: {expected}")  # Beklenen sonucu yazdır
            print(f"   Tahmin: {predicted_name} ({confidence:.1f}%)")  # Tahmin ve güven skorunu yazdır
            print(f"   Doğru: {'✅' if is_correct else '❌'}")  # Doğru veya yanlış işareti
            print()  # Boş satır
            
        except Exception as e:
            print(f"❌ Hata: {img_path} - {e}")  # Hata durumunda mesaj yazdır
    
    # Özet - tüm test sonuçlarının özeti
    correct_count = sum(1 for r in results if r['correct'])  # Doğru tahmin sayısını hesapla
    total_count = len(results)  # Toplam test sayısı
    accuracy = (correct_count / total_count) * 100  # Genel doğruluğu yüzde olarak hesapla
    
    # Özet bilgileri yazdır
    print(f"=== ÖZET ===")
    print(f"Toplam Test: {total_count}")  # Toplam test sayısı
    print(f"Doğru Tahmin: {correct_count}")  # Doğru tahmin sayısı
    print(f"Genel Doğruluk: {accuracy:.1f}%")  # Genel doğruluk yüzdesi
    
    return results, accuracy  # Sonuçları ve doğruluğu döndür

def create_improved_config():
    """İyileştirilmiş model eğitimi için konfigürasyon oluştur"""
    print("\n=== İYİLEŞTİRİLMİŞ KONFİGÜRASYON ===")
    
    # İyileştirilmiş konfigürasyon sözlüğü
    config = {
        # Sınıf ağırlıkları - dengesiz veri seti için
        'class_weights': {
            'YabanciMaddeOlan': 2.0,      # Tehlikeli madde sınıfına 2x ağırlık ver (daha önemli)
            'YabanciMaddeOlmayan': 1.0   # Temiz sınıfı normal ağırlıkta
        },
        # Veri artırma parametreleri - görüntü çeşitliliğini artırmak için
        'data_augmentation': {
            'rotation_range': 60,           # ±60 derece rotasyon (daha fazla açı çeşitliliği)
            'width_shift_range': 0.4,      # %40 genişlik kaydırma (daha fazla kaydırma)
            'height_shift_range': 0.4,     # %40 yükseklik kaydırma
            'shear_range': 0.4,            # %40 eğme (shear transformation)
            'zoom_range': 0.4,             # %40 zoom (daha fazla yakınlaştırma/uzaklaştırma)
            'brightness_range': [0.3, 1.7], # %30-170 parlaklık aralığı (daha geniş ışık koşulları)
            'channel_shift_range': 80.0     # ±80 renk kanalı kayması (daha geniş renk varyasyonu)
        },
        # Eğitim parametreleri - model eğitimi için
        'training_params': {
            'learning_rate': 0.00003,      # Çok düşük öğrenme hızı (stabil eğitim için)
            'batch_size': 4,               # Küçük batch size (overfitting'i önlemek için)
            'epochs': 200,                 # Daha fazla epoch (daha uzun eğitim)
            'patience': 30                 # Daha fazla patience (erken durdurma için daha uzun bekleme)
        }
    }
    
    # Sınıf ağırlıklarını yazdır
    print("✅ Class Weights:")
    for class_name, weight in config['class_weights'].items():
        print(f"   {class_name}: {weight}x")  # Her sınıf için ağırlık bilgisini yazdır
    
    # Veri artırma parametrelerini yazdır
    print("\n✅ Veri Artırma:")
    for param, value in config['data_augmentation'].items():
        print(f"   {param}: {value}")  # Her parametre için değer bilgisini yazdır
    
    # Eğitim parametrelerini yazdır
    print("\n✅ Eğitim Parametreleri:")
    for param, value in config['training_params'].items():
        print(f"   {param}: {value}")  # Her parametre için değer bilgisini yazdır
    
    # Konfigürasyonu JSON dosyasına kaydet
    with open('improved_config.json', 'w', encoding='utf-8') as f:  # UTF-8 encoding ile Türkçe karakter desteği
        json.dump(config, f, ensure_ascii=False, indent=2)  # JSON formatında kaydet (güzel format)
    
    print(f"\n💾 Konfigürasyon kaydedildi: improved_config.json")
    
    return config  # Konfigürasyonu döndür

def main():
    """Ana fonksiyon - hızlı model test ve iyileştirme önerileri"""
    print("🚀 === HIZLI MODEL İYİLEŞTİRME ===")
    
    # 1. Mevcut modeli test et - model performansını ölç
    results, current_accuracy = test_current_model()
    
    # 2. İyileştirilmiş konfigürasyon oluştur - yeni eğitim için ayarlar
    config = create_improved_config()
    
    # 3. Öneriler - performansa göre iyileştirme önerileri
    print("\n=== ÖNERİLER ===")
    
    # Performansa göre öneriler
    if current_accuracy < 80:
        # Düşük performans için öneriler
        print("🔴 Model performansı düşük! İyileştirme gerekli.")
        print("   - Class weights kullan")  # Sınıf ağırlıkları kullanılmalı
        print("   - Daha agresif veri artırma")  # Daha fazla veri artırma gerekli
        print("   - Base model katmanlarını eğitilebilir yap")  # Fine-tuning yapılmalı
    elif current_accuracy < 90:
        # Orta performans için öneriler
        print("🟡 Model performansı orta. İyileştirme faydalı olabilir.")
        print("   - Class weights ekle")  # Sınıf ağırlıkları eklenebilir
        print("   - Veri artırma iyileştir")  # Veri artırma iyileştirilebilir
    else:
        # İyi performans için mesaj
        print("🟢 Model performansı iyi! Küçük iyileştirmeler yeterli.")
    
    # Mevcut ve hedef doğruluğu yazdır
    print(f"\n📊 Mevcut Doğruluk: {current_accuracy:.1f}%")  # Mevcut doğruluk
    print(f"🎯 Hedef Doğruluk: 95%+")  # Hedef doğruluk
    
    # Tamamlanma mesajı
    print("\n✅ Hızlı iyileştirme tamamlandı!")
    print("💡 Şimdi improved_training.py ile eğitim yapabilirsiniz.")  # Sonraki adım önerisi

if __name__ == "__main__":
    main()  # Script doğrudan çalıştırıldığında main fonksiyonunu çağır
