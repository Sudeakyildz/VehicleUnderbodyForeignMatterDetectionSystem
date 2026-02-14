import numpy as np  # Sayısal hesaplamalar için numpy kütüphanesi
import pandas as pd  # Veri işleme için pandas kütüphanesi (kullanılmıyor ama import edilmiş)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # Görüntü veri artırma ve ön işleme için
from tensorflow.keras.models import Sequential  # Sıralı model oluşturmak için
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization  # Sinir ağı katmanları
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input  # ResNet50 transfer learning modeli (kullanılmıyor ama import edilmiş)
from tensorflow.keras.applications.efficientnet import EfficientNetB0  # EfficientNetB0 transfer learning modeli
from PIL import Image  # Görüntü işleme için PIL kütüphanesi
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib
import os  # İşletim sistemi işlemleri için
import cv2  # Bilgisayarlı görü için OpenCV kütüphanesi
from sklearn.metrics import classification_report, confusion_matrix  # Model performans metrikleri
import seaborn as sns  # İstatistiksel görselleştirme için
from tensorflow.keras.metrics import Precision, Recall  # Precision ve Recall metrikleri
import json  # JSON dosya işlemleri için
from tensorflow.keras.optimizers import Adam  # Adam optimizer (gradient descent algoritması)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # Eğitim sırasında kullanılacak callback'ler

# İyileştirilmiş konfigürasyon sınıfı - tüm eğitim parametrelerini içerir
class ImprovedConfig:
    TRAIN_PATH = "AracAltiGorseller_Train/"  # Eğitim görüntülerinin bulunduğu klasör yolu
    TEST_PATH = "AracAltiGorseller_Test/"  # Test görüntülerinin bulunduğu klasör yolu
    
    # İyileştirilmiş parametreler
    IMAGE_SIZE = (224, 224)  # Görüntülerin yeniden boyutlandırılacağı boyut (EfficientNet standart boyutu)
    BATCH_SIZE = 4   # Her eğitim adımında işlenecek görüntü sayısı (küçük batch size overfitting'i önler)
    EPOCHS = 150     # Modelin tüm veri seti üzerinden kaç kez geçeceği (daha fazla epoch = daha uzun eğitim)
    LEARNING_RATE = 0.00005  # Öğrenme hızı (düşük LR = daha yavaş ama daha stabil öğrenme)
    
    ARAÇ_ALTI_KATEGORİLERİ = [  # Sınıflandırılacak kategoriler
        "YabanciMaddeOlan",  # Araç altında yabancı madde var
        "YabanciMaddeOlmayan"  # Araç altı temiz
    ]
    
    SINIF_SAYISI = len(ARAÇ_ALTI_KATEGORİLERİ)  # Toplam sınıf sayısı (2 sınıf)
    
    # Class weights - dengesiz veri seti için sınıf ağırlıkları
    CLASS_WEIGHTS = {
        0: 1.5,  # YabanciMaddeOlan sınıfına 1.5x ağırlık ver (daha az örnek olduğu için daha önemli)
        1: 1.0   # YabanciMaddeOlmayan sınıfı normal ağırlıkta
    }

class ImprovedVehicleDetector:
    """Araç altı görüntü tespit sistemi için iyileştirilmiş model sınıfı"""
    def __init__(self, config):
        """Sınıf başlatıcı - tüm değişkenleri None olarak ayarla"""
        self.config = config  # Konfigürasyon nesnesini sakla
        self.model = None  # Eğitilecek model (henüz oluşturulmadı)
        self.train_generator = None  # Eğitim veri generator'ı (henüz oluşturulmadı)
        self.test_generator = None  # Test veri generator'ı (henüz oluşturulmadı)
        self.val_generator = None  # Doğrulama veri generator'ı (henüz oluşturulmadı)
        self.class_names = None  # Sınıf isimleri listesi (henüz yüklenmedi)
        
    def create_improved_data_generators(self):
        """İyileştirilmiş veri artırma ve veri yükleyicileri oluştur"""
        # Çok agresif veri artırma - tehlikeli madde tespiti için daha fazla çeşitlilik
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # EfficientNet için ön işleme fonksiyonu
            rotation_range=45,        # Görüntüleri ±45 derece arası rastgele döndür
            width_shift_range=0.3,    # Görüntüleri genişlik yönünde %30'a kadar kaydır
            height_shift_range=0.3,   # Görüntüleri yükseklik yönünde %30'a kadar kaydır
            shear_range=0.3,          # Görüntüleri %30'a kadar eğ (shear transformation)
            zoom_range=0.3,           # Görüntüleri %30'a kadar yakınlaştır/uzaklaştır
            horizontal_flip=True,     # Görüntüleri yatay olarak çevir (ayna efekti)
            vertical_flip=False,      # Dikey çevirme yapma (araç altı görüntüleri için mantıklı değil)
            fill_mode='nearest',      # Dönüşüm sonrası boş kalan alanları en yakın pikselle doldur
            brightness_range=[0.5, 1.5],  # Parlaklığı %50-150 arası değiştir (ışık koşullarına karşı dayanıklılık)
            channel_shift_range=50.0,     # RGB kanallarını ±50 değer aralığında kaydır (renk varyasyonu)
            validation_split=0.2  # Verinin %20'sini doğrulama için ayır
        )
        
        # Test verisi için veri artırma yok (sadece ön işleme)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        # Eğitim verisi generator'ı oluştur
        self.train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_PATH,  # Eğitim görüntülerinin bulunduğu klasör
            target_size=self.config.IMAGE_SIZE,  # Görüntüleri bu boyuta yeniden boyutlandır
            batch_size=self.config.BATCH_SIZE,  # Her seferinde bu kadar görüntü işle
            class_mode='categorical',  # Çoklu sınıf kategorik mod (one-hot encoding)
            shuffle=True,  # Görüntüleri karıştır (her epoch'ta farklı sıralama)
            subset='training',  # Eğitim verisi olarak kullan (validation_split'in geri kalanı)
            seed=42  # Rastgelelik için seed değeri (tutarlılık için)
        )
        
        # Doğrulama verisi generator'ı oluştur
        self.val_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_PATH,  # Aynı klasörden doğrulama verisi al
            target_size=self.config.IMAGE_SIZE,  # Aynı boyut
            batch_size=self.config.BATCH_SIZE,  # Aynı batch size
            class_mode='categorical',  # Aynı class mode
            shuffle=False,  # Doğrulama verisini karıştırma (tutarlılık için)
            subset='validation',  # Doğrulama verisi olarak kullan (validation_split'in %20'si)
            seed=42  # Aynı seed değeri
        )
        
        # Test verisi generator'ı oluştur
        self.test_generator = test_datagen.flow_from_directory(
            self.config.TEST_PATH,  # Test görüntülerinin bulunduğu klasör
            target_size=self.config.IMAGE_SIZE,  # Aynı boyut
            batch_size=self.config.BATCH_SIZE,  # Aynı batch size
            class_mode='categorical',  # Aynı class mode
            shuffle=False  # Test verisini karıştırma (tutarlılık için)
        )
        
        # Bilgilendirme mesajları yazdır
        print(f"İyileştirilmiş veri seti yüklendi!")
        print(f"Eğitim sınıfları: {self.train_generator.class_indices}")  # Sınıf isimlerini ve indekslerini göster
        print(f"Eğitim örnekleri: {self.train_generator.samples}")  # Toplam eğitim görüntü sayısı
        print(f"Test örnekleri: {self.test_generator.samples}")  # Toplam test görüntü sayısı

        # Sınıf isimlerini indeks sırasına göre sırala ve listeye çevir
        self.class_names = [name for name, idx in sorted(self.train_generator.class_indices.items(), key=lambda x: x[1])]

        # Sınıf eşlemesini JSON dosyasına kaydet (gerçek zamanlı tespit için gerekli)
        try:
            with open('improved_class_names.json', 'w', encoding='utf-8') as f:  # UTF-8 encoding ile Türkçe karakter desteği
                json.dump({"class_names": self.class_names}, f, ensure_ascii=False, indent=2)  # JSON formatında kaydet
        except Exception as e:
            print(f"Sınıf isimleri kaydedilemedi: {e}")  # Hata durumunda mesaj yazdır
    
    def build_improved_model(self):
        """İyileştirilmiş model mimarisi oluştur - Base model katmanları eğitilebilir"""
        # EfficientNetB0 transfer learning modelini kullan
        base_model = EfficientNetB0(
            weights='imagenet',  # ImageNet veri setinde önceden eğitilmiş ağırlıkları yükle
            include_top=False,  # Üst sınıflandırma katmanlarını dahil etme (kendi katmanlarımızı ekleyeceğiz)
            input_shape=(*self.config.IMAGE_SIZE, 3)  # Giriş görüntü boyutu: (224, 224, 3) - RGB
        )
        
        # TÜM base model katmanlarını eğitilebilir yap (transfer learning iyileştirmesi)
        # Varsayılan olarak base model katmanları dondurulur, ama biz tam fine-tuning yapıyoruz
        for layer in base_model.layers:
            layer.trainable = True  # Her katmanı eğitilebilir yap
        
        # İyileştirilmiş model mimarisi - daha derin sınıflandırıcı katmanları
        self.model = Sequential([
            base_model,  # EfficientNetB0 base modeli (özellik çıkarıcı)
            GlobalAveragePooling2D(),  # 2D özellik haritasını 1D vektöre çevir (ortalama alma)
            Dense(1024, activation='relu'),  # 1024 nöronlu tam bağlı katman (ReLU aktivasyon)
            BatchNormalization(),  # Batch normalizasyon (eğitimi stabilize eder)
            Dropout(0.4),  # %40 dropout (overfitting'i önlemek için rastgele nöronları kapat)
            Dense(512, activation='relu'),  # 512 nöronlu tam bağlı katman
            BatchNormalization(),  # Batch normalizasyon
            Dropout(0.3),  # %30 dropout
            Dense(256, activation='relu'),  # 256 nöronlu tam bağlı katman
            BatchNormalization(),  # Batch normalizasyon
            Dropout(0.2),  # %20 dropout
            Dense(128, activation='relu'),  # 128 nöronlu tam bağlı katman
            BatchNormalization(),  # Batch normalizasyon
            Dropout(0.1),  # %10 dropout
            Dense(self.config.SINIF_SAYISI, activation='softmax')  # Çıkış katmanı: 2 sınıf için softmax (olasılık dağılımı)
        ])
        
        # İyileştirilmiş optimizer - daha düşük learning rate ile Adam optimizer
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE, beta_1=0.9, beta_2=0.999)  # Adam optimizer parametreleri
        
        # Modeli derle (compile) - optimizer, kayıp fonksiyonu ve metrikleri belirle
        self.model.compile(
            optimizer=optimizer,  # Adam optimizer kullan
            loss='categorical_crossentropy',  # Çoklu sınıf kategorik cross-entropy kayıp fonksiyonu
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]  # İzlenecek metrikler: doğruluk, precision, recall
        )
        
        # Bilgilendirme mesajları yazdır
        print("İyileştirilmiş araç altı tespit modeli oluşturuldu:")
        print("Base model katmanları eğitilebilir yapıldı")  # Fine-tuning yapılıyor
        print("Daha derin mimari eklendi")  # Çok katmanlı sınıflandırıcı
        print("Class weights kullanılacak")  # Dengesiz veri seti için ağırlıklandırma
        self.model.summary()  # Model mimarisini detaylı olarak yazdır
    
    def train_improved_model(self):
        """İyileştirilmiş model eğitimi başlat"""
        print("İyileştirilmiş model eğitimi başlıyor...")
        
        # İyileştirilmiş callbacks - eğitim sırasında otomatik kontrol ve iyileştirme
        callbacks = [
            # EarlyStopping: Doğrulama kaybı iyileşmezse erken durdur
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),  # 25 epoch sabırla bekle, en iyi ağırlıkları geri yükle
            # ReduceLROnPlateau: Doğrulama kaybı iyileşmezse öğrenme hızını azalt
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-9),  # LR'ı 5'te 1'e düşür, 10 epoch bekle, min LR sınırı
            # ModelCheckpoint: En iyi modeli kaydet
            ModelCheckpoint('improved_vehicle_underneath_model.h5', 
                          monitor='val_accuracy', save_best_only=True, verbose=1)  # En yüksek doğrulama doğruluğuna sahip modeli kaydet
        ]
        
        # Model eğitimini başlat
        history = self.model.fit(
            self.train_generator,  # Eğitim verisi generator'ı
            steps_per_epoch=self.train_generator.samples // self.config.BATCH_SIZE,  # Her epoch'ta kaç adım (batch) işlenecek
            epochs=self.config.EPOCHS,  # Toplam epoch sayısı
            validation_data=self.val_generator,  # Doğrulama verisi generator'ı
            validation_steps=self.val_generator.samples // self.config.BATCH_SIZE,  # Doğrulama için adım sayısı
            callbacks=callbacks,  # Callback'leri kullan
            class_weight=self.config.CLASS_WEIGHTS,  # Sınıf ağırlıklarını uygula (dengesiz veri seti için)
            verbose=1  # Eğitim ilerlemesini detaylı göster (1 = progress bar göster)
        )
        
        # Eğitim geçmişini kaydet (grafikler için)
        self.save_improved_training_history(history)
        return history  # Eğitim geçmişini döndür (daha sonra analiz için)
    
    def save_improved_training_history(self, history):
        """İyileştirilmiş eğitim geçmişini görselleştir ve kaydet"""
        # 2x2 subplot oluştur (4 grafik için)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Grafik: Doğruluk (Accuracy) - Eğitim ve doğrulama doğruluğunu karşılaştır
        axes[0, 0].plot(history.history['accuracy'], label='Eğitim Doğruluğu')  # Eğitim doğruluğu çizgisi
        axes[0, 0].plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')  # Doğrulama doğruluğu çizgisi
        axes[0, 0].set_title('İyileştirilmiş Model Doğruluğu')  # Grafik başlığı
        axes[0, 0].set_xlabel('Epoch')  # X ekseni etiketi
        axes[0, 0].set_ylabel('Doğruluk')  # Y ekseni etiketi
        axes[0, 0].legend()  # Göstergeyi ekle
        
        # 2. Grafik: Kayıp (Loss) - Eğitim ve doğrulama kaybını karşılaştır
        axes[0, 1].plot(history.history['loss'], label='Eğitim Kaybı')  # Eğitim kaybı çizgisi
        axes[0, 1].plot(history.history['val_loss'], label='Doğrulama Kaybı')  # Doğrulama kaybı çizgisi
        axes[0, 1].set_title('İyileştirilmiş Model Kaybı')  # Grafik başlığı
        axes[0, 1].set_xlabel('Epoch')  # X ekseni etiketi
        axes[0, 1].set_ylabel('Kayıp')  # Y ekseni etiketi
        axes[0, 1].legend()  # Göstergeyi ekle
        
        # 3. Grafik: Precision - Pozitif tahminlerin ne kadarının doğru olduğu
        axes[1, 0].plot(history.history['precision'], label='Eğitim Precision')  # Eğitim precision çizgisi
        axes[1, 0].plot(history.history['val_precision'], label='Doğrulama Precision')  # Doğrulama precision çizgisi
        axes[1, 0].set_title('İyileştirilmiş Model Precision')  # Grafik başlığı
        axes[1, 0].set_xlabel('Epoch')  # X ekseni etiketi
        axes[1, 0].set_ylabel('Precision')  # Y ekseni etiketi
        axes[1, 0].legend()  # Göstergeyi ekle
        
        # 4. Grafik: Recall - Gerçek pozitiflerin ne kadarının bulunduğu
        axes[1, 1].plot(history.history['recall'], label='Eğitim Recall')  # Eğitim recall çizgisi
        axes[1, 1].plot(history.history['val_recall'], label='Doğrulama Recall')  # Doğrulama recall çizgisi
        axes[1, 1].set_title('İyileştirilmiş Model Recall')  # Grafik başlığı
        axes[1, 1].set_xlabel('Epoch')  # X ekseni etiketi
        axes[1, 1].set_ylabel('Recall')  # Y ekseni etiketi
        axes[1, 1].legend()  # Göstergeyi ekle
        
        plt.tight_layout()  # Subplot'ları düzgün yerleştir (çakışmaları önle)
        plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')  # Yüksek çözünürlükte kaydet
        plt.show()  # Grafiği göster
    
    def evaluate_improved_model(self):
        """İyileştirilmiş modeli test verisi üzerinde değerlendir"""
        print("İyileştirilmiş model değerlendirmesi...")
        
        # Test verisi üzerinde tahmin yap
        predictions = self.model.predict(self.test_generator)  # Modelden tahminleri al (olasılık dağılımları)
        y_pred = np.argmax(predictions, axis=1)  # Her tahmin için en yüksek olasılıklı sınıfı seç (sınıf indeksi)
        y_true = self.test_generator.classes  # Gerçek sınıf etiketlerini al
        
        # Sınıf isimlerini al
        class_names = list(self.test_generator.class_indices.keys())
        
        # Sınıflandırma raporu yazdır (precision, recall, f1-score, support)
        print("\nİyileştirilmiş Araç Altı Tespit Sınıflandırma Raporu:")
        print(classification_report(y_true, y_pred, target_names=class_names))  # Detaylı performans metrikleri
        
        # Confusion matrix oluştur ve görselleştir
        cm = confusion_matrix(y_true, y_pred)  # Karışıklık matrisini hesapla (gerçek vs tahmin)
        plt.figure(figsize=(8, 6))  # Grafik boyutunu ayarla
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # Heatmap oluştur: değerleri göster, mavi renk paleti
                   xticklabels=class_names, yticklabels=class_names)  # X ve Y eksenlerine sınıf isimlerini ekle
        plt.title('İyileştirilmiş Araç Altı Tespit Confusion Matrix')  # Grafik başlığı
        plt.xlabel('Tahmin Edilen')  # X ekseni etiketi
        plt.ylabel('Gerçek')  # Y ekseni etiketi
        plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')  # Yüksek çözünürlükte kaydet
        plt.show()  # Grafiği göster
        
        return predictions, y_pred, y_true  # Tahminleri, tahmin edilen sınıfları ve gerçek sınıfları döndür
    
    def save_improved_model(self, model_path='improved_vehicle_underneath_model.h5'):
        """İyileştirilmiş modeli dosyaya kaydet"""
        self.model.save(model_path)  # Modeli H5 formatında kaydet (tüm ağırlıklar ve mimari dahil)
        print(f"İyileştirilmiş araç altı tespit modeli kaydedildi: {model_path}")

def main():
    """Ana fonksiyon - İyileştirilmiş eğitim sürecini başlat"""
    print("=== İyileştirilmiş Araç Altı Görüntü Tespit Sistemi ===")
    
    # Konfigürasyon nesnesini oluştur
    config = ImprovedConfig()
    
    # Detector sınıfını oluştur
    detector = ImprovedVehicleDetector(config)
    
    # Veri yükleyicileri oluştur (eğitim, doğrulama, test)
    detector.create_improved_data_generators()
    
    # Model mimarisini oluştur
    detector.build_improved_model()
    
    # Model eğitimini başlat
    history = detector.train_improved_model()
    
    # Model değerlendirmesi yap (test verisi üzerinde)
    detector.evaluate_improved_model()
    
    # Modeli dosyaya kaydet
    detector.save_improved_model()
    
    print("İyileştirilmiş araç altı tespit modeli eğitimi tamamlandı!")

if __name__ == "__main__":
    main()  # Script doğrudan çalıştırıldığında main fonksiyonunu çağır
