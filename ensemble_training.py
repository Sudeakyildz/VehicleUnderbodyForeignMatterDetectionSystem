import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input, Concatenate
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.metrics import Precision, Recall
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Ensemble konfigürasyon
class EnsembleConfig:
    TRAIN_PATH = "AracAltiGorseller_Train/"
    TEST_PATH = "AracAltiGorseller_Test/"
    
    # Ensemble parametreleri
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 4
    EPOCHS = 200
    LEARNING_RATE = 0.00003  # Çok düşük learning rate
    
    ARAÇ_ALTI_KATEGORİLERİ = [
        "YabanciMaddeOlan",
        "YabanciMaddeOlmayan"
    ]
    
    SINIF_SAYISI = len(ARAÇ_ALTI_KATEGORİLERİ)
    
    # Class weights for imbalanced dataset
    CLASS_WEIGHTS = {
        0: 2.0,  # YabanciMaddeOlan - çok daha ağırlıklı
        1: 1.0   # YabanciMaddeOlmayan
    }

class EnsembleVehicleDetector:
    def __init__(self, config):
        self.config = config
        self.models = []
        self.ensemble_model = None
        self.train_generator = None
        self.test_generator = None
        self.val_generator = None
        self.class_names = None
        
    def create_ensemble_data_generators(self):
        """Ensemble için veri artırma"""
        # Çok agresif veri artırma
        train_datagen = ImageDataGenerator(
            preprocessing_function=None,  # Her model kendi preprocessing kullanacak
            rotation_range=60,        # Çok daha fazla rotasyon
            width_shift_range=0.4,    # Çok daha fazla kaydırma
            height_shift_range=0.4,
            shear_range=0.4,          # Çok daha fazla eğme
            zoom_range=0.4,           # Çok daha fazla zoom
            horizontal_flip=True,     # Yatay çevirme
            vertical_flip=False,      # Dikey çevirme yok
            fill_mode='nearest',
            brightness_range=[0.3, 1.7],  # Çok daha geniş ışık ayarı
            channel_shift_range=80.0,     # Çok daha geniş renk kayması
            validation_split=0.2
        )
        
        # Eğitim verisi
        self.train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_PATH,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            subset='training',
            seed=42
        )
        
        # Doğrulama verisi
        self.val_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_PATH,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            subset='validation',
            seed=42
        )
        
        # Test verisi
        self.test_generator = train_datagen.flow_from_directory(
            self.config.TEST_PATH,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"🚀 Ensemble veri seti yüklendi!")
        print(f"Eğitim sınıfları: {self.train_generator.class_indices}")
        print(f"Eğitim örnekleri: {self.train_generator.samples}")
        print(f"Test örnekleri: {self.test_generator.samples}")

        self.class_names = [name for name, idx in sorted(self.train_generator.class_indices.items(), key=lambda x: x[1])]

        # Sınıf eşlemesini kaydet
        try:
            with open('ensemble_class_names.json', 'w', encoding='utf-8') as f:
                json.dump({"class_names": self.class_names}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Sınıf isimleri kaydedilemedi: {e}")
    
    def build_individual_models(self):
        """Bireysel modelleri oluştur"""
        print("🔧 Bireysel modeller oluşturuluyor...")
        
        # Model 1: EfficientNetB0
        print("1️⃣ EfficientNetB0 oluşturuluyor...")
        efficientnet_base = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMAGE_SIZE, 3)
        )
        
        # Tüm katmanları eğitilebilir yap
        for layer in efficientnet_base.layers:
            layer.trainable = True
        
        efficientnet_model = Sequential([
            efficientnet_base,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.config.SINIF_SAYISI, activation='softmax')
        ])
        
        efficientnet_model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models.append(('EfficientNetB0', efficientnet_model))
        
        # Model 2: ResNet50
        print("2️⃣ ResNet50 oluşturuluyor...")
        resnet_base = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMAGE_SIZE, 3)
        )
        
        # Tüm katmanları eğitilebilir yap
        for layer in resnet_base.layers:
            layer.trainable = True
        
        resnet_model = Sequential([
            resnet_base,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.config.SINIF_SAYISI, activation='softmax')
        ])
        
        resnet_model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models.append(('ResNet50', resnet_model))
        
        # Model 3: VGG16
        print("3️⃣ VGG16 oluşturuluyor...")
        vgg_base = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMAGE_SIZE, 3)
        )
        
        # Tüm katmanları eğitilebilir yap
        for layer in vgg_base.layers:
            layer.trainable = True
        
        vgg_model = Sequential([
            vgg_base,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.config.SINIF_SAYISI, activation='softmax')
        ])
        
        vgg_model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models.append(('VGG16', vgg_model))
        
        print(f"✅ {len(self.models)} model oluşturuldu!")
    
    def train_individual_models(self):
        """Bireysel modelleri eğit"""
        print("🚀 Bireysel model eğitimi başlıyor...")
        
        for name, model in self.models:
            print(f"\n🎯 {name} eğitiliyor...")
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-9),
                ModelCheckpoint(f'{name.lower()}_model.h5', 
                              monitor='val_accuracy', save_best_only=True, verbose=1)
            ]
            
            # Eğitim
            history = model.fit(
                self.train_generator,
                steps_per_epoch=self.train_generator.samples // self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                validation_data=self.val_generator,
                validation_steps=self.val_generator.samples // self.config.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=self.config.CLASS_WEIGHTS,
                verbose=1
            )
            
            print(f"✅ {name} eğitimi tamamlandı!")
    
    def build_ensemble_model(self):
        """Ensemble model oluştur"""
        print("🔧 Ensemble model oluşturuluyor...")
        
        # Her model için input
        inputs = []
        model_outputs = []
        
        for name, model in self.models:
            # Model'in son katmanını çıkar
            base_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            base_model.trainable = False  # Ensemble'da eğitilemez
            
            input_layer = Input(shape=(*self.config.IMAGE_SIZE, 3))
            inputs.append(input_layer)
            
            # Her model için preprocessing
            if 'EfficientNet' in name:
                processed = efficientnet_preprocess(input_layer)
            elif 'ResNet' in name:
                processed = resnet_preprocess(input_layer)
            else:  # VGG
                processed = vgg_preprocess(input_layer)
            
            # Model çıktısı
            output = base_model(processed)
            model_outputs.append(output)
        
        # Çıktıları birleştir
        if len(model_outputs) > 1:
            concatenated = Concatenate()(model_outputs)
        else:
            concatenated = model_outputs[0]
        
        # Ensemble sınıflandırıcı
        ensemble_output = Dense(512, activation='relu')(concatenated)
        ensemble_output = BatchNormalization()(ensemble_output)
        ensemble_output = Dropout(0.5)(ensemble_output)
        
        ensemble_output = Dense(256, activation='relu')(ensemble_output)
        ensemble_output = BatchNormalization()(ensemble_output)
        ensemble_output = Dropout(0.3)(ensemble_output)
        
        ensemble_output = Dense(self.config.SINIF_SAYISI, activation='softmax')(ensemble_output)
        
        # Ensemble model
        self.ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
        
        self.ensemble_model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE * 0.1),  # Daha düşük LR
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        
        print("✅ Ensemble model oluşturuldu!")
        self.ensemble_model.summary()
    
    def train_ensemble_model(self):
        """Ensemble modeli eğit"""
        print("🚀 Ensemble model eğitimi başlıyor...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-10),
            ModelCheckpoint('ensemble_vehicle_underneath_model.h5', 
                          monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        # Eğitim
        history = self.ensemble_model.fit(
            [self.train_generator] * len(self.models),  # Her model için aynı veri
            steps_per_epoch=self.train_generator.samples // self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=[self.val_generator] * len(self.models),
            validation_steps=self.val_generator.samples // self.config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=self.config.CLASS_WEIGHTS,
            verbose=1
        )
        
        return history
    
    def evaluate_ensemble(self):
        """Ensemble modeli değerlendir"""
        print("🔍 Ensemble model değerlendiriliyor...")
        
        # Test verisi ile değerlendir
        test_loss, test_accuracy, test_precision, test_recall = self.ensemble_model.evaluate(
            [self.test_generator] * len(self.models),
            steps=self.test_generator.samples // self.config.BATCH_SIZE,
            verbose=1
        )
        
        print(f"\n📊 Ensemble Model Test Sonuçları:")
        print(f"Test Doğruluğu: {test_accuracy:.4f}")
        print(f"Test Kaybı: {test_loss:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Tahminler
        predictions = self.ensemble_model.predict(
            [self.test_generator] * len(self.models),
            steps=self.test_generator.samples // self.config.BATCH_SIZE,
            verbose=1
        )
        
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        print("\n📋 Ensemble Model Sınıflandırma Raporu:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Ensemble Model Confusion Matrix')
        plt.ylabel('Gerçek Sınıf')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.savefig('ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_ensemble_training(self):
        """Ensemble eğitim sürecini çalıştır"""
        print("🚀 === ENSEMBLE ARAÇ ALTI TESPİT SİSTEMİ ===")
        
        # 1. Veri setini yükle
        self.create_ensemble_data_generators()
        
        # 2. Bireysel modelleri oluştur
        self.build_individual_models()
        
        # 3. Bireysel modelleri eğit
        self.train_individual_models()
        
        # 4. Ensemble modeli oluştur
        self.build_ensemble_model()
        
        # 5. Ensemble modeli eğit
        history = self.train_ensemble_model()
        
        # 6. Değerlendir
        self.evaluate_ensemble()
        
        print("✅ Ensemble eğitim tamamlandı!")

def main():
    """Ana fonksiyon"""
    config = EnsembleConfig()
    detector = EnsembleVehicleDetector(config)
    detector.run_ensemble_training()

if __name__ == "__main__":
    main()
