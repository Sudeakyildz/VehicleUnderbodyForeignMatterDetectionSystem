# -*- coding: utf-8 -*-
import sys
import io
# Windows terminalinde encoding sorununu çöz
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2  # Bilgisayarlı görü ve video işleme için OpenCV kütüphanesi
import numpy as np  # Sayısal hesaplamalar için numpy kütüphanesi
import time  # Zaman işlemleri için (uyarı cooldown kontrolü)
import winsound  # Windows ses uyarıları için (tehlikeli madde tespitinde bip sesi)
from tensorflow.keras.models import load_model  # Kaydedilmiş modeli yüklemek için
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess  # EfficientNet için ön işleme fonksiyonu
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess  # ResNet50 için ön işleme (kullanılmıyor ama import edilmiş)
from tensorflow.keras.preprocessing.image import img_to_array  # Görüntüyü numpy array'e çevirmek için
from PIL import Image  # Görüntü işleme için PIL kütüphanesi
import json  # JSON dosya işlemleri için (sınıf isimlerini yüklemek için)

class ImprovedRealTimeDetector:
    """Gerçek zamanlı araç altı tespit sistemi sınıfı"""
    def __init__(self):
        """Sınıf başlatıcı - tüm değişkenleri başlangıç değerlerine ayarla"""
        self.models = []  # Yüklenecek modellerin listesi (şu anda tek model kullanılıyor)
        self.class_names = None  # Sınıf isimleri listesi (henüz yüklenmedi)
        self.cap = None  # VideoCapture nesnesi (kamera bağlantısı)
        self.vehicle_detected = False  # Araç altı tespit edildi mi? (flag)
        self.last_warning_time = 0  # Son uyarı zamanı (ses uyarısı cooldown için)
        self.warning_cooldown = 2  # Uyarılar arası bekleme süresi (saniye) - sürekli bip sesini önlemek için
        self.debug_info = {}  # Debug bilgileri için sözlük (araç tespit kriterleri bilgileri)
        self.weights = [0.6, 0.4]  # Model ağırlıkları (ensemble için, şu anda kullanılmıyor)
        print("=== İyileştirilmiş Araç Altı Tespit Sistemi ===")  # Başlangıç mesajı
        print("Bu versiyon EfficientNet modeli kullanıyor")  # Model bilgisi
        self.load_models()  # Modeli yükle
    
    def load_models(self):
        """Eğitilmiş modeli ve sınıf isimlerini yükle"""
        try:
            print("İyileştirilmiş model yükleniyor...")
            
            # İyileştirilmiş model (EfficientNet) yükleme
            try:
                improved_model = load_model('improved_vehicle_underneath_model.h5')  # Model dosyasından yükle
                self.models.append(('improved', improved_model))  # Modeli listeye ekle (isim, model) tuple olarak
                print("✅ İyileştirilmiş model (EfficientNet) yüklendi")  # Başarı mesajı
            except Exception as e:
                print(f"❌ İyileştirilmiş model yüklenemedi: {e}")  # Hata mesajı
                return False  # Başarısız olduğunu belirt
            
            # Sınıf isimlerini JSON dosyasından yükle
            try:
                with open('improved_class_names.json', 'r') as f:  # JSON dosyasını aç
                    class_data = json.load(f)  # JSON verisini oku
                    self.class_names = class_data.get('class_names', ['YabanciMaddeOlan', 'YabanciMaddeOlmayan'])  # Sınıf isimlerini al, yoksa varsayılan kullan
                print(f"✅ Sınıf isimleri yüklendi: {self.class_names}")  # Başarı mesajı
            except:
                # JSON dosyası yoksa varsayılan sınıf isimlerini kullan
                self.class_names = ['YabanciMaddeOlan', 'YabanciMaddeOlmayan']  # Varsayılan sınıf isimleri
                print(f"⚠️ Varsayılan sınıf isimleri kullanılıyor: {self.class_names}")  # Uyarı mesajı
            
            print(f"✅ Model başarıyla yüklendi!")  # Genel başarı mesajı
            return True  # Başarılı olduğunu belirt
            
        except Exception as e:
            print(f"❌ Model yüklenirken hata: {e}")  # Genel hata mesajı
            return False  # Başarısız olduğunu belirt
    
    def start_camera(self, camera_id=0):
        """Kamerayı başlat ve ayarlarını yapılandır"""
        try:
            self.cap = cv2.VideoCapture(camera_id)  # Belirtilen kamera ID'si ile VideoCapture nesnesi oluştur
            if not self.cap.isOpened():  # Kamera açılabildi mi kontrol et
                print(f"❌ Kamera {camera_id} açılamadı!")  # Hata mesajı
                return False  # Başarısız olduğunu belirt
            
            # Kamera ayarları - performans ve kalite için optimize edilmiş değerler
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Frame genişliğini 640 piksel yap
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Frame yüksekliğini 480 piksel yap
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Frame rate'i 30 FPS yap (gerçek zamanlı işleme için yeterli)
            
            print(f"✅ Kamera {camera_id} başarıyla açıldı!")  # Başarı mesajı
            return True  # Başarılı olduğunu belirt
        except Exception as e:
            print(f"❌ Kamera başlatılırken hata: {e}")  # Hata mesajı
            return False  # Başarısız olduğunu belirt
    
    def detect_vehicle_underneath(self, frame):
        """Detect if frame contains vehicle underneath"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Balanced vehicle underneath detection - specific but not too strict
            
            # 1. Check frame brightness (vehicle underneath is usually dark)
            avg_brightness = np.mean(gray)
            
            # 2. Check brightness distribution (vehicle underneath has more dark pixels)
            very_dark_pixels = np.sum(gray < 70)  # Slightly increased threshold
            dark_pixels = np.sum(gray < 100)      # Increased threshold
            total_pixels = gray.size
            very_dark_ratio = very_dark_pixels / total_pixels
            dark_ratio = dark_pixels / total_pixels
            
            # 3. Check for large, rectangular dark areas with specific characteristics
            _, thresh = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)  # Increased threshold
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours for vehicle-like characteristics
            vehicle_contours = []
            large_dark_areas = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 8000:  # Reduced area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (vehicle underneath is usually wide and not too tall)
                    aspect_ratio = w / h
                    if 1.6 < aspect_ratio < 3.8:  # Wider range
                        # Check if it's in the center/lower part of the frame
                        frame_center_y = frame.shape[0] // 2
                        if y > frame_center_y * 0.5:  # Lower 50% of frame
                            vehicle_contours.append(contour)
                
                # Check for very large dark areas
                if area > 15000:  # Reduced threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 1.4 < aspect_ratio < 4.2:  # Wider range
                        large_dark_areas += 1
            
            # 4. Check for edge density (vehicle underneath has moderate edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # 5. Check for texture consistency (vehicle underneath has consistent dark texture)
            # Calculate standard deviation of brightness in dark regions
            dark_region = gray[gray < 100]  # Increased threshold
            if len(dark_region) > 0:
                dark_std = np.std(dark_region)
            else:
                dark_std = 255
            
            # 6. Check for horizontal line patterns (vehicle underneath has horizontal structures)
            # Apply horizontal edge detection
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_edge_density = np.sum(horizontal_edges > 0) / total_pixels
            
            # 7. Combined detection logic - balanced criteria
            is_detected = False
            
            # Primary method: Multiple balanced criteria for vehicle underneath
            if (avg_brightness < 95 and                    # Increased brightness threshold
                very_dark_ratio > 0.12 and                 # Reduced very dark pixel ratio
                dark_ratio > 0.30 and                      # Reduced dark pixel ratio
                len(vehicle_contours) > 0 and              # Vehicle-like contours found
                0.006 < edge_density < 0.14 and            # Wider edge density range
                dark_std < 45 and                          # Increased texture threshold
                horizontal_edge_density > 0.001):          # Reduced horizontal threshold
                is_detected = True
            
            # Secondary method: Very large dark area with good characteristics
            if not is_detected:
                if (large_dark_areas > 0 and 
                    avg_brightness < 100 and 
                    very_dark_ratio > 0.10 and
                    dark_std < 50 and
                    horizontal_edge_density > 0.0005):     # Further reduced
                    is_detected = True
            
            # Tertiary method: Simple but effective detection
            if not is_detected:
                if (avg_brightness < 90 and 
                    dark_ratio > 0.35 and 
                    len(vehicle_contours) > 0 and
                    horizontal_edge_density > 0.001):
                    is_detected = True
            
            # Debug info - store for display
            self.debug_info = {
                'brightness': avg_brightness,
                'dark_ratio': dark_ratio,
                'very_dark_ratio': very_dark_ratio,
                'contours_found': len(vehicle_contours),
                'edge_density': edge_density,
                'horizontal_edges': horizontal_edge_density,
                'large_dark_areas': large_dark_areas,
                'dark_std': dark_std,
                'method1': (avg_brightness < 95 and very_dark_ratio > 0.12 and dark_ratio > 0.30 and len(vehicle_contours) > 0 and 0.006 < edge_density < 0.14 and dark_std < 45 and horizontal_edge_density > 0.001),
                'method2': (large_dark_areas > 0 and avg_brightness < 100 and very_dark_ratio > 0.10 and dark_std < 50 and horizontal_edge_density > 0.0005),
                'method3': (avg_brightness < 90 and dark_ratio > 0.35 and len(vehicle_contours) > 0 and horizontal_edge_density > 0.001)
            }
            
            return is_detected
            
        except Exception as e:
            print(f"❌ Araç tespitinde hata: {e}")
            return False
    
    def predict_single_model(self, frame):
        """Tek model tahmin yap"""
        try:
            # Resize image
            frame_resized = cv2.resize(frame, (224, 224))
            
            # Convert to PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            
            # Convert to numpy array
            img_array = img_to_array(img_pil)
            img_array = np.expand_dims(img_array, axis=0)
            
            # EfficientNet preprocess
            processed_img = efficientnet_preprocess(img_array)
            
            # Tahmin yap
            model = self.models[0][1]  # İlk modeli al
            prediction = model.predict(processed_img, verbose=0)
            
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100  # Convert to percentage
            predicted_label = self.class_names[predicted_class]
            
            return predicted_label, confidence, prediction[0]
            
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            return "Error", 0.0, None
    
    def process_frame(self, frame):
        """Process single frame"""
        try:
            # First check if vehicle underneath is detected
            self.vehicle_detected = self.detect_vehicle_underneath(frame)
            
            if not self.vehicle_detected:
                return "No Vehicle", 0.0
            
            print("🚗 Araç tespit edildi! Tahmin yapılıyor...")  # Debug
            
            # Single model prediction
            result, confidence, prediction = self.predict_single_model(frame)
            
            print(f"🎯 Tahmin: {result} ({confidence:.2f}%)")  # Debug
            
            return result, confidence
            
        except Exception as e:
            print(f"❌ Frame işleme hatası: {e}")
            return "Error", 0.0
    
    def play_warning_sound(self):
        """Play warning sound for foreign material detection"""
        current_time = time.time()
        if current_time - self.last_warning_time > self.warning_cooldown:
            try:
                # Play beep sound (frequency: 1000Hz, duration: 500ms)
                winsound.Beep(1000, 500)
                self.last_warning_time = current_time
            except:
                # Fallback: print warning
                print("🚨 UYARI: Yabancı madde tespit edildi!")
    
    def draw_results(self, frame, result, confidence, fps, processing_time):
        """Draw results on frame"""
        # Draw main result
        if not self.vehicle_detected:
            cv2.putText(frame, "ARAC TESPIT EDILMEDI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            cv2.putText(frame, "Arac bekleniyor...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        else:
            if result == "YabanciMaddeOlan":
                color = (0, 0, 255)  # Red
                text = "YABANCI MADDE TESPIT EDILDI!"
                self.play_warning_sound()
            else:
                color = (0, 255, 0)  # Green
                text = "YABANCI MADDE YOK"
            
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Güven: {confidence:.2f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw debug information
        if hasattr(self, 'debug_info') and self.debug_info:
            y_offset = 100
            cv2.putText(frame, "DEBUG BILGILERI:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
            cv2.putText(frame, f"Parlaklık: {self.debug_info.get('brightness', 0):.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Koyu Oranı: {self.debug_info.get('dark_ratio', 0):.3f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Çok Koyu: {self.debug_info.get('very_dark_ratio', 0):.3f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Konturlar: {self.debug_info.get('contours_found', 0)}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Kenar Yoğunluğu: {self.debug_info.get('edge_density', 0):.4f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Yatay: {self.debug_info.get('horizontal_edges', 0):.4f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Koyu Std: {self.debug_info.get('dark_std', 0):.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Yöntem1: {'EVET' if self.debug_info.get('method1', False) else 'HAYIR'}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Yöntem2: {'EVET' if self.debug_info.get('method2', False) else 'HAYIR'}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 15
            cv2.putText(frame, f"Yöntem3: {'EVET' if self.debug_info.get('method3', False) else 'HAYIR'}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw status
        status_color = (0, 255, 0) if self.vehicle_detected else (128, 128, 128)
        status_text = "ARAC TESPIT EDILDI" if self.vehicle_detected else "ARAC YOK"
        cv2.putText(frame, status_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Draw model info
        cv2.putText(frame, f"EfficientNet Model", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw performance info
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"İşlem: {processing_time:.1f}ms", (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw controls
        cv2.putText(frame, "Q: Çıkış | R: Yeniden Başlat", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run main loop"""
        if not self.load_models():
            return
        
        if not self.start_camera():
            return
        
        print("🚀 İyileştirilmiş gerçek zamanlı tespit başladı!")
        print("Kontroller: Q = Çıkış, R = Yeniden Başlat")
        print("Sistem sadece araç altı görünür olduğunda tespit yapar")
        print("Model: EfficientNetB0")
        
        fps_counter = 0
        fps_start_time = time.time()
        last_fps = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame okunamadı")
                    break
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    last_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Process frame
                start_time = time.time()
                result, confidence = self.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Draw results on frame
                frame = self.draw_results(frame, result, confidence, last_fps, processing_time)
                
                # Show frame
                cv2.imshow('İyileştirilmiş Araç Altı Tespit Sistemi', frame)
                
                # Key controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("👋 Kullanıcı tarafından durduruldu")
                    break
                elif key == ord('r') or key == ord('R'):
                    print("🔄 Sistem yeniden başlatılıyor...")
                    self.cap.release()
                    time.sleep(1)
                    if not self.start_camera():
                        break
                
        except KeyboardInterrupt:
            print("\n👋 Program kullanıcı tarafından durduruldu")
        except Exception as e:
            print(f"❌ Program hatası: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Sistem temizlendi")

def main():
    """Main function"""
    print("=== İyileştirilmiş Araç Altı Tespit Sistemi ===")
    print("Bu versiyon EfficientNet modeli kullanıyor")
    
    # Create detector
    detector = ImprovedRealTimeDetector()
    
    # Run system
    detector.run()

if __name__ == "__main__":
    main()
