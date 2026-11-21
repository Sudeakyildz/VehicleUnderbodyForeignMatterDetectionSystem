# -*- coding: utf-8 -*-
"""Gerçek zamanlı araç altı yabancı madde tespiti.

Eski sürüme göre iyileştirmeler:
  * Eğitimle aynı ön işleme ve eğitimde seçilen karar eşiği (common.py / model_meta.json).
  * Zamansal yumuşatma: tek kareye göre değil son N karenin çoğunluğuna göre alarm (titreşim/yanlış alarm azalır).
  * Modeli her kare yerine --every karede bir çalıştırır (daha akıcı FPS).
  * Alarmda kare görüntüsü detections/ klasörüne ve detections/log.csv dosyasına kaydedilir.
  * Kamera yerine video dosyası ile test (--video), Windows dışında da çalışır (ses yoksa terminal zili).

Kullanım:
  python improved_realtime.py [--camera 0] [--video dosya.mp4] [--meta ensemble_meta.json]
                              [--no-gate] [--no-sound] [--every 3]
Tuşlar: Q = çıkış, R = kamerayı yeniden aç, G = araç-altı ön filtresini aç/kapat
"""
import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from common import BASE_DIR, Predictor

try:
    import winsound  # sadece Windows
except ImportError:
    winsound = None

DETECTION_DIR = os.path.join(BASE_DIR, "detections")


def looks_like_underbody(frame):
    """Klasik görüntü işleme ile ön filtre: kadrajda koyu, geniş, yatay yapılı bir araç altı var mı?
    Eşikler kameraya/ışığa göre ayar gerektirir; uygun değilse --no-gate ile kapatın."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total = gray.size
    avg = float(np.mean(gray))
    very_dark_ratio = float(np.sum(gray < 70)) / total
    dark_ratio = float(np.sum(gray < 100)) / total

    _, thresh = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wide_dark, huge_dark = 0, 0
    half_h = frame.shape[0] // 2
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 8000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / max(h, 1)
        if 1.6 < ar < 3.8 and y > half_h * 0.5:
            wide_dark += 1
        if area > 15000 and 1.4 < ar < 4.2:
            huge_dark += 1

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / total
    dark_px = gray[gray < 100]
    dark_std = float(np.std(dark_px)) if dark_px.size else 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz = float(np.sum(cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel) > 0)) / total

    m1 = (avg < 95 and very_dark_ratio > 0.12 and dark_ratio > 0.30 and wide_dark > 0
          and 0.006 < edge_density < 0.14 and dark_std < 45 and horiz > 0.001)
    m2 = huge_dark > 0 and avg < 100 and very_dark_ratio > 0.10 and dark_std < 50 and horiz > 0.0005
    m3 = avg < 90 and dark_ratio > 0.35 and wide_dark > 0 and horiz > 0.001
    return (m1 or m2 or m3), {"parlaklik": avg, "koyu": dark_ratio, "cok_koyu": very_dark_ratio,
                              "kenar": edge_density, "yatay": horiz, "std": dark_std}


class RealTimeDetector:
    def __init__(self, args):
        self.args = args
        self.predictor = Predictor(args.meta)
        self.threshold = self.predictor.threshold
        self.history = deque(maxlen=args.window)   # son N karenin P(tehlikeli) değerleri
        self.use_gate = not args.no_gate
        self.cap = None
        self.last_p = None
        self.alarm = False
        self.last_beep = 0.0
        self.last_save = 0.0
        os.makedirs(DETECTION_DIR, exist_ok=True)
        self.log_path = os.path.join(DETECTION_DIR, "log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["zaman", "olasilik", "esik", "goruntu"])
        print(f"Model(ler): {[os.path.basename(p) for p in self.predictor.model_paths]}  eşik={self.threshold:.2f}")

    # --- kamera ------------------------------------------------------------
    def open_source(self):
        src = self.args.video if self.args.video else self.args.camera
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"Kaynak açılamadı: {src}")
            return False
        if not self.args.video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    # --- karar -------------------------------------------------------------
    def update(self, frame, frame_idx):
        present, info = (looks_like_underbody(frame) if self.use_gate else (True, {}))
        if not present:
            self.history.clear()
            self.last_p = None
            self.alarm = False
            return present, info
        if self.last_p is None or frame_idx % self.args.every == 0:
            self.last_p = self.predictor.predict_bgr_frame(frame, tta=False)
            self.history.append(self.last_p)
        # Çoğunluk kuralı: pencere dolduğunda karelerin >= yarısı eşiği aşmalı ve ortalama da yüksek olmalı
        h = np.array(self.history)
        votes = (h >= self.threshold).sum()
        self.alarm = len(h) >= max(3, self.history.maxlen // 2) and votes >= (len(h) + 1) // 2 \
            and float(h.mean()) >= self.threshold * 0.9
        return present, info

    def beep(self):
        if time.time() - self.last_beep < 2 or self.args.no_sound:
            return
        self.last_beep = time.time()
        if winsound:
            try:
                winsound.Beep(1000, 400)
                return
            except Exception:
                pass
        print("\a UYARI: yabanci madde!", flush=True)

    def save_detection(self, frame):
        if time.time() - self.last_save < 3:
            return
        self.last_save = time.time()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(DETECTION_DIR, f"alarm_{stamp}.jpg")
        cv2.imwrite(img_path, frame)
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([stamp, f"{float(np.mean(self.history)):.3f}", f"{self.threshold:.2f}", img_path])

    # --- ekran -------------------------------------------------------------
    def draw(self, frame, present, info, fps):
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = frame.shape[:2]
        if not present:
            cv2.putText(frame, "ARAC ALTI BEKLENIYOR", (20, 40), font, 0.7, (160, 160, 160), 2)
        elif self.alarm:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(frame, "YABANCI MADDE TESPIT EDILDI!", (20, 40), font, 0.85, (0, 0, 255), 2)
        elif self.last_p is not None:
            cv2.putText(frame, "TEMIZ", (20, 40), font, 0.85, (0, 200, 0), 2)
        if self.last_p is not None and present:
            cv2.putText(frame, f"P(tehlikeli)={self.last_p:.2f}  ort={np.mean(self.history):.2f}  esik={self.threshold:.2f}",
                        (20, 70), font, 0.5, (255, 255, 255), 1)
        y = 100
        if info and self.args.debug:
            for k, v in info.items():
                cv2.putText(frame, f"{k}: {v:.3f}", (20, y), font, 0.4, (255, 255, 0), 1)
                y += 16
        cv2.putText(frame, f"FPS: {fps:.1f}  filtre: {'acik' if self.use_gate else 'kapali'}", (w - 220, 25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Q: cikis | R: yeniden ac | G: filtre", (10, h - 15), font, 0.5, (255, 255, 255), 1)
        return frame

    def run(self):
        if not self.open_source():
            return
        print("Başladı. Q = çıkış, R = yeniden aç, G = ön filtre aç/kapat")
        frame_idx, fps, count, t0 = 0, 0.0, 0, time.time()
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Kare okunamadı / video bitti.")
                    break
                frame_idx += 1
                present, info = self.update(frame, frame_idx)
                if self.alarm:
                    self.beep()
                    self.save_detection(frame)
                count += 1
                if time.time() - t0 >= 1:
                    fps, count, t0 = count / (time.time() - t0), 0, time.time()
                cv2.imshow("Arac Alti Tespit Sistemi", self.draw(frame, present, info, fps))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    self.cap.release()
                    time.sleep(0.5)
                    if not self.open_source():
                        break
                if key in (ord("g"), ord("G")):
                    self.use_gate = not self.use_gate
        except KeyboardInterrupt:
            pass
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="kamera yerine video dosyası")
    ap.add_argument("--meta", default=None, help="model_meta.json veya ensemble_meta.json")
    ap.add_argument("--window", type=int, default=7, help="yumuşatma penceresi (tahmin sayısı)")
    ap.add_argument("--every", type=int, default=3, help="modeli kaç karede bir çalıştır")
    ap.add_argument("--no-gate", action="store_true", help="araç-altı ön filtresini kapat")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--debug", action="store_true", help="ön filtre ölçümlerini ekranda göster")
    RealTimeDetector(ap.parse_args()).run()


if __name__ == "__main__":
    main()
