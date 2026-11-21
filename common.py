# -*- coding: utf-8 -*-
"""Tüm scriptlerin ortak kullandığı ayarlar, ön işleme ve tahmin (inference) kodu.

ÖNEMLİ: Eğitim, test ve gerçek zamanlı tespit AYNI ön işlemeyi bu dosyadan kullanır.
Eskiden eğitimde ResNet50, canlı kodda EfficientNet ön işlemesi kullanılıyordu (uyuşmazlık).
Artık: görüntü RGB, 224x224, 0-255 aralığında verilir; EfficientNet kendi içinde normalize eder.
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "AracAltiGorseller_Train")
TEST_DIR = os.path.join(BASE_DIR, "AracAltiGorseller_Test")

IMAGE_SIZE = (224, 224)
# Sıra önemli: indeks 0 = tehlikeli (pozitif) sınıf
CLASS_NAMES = ["YabanciMaddeOlan", "YabanciMaddeOlmayan"]
POSITIVE_INDEX = 0

MODEL_PATH = os.path.join(BASE_DIR, "improved_vehicle_underneath_model.keras")
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, "improved_vehicle_underneath_model.h5")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")
ENSEMBLE_META_PATH = os.path.join(BASE_DIR, "ensemble_meta.json")

DEFAULT_THRESHOLD = 0.40  # P(YabanciMaddeOlan) bu değerin üstündeyse alarm (güvenlik için 0.5'ten düşük)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder):
    """Bir klasördeki görüntü yollarını sıralı döndür."""
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def load_test_set(root=TEST_DIR):
    """root/<sınıf adı>/* yapısından (yollar, pozitif_mi etiketleri) döndür."""
    paths, labels = [], []
    for idx, name in enumerate(CLASS_NAMES):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for p in list_images(folder):
            paths.append(p)
            labels.append(1 if idx == POSITIVE_INDEX else 0)
    return paths, np.array(labels)


def choose_threshold(y_true_pos, p_pos, min_recall=0.95):
    """Tehlikeli sınıf recall'ı >= min_recall olacak en yüksek eşiği seç (yanlış alarmı en aza indirir).
    Veri çok azsa güvenilir değildir; sonuç [0.2, 0.6] aralığına kırpılır."""
    y_true_pos = np.asarray(y_true_pos)
    p_pos = np.asarray(p_pos)
    if y_true_pos.sum() == 0 or len(p_pos) == 0:
        return DEFAULT_THRESHOLD
    best = None
    for t in np.arange(0.05, 0.96, 0.01):
        pred = p_pos >= t
        recall = (pred & (y_true_pos == 1)).sum() / y_true_pos.sum()
        if recall >= min_recall:
            best = t
    if best is None:
        best = 0.2
    return float(np.clip(best, 0.2, 0.6))


class Predictor:
    """Kaydedilmiş modeli (veya ensemble'ı) yükler ve P(YabanciMaddeOlan) üretir."""

    def __init__(self, meta_path=None):
        import tensorflow as tf  # geç import: sadece gerekince yüklensin

        self.tf = tf
        meta = None
        for candidate in ([meta_path] if meta_path else [META_PATH]):
            if candidate and os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta_dir = os.path.dirname(os.path.abspath(candidate))
                break

        if meta is not None:
            self.model_paths = [os.path.join(meta_dir, m) for m in meta["models"]]
            self.preprocess = meta.get("preprocess", "raw")
            self.threshold = float(meta.get("threshold", DEFAULT_THRESHOLD))
            self.class_names = meta.get("class_names", CLASS_NAMES)
        else:
            # Eski .h5 model yeni Keras'ta yüklenemiyor ve hatalı ön işlemeyle eğitilmişti; desteklenmiyor.
            raise FileNotFoundError("model_meta.json bulunamadı. Önce eğitin: python improved_training.py")

        self.models = [tf.keras.models.load_model(p, compile=False) for p in self.model_paths]
        self.image_size = tuple(IMAGE_SIZE)

    # --- ön işleme -------------------------------------------------------
    def _prepare(self, rgb_batch):
        x = self.tf.image.resize(rgb_batch.astype("float32"), self.image_size).numpy()  # eğitimle aynı yöntem
        if self.preprocess == "resnet_legacy":
            from tensorflow.keras.applications.resnet50 import preprocess_input
            x = preprocess_input(x)
        return x

    # --- tahmin ----------------------------------------------------------
    def predict_rgb_batch(self, rgb_batch, tta=False):
        """rgb_batch: (N,H,W,3) uint8/float RGB. Dönen: (N,) P(YabanciMaddeOlan)."""
        x = self._prepare(np.asarray(rgb_batch))
        views = [x]
        if tta:
            views.append(x[:, :, ::-1, :])  # yatay çevrilmiş görünüm
        probs = []
        for m in self.models:
            for v in views:
                probs.append(m.predict(v, verbose=0)[:, POSITIVE_INDEX])
        return np.mean(probs, axis=0)

    def predict_bgr_frame(self, frame, tta=True):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return float(self.predict_rgb_batch(rgb[None, ...], tta=tta)[0])

    def predict_paths(self, paths, tta=False, batch_size=8):
        from PIL import Image
        out = []
        for i in range(0, len(paths), batch_size):
            imgs = []
            for p in paths[i:i + batch_size]:
                arr = np.asarray(Image.open(p).convert("RGB"), dtype="float32")
                # eğitimdeki yükleyiciyle aynı yeniden boyutlandırma (tf bilinear)
                imgs.append(self.tf.image.resize(arr, self.image_size).numpy())
            out.append(self.predict_rgb_batch(np.stack(imgs), tta=tta))
        return np.concatenate(out) if out else np.array([])
