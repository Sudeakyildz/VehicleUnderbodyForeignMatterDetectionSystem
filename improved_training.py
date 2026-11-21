# -*- coding: utf-8 -*-
"""Araç altı yabancı madde sınıflandırıcısı eğitimi (EfficientNet, transfer learning).

Eski sürüme göre düzeltmeler:
  * Eğitim/test/canlı tespit aynı ön işlemeyi kullanır (bkz. common.py).
  * Doğrulama verisine veri artırma UYGULANMAZ (eskiden uygulanıyordu).
  * İki aşamalı eğitim: önce sadece sınıflandırıcı başlık, sonra son katmanlar ince ayar.
    (75 görüntüyle tüm ağı en baştan eğitmek aşırı öğrenmeye yol açıyordu.)
  * Sınıf ağırlıkları veriden otomatik hesaplanır; tehlikeli sınıfa ek güvenlik çarpanı uygulanır.
  * Karar eşiği doğrulama setinden seçilir ve model_meta.json içine kaydedilir.

Kullanım:  python improved_training.py [--backbone EfficientNetB0] [--epochs-head 15] [--epochs-ft 60]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # pencere açmadan dosyaya kaydet
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

from common import (BASE_DIR, CLASS_NAMES, DEFAULT_THRESHOLD, IMAGE_SIZE, META_PATH, MODEL_PATH,
                    POSITIVE_INDEX, TRAIN_DIR, choose_threshold, load_test_set)

SAFETY_WEIGHT_FACTOR = 1.5  # tehlikeli sınıfı kaçırmak daha pahalı


def _load_image(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, IMAGE_SIZE), tf.one_hot(label, len(CLASS_NAMES))


def _paths_to_ds(paths, labels, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((np.array(paths), np.array(labels, dtype="int32")))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)


def make_datasets(batch_size=8, seed=42, val_split=0.25):
    """Sınıf dengeli (stratified) eğitim/doğrulama ayrımı. (Keras'ın otomatik ayrımı sınıfa bakmadığı için
    doğrulama setine hiç tehlikeli örnek düşebiliyordu.)"""
    from sklearn.model_selection import train_test_split
    from common import list_images
    paths, labels = [], []
    for idx, name in enumerate(CLASS_NAMES):
        for p in list_images(os.path.join(TRAIN_DIR, name)):
            paths.append(p)
            labels.append(idx)
    tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=val_split,
                                              stratify=labels, random_state=seed)
    print(f"Eğitim: {len(tr_p)}  Doğrulama: {len(va_p)} (sınıf dağılımı: {np.bincount(va_l)})")
    test_p, test_l = load_test_set()
    # load_test_set pozitif=1 etiketi döndürür; burada sınıf indeksine çeviriyoruz (pozitif=0)
    test_idx = [POSITIVE_INDEX if l == 1 else 1 - POSITIVE_INDEX for l in test_l]
    train = _paths_to_ds(tr_p, tr_l, batch_size, True, seed)
    val = _paths_to_ds(va_p, va_l, batch_size, False, seed)
    test = _paths_to_ds(test_p, test_idx, batch_size, False, seed)
    return train, val, test


def build_augmenter():
    """Sadece eğitim verisine uygulanır. Araç altı için yatay çevirme mantıklı, dikey değil."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomTranslation(0.12, 0.12),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.3, value_range=(0, 255)),
    ], name="augment")


def class_weights_from(ds):
    counts = np.zeros(len(CLASS_NAMES))
    for _, y in ds:
        counts += y.numpy().sum(axis=0)
    total = counts.sum()
    weights = {i: float(total / (len(CLASS_NAMES) * max(c, 1))) for i, c in enumerate(counts)}
    weights[POSITIVE_INDEX] *= SAFETY_WEIGHT_FACTOR
    print(f"Eğitim sınıf sayıları: {dict(zip(CLASS_NAMES, counts.astype(int)))}  ağırlıklar: {weights}")
    return weights


def build_model(backbone="EfficientNetB0"):
    base_cls = getattr(keras.applications, backbone)
    # Not: Keras EfficientNet 0-255 RGB girişi bekler, normalizasyonu kendi içinde yapar.
    base = base_cls(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"underbody_{backbone}")
    return model, base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])


def unfreeze_top(base, n_layers=30):
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers[-n_layers:]:  # BatchNorm'u dondurulmuş tut (küçük veri için daha stabil)
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def collect(model, ds):
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in ds])
    p_pos = model.predict(ds, verbose=0)[:, POSITIVE_INDEX]
    return (y_true == POSITIVE_INDEX).astype(int), p_pos


def report(y_pos, p_pos, threshold, title, prefix):
    pred = (p_pos >= threshold).astype(int)
    print(f"\n=== {title} (eşik={threshold:.2f}) ===")
    names = ["Temiz", "Tehlikeli"]
    print(classification_report(y_pos, pred, labels=[0, 1], target_names=names, zero_division=0))
    if len(set(y_pos)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_pos, p_pos):.3f}")
    cm = confusion_matrix(y_pos, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title(f"{title} confusion matrix"); plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
    plt.savefig(os.path.join(BASE_DIR, f"{prefix}_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    missed = int(((pred == 0) & (y_pos == 1)).sum())
    print(f"Kaçırılan tehlikeli örnek: {missed} / {int(y_pos.sum())}")


def save_history(histories, path):
    keys = ["loss", "accuracy", "auc"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k in zip(axes, keys):
        tr = sum((h.history.get(k, []) for h in histories), [])
        va = sum((h.history.get("val_" + k, []) for h in histories), [])
        ax.plot(tr, label="eğitim"); ax.plot(va, label="doğrulama"); ax.set_title(k); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def train_backbone(backbone, model_path, epochs_head=15, epochs_ft=60, batch_size=8, seed=42):
    """Tek bir backbone eğitir, modeli kaydeder. (model, val_y, val_p, test_y, test_p) döndürür."""
    keras.utils.set_random_seed(seed)
    train_raw, val_ds, test_ds = make_datasets(batch_size, seed)
    cw = class_weights_from(train_raw)
    aug = build_augmenter()
    train_ds = train_raw.map(lambda x, y: (aug(x, training=True), y),
                             num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    model, base = build_model(backbone)
    cbs = lambda patience: [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(patience // 2, 2), min_lr=1e-7),
    ]

    print(f"\n--- Aşama 1: sadece sınıflandırıcı başlık ({backbone}) ---")
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, class_weight=cw,
                   callbacks=cbs(6), verbose=1)

    print(f"\n--- Aşama 2: son katmanların ince ayarı ({backbone}) ---")
    unfreeze_top(base)
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_ft, class_weight=cw,
                   callbacks=cbs(12), verbose=1)

    model.save(model_path)
    print(f"Model kaydedildi: {model_path}")
    save_history([h1, h2], os.path.join(BASE_DIR, f"improved_training_history_{backbone}.png"))

    val_y, val_p = collect(model, val_ds)
    test_y, test_p = collect(model, test_ds)
    return model, val_y, val_p, test_y, test_p


def write_meta(path, model_files, threshold, backbones):
    meta = {"models": model_files, "backbones": backbones, "preprocess": "raw",
            "threshold": threshold, "class_names": CLASS_NAMES, "image_size": list(IMAGE_SIZE)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # Eski kodla uyumluluk için
    with open(os.path.join(BASE_DIR, "improved_class_names.json"), "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASS_NAMES}, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="EfficientNetB0")
    ap.add_argument("--epochs-head", type=int, default=15)
    ap.add_argument("--epochs-ft", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    _, val_y, val_p, test_y, test_p = train_backbone(
        a.backbone, MODEL_PATH, a.epochs_head, a.epochs_ft, a.batch_size, a.seed)

    thr = choose_threshold(val_y, val_p) if val_y.sum() else DEFAULT_THRESHOLD
    print(f"\nDoğrulama setinden seçilen karar eşiği: {thr:.2f}")
    report(val_y, val_p, thr, "Doğrulama", "improved_val")
    report(test_y, test_p, thr, "Test", "improved")
    write_meta(META_PATH, [os.path.basename(MODEL_PATH)], thr, [a.backbone])
    print("\nTamamlandı. Test: python quick_improve.py   Canlı: python improved_realtime.py")
    print("NOT: Test seti çok küçük (16 görüntü); sonuçlar kesin performansı göstermez.")


if __name__ == "__main__":
    main()
