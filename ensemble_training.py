# -*- coding: utf-8 -*-
"""Ensemble eğitimi: birden fazla EfficientNet varyantını ayrı ayrı eğitir, olasılıklarını ortalar.

Eski sürüm çalışmıyordu (generator listesiyle fit, ön işleme eksikliği, test verisine augmentation).
Bu sürümde tüm modeller aynı ham RGB (0-255) girişini kullanır, bu yüzden ön işleme uyuşmazlığı olmaz.
Sonuç ensemble_meta.json'a yazılır; canlı kullanım:  python improved_realtime.py --meta ensemble_meta.json

Kullanım:  python ensemble_training.py [--backbones EfficientNetB0 EfficientNetB1 EfficientNetB2]
"""
import argparse
import os

import numpy as np

from common import BASE_DIR, DEFAULT_THRESHOLD, ENSEMBLE_META_PATH, choose_threshold
from improved_training import report, train_backbone, write_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["EfficientNetB0", "EfficientNetB1", "EfficientNetB2"])
    ap.add_argument("--epochs-head", type=int, default=15)
    ap.add_argument("--epochs-ft", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    files, val_ps, test_ps = [], [], []
    val_y = test_y = None
    for i, bb in enumerate(a.backbones):
        fname = f"ensemble_{bb.lower()}.keras"
        _, val_y, vp, test_y, tp = train_backbone(
            bb, os.path.join(BASE_DIR, fname), a.epochs_head, a.epochs_ft, a.batch_size, seed=42 + i)
        files.append(fname)
        val_ps.append(vp)
        test_ps.append(tp)

    val_p = np.mean(val_ps, axis=0)
    test_p = np.mean(test_ps, axis=0)
    thr = choose_threshold(val_y, val_p) if val_y.sum() else DEFAULT_THRESHOLD
    report(val_y, val_p, thr, "Ensemble doğrulama", "ensemble_val")
    report(test_y, test_p, thr, "Ensemble test", "ensemble")
    write_meta(ENSEMBLE_META_PATH, files, thr, a.backbones)
    print(f"\nEnsemble hazır: {ENSEMBLE_META_PATH}")


if __name__ == "__main__":
    main()
