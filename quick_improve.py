# -*- coding: utf-8 -*-
"""Kaydedilmiş modeli TÜM test setinde değerlendirir ve eşik analizi yapar.

Kullanım:
  python quick_improve.py                 # AracAltiGorseller_Test/ üzerinde değerlendir
  python quick_improve.py --meta ensemble_meta.json
  python quick_improve.py --image yol/gorsel.jpg   # tek görüntü tahmini
"""
import argparse

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from common import Predictor, load_test_set


def evaluate(pred, tta):
    paths, y = load_test_set()
    if len(paths) == 0:
        print("Test görüntüsü bulunamadı.")
        return
    p = pred.predict_paths(paths, tta=tta)
    print(f"=== Test seti: {len(paths)} görüntü, karar eşiği = {pred.threshold:.2f} ===\n")
    for path, label, prob in zip(paths, y, p):
        ok = (prob >= pred.threshold) == bool(label)
        print(f"{'OK ' if ok else 'HATA'} P(tehlikeli)={prob:.2f}  gerçek={'tehlikeli' if label else 'temiz':9s} {path}")

    yhat = (p >= pred.threshold).astype(int)
    print("\n" + classification_report(y, yhat, labels=[0, 1], target_names=["Temiz", "Tehlikeli"], zero_division=0))
    print("Confusion matrix (satır=gerçek, sütun=tahmin) [Temiz, Tehlikeli]:")
    print(confusion_matrix(y, yhat, labels=[0, 1]))
    if len(set(y)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y, p):.3f}")

    print("\nEşik analizi (tehlikeli sınıf için):")
    print("eşik  kaçırılan  yanlış_alarm")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        pr = p >= t
        missed = int(((~pr) & (y == 1)).sum())
        fa = int((pr & (y == 0)).sum())
        print(f"{t:.1f}   {missed:3d}/{int(y.sum())}      {fa:3d}/{int((y == 0).sum())}")
    print("\nNot: Test seti küçük olduğundan bu sayılar yalnızca kaba bir fikir verir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()

    pred = Predictor(a.meta)
    tta = not a.no_tta
    if a.image:
        p = float(pred.predict_paths([a.image], tta=tta)[0])
        verdict = "YABANCI MADDE VAR" if p >= pred.threshold else "temiz"
        print(f"{a.image}: P(tehlikeli)={p:.3f} -> {verdict}")
    else:
        evaluate(pred, tta)


if __name__ == "__main__":
    main()
