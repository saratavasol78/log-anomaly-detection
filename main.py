"""اجرای کامل پایپ‌لاین: تولید لاگ ← پارس ← ویژگی ← آموزش ← ارزیابی ← نمودار."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from src.features import build_vocab, make_labels, window_features
from src.generate_logs import generate, save
from src.models import AutoEncoderDetector, IForestDetector
from src.parse_logs import parse_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="مسیر فایل لاگ واقعی (اختیاری)")
    ap.add_argument("--minutes", type=int, default=600)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ۱) داده
    log_path = args.log or os.path.join(args.out, "server.log")
    intervals, t0 = [], None
    if args.log is None:
        text, intervals, t0 = generate(minutes=args.minutes)
        save(log_path, text)
        print(f"[+] {len(text)} خط لاگ تولید شد -> {log_path}")
    df = parse_file(log_path)
    t0 = t0 or df["ts"].min().floor("min")

    # ۲) تقسیم زمانی: نیمه‌ی اول = آموزش (عادی)، نیمه‌ی دوم = آزمون
    split_time = t0 + (df["ts"].max() - t0) / 2
    df_train, df_all = df[df["ts"] < split_time], df
    vocab = build_vocab(df_train)
    X = window_features(df_all, vocab, t0)
    n_train = int((split_time - t0) / np.timedelta64(1, "m"))
    X_train, X_test = X[:n_train], X[n_train:]
    print(f"[+] الگوهای یکتا در آموزش: {len(vocab)} | پنجره‌های آموزش: {len(X_train)} | آزمون: {len(X_test)}")

    # ۳) آموزش و امتیازدهی
    detectors = {
        "IsolationForest": IForestDetector().fit(X_train),
        "AutoEncoder": AutoEncoderDetector().fit(X_train),
    }

    fig, axes = plt.subplots(len(detectors), 1, figsize=(11, 6), sharex=True)
    y_test = make_labels(len(X), intervals)[n_train:] if intervals else None

    for ax, (name, det) in zip(axes, detectors.items()):
        s_train, s_test = det.score(X_train), det.score(X_test)
        thr = np.percentile(s_train, 99)       # آستانه = صدک ۹۹ امتیاز داده‌ی عادی
        pred = (s_test > thr).astype(int)
        print(f"\n===== {name} (threshold={thr:.4f}) =====")
        if y_test is not None:
            print(f"ROC-AUC: {roc_auc_score(y_test, s_test):.3f}")
            print(classification_report(y_test, pred, target_names=["normal", "anomaly"], zero_division=0))
            for i, yv in enumerate(y_test):
                if yv:
                    ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15)
        ax.plot(s_test, lw=1, label="anomaly score")
        ax.axhline(thr, color="k", ls="--", lw=0.8, label="threshold")
        ax.set_title(name)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("minute (test period)")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "scores.png"), dpi=130)
    print(f"\n[+] نمودار ذخیره شد -> {args.out}/scores.png")


if __name__ == "__main__":
    main()
