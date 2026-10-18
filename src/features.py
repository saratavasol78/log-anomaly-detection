"""ساخت بردار ویژگی از پنجره‌های زمانی."""
import numpy as np
import pandas as pd


def build_vocab(df_train):
    """واژگان الگوها فقط از داده‌ی آموزش ساخته می‌شود (برای جلوگیری از نشت داده)."""
    return sorted(df_train["template"].unique())


def window_features(df, vocab, t0, window="1min"):
    """هر پنجره = یک بردار شامل شمارش هر الگو + آمار کلی.

    الگوهای خارج از vocab در ستون `unknown_count` جمع می‌شوند؛
    ظاهر شدن الگوی کاملاً جدید خودش نشانه‌ی ناهنجاری است.
    """
    df = df.copy()
    df["win"] = ((df["ts"] - t0) // pd.Timedelta(window)).astype(int)
    n_win = int(df["win"].max()) + 1

    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((n_win, len(vocab) + 4), dtype=np.float32)

    for w, g in df.groupby("win"):
        total = len(g)
        for tpl, cnt in g["template"].value_counts().items():
            if tpl in idx:
                X[w, idx[tpl]] = cnt
            else:
                X[w, len(vocab)] += cnt          # unknown_count
        X[w, len(vocab) + 1] = total              # total_count
        X[w, len(vocab) + 2] = (g["level"] == "ERROR").mean()
        X[w, len(vocab) + 3] = (g["level"] == "WARN").mean()
    return X


def make_labels(n_win, intervals):
    """برچسب واقعی هر پنجره (۱ = ناهنجار) برای ارزیابی."""
    y = np.zeros(n_win, dtype=int)
    for a, b, _ in intervals:
        y[a:b] = 1
    return y
