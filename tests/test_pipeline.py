"""تست‌های سریع برای اطمینان از سالم بودن پایپ‌لاین."""
import numpy as np

from src.features import build_vocab, make_labels, window_features
from src.generate_logs import generate, save
from src.models import AutoEncoderDetector, IForestDetector
from src.parse_logs import parse_file, to_template


def test_template_masks_numbers_and_ips():
    msg = "Request GET /users/42 completed in 120ms from 192.168.1.5"
    expected = "Request GET /users/<NUM> completed in <NUM>ms from <IP>"
    assert to_template(msg) == expected


def test_parse_roundtrip(tmp_path):
    text, _, _ = generate(minutes=20, n_anomalies=1, anomaly_start_ratio=0.5)
    path = tmp_path / "server.log"
    save(path, text)
    df = parse_file(path)
    assert len(df) == len(text)
    # تعداد الگوهای یکتا باید کم باشد (ماسک اعداد کار می‌کند)
    assert df["template"].nunique() < 30


def test_end_to_end_detects_injected_anomalies(tmp_path):
    text, intervals, t0 = generate(minutes=120, n_anomalies=4)
    path = tmp_path / "server.log"
    save(path, text)
    df = parse_file(path)

    split = t0 + (df["ts"].max() - t0) / 2
    vocab = build_vocab(df[df["ts"] < split])
    X = window_features(df, vocab, t0)
    n_train = int((split - t0) / np.timedelta64(1, "m"))
    y = make_labels(len(X), intervals)[n_train:]

    scores = AutoEncoderDetector().fit(X[:n_train]).score(X[n_train:])
    # میانگین امتیاز پنجره‌های ناهنجار باید بیشتر از عادی باشد
    assert scores[y == 1].mean() > scores[y == 0].mean()

    iso = IForestDetector().fit(X[:n_train]).score(X[n_train:])
    assert iso.shape == scores.shape
