"""تولید لاگ سرور مصنوعی همراه با ناهنجاری‌های تزریق‌شده."""
import random
from datetime import datetime, timedelta

import numpy as np

SERVICES = ["api", "auth", "db", "cache"]

NORMAL_TEMPLATES = {
    "api": [
        ("INFO", "Request GET /users/{n} completed in {ms}ms status=200"),
        ("INFO", "Request POST /orders completed in {ms}ms status=201"),
        ("WARN", "Slow request GET /reports took {ms}ms"),
    ],
    "auth": [
        ("INFO", "User {n} logged in from {ip}"),
        ("INFO", "Token refreshed for user {n}"),
        ("WARN", "Failed login for user {n} from {ip}"),
    ],
    "db": [
        ("INFO", "Query executed in {ms}ms rows={n}"),
        ("INFO", "Connection pool size={n}"),
    ],
    "cache": [
        ("INFO", "Cache hit key=item:{n}"),
        ("INFO", "Cache miss key=item:{n}"),
    ],
}
# وزن انتخاب هر الگو در حالت عادی (WARN نادر است)
NORMAL_WEIGHTS = {
    "api": [0.60, 0.30, 0.10],
    "auth": [0.55, 0.40, 0.05],
    "db": [0.8, 0.2],
    "cache": [0.7, 0.3],
}


def _fill(template, rng):
    """پر کردن جای‌نگهدارها با مقادیر تصادفی."""
    return template.format(
        n=rng.randint(1, 5000),
        ms=rng.randint(5, 400),
        ip=f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
    )


def _normal_line(ts, rng):
    service = rng.choice(SERVICES)
    idx = rng.choices(range(len(NORMAL_TEMPLATES[service])), NORMAL_WEIGHTS[service])[0]
    level, tpl = NORMAL_TEMPLATES[service][idx]
    return ts, level, service, _fill(tpl, rng)


def _anomaly_line(kind, ts, rng):
    """سه نوع ناهنجاری: موج خطا، حمله brute-force، الگوی ناشناخته."""
    if kind == "error_burst":
        msg = f"Connection timeout to db-10.0.0.{rng.randint(1, 9)}:5432 after {rng.randint(3000, 9000)}ms"
        return ts, "ERROR", "db", msg
    if kind == "brute_force":
        msg = f"Failed login for user admin from 10.66.{rng.randint(0, 5)}.{rng.randint(1, 254)}"
        return ts, "WARN", "auth", msg
    msg = f"Out of memory: killed process {rng.randint(1000, 9999)} (worker)"
    return ts, "ERROR", "api", msg


def generate(minutes=600, seed=42, anomaly_start_ratio=0.5, n_anomalies=12):
    """لاگ‌ها را تولید می‌کند و (خطوط، بازه‌های ناهنجاری) را برمی‌گرداند.

    ناهنجاری‌ها فقط در نیمه‌ی دوم زمان تزریق می‌شوند تا نیمه‌ی اول
    به‌عنوان دوره‌ی پایه‌ی (baseline) عادی برای آموزش استفاده شود.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    start = datetime(2026, 10, 1, 0, 0, 0)

    # تعیین بازه‌های ناهنجاری (هرکدام ۲ تا ۴ دقیقه)
    kinds = ["error_burst", "brute_force", "unknown_pattern"]
    first_minute = int(minutes * anomaly_start_ratio)
    slots = np_rng.choice(range(first_minute, minutes - 5, 8), size=n_anomalies, replace=False)
    intervals = []
    for s in sorted(slots):
        dur = int(rng.randint(2, 4))
        intervals.append((int(s), int(s) + dur, rng.choice(kinds)))

    lines = []
    for minute in range(minutes):
        active = next((k for a, b, k in intervals if a <= minute < b), None)
        # تعداد رویداد عادی در دقیقه (با نوسان روزانه‌ی ملایم)
        base_rate = 120 + 30 * np.sin(2 * np.pi * minute / 240)
        n_events = int(np_rng.poisson(base_rate))
        for _ in range(n_events):
            ts = start + timedelta(minutes=minute, seconds=rng.random() * 60)
            lines.append(_normal_line(ts, rng))
        if active:
            for _ in range(int(np_rng.integers(80, 160))):
                ts = start + timedelta(minutes=minute, seconds=rng.random() * 60)
                lines.append(_anomaly_line(active, ts, rng))

    lines.sort(key=lambda x: x[0])
    text = [
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{int(ts.microsecond/1000):03d} {lvl} [{svc}] {msg}"
        for ts, lvl, svc, msg in lines
    ]
    return text, intervals, start


def save(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
