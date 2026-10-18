"""پارس کردن لاگ و استخراج الگو (template) از هر خط."""
import re

import pandas as pd

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} "
    r"(?P<level>[A-Z]+) \[(?P<service>[\w-]+)\] (?P<msg>.*)$"
)

# ترتیب مهم است: ابتدا IP، سپس hex، سپس هر عدد (حتی چسبیده به واحد مثل 120ms)
MASKS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\d+"), "<NUM>"),
]


def to_template(message):
    """اعداد و IP را با placeholder جایگزین می‌کند تا الگوی ثابت به‌دست آید."""
    for pattern, token in MASKS:
        message = pattern.sub(token, message)
    return message


def parse_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue  # خطوط خراب نادیده گرفته می‌شوند
            d = m.groupdict()
            d["template"] = f"{d['service']}|{d['level']}|{to_template(d['msg'])}"
            rows.append(d)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df
