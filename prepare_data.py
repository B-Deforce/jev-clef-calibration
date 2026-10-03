"""Prepare stable, deduplicated JSONL datasets from UCI's original ZIP files."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile


ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SEED = 20261002


def parse_labeled_lines(text: str, labels: dict[str, int]):
    """A labeled row ends with TAB + label; internal newlines stay in its text."""
    pending = []
    for line in text.splitlines():
        if "\t" in line:
            body, label = line.rsplit("\t", 1)
            if label in labels:
                pending.append(body)
                yield "\n".join(pending), labels[label]
                pending = []
                continue
        pending.append(line)
    if pending:
        raise ValueError("Source ended with an unlabeled text fragment")


def deduplicate(rows):
    unique = {}
    duplicates = 0
    for row in rows:
        digest = sha256(row["text"].strip().casefold().encode("utf-8")).hexdigest()
        if digest in unique:
            if unique[digest]["label"] != row["label"]:
                raise ValueError(f"Conflicting labels for text hash {digest}")
            duplicates += 1
            continue
        row["id"] = digest[:20]
        unique[digest] = row
    return list(unique.values()), duplicates


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_sms():
    path = RAW / "sms_spam_collection.zip"
    with ZipFile(path) as archive:
        source = archive.read("SMSSpamCollection").decode("utf-8", errors="replace")
    rows = []
    for index, line in enumerate(source.splitlines(), start=1):
        label_text, text = line.split("\t", 1)
        rows.append(
            {
                "dataset": "sms_spam",
                "source_row": index,
                "text": text,
                "label": {"ham": 0, "spam": 1}[label_text],
            }
        )
    return path, rows


def prepare_sms_redacted():
    path, rows = prepare_sms()
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_pattern = re.compile(r"\b\+?\d[\d\s().-]{6,}\d\b")
    for row in rows:
        row["text"] = phone_pattern.sub("[PHONE]", email_pattern.sub("[EMAIL]", row["text"]))
    return path, rows


def prepare_sentiment():
    path = RAW / "sentiment_labelled_sentences.zip"
    rows = []
    with ZipFile(path) as archive:
        for source_name in ("amazon_cells", "imdb", "yelp"):
            name = f"sentiment labelled sentences/{source_name}_labelled.txt"
            source = archive.read(name).decode("utf-8", errors="replace")
            for index, (text, label) in enumerate(
                parse_labeled_lines(source, {"0": 0, "1": 1}), start=1
            ):
                rows.append(
                    {
                        "dataset": "sentiment",
                        "source": source_name,
                        "source_row": index,
                        "text": text,
                        "label": label,
                    }
                )
    return path, rows


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "label_meaning": {
            "sms_spam": "spam",
            "sms_spam_redacted": "spam",
            "sentiment": "positive",
        },
    }
    for name, producer in (
        ("sms_spam", prepare_sms),
        ("sms_spam_redacted", prepare_sms_redacted),
        ("sentiment", prepare_sentiment),
    ):
        path, source_rows = producer()
        rows, duplicates = deduplicate(source_rows)
        rows.sort(key=lambda row: row["id"])
        write_jsonl(PROCESSED / f"{name}.jsonl", rows)
        pilot = random.Random(SEED).sample(rows, min(100, len(rows)))
        pilot.sort(key=lambda row: row["id"])
        write_jsonl(PROCESSED / f"{name}_pilot100.jsonl", pilot)
        pilot500 = random.Random(SEED + 1).sample(rows, min(500, len(rows)))
        pilot500.sort(key=lambda row: row["id"])
        write_jsonl(PROCESSED / f"{name}_pilot500.jsonl", pilot500)
        manifest[name] = {
            "source_url": {
                "sms_spam": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
                "sms_spam_redacted": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
                "sentiment": "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
            }[name],
            "source_zip_sha256": sha256(path.read_bytes()).hexdigest(),
            "source_rows": len(source_rows),
            "unique_rows": len(rows),
            "duplicates_removed": duplicates,
            "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
            "pilot_rows": len(pilot),
            "pilot500_rows": len(pilot500),
        }
    (PROCESSED / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
