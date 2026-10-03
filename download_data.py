"""Download the two UCI ZIP files used for the original samples.

The files are saved under ignored data/raw/. SHA-256 checks keep the sampled
rows reproducible if UCI updates an archive in the future.
"""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
ARCHIVES = {
    "sms_spam_collection.zip": (
        "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip",
        "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
    ),
    "sentiment_labelled_sentences.zip": (
        "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip",
        "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343",
    ),
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ARCHIVES.items():
        path = RAW / name
        if path.exists():
            content = path.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                content = response.read()
        actual = sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        if not path.exists():
            path.write_bytes(content)
        print(f"Verified {name}: {actual}")


if __name__ == "__main__":
    main()
