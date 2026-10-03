"""Run one prepared JSONL dataset through Jev or Clef; resume by row ID."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
QUESTIONS = {
    "sentiment": "Does this text express positive rather than negative sentiment?",
    "sms_spam": "Is this text message spam rather than a legitimate personal message?",
}


def settings():
    values = dict(os.environ)
    path = ROOT / ".env.local"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if not values.get(key.strip()):
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def request_config(provider, values):
    if provider == "clef":
        account_id = values.get("CLOUDFLARE_ACCOUNT_ID", "")
        token = values.get("CLOUDFLARE_AUTH_TOKEN", "")
        if not account_id or not token:
            raise ValueError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN in .env.local")
        return (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef",
            token,
            "clef",
        )
    token = values.get("TYPESAFE_API_KEY", "")
    if not token:
        raise ValueError("Set TYPESAFE_API_KEY in .env.local")
    return "https://api.typesafe.ai/v1/systemone", token, values.get("JEV_MODEL") or "jev-latest"


def send(provider, endpoint, token, model, row):
    question = QUESTIONS[row["dataset"]]
    payload = {
        "model": model,
        "state": row["text"],
        "questions": {"target": {"type": "noul", "instructions": question}},
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    start = perf_counter()
    with urlopen(request, timeout=60) as response:
        body = json.load(response)
    elapsed_ms = round((perf_counter() - start) * 1000, 1)
    if provider == "clef":
        if not body.get("success"):
            raise ValueError("Cloudflare returned an unsuccessful response")
        result = body["result"]
    else:
        result = body
    probability = float(result["answers"]["target"]["noul"])
    if not 0 <= probability <= 1:
        raise ValueError("Provider returned an invalid probability")
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "provider": provider,
        "model": result.get("model", model),
        "requested_model": model,
        "question": question,
        "probability": probability,
        "elapsed_ms": elapsed_ms,
        "usage": result.get("usage", {}),
        "answer": result["answers"]["target"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("clef", "jev"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset must have unique IDs and at least one row")
    if len({row["dataset"] for row in rows}) != 1:
        raise ValueError("Run one dataset at a time")
    endpoint, token, model = request_config(args.provider, settings())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous = json.loads(line)
                if previous.get("provider") != args.provider or previous.get("requested_model") != model:
                    raise ValueError("Output file already contains a different provider or model")
                completed.add(previous["id"])
    for index, row in enumerate(rows, start=1):
        if row["id"] in completed:
            continue
        try:
            result = send(args.provider, endpoint, token, model, row)
        except HTTPError as error:
            print(f"Stopped at row {index}: HTTP {error.code}. Previous results are saved.", file=sys.stderr)
            return 1
        except (URLError, TimeoutError, ValueError, KeyError) as error:
            print(f"Stopped at row {index}: {type(error).__name__}. Previous results are saved.", file=sys.stderr)
            return 1
        with args.output.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{args.provider}: {index}/{len(rows)}", end="\r", flush=True)
    print(f"{args.provider}: complete ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
