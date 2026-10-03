"""One synthetic Clef smoke test. Reads local credentials without displaying them."""

import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def local_settings() -> dict[str, str]:
    settings = dict(os.environ)
    path = Path(__file__).with_name(".env.local")
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            settings.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    return settings


def main() -> int:
    settings = local_settings()
    account_id = settings.get("CLOUDFLARE_ACCOUNT_ID", "")
    token = settings.get("CLOUDFLARE_AUTH_TOKEN", "")
    if not account_id or not token:
        print("Add CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN to .env.local first.")
        return 2

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/@cf/cloudflare/clef"
    )
    payload = {
        "model": "clef",
        "state": "Congratulations! You won a cash prize. Text CLAIM to 55555 now.",
        "questions": {
            "spam": {
                "type": "noul",
                "instructions": "Is this text message spam rather than a legitimate personal message?",
            }
        },
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = json.load(response)
    except HTTPError as error:
        print(f"Cloudflare returned HTTP {error.code}.", file=sys.stderr)
        return 1
    except URLError as error:
        print(f"Cloudflare request failed: {error.reason}", file=sys.stderr)
        return 1

    if not body.get("success"):
        print("Cloudflare returned an unsuccessful response.", file=sys.stderr)
        return 1
    result = body.get("result", {})
    answer = result.get("answers", {}).get("spam", {})
    probability = answer.get("noul")
    if probability is None:
        print("Clef response had no spam probability.", file=sys.stderr)
        return 1
    print(f"model: {result.get('model', 'unknown')}")
    print(f"P(spam): {probability}")
    print(f"usage: {json.dumps(result.get('usage', {}), sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
