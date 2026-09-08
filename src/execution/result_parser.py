from __future__ import annotations

import json
from typing import Any


def parse_summary_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("SUMMARY_JSON:"):
            payload = line.split("SUMMARY_JSON:", 1)[1].strip()
            return json.loads(payload)
    return {}

