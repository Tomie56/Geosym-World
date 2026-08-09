"""Record readers for raw real-geometry datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict, Any
import json


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
