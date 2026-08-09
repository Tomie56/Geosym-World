"""Lightweight model-output parser.

This parser is intentionally conservative. It supports JSON outputs and a small
set of natural-language answer patterns. Rich CoT parsing can be added later.
"""

from __future__ import annotations

from typing import Any, Dict
import json
import re


class ModelOutputParser:
    def parse(self, output: Any) -> Dict[str, Any]:
        if isinstance(output, dict):
            return output
        text = str(output or "").strip()
        parsed = self._try_json(text)
        if parsed is not None:
            return parsed
        return {
            "answer": self._extract_answer(text),
            "referenced_objects": self._extract_objects(text),
            "claims": [],
            "raw_text": text,
        }

    def _try_json(self, text: str) -> Dict[str, Any] | None:
        try:
            data = json.loads(text)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _extract_answer(self, text: str) -> str:
        boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            return boxed.group(1).strip()
        patterns = [
            r"(?:answer|ans|therefore|so)\s*(?:is|=|:)\s*([A-Za-z0-9_+\-*/^().\\{} ]+)",
            r"=\s*([A-Za-z0-9_+\-*/^().\\{} ]+)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".")
        return text

    def _extract_objects(self, text: str) -> list[str]:
        candidates = set(re.findall(r"\b[A-Z]{1,3}\b", text))
        return sorted(candidates)
