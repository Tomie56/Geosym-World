"""Relation extractor interface.

The implementation supports manual JSON, mock outputs, and an OpenRouter VLM
backend controlled by OPENROUTER_API_KEY.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request


DEFAULT_OPENROUTER_MODEL = os.environ.get("GEOSYM_OPENROUTER_MODEL", "qwen/qwen3.7-plus")


class RelationExtractor:
    def __init__(
        self,
        backend: str = "manual_json",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        site_url: str = "https://geosym.local",
        app_name: str = "GeoSym-Hard",
    ):
        self.backend = backend
        self.model = model or DEFAULT_OPENROUTER_MODEL
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name

    def extract(
        self,
        image_path: Optional[str] = None,
        problem_text: str = "",
        solution_text: Optional[str] = None,
        manual_json: Optional[Dict[str, Any] | str] = None,
    ) -> Dict[str, Any]:
        if self.backend == "manual_json":
            if manual_json is None:
                raise ValueError("manual_json backend requires manual_json input")
            if isinstance(manual_json, str):
                return json.loads(manual_json)
            return manual_json
        if self.backend == "mock":
            return {
                "objects": ["Point A", "Point B", "Point C", "Segment AB", "Segment BC", "Segment AC"],
                "relations_text": [
                    "AB is perpendicular to BC",
                    "AB has length 3",
                    "BC has length 4",
                    "Find AC",
                ],
                "metadata": {
                    "image_path": image_path,
                    "problem_text": problem_text,
                    "solution_text": solution_text,
                },
            }
        if self.backend == "openrouter":
            return self._extract_openrouter(
                image_path=image_path,
                problem_text=problem_text,
                solution_text=solution_text,
            )
        raise NotImplementedError(f"Unsupported relation extractor backend: {self.backend}")

    def _extract_openrouter(
        self,
        image_path: Optional[str],
        problem_text: str,
        solution_text: Optional[str],
    ) -> Dict[str, Any]:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for backend='openrouter'")
        prompt = self._prompt(problem_text=problem_text, solution_text=solution_text)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_data_url(image_path)},
            })
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:1000]}") from exc
        message = data["choices"][0]["message"]["content"]
        return self._parse_json_response(message)

    def _prompt(self, problem_text: str, solution_text: Optional[str]) -> str:
        solution_block = f"\nKnown solution/answer:\n{solution_text}\n" if solution_text else ""
        return f"""
You are extracting geometry relations for GeoSym.

Return JSON only. Do not include markdown.

Schema:
{{
  "objects": ["Point A", "Segment AB", "Circle O"],
  "relations_text": [
    "AB is perpendicular to BC",
    "AB has length 3",
    "Find AC"
  ],
  "uncertain_relations_text": [],
  "metadata": {{
    "notes": ""
  }}
}}

Rules:
- Extract visible and textual geometry facts: points, segments, circles, arcs, shaded regions, labels, lengths, angles, perpendicular, parallel, equal length, collinear, tangent, midpoint, target.
- Do not infer pixel coordinates as mathematical coordinates.
- If uncertain, put the statement in uncertain_relations_text.
- Prefer concise relation sentences that can be normalized into constraints.

Problem text:
{problem_text}
{solution_block}
""".strip()

    def _image_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"OpenRouter response did not contain JSON: {text[:500]}")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("OpenRouter response JSON is not an object")
        return data
