"""LLM judge soft correctness signal utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request


DEFAULT_OPENROUTER_JUDGE_MODEL = os.environ.get(
    "GEOSYM_OPENROUTER_JUDGE_MODEL",
    "qwen/qwen3-vl-32b-instruct",
)


@dataclass
class LLMJudgeSignal:
    used: bool = False
    pass_signal: str = "not_used"
    confidence: float | None = None
    blocking_recommendation: bool = False
    checks: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used": self.used,
            "pass_signal": self.pass_signal,
            "confidence": self.confidence,
            "blocking_recommendation": self.blocking_recommendation,
            "checks": self.checks,
            "reason": self.reason,
        }


def merge_hard_and_llm_signals(hard_valid: bool, llm_signal: LLMJudgeSignal | Dict[str, Any] | None) -> str:
    if llm_signal is None:
        return "auto_accept" if hard_valid else "reject_or_repair"
    if isinstance(llm_signal, dict):
        signal = llm_signal.get("pass_signal", "not_used")
    else:
        signal = llm_signal.pass_signal
    if signal in {None, "", "not_used"}:
        return "auto_accept" if hard_valid else "reject_or_repair"
    if hard_valid and signal == "pass":
        return "auto_accept"
    if hard_valid and signal in {"needs_review", "fail"}:
        return "human_review"
    if not hard_valid and signal == "pass":
        return "human_review"
    return "reject_or_repair"


class OpenRouterLLMJudge:
    """Optional OpenRouter judge.

    This is a soft correctness signal only. It does not replace hard verifier
    checks and should be used as a triage aid.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        site_url: str = "https://geosym.local",
        app_name: str = "GeoSym-Hard",
    ):
        self.model = model or DEFAULT_OPENROUTER_JUDGE_MODEL
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name

    def judge(
        self,
        *,
        image_path: str | None,
        problem_text: str,
        original_answer: Any,
        canonical_state: Dict[str, Any],
        verification_report: Dict[str, Any],
    ) -> LLMJudgeSignal:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouterLLMJudge")
        content: list[dict[str, Any]] = [{"type": "text", "text": self._prompt(
            problem_text=problem_text,
            original_answer=original_answer,
            canonical_state=canonical_state,
            verification_report=verification_report,
        )}]
        if image_path:
            path = Path(image_path)
            if path.exists():
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(path)},
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
            raise RuntimeError(f"OpenRouter judge HTTP {exc.code}: {body[:1000]}") from exc
        response = data["choices"][0]["message"]["content"]
        parsed = self._parse_json_response(response)
        return LLMJudgeSignal(
            used=True,
            pass_signal=parsed.get("pass_signal", "needs_review"),
            confidence=parsed.get("confidence"),
            blocking_recommendation=bool(parsed.get("blocking_recommendation", False)),
            checks=dict(parsed.get("checks") or {}),
            reason=str(parsed.get("reason", "")),
        )

    def _prompt(
        self,
        *,
        problem_text: str,
        original_answer: Any,
        canonical_state: Dict[str, Any],
        verification_report: Dict[str, Any],
    ) -> str:
        return f"""
You are a soft judge for GeoSym real-geometry reconstruction.

Return JSON only:
{{
  "pass_signal": "pass|needs_review|fail",
  "confidence": 0.0,
  "blocking_recommendation": false,
  "checks": {{
    "image_matches_canonical": "pass|needs_review|fail",
    "answer_consistency": "pass|needs_review|fail",
    "relation_completeness": "pass|needs_review|fail"
  }},
  "reason": ""
}}

Judge whether the canonical GeoSym reconstruction is plausibly correct for the
original problem image/text/answer. This is only a soft signal; prefer
needs_review when uncertain.

Problem text:
{problem_text}

Original answer:
{original_answer}

Canonical state JSON:
{json.dumps(canonical_state, ensure_ascii=False)[:12000]}

Hard verification report:
{json.dumps(verification_report, ensure_ascii=False)[:4000]}
""".strip()

    def _image_data_url(self, path: Path) -> str:
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
            raise ValueError(f"OpenRouter judge response did not contain JSON: {text[:500]}")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("OpenRouter judge response JSON is not an object")
        return data
