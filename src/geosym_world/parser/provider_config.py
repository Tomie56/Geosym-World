"""Local configuration for OpenAI-compatible vision providers."""

from __future__ import annotations

import os
import re
from pathlib import Path


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_project_env(path: Path | None = None) -> Path | None:
    env_path = path or Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return None
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at {env_path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(f"Invalid .env key at {env_path}:{line_number}: {key!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return env_path


def provider_base_url() -> str:
    value = os.environ.get("GEOSYM_OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip()
    if not value:
        raise RuntimeError("GEOSYM_OPENROUTER_BASE_URL must not be empty")
    return value.rstrip("/")


def chat_completions_url() -> str:
    return provider_base_url() + "/chat/completions"
