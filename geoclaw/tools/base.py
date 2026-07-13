"""Base interfaces for GeoClaw tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class ToolResult:
    status: str
    result: Dict[str, Any] = field(default_factory=dict)
    evidence: list[Dict[str, Any]] = field(default_factory=list)
    errors: list[Dict[str, Any]] = field(default_factory=list)
    state_changed: bool = False
    image_changed: bool = False
    audit: Dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str

    def execute(self, environment: Any, arguments: Dict[str, Any]) -> ToolResult:
        ...
