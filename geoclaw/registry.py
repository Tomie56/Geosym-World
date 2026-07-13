"""Tool registry for GeoClaw environments."""

from __future__ import annotations

from typing import Iterable

from .tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools = {tool.name: tool for tool in tools}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def restricted(self, allowed: Iterable[str]) -> "ToolRegistry":
        allowed_set = set(allowed)
        return ToolRegistry(tool for name, tool in self._tools.items() if name in allowed_set)
