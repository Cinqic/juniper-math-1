"""Tool registry: the closed set of tool names this runtime recognizes.

No tool is ever dispatched by dynamically importing or reflecting on a
model-supplied name — every handler here is a statically wired Python
callable, and an unknown name never reaches import machinery. See
docs/TOOLS.md "No dynamic dispatch".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from juniper_math.tools.config import ToolsConfig


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    handler: Callable[..., dict]
    available: bool = True


class ToolRegistry:
    def __init__(self, config: ToolsConfig) -> None:
        self._config = config
        self._registrations: dict[str, ToolRegistration] = {}

    def register(self, name: str, handler: Callable[..., dict], *, available: bool = True) -> None:
        if name not in self._config.tools:
            raise ValueError(f"{name!r} is not an approved tool in config/tools.yaml")
        self._registrations[name] = ToolRegistration(name=name, handler=handler, available=available)

    def set_available(self, name: str, available: bool) -> None:
        if name not in self._registrations:
            raise KeyError(f"Tool not registered: {name!r}")
        existing = self._registrations[name]
        self._registrations[name] = ToolRegistration(existing.name, existing.handler, available)

    def is_known(self, name: str) -> bool:
        return name in self._config.tools

    def get(self, name: str) -> ToolRegistration | None:
        return self._registrations.get(name)
