from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal


RetrySafety = Literal["safe", "verify", "never"]
RiskLevel = Literal["read", "write", "destructive"]


@dataclass(frozen=True)
class ToolContext:
    work_thread_id: int
    work_run_id: int
    owner_agent_id: int
    execution_token: int


ToolHandler = Callable[[ToolContext, dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    risk_level: RiskLevel = "read"
    retry_safety: RetrySafety = "safe"
    requires_approval: bool = False

    def prompt_description(self, permission: str) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "risk_level": self.risk_level,
            "retry_safety": self.retry_safety,
            "permission": permission,
            "requires_approval": self.requires_approval,
        }

    def compact_prompt_description(self, permission: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        if permission != "allow":
            result["permission"] = permission
        if self.risk_level != "read":
            result["risk_level"] = self.risk_level
        if self.retry_safety != "safe":
            result["retry_safety"] = self.retry_safety
        if self.requires_approval:
            result["requires_approval"] = True
        return result


class ToolRegistryError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ToolRegistryError("工具名称不能为空")
        if name in self._tools:
            raise ToolRegistryError(f"工具已注册：{name}")
        self._tools[name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError(f"工具不存在：{name}") from exc

    def list(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools)]


def current_time_tool(_: ToolContext, arguments: dict[str, Any]) -> dict[str, str]:
    if arguments:
        raise ToolRegistryError("current_time不接受参数")
    return {"utc": datetime.now(timezone.utc).isoformat()}
