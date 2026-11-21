from __future__ import annotations

import threading
from pathlib import Path
from uuid import uuid4


class EnvStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def create_secret(self, value: str) -> str | None:
        value = value.strip()
        if not value:
            return None
        name = f"MODEL_API_KEY_{uuid4().hex.upper()}"
        self.set(name, value)
        return name

    def set(self, name: str, value: str) -> None:
        if "\n" in value or "\r" in value:
            raise ValueError("Secret value cannot contain line breaks")
        with self._lock:
            values, comments = self._read()
            values[name] = value
            self._write(values, comments)

    def remove(self, name: str | None) -> None:
        if not name:
            return
        with self._lock:
            values, comments = self._read()
            values.pop(name, None)
            self._write(values, comments)

    def get(self, name: str | None) -> str | None:
        if not name:
            return None
        values, _ = self._read()
        return values.get(name)

    def _read(self) -> tuple[dict[str, str], list[str]]:
        values: dict[str, str] = {}
        comments: list[str] = []
        if not self.path.exists():
            return values, comments
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in raw_line:
                comments.append(raw_line)
                continue
            name, value = raw_line.split("=", 1)
            values[name.strip()] = value
        return values, comments

    def _write(self, values: dict[str, str], comments: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = comments + [f"{name}={value}" for name, value in sorted(values.items())]
        content = "\n".join(lines).rstrip() + "\n"
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(self.path)
