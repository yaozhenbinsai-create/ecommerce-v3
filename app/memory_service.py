from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

from sqlalchemy import or_, select

from .database import Database
from .models import MemoryEntry, MemoryOperationLog


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LocalVectorizer:
    """Dependency-free local vectors for the first SQLite memory implementation."""

    backend = "local_hash_v1"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        tokens = re.findall(r"[\u3400-\u9fff]|[a-z0-9_./:-]+", normalized)
        features = tokens + [
            normalized[index : index + 2]
            for index in range(max(0, len(normalized) - 1))
            if not normalized[index : index + 2].isspace()
        ]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            vector[index] += -1.0 if value & 1 else 1.0
        length = math.sqrt(sum(value * value for value in vector))
        if length:
            vector = [value / length for value in vector]
        return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


class MemoryService:
    def __init__(self, database: Database, settings: dict[str, Any]) -> None:
        self.database = database
        self.settings = settings

    def _config(self) -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "vector_dimensions": 256,
            "top_k": 5,
            "minimum_score": 0.18,
        }
        defaults.update(self.settings.get("memory") or {})
        return defaults

    def _vectorizer(self) -> LocalVectorizer:
        return LocalVectorizer(int(self._config()["vector_dimensions"]))

    def search(
        self,
        query: str,
        *,
        agent_id: int | None = None,
        include_all_agents: bool = False,
        include_deleted: bool = False,
        top_k: int | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        config = self._config()
        if not config["enabled"] or not query.strip():
            return []
        vectorizer = self._vectorizer()
        query_vector = vectorizer.embed(query)
        with self.database.sessions() as session:
            statement = select(MemoryEntry)
            if not include_deleted:
                statement = statement.where(MemoryEntry.status == "active")
            if not include_all_agents:
                statement = statement.where(
                    or_(MemoryEntry.agent_id == agent_id, MemoryEntry.agent_id.is_(None))
                )
            entries = session.scalars(statement).all()
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in entries:
            try:
                vector = json.loads(entry.vector_json)
            except (TypeError, ValueError):
                continue
            score = cosine_similarity(query_vector, vector)
            if score >= float(config["minimum_score"]):
                scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: int(top_k or config["top_k"])]

    def add(
        self,
        agent_id: int | None,
        content: str,
        *,
        source_message_id: int | None = None,
        reason: str | None = None,
    ) -> MemoryEntry:
        vectorizer = self._vectorizer()
        with self.database.sessions() as session:
            entry = MemoryEntry(
                agent_id=agent_id,
                content=content.strip(),
                vector_json=json.dumps(vectorizer.embed(content)),
                vector_backend=vectorizer.backend,
                source_message_id=source_message_id,
            )
            session.add(entry)
            session.flush()
            session.add(
                MemoryOperationLog(
                    memory_id=entry.id,
                    agent_id=agent_id,
                    operation="add",
                    after_content=entry.content,
                    reason=reason,
                )
            )
            session.commit()
            session.refresh(entry)
            return entry

    def update(self, memory_id: int, content: str, reason: str | None = None) -> MemoryEntry:
        vectorizer = self._vectorizer()
        with self.database.sessions() as session:
            entry = session.get(MemoryEntry, memory_id)
            if not entry:
                raise LookupError("记忆不存在")
            before = entry.content
            entry.content = content.strip()
            entry.vector_json = json.dumps(vectorizer.embed(content))
            entry.vector_backend = vectorizer.backend
            entry.status = "active"
            entry.deleted_at = None
            session.add(
                MemoryOperationLog(
                    memory_id=entry.id,
                    agent_id=entry.agent_id,
                    operation="update",
                    before_content=before,
                    after_content=entry.content,
                    reason=reason,
                )
            )
            session.commit()
            session.refresh(entry)
            return entry

    def soft_delete(self, memory_id: int, reason: str | None = None) -> MemoryEntry:
        with self.database.sessions() as session:
            entry = session.get(MemoryEntry, memory_id)
            if not entry:
                raise LookupError("记忆不存在")
            entry.status = "deleted"
            entry.deleted_at = _utc_now()
            session.add(
                MemoryOperationLog(
                    memory_id=entry.id,
                    agent_id=entry.agent_id,
                    operation="delete",
                    before_content=entry.content,
                    reason=reason,
                )
            )
            session.commit()
            session.refresh(entry)
            return entry

    def restore(self, memory_id: int, reason: str | None = None) -> MemoryEntry:
        with self.database.sessions() as session:
            entry = session.get(MemoryEntry, memory_id)
            if not entry:
                raise LookupError("记忆不存在")
            entry.status = "active"
            entry.deleted_at = None
            session.add(
                MemoryOperationLog(
                    memory_id=entry.id,
                    agent_id=entry.agent_id,
                    operation="restore",
                    after_content=entry.content,
                    reason=reason,
                )
            )
            session.commit()
            session.refresh(entry)
            return entry

