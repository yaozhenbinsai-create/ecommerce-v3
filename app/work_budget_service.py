from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
from typing import Any

from sqlalchemy import func, select

from .database import Database
from .models import ToolCall, WorkModelCall, WorkThread


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def estimate_tokens(value: str) -> int:
    ascii_count = sum(1 for char in value if ord(char) < 128)
    other_count = len(value) - ascii_count
    return max(1, math.ceil(ascii_count / 4 + other_count / 1.5))


@dataclass
class ModelUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    source: str = "estimated"


def derive_model_circuit_states(
    calls: list[Any],
    model_ids: list[int],
    *,
    failure_threshold: int = 3,
    recovery_decisions: int = 5,
    successful_probes_required: int = 2,
    probe_cooldown_seconds: int = 300,
    now: datetime | None = None,
    reset_at_by_model: dict[int, datetime] | None = None,
) -> dict[int, dict[str, Any]]:
    """Rebuild event-driven circuit state from the persistent model-call ledger."""
    states = {
        model_id: {
            "state": "healthy",
            "consecutive_transient_failures": 0,
            "recovery_successes": 0,
            "successful_probes": 0,
            "last_error": None,
            "last_failure_at": None,
        }
        for model_id in model_ids
    }
    resets = reset_at_by_model or {}
    for call in calls:
        model_id = getattr(call, "model_id", None)
        if model_id not in states:
            continue
        reset_at = resets.get(model_id)
        call_at = getattr(call, "finished_at", None) or getattr(
            call, "started_at", None
        )
        if reset_at is not None and call_at is not None:
            if reset_at.tzinfo is None:
                reset_at = reset_at.replace(tzinfo=timezone.utc)
            if call_at.tzinfo is None:
                call_at = call_at.replace(tzinfo=timezone.utc)
            if call_at <= reset_at:
                continue
        status = str(getattr(call, "status", ""))
        error_kind = str(getattr(call, "error_kind", "") or "")
        if status == "completed":
            for other_id, other in states.items():
                if other_id != model_id and other["state"] == "open":
                    other["recovery_successes"] += 1
            current = states[model_id]
            if current["state"] in {"open", "half_open"}:
                current["state"] = "half_open"
                current["successful_probes"] += 1
                if current["successful_probes"] >= successful_probes_required:
                    current.update(
                        state="healthy",
                        consecutive_transient_failures=0,
                        recovery_successes=0,
                        successful_probes=0,
                        last_error=None,
                    )
            else:
                current.update(
                    consecutive_transient_failures=0,
                    recovery_successes=0,
                    successful_probes=0,
                    last_error=None,
                )
            continue
        # Authentication failures can also be temporary (for example a
        # provider balance being replenished).  They must leave the normal
        # route after repeated failures, but remain eligible for a timed probe.
        if status != "failed" or error_kind not in {"transient", "auth"}:
            continue
        current = states[model_id]
        current["last_error"] = str(getattr(call, "error", "") or "")[:500]
        current["last_failure_at"] = call_at
        if current["state"] in {"open", "half_open"}:
            current.update(
                state="open",
                consecutive_transient_failures=failure_threshold,
                recovery_successes=0,
                successful_probes=0,
            )
            continue
        current["consecutive_transient_failures"] += 1
        if current["consecutive_transient_failures"] >= failure_threshold:
            current.update(
                state="open",
                recovery_successes=0,
                successful_probes=0,
            )
    current_time = now or utc_now()
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    for state in states.values():
        last_failure_at = state.get("last_failure_at")
        if last_failure_at is not None and last_failure_at.tzinfo is None:
            last_failure_at = last_failure_at.replace(tzinfo=timezone.utc)
        cooldown_elapsed = bool(
            last_failure_at is not None
            and (current_time - last_failure_at).total_seconds()
            >= max(1, int(probe_cooldown_seconds))
        )
        # A different provider succeeding says nothing about this provider's
        # recovery.  Only the timed cooldown (or an already successful probe)
        # may put an open provider back on the route.
        state["probe_ready"] = state["state"] == "half_open" or (
            state["state"] == "open" and cooldown_elapsed
        )
        state["attempt_limit"] = 1 if state["state"] in {"open", "half_open"} else 3
    return states


def select_model_routes(
    models: list[Any],
    circuit_states: dict[int, dict[str, Any]],
    *,
    attempts_per_model: int,
) -> tuple[list[Any], dict[int, int]]:
    """Choose work-model routes without hiding a recovered backup provider."""
    # Preserve the routing plan's configured priority across healthy models
    # and providers whose timed recovery probe is due.  In particular, an
    # open primary must get one probe before a healthy backup once cooldown
    # expires; otherwise the backup silently becomes the permanent primary.
    routed_models = [
        model
        for model in models
        if circuit_states[model.id]["state"] == "healthy"
        or (
            circuit_states[model.id]["state"] in {"open", "half_open"}
            and circuit_states[model.id]["probe_ready"]
        )
    ]
    if not any(
        circuit_states[model.id]["state"] == "healthy" for model in models
    ):
        routed_ids = {model.id for model in routed_models}
        routed_models.extend(model for model in models if model.id not in routed_ids)
    attempts_by_model = {
        model.id: (
            1
            if circuit_states[model.id]["state"] in {"open", "half_open"}
            else max(1, int(attempts_per_model))
        )
        for model in routed_models
    }
    return routed_models, attempts_by_model


class WorkBudgetService:
    def __init__(self, database: Database, settings: dict[str, Any] | None = None) -> None:
        self.database = database
        self.configure(settings or {})

    def configure(self, settings: dict[str, Any]) -> None:
        config = (settings or {}).get("work_budget") or {}
        self.max_model_tokens_per_tree = min(
            100_000_000, max(1_000, int(config.get("max_model_tokens_per_tree", 500_000)))
        )
        self.max_active_minutes_per_tree = min(
            10_080, max(1, int(config.get("max_active_minutes_per_tree", 240)))
        )

    def tree_thread_ids(self, session: Any, thread_id: int) -> list[int]:
        current = session.get(WorkThread, thread_id)
        if not current:
            return [thread_id]
        seen: set[int] = set()
        while current.parent_thread_id and current.id not in seen:
            seen.add(current.id)
            parent = session.get(WorkThread, current.parent_thread_id)
            if not parent:
                break
            current = parent
        tree_ids = [current.id]
        frontier = [current.id]
        while frontier:
            children = session.scalars(
                select(WorkThread.id).where(WorkThread.parent_thread_id.in_(frontier))
            ).all()
            new_ids = [child_id for child_id in children if child_id not in tree_ids]
            tree_ids.extend(new_ids)
            frontier = new_ids
        return tree_ids

    def usage(self, thread_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            tree_ids = self.tree_thread_ids(session, thread_id)
            model_rows = session.execute(
                select(
                    WorkModelCall.status,
                    WorkModelCall.usage_source,
                    WorkModelCall.input_tokens,
                    WorkModelCall.output_tokens,
                    WorkModelCall.total_tokens,
                    WorkModelCall.duration_ms,
                    WorkModelCall.prompt_hash,
                    WorkModelCall.id,
                    WorkModelCall.work_run_id,
                    WorkModelCall.attempt_number,
                )
                .where(WorkModelCall.work_thread_id.in_(tree_ids))
                .order_by(WorkModelCall.id.asc())
            ).all()
            tool_duration = int(
                session.scalar(
                    select(func.coalesce(func.sum(ToolCall.duration_ms), 0)).where(
                        ToolCall.work_thread_id.in_(tree_ids)
                    )
                ) or 0
            )
        attempted_input_tokens = sum(max(0, int(row[2] or 0)) for row in model_rows)
        attempted_output_tokens = sum(max(0, int(row[3] or 0)) for row in model_rows)
        attempted_tokens = sum(max(0, int(row[4] or 0)) for row in model_rows)
        # A successful response consumed inference even when the provider omitted
        # usage and we had to estimate it. A failed request only consumes the hard
        # budget when the provider explicitly reports usage. Local estimates for
        # network/protocol failures remain auditable but cannot terminate work.
        budget_rows = [
            row
            for row in model_rows
            if row[0] == "completed" or row[1] == "provider"
        ]
        input_tokens = sum(max(0, int(row[2] or 0)) for row in budget_rows)
        output_tokens = sum(max(0, int(row[3] or 0)) for row in budget_rows)
        tokens = sum(max(0, int(row[4] or 0)) for row in budget_rows)
        provider_tokens = sum(
            max(0, int(row[4] or 0)) for row in model_rows if row[1] == "provider"
        )
        failed_estimated_tokens_raw = sum(
            max(0, int(row[4] or 0))
            for row in model_rows
            if row[0] != "completed" and row[1] != "provider"
        )
        unique_failed_prompts: dict[str, int] = {}
        failed_usage_unknown_count = 0
        previous_legacy_run_id: int | None = None
        previous_legacy_attempt: int | None = None
        legacy_group_key = ""
        for index, row in enumerate(model_rows):
            if row[0] == "completed" or row[1] == "provider":
                previous_legacy_run_id = None
                previous_legacy_attempt = None
                legacy_group_key = ""
                continue
            failed_usage_unknown_count += 1
            prompt_key = str(row[6] or "").strip()
            if not prompt_key:
                run_id = int(row[8] or 0)
                attempt = int(row[9] or 0)
                if not (
                    legacy_group_key
                    and previous_legacy_run_id == run_id
                    and previous_legacy_attempt is not None
                    and attempt == previous_legacy_attempt + 1
                ):
                    legacy_group_key = f"legacy-call-{int(row[7] or index)}"
                prompt_key = legacy_group_key
                previous_legacy_run_id = run_id
                previous_legacy_attempt = attempt
            unique_failed_prompts[prompt_key] = max(
                unique_failed_prompts.get(prompt_key, 0), max(0, int(row[4] or 0))
            )
        failed_estimated_tokens = sum(unique_failed_prompts.values())
        completed_calls = sum(1 for row in model_rows if row[0] == "completed")
        failed_calls = len(model_rows) - completed_calls
        active_ms = sum(max(0, int(row[5] or 0)) for row in model_rows) + max(
            0, tool_duration
        )
        active_limit_ms = self.max_active_minutes_per_tree * 60_000
        model_review_cycle = tokens // self.max_model_tokens_per_tree
        active_review_cycle = active_ms // active_limit_ms
        review_required = model_review_cycle > 0 or active_review_cycle > 0
        return {
            "scope": "work_tree",
            "thread_ids": tree_ids,
            "model_calls": len(model_rows),
            "completed_model_calls": completed_calls,
            "failed_model_calls": failed_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_tokens": tokens,
            "provider_reported_tokens": provider_tokens,
            "estimated_failed_attempt_tokens": failed_estimated_tokens,
            "estimated_failed_attempt_tokens_raw": failed_estimated_tokens_raw,
            "unique_failed_prompt_count": len(unique_failed_prompts),
            "failed_usage_unknown_count": failed_usage_unknown_count,
            "duplicate_retry_attempts": max(
                0, failed_usage_unknown_count - len(unique_failed_prompts)
            ),
            "attempted_input_tokens": attempted_input_tokens,
            "attempted_output_tokens": attempted_output_tokens,
            "attempted_model_tokens": attempted_tokens,
            "model_token_limit": self.max_model_tokens_per_tree,
            "model_tokens_remaining": max(
                0, self.max_model_tokens_per_tree - tokens
            ),
            "model_tokens_until_next_review": (
                self.max_model_tokens_per_tree
                - (tokens % self.max_model_tokens_per_tree)
                if tokens
                else self.max_model_tokens_per_tree
            ),
            "active_ms": active_ms,
            "active_minutes": round(active_ms / 60_000, 3),
            "active_limit_minutes": self.max_active_minutes_per_tree,
            "active_ms_remaining": max(0, active_limit_ms - active_ms),
            "review_required": review_required,
            "review_cycle": max(model_review_cycle, active_review_cycle),
            "review_reasons": [
                reason
                for condition, reason in (
                    (tokens >= self.max_model_tokens_per_tree, "model_tokens"),
                    (active_ms >= active_limit_ms, "active_time"),
                )
                if condition
            ],
            # Compatibility fields: resource thresholds are now autonomous
            # reflection triggers, not terminal project limits.
            "exceeded": False,
            "exceeded_reasons": [],
        }

    def record_model_call(
        self,
        *,
        thread_id: int,
        run_id: int,
        owner_agent_id: int,
        execution_token: int,
        purpose: str,
        attempt_number: int,
        model_id: int | None,
        model_name: str,
        status: str,
        prompt: str,
        output: str = "",
        usage: ModelUsage | None = None,
        error_kind: str | None = None,
        error: str | None = None,
        started_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> WorkModelCall:
        usage = usage or ModelUsage(
            estimate_tokens(prompt), estimate_tokens(output) if output else 0,
            estimate_tokens(prompt) + (estimate_tokens(output) if output else 0), "estimated"
        )
        finished_at = utc_now()
        if started_at is None:
            started_at = (
                finished_at - timedelta(milliseconds=max(0, duration_ms))
                if duration_ms is not None
                else finished_at
            )
        if duration_ms is None:
            duration_ms = max(0, int((finished_at - started_at).total_seconds() * 1000))
        with self.database.sessions() as session:
            prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
            row = WorkModelCall(
                work_thread_id=thread_id,
                work_run_id=run_id,
                owner_agent_id=owner_agent_id,
                execution_token=execution_token,
                purpose=purpose,
                attempt_number=attempt_number,
                model_id=model_id,
                model_name=model_name,
                status=status,
                error_kind=error_kind,
                error=error,
                prompt_hash=prompt_hash,
                prompt_chars=len(prompt),
                input_tokens=max(0, usage.input_tokens),
                output_tokens=max(0, usage.output_tokens),
                total_tokens=max(0, usage.total_tokens),
                usage_source=usage.source,
                duration_ms=max(0, duration_ms),
                started_at=started_at,
                finished_at=finished_at,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def list_calls(self, thread_id: int, offset: int = 0, limit: int = 50) -> list[WorkModelCall]:
        offset = min(10_000, max(0, int(offset)))
        limit = min(100, max(1, int(limit)))
        with self.database.sessions() as session:
            return session.scalars(
                select(WorkModelCall)
                .where(WorkModelCall.work_thread_id == thread_id)
                .order_by(WorkModelCall.id.desc())
                .offset(offset)
                .limit(limit)
            ).all()


def serialize_model_call(row: WorkModelCall) -> dict[str, Any]:
    return {
        "id": row.id,
        "work_thread_id": row.work_thread_id,
        "work_run_id": row.work_run_id,
        "owner_agent_id": row.owner_agent_id,
        "execution_token": row.execution_token,
        "purpose": row.purpose,
        "attempt_number": row.attempt_number,
        "model_id": row.model_id,
        "model_name": row.model_name,
        "status": row.status,
        "error_kind": row.error_kind,
        "error": row.error,
        "prompt_hash": row.prompt_hash,
        "prompt_chars": row.prompt_chars,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "total_tokens": row.total_tokens,
        "usage_source": row.usage_source,
        "duration_ms": row.duration_ms,
        "started_at": row.started_at.isoformat(),
        "finished_at": row.finished_at.isoformat(),
    }
