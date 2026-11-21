from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .database import Database
from .models import AgentProfile, Evidence, ToolCall, WorkContextSummary, WorkEvent, WorkModelCall, WorkProject, WorkThread
from .work_budget_service import serialize_model_call
from .tool_runtime import ToolContext, ToolRegistryError


def stored_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def estimate_tokens(value: str) -> int:
    ascii_count = sum(1 for char in value if ord(char) < 128)
    other_count = len(value) - ascii_count
    return max(1, math.ceil(ascii_count / 4 + other_count / 1.5))


def clip(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "…"


def compact_unique(values: Any, *, limit: int, item_chars: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in reversed(list(values or [])):
        text = clip(value, item_chars).strip()
        key = "".join(text.lower().split())
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return list(reversed(result))


def compact_work_state(value: Any) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    return {
        "phase": clip(state.get("phase"), 400),
        "current_judgment": clip(state.get("current_judgment"), 2500),
        "current_strategy": clip(state.get("current_strategy"), 1200),
        "strategy_change_reason": clip(
            state.get("strategy_change_reason"), 900
        ),
        "working_hypotheses": [
            {
                "hypothesis_id": clip(item.get("hypothesis_id"), 120),
                "statement": clip(item.get("statement"), 700),
                "status": item.get("status"),
                "evidence_refs": compact_unique(
                    item.get("evidence_refs"), limit=6, item_chars=180
                ),
                "next_test": clip(item.get("next_test"), 500),
            }
            for item in list(state.get("working_hypotheses") or [])[-10:]
            if isinstance(item, dict) and item.get("hypothesis_id")
        ],
        "strategy_assessments": [
            {
                "strategy_id": clip(item.get("strategy_id"), 120),
                "approach": clip(item.get("approach"), 700),
                "outcome": item.get("outcome"),
                "evidence_refs": compact_unique(
                    item.get("evidence_refs"), limit=6, item_chars=180
                ),
                "lesson": clip(item.get("lesson"), 600),
                "next_change": clip(item.get("next_change"), 600),
            }
            for item in list(state.get("strategy_assessments") or [])[-10:]
            if isinstance(item, dict) and item.get("strategy_id")
        ],
        "criteria_status": [
            {
                "criterion_id": clip(item.get("criterion_id"), 120),
                "status": item.get("status"),
                "evidence_refs": [clip(ref, 120) for ref in item.get("evidence_refs", [])[:12]],
                "note": clip(item.get("note"), 500),
            }
            for item in list(state.get("criteria_status") or [])[:30]
            if isinstance(item, dict)
        ],
        "verified_facts": compact_unique(
            state.get("verified_facts"), limit=12, item_chars=450
        ),
        "open_questions": compact_unique(
            state.get("open_questions"), limit=5, item_chars=400
        ),
        "rejected_paths": compact_unique(
            state.get("rejected_paths"), limit=8, item_chars=400
        ),
        "risks": compact_unique(state.get("risks"), limit=5, item_chars=400),
        "next_actions": compact_unique(
            state.get("next_actions"), limit=3, item_chars=400
        ),
        "market_insights": [
            {
                "title": clip(item.get("title"), 300),
                "category": clip(item.get("category"), 80),
                "signal": clip(item.get("signal"), 700),
                "phenomenon": clip(item.get("phenomenon"), 900),
                "why_interesting": clip(item.get("why_interesting"), 700),
                "drivers": compact_unique(item.get("drivers"), limit=4, item_chars=250),
                "possible_waves": compact_unique(item.get("possible_waves"), limit=4, item_chars=250),
                "adjacent_searches": compact_unique(item.get("adjacent_searches"), limit=4, item_chars=250),
                "implications": compact_unique(item.get("implications"), limit=4, item_chars=250),
                "discussion_signals": compact_unique(item.get("discussion_signals"), limit=4, item_chars=250),
                "participant_groups": compact_unique(item.get("participant_groups"), limit=4, item_chars=200),
                "participation_behaviors": compact_unique(item.get("participation_behaviors"), limit=4, item_chars=250),
                "participation_evidence": compact_unique(item.get("participation_evidence"), limit=4, item_chars=300),
                "ordinary_people_relevance": clip(item.get("ordinary_people_relevance"), 700),
                "source_refs": compact_unique(item.get("source_refs"), limit=4, item_chars=250),
                "confidence": item.get("confidence", "medium"),
            }
            for item in list(state.get("market_insights") or [])[-8:]
            if isinstance(item, dict) and item.get("title")
        ],
        "market_lens_reviews": [
            {
                "lens": clip(item.get("lens"), 80),
                "status": item.get("status"),
                "candidate_refs": compact_unique(
                    item.get("candidate_refs"), limit=6, item_chars=250
                ),
                "note": clip(item.get("note"), 600),
            }
            for item in list(state.get("market_lens_reviews") or [])[-12:]
            if isinstance(item, dict) and item.get("lens")
        ],
        "market_candidates": [
            {
                "candidate_ref": clip(item.get("candidate_ref"), 500),
                "title": clip(item.get("title"), 300),
                "lens": clip(item.get("lens"), 80),
                "decision": item.get("decision"),
                "reason": clip(item.get("reason"), 600),
                "source_url": clip(item.get("source_url"), 800),
            }
            for item in list(state.get("market_candidates") or [])[-16:]
            if isinstance(item, dict) and item.get("candidate_ref")
        ],
    }


def serialize_summary(row: WorkContextSummary) -> dict[str, Any]:
    return {
        "id": row.id,
        "work_thread_id": row.work_thread_id,
        "version": row.version,
        "source_hash": row.source_hash,
        "summary": stored_json(row.summary_json, {}),
        "through_tool_call_id": row.through_tool_call_id,
        "through_event_id": row.through_event_id,
        "estimated_tokens": row.estimated_tokens,
        "created_at": row.created_at.isoformat(),
    }


class WorkContextService:
    def __init__(self, database: Database, settings: dict[str, Any] | None = None) -> None:
        self.database = database
        self.configure(settings or {})

    def configure(self, settings: dict[str, Any]) -> None:
        config = (settings or {}).get("work_context") or {}
        self.enabled = bool(config.get("enabled", True))
        self.max_prompt_chars = min(200_000, max(12_000, int(config.get("max_prompt_chars", 16_000))))
        self.checkpoint_max_chars = min(80_000, max(4_000, int(config.get("checkpoint_max_chars", 8_000))))
        self.recent_tool_calls = min(20, max(1, int(config.get("recent_tool_calls", 2))))
        self.recent_result_chars = min(20_000, max(500, int(config.get("recent_result_chars", 1_200))))
        self.summary_interval_tool_calls = min(100, max(1, int(config.get("summary_interval_tool_calls", 5))))
        self.evidence_limit = min(100, max(1, int(config.get("evidence_limit", 20))))
        self.failure_limit = min(100, max(1, int(config.get("failure_limit", 20))))
        # One article is an atomic input artifact.  It may be larger than the
        # compact rolling state, but it must never be replaced by a hand-picked
        # excerpt before the editorial decision has been made.
        self.active_input_chars = min(
            100_000, max(10_000, int(config.get("active_input_chars", 40_000)))
        )

    def latest(self, thread_id: int) -> WorkContextSummary | None:
        with self.database.sessions() as session:
            return session.scalar(
                select(WorkContextSummary)
                .where(WorkContextSummary.work_thread_id == thread_id)
                .order_by(WorkContextSummary.version.desc())
            )

    def list(self, thread_id: int) -> list[WorkContextSummary]:
        with self.database.sessions() as session:
            return session.scalars(
                select(WorkContextSummary)
                .where(WorkContextSummary.work_thread_id == thread_id)
                .order_by(WorkContextSummary.version.desc())
            ).all()

    def _payload(self, session: Any, thread: WorkThread, checkpoint: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        latest_tool_id = int(
            session.scalar(
                select(func.max(ToolCall.id)).where(ToolCall.work_thread_id == thread.id)
            )
            or 0
        )
        latest_event_id = int(
            session.scalar(
                select(func.max(WorkEvent.id)).where(WorkEvent.work_thread_id == thread.id)
            )
            or 0
        )
        failed_calls = session.scalars(
            select(ToolCall)
            .where(
                ToolCall.work_thread_id == thread.id,
                ToolCall.status.in_(("failed", "denied", "unknown_outcome")),
            )
            .order_by(ToolCall.id.desc())
            .limit(self.failure_limit)
        ).all()
        evidence = session.scalars(
            select(Evidence)
            .where(Evidence.work_thread_id == thread.id)
            .order_by(Evidence.id.desc())
            .limit(self.evidence_limit)
        ).all()
        plan = checkpoint.get("plan") if isinstance(checkpoint.get("plan"), list) else []
        errors = {
            key: clip(value, 1200)
            for key, value in checkpoint.items()
            if key.startswith("last_") and (key.endswith("error") or "error" in key)
        }
        payload = {
            "objective": thread.objective,
            "success_criteria": stored_json(thread.success_criteria_json, []),
            "constraints": stored_json(thread.constraints_json, []),
            "state_summary": clip(checkpoint.get("state_summary") or thread.summary, 5000),
            "work_state_version": checkpoint.get("work_state_version", 0),
            "work_state": compact_work_state(checkpoint.get("work_state")),
            "research_material_count": len(
                list(checkpoint.get("research_materials") or [])
            ),
            "pending_research_material_count": len(
                list(checkpoint.get("pending_research_material_ids") or [])
            ),
            "progress": checkpoint.get("progress", 0),
            "plan_version": checkpoint.get("plan_version", 0),
            "completed_steps": [clip(item.get("title"), 500) for item in plan if isinstance(item, dict) and item.get("status") == "completed"],
            "blocked_steps": [clip(item.get("title"), 500) for item in plan if isinstance(item, dict) and item.get("status") == "blocked"],
            "current_plan": plan[:50],
            "last_action": checkpoint.get("last_action"),
            "waiting_collaboration_id": checkpoint.get("waiting_collaboration_id"),
            "last_collaboration_error": checkpoint.get("last_collaboration_error"),
            "last_scheduled_wait_error": checkpoint.get(
                "last_scheduled_wait_error"
            ),
            "collaboration_results": list(checkpoint.get("collaboration_results") or [])[-10:],
            "criteria_evidence": checkpoint.get("unmet_criteria_evidence") or [],
            "unresolved_errors": errors,
            "failed_methods": [
                {
                    "tool_call_id": call.id,
                    "tool_name": call.tool_name,
                    "arguments": stored_json(call.arguments_json, {}),
                    "status": call.status,
                    "error": clip(call.error, 1200),
                }
                for call in reversed(failed_calls)
            ],
            "evidence_refs": [
                {
                    "evidence_id": item.id,
                    "stance": item.stance,
                    "confidence": item.confidence,
                    "source_type": item.source_type,
                    "source_publisher": item.source_publisher,
                    "claim_key": item.claim_key,
                    "review_status": item.review_status,
                    "claim": clip(item.claim, 1500),
                    "source_url": item.source_url,
                }
                for item in reversed(evidence)
            ],
            "continuation": {
                "last_action": checkpoint.get("last_action"),
                "next_wake_at": thread.next_wake_at.isoformat() if thread.next_wake_at else None,
                "thread_status": thread.status,
            },
            "source_coverage": {
                "through_tool_call_id": latest_tool_id,
                "through_event_id": latest_event_id,
                "checkpoint_steps_executed": checkpoint.get("steps_executed", 0),
            },
        }
        return payload, latest_tool_id, latest_event_id

    def ensure_summary(
        self, thread_id: int, checkpoint: dict[str, Any], *, force: bool = False
    ) -> WorkContextSummary | None:
        if not self.enabled:
            return None
        with self.database.sessions() as session:
            thread = session.get(WorkThread, thread_id)
            if not thread:
                raise ToolRegistryError("工作线程不存在")
            latest = session.scalar(
                select(WorkContextSummary)
                .where(WorkContextSummary.work_thread_id == thread_id)
                .order_by(WorkContextSummary.version.desc())
            )
            latest_tool_id = int(
                session.scalar(select(func.max(ToolCall.id)).where(ToolCall.work_thread_id == thread_id)) or 0
            )
            calls_since = session.scalar(
                select(func.count(ToolCall.id)).where(
                    ToolCall.work_thread_id == thread_id,
                    ToolCall.id > (latest.through_tool_call_id if latest else 0),
                )
            ) or 0
            checkpoint_text = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, default=str)
            if not force and latest and calls_since < self.summary_interval_tool_calls and len(checkpoint_text) <= self.checkpoint_max_chars:
                return latest
            if not force and not latest and calls_since < self.summary_interval_tool_calls and len(checkpoint_text) <= self.checkpoint_max_chars:
                return None
            payload, latest_tool_id, latest_event_id = self._payload(session, thread, checkpoint)
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            source_hash = sha256(encoded.encode("utf-8")).hexdigest()
            if latest and latest.source_hash == source_hash:
                return latest
            row = WorkContextSummary(
                work_thread_id=thread_id,
                version=(latest.version + 1 if latest else 1),
                source_hash=source_hash,
                summary_json=encoded,
                through_tool_call_id=latest_tool_id,
                through_event_id=latest_event_id,
                estimated_tokens=estimate_tokens(encoded),
            )
            session.add(row)
            thread.summary = clip(payload.get("state_summary"), 20_000)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return session.scalar(
                    select(WorkContextSummary)
                    .where(WorkContextSummary.work_thread_id == thread_id)
                    .order_by(WorkContextSummary.version.desc())
                )
            session.refresh(row)
            return row

    def prompt_context(self, thread_id: int, checkpoint: dict[str, Any]) -> dict[str, Any]:
        summary = self.ensure_summary(thread_id, checkpoint)
        plan = [
            item
            for item in list(checkpoint.get("plan") or [])
            if isinstance(item, dict)
            and item.get("status") in {"pending", "in_progress", "blocked"}
        ][:12]
        pending_ids = {
            int(item)
            for item in list(checkpoint.get("pending_research_material_ids") or [])
            if str(item).isdigit() and int(item) > 0
        }
        context_runtime = (
            checkpoint.get("context_runtime")
            if isinstance(checkpoint.get("context_runtime"), dict)
            else {}
        )
        active_unit = (
            context_runtime.get("active_unit")
            if isinstance(context_runtime.get("active_unit"), dict)
            else {}
        )
        dependency_ids = {
            int(item)
            for item in list(active_unit.get("dependency_tool_call_ids") or [])
            if str(item).isdigit() and int(item) > 0
        }
        # Research pages are judged one at a time.  The next page is not part of
        # this model call, and already consumed pages remain only in the ledger.
        active_pending_id = next(
            (
                int(item.get("tool_call_id") or 0)
                for item in list(checkpoint.get("research_materials") or [])
                if isinstance(item, dict)
                and int(item.get("tool_call_id") or 0) in pending_ids
                and item.get("kind") == "page"
            ),
            0,
        )
        if active_pending_id:
            dependency_ids = {active_pending_id}
        pending_materials = [
            {
                "tool_call_id": item.get("tool_call_id"),
                "tool_name": item.get("tool_name"),
                "kind": item.get("kind"),
                "query": clip(item.get("query"), 500),
                "source_url": clip(item.get("source_url"), 1200),
                "title": clip(item.get("title"), 400),
                "content_chars": item.get("content_chars"),
                "has_substantial_content": item.get("has_substantial_content"),
                "content_excerpt": clip(item.get("content_excerpt"), 3600),
                "candidates": [
                    {
                        "title": clip(candidate.get("title"), 300),
                        "url": clip(candidate.get("url"), 800),
                        "snippet": clip(candidate.get("snippet"), 500),
                    }
                    for candidate in list(item.get("candidates") or [])[:8]
                    if isinstance(candidate, dict)
                ],
            }
            for item in list(checkpoint.get("research_materials") or [])[-20:]
            if isinstance(item, dict)
            and int(item.get("tool_call_id") or 0) in pending_ids
            and (
                not active_pending_id
                or int(item.get("tool_call_id") or 0) == active_pending_id
            )
        ]
        pending_materials = pending_materials[:1] if active_pending_id else pending_materials[-3:]
        budget_pending_materials = [
            {
                "tool_call_id": item.get("tool_call_id"),
                "tool_name": item.get("tool_name"),
                "kind": item.get("kind"),
                "query": clip(item.get("query"), 300),
                "source_url": clip(item.get("source_url"), 600),
                "title": clip(item.get("title"), 260),
                "content_chars": item.get("content_chars"),
                "has_substantial_content": item.get("has_substantial_content"),
                "content_excerpt": clip(item.get("content_excerpt"), 1200),
                "candidates": [
                    {
                        "title": clip(candidate.get("title"), 180),
                        "url": clip(candidate.get("url"), 400),
                        "snippet": clip(candidate.get("snippet"), 180),
                    }
                    for candidate in list(item.get("candidates") or [])[:2]
                    if isinstance(candidate, dict)
                ],
                "more_candidate_count": max(
                    0, len(list(item.get("candidates") or [])) - 2
                ),
            }
            for item in pending_materials
        ]
        temporal = checkpoint.get("temporal_contract")
        compact_checkpoint = {
            "state_summary": clip(checkpoint.get("state_summary"), 2000),
            "work_state_version": checkpoint.get("work_state_version", 0),
            "work_state": compact_work_state(checkpoint.get("work_state")),
            "progress": checkpoint.get("progress"),
            "plan_version": checkpoint.get("plan_version"),
            "active_plan": plan,
            "last_action": checkpoint.get("last_action"),
            "pending_research_materials": pending_materials,
            "waiting_collaboration_id": checkpoint.get("waiting_collaboration_id"),
            "latest_collaboration_progress": checkpoint.get(
                "latest_collaboration_progress"
            ),
            "collaboration_results": list(
                checkpoint.get("collaboration_results") or []
            )[-3:],
            "action_required": checkpoint.get("action_required"),
            "reflection_required": checkpoint.get("reflection_required"),
            "strategy_change_required": checkpoint.get(
                "strategy_change_required"
            ),
            "observation_assessments": list(
                checkpoint.get("observation_assessments") or []
            )[-5:],
            "administrative_actions_since_progress": checkpoint.get(
                "administrative_actions_since_progress", 0
            ),
            "efficiency_review": checkpoint.get("efficiency_review"),
            "temporal_contract": (
                {
                    "contract_version": temporal.get("contract_version", 3),
                    "completion_basis": temporal.get(
                        "completion_basis", "fixed_duration_then_finalize"
                    ),
                    "started_at": temporal.get("started_at"),
                    "earliest_finish_at": temporal.get("earliest_finish_at"),
                    "deadline_at": temporal.get("deadline_at")
                    or temporal.get("earliest_finish_at"),
                    "required_minutes": temporal.get("required_minutes"),
                    "meaningful_activities": list(
                        temporal.get("meaningful_activities")
                        or temporal.get("meaningful_windows")
                        or []
                    )[-30:],
                    "activity_scope": temporal.get("activity_scope"),
                }
                if isinstance(temporal, dict)
                else None
            ),
            "context_unit": {
                "unit_id": active_unit.get("unit_id"),
                "purpose": active_unit.get("purpose"),
                "dependency_tool_call_ids": sorted(dependency_ids),
                "input_contract": active_unit.get("input_contract"),
                "output_contract": active_unit.get("output_contract"),
            },
            "current_artifacts": [
                {
                    "kind": clip(item.get("kind"), 80),
                    "ref": clip(item.get("ref"), 2000),
                    "source_tool_call_id": item.get("source_tool_call_id"),
                }
                for item in list(checkpoint.get("current_artifacts") or [])[-10:]
                if isinstance(item, dict) and item.get("ref")
            ],
        }
        with self.database.sessions() as session:
            calls = (
                session.scalars(
                    select(ToolCall)
                    .where(
                        ToolCall.work_thread_id == thread_id,
                        ToolCall.id.in_(dependency_ids),
                    )
                    .order_by(ToolCall.id.asc())
                ).all()
                if dependency_ids
                else (
                    session.scalars(
                        select(ToolCall)
                        .where(ToolCall.work_thread_id == thread_id)
                        .order_by(ToolCall.id.desc())
                        .limit(self.recent_tool_calls)
                    ).all()
                    if not context_runtime
                    else []
                )
            )
            active_call = session.get(ToolCall, active_pending_id) if active_pending_id else None
        recent = [
            {
                "tool_call_id": call.id,
                "tool_name": call.tool_name,
                "status": call.status,
                "arguments": stored_json(call.arguments_json, {}),
                "result_preview": clip(call.result_json, self.recent_result_chars),
                "error": clip(call.error, 1000),
            }
            for call in reversed(calls)
        ]
        value = {
            "latest_summary": serialize_summary(summary) if summary else None,
            "current_checkpoint": compact_checkpoint,
            "recent_tool_calls": recent,
            "history_access": (
                "当前工作事实已由系统归约。只有核验某个已知tool_call_id的原始内容时"
                "才调用read_work_history；不得分页扫描历史来恢复进度。"
            ),
        }
        if active_call is not None:
            active_result = stored_json(active_call.result_json, {})
            if not isinstance(active_result, dict):
                active_result = {}
            full_text = str(active_result.get("text") or "")
            value["active_input"] = {
                "kind": "web_article",
                "tool_call_id": active_call.id,
                "title": clip(active_result.get("title"), 600),
                "source_url": clip(
                    active_result.get("resolved_url")
                    or active_result.get("url")
                    or stored_json(active_call.arguments_json, {}).get("url"),
                    2000,
                ),
                "content_chars": len(full_text),
                "content_truncated_by_source_tool": bool(active_result.get("truncated")),
                "text": clip(full_text, self.active_input_chars),
            }
        active_input_chars = len(
            json.dumps(value.get("active_input") or {}, ensure_ascii=False, default=str)
        )
        effective_prompt_chars = self.max_prompt_chars + active_input_chars
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > effective_prompt_chars:
            value["recent_tool_calls"] = recent[-2:]
            value["context_budget_truncated"] = True
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > effective_prompt_chars and value.get("latest_summary"):
            latest = value["latest_summary"]
            summary_payload = latest.get("summary") or {}
            latest["summary"] = {
                "objective": clip(summary_payload.get("objective"), 2000),
                "success_criteria": [
                    clip(item, 300)
                    for item in list(summary_payload.get("success_criteria") or [])[:20]
                ],
                "constraints": [
                    clip(item, 300)
                    for item in list(summary_payload.get("constraints") or [])[:20]
                ],
                "state_summary": clip(summary_payload.get("state_summary"), 3000),
                "work_state_version": summary_payload.get("work_state_version", 0),
                "work_state": compact_work_state(summary_payload.get("work_state")),
                "progress": summary_payload.get("progress"),
                "completed_steps": [
                    clip(item, 200)
                    for item in list(summary_payload.get("completed_steps") or [])[-20:]
                ],
                "blocked_steps": [
                    clip(item, 200)
                    for item in list(summary_payload.get("blocked_steps") or [])[-20:]
                ],
                "continuation": summary_payload.get("continuation"),
                "source_coverage": summary_payload.get("source_coverage"),
                "note": "摘要详情已按提示预算折叠，可用read_work_history(kind=summaries)读取完整版本。",
            }
            value["recent_tool_calls"] = recent[-1:]
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > effective_prompt_chars:
            value["current_checkpoint"] = {
                "state_summary": clip(checkpoint.get("state_summary"), 2000),
                "work_state_version": checkpoint.get("work_state_version", 0),
                "work_state": compact_work_state(checkpoint.get("work_state")),
                "progress": checkpoint.get("progress"),
                "last_action": checkpoint.get("last_action"),
                "pending_research_materials": budget_pending_materials,
                "context_unit": compact_checkpoint.get("context_unit"),
                "current_artifacts": compact_checkpoint.get("current_artifacts"),
                "strategy_change_required": checkpoint.get(
                    "strategy_change_required"
                ),
                "observation_assessments": list(
                    checkpoint.get("observation_assessments") or []
                )[-3:],
                "note": "Checkpoint详情已折叠，可用read_work_history读取原账本。",
            }
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > effective_prompt_chars:
            value["recent_tool_calls"] = []
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > effective_prompt_chars and value.get("latest_summary"):
            latest = value["latest_summary"]
            summary_payload = latest.get("summary") or {}
            latest["summary"] = {
                "objective": clip(summary_payload.get("objective"), 1000),
                "state_summary": clip(summary_payload.get("state_summary"), 1500),
                "work_state_version": summary_payload.get("work_state_version", 0),
                "work_state": compact_work_state(summary_payload.get("work_state")),
                "progress": summary_payload.get("progress"),
                "continuation": summary_payload.get("continuation"),
                "source_coverage": summary_payload.get("source_coverage"),
                "note": "完整摘要可用read_work_history(kind=summaries)读取。",
            }
        final_encoded = json.dumps(value, ensure_ascii=False, default=str)
        budget = {
            "max_chars": effective_prompt_chars,
            "actual_chars": 0,
            "estimated_tokens": estimate_tokens(final_encoded),
        }
        value["context_budget"] = budget
        final_encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(final_encoded) > effective_prompt_chars:
            latest_summary = value.get("latest_summary") or {}
            summary_payload = latest_summary.get("summary") or {}
            active_input = value.get("active_input")
            value = {
                "latest_summary": {
                    "id": latest_summary.get("id"),
                    "version": latest_summary.get("version"),
                    "estimated_tokens": latest_summary.get("estimated_tokens"),
                    "summary": {
                        "objective": clip(summary_payload.get("objective"), 800),
                        "state_summary": clip(summary_payload.get("state_summary"), 1200),
                        "work_state_version": summary_payload.get("work_state_version", 0),
                        "work_state": compact_work_state(summary_payload.get("work_state")),
                        "progress": summary_payload.get("progress"),
                        "continuation": summary_payload.get("continuation"),
                        "source_coverage": summary_payload.get("source_coverage"),
                    },
                } if latest_summary else None,
                "current_checkpoint": {
                    "state_summary": clip(checkpoint.get("state_summary"), 1200),
                    "work_state_version": checkpoint.get("work_state_version", 0),
                    "work_state": compact_work_state(checkpoint.get("work_state")),
                    "progress": checkpoint.get("progress"),
                    "last_action": checkpoint.get("last_action"),
                    "pending_research_materials": budget_pending_materials,
                    "context_unit": compact_checkpoint.get("context_unit"),
                    "current_artifacts": compact_checkpoint.get("current_artifacts"),
                    "strategy_change_required": checkpoint.get(
                        "strategy_change_required"
                    ),
                    "observation_assessments": list(
                        checkpoint.get("observation_assessments") or []
                    )[-3:],
                },
                "recent_tool_calls": [],
                "history_access": "上下文已达到硬预算；用read_work_history读取当前线程原账本。",
                "context_budget_truncated": True,
                "context_budget": budget,
            }
            if active_input:
                value["active_input"] = active_input
            final_encoded = json.dumps(value, ensure_ascii=False, default=str)
        budget["actual_chars"] = len(final_encoded)
        budget["estimated_tokens"] = estimate_tokens(final_encoded)
        final_encoded = json.dumps(value, ensure_ascii=False, default=str)
        budget["actual_chars"] = len(final_encoded)
        return value

    def history(self, thread_id: int, kind: str, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        if kind not in {"tool_calls", "events", "evidence", "summaries", "model_calls"}:
            raise ToolRegistryError("kind必须是tool_calls、events、evidence、summaries或model_calls")
        offset = min(10_000, max(0, int(offset)))
        limit = min(50, max(1, int(limit)))
        with self.database.sessions() as session:
            if not session.get(WorkThread, thread_id):
                raise ToolRegistryError("工作线程不存在")
            if kind == "tool_calls":
                rows = session.scalars(select(ToolCall).where(ToolCall.work_thread_id == thread_id).order_by(ToolCall.id.desc()).offset(offset).limit(limit)).all()
                items = [{"id": row.id, "tool_name": row.tool_name, "status": row.status, "arguments": stored_json(row.arguments_json, {}), "result": stored_json(row.result_json, None), "error": row.error} for row in rows]
            elif kind == "events":
                rows = session.scalars(select(WorkEvent).where(WorkEvent.work_thread_id == thread_id).order_by(WorkEvent.id.desc()).offset(offset).limit(limit)).all()
                items = [{"id": row.id, "event_type": row.event_type, "payload": stored_json(row.payload_json, {}), "created_at": row.created_at.isoformat()} for row in rows]
            elif kind == "evidence":
                rows = session.scalars(select(Evidence).where(Evidence.work_thread_id == thread_id).order_by(Evidence.id.desc()).offset(offset).limit(limit)).all()
                items = [{"id": row.id, "source_url": row.source_url, "claim": row.claim, "excerpt": row.excerpt, "confidence": row.confidence, "stance": row.stance, "source_type": row.source_type, "source_publisher": row.source_publisher, "claim_key": row.claim_key, "verification_notes": row.verification_notes, "review_status": row.review_status, "review_notes": row.review_notes} for row in rows]
            elif kind == "summaries":
                rows = session.scalars(select(WorkContextSummary).where(WorkContextSummary.work_thread_id == thread_id).order_by(WorkContextSummary.version.desc()).offset(offset).limit(limit)).all()
                items = [serialize_summary(row) for row in rows]
            else:
                rows = session.scalars(select(WorkModelCall).where(WorkModelCall.work_thread_id == thread_id).order_by(WorkModelCall.id.desc()).offset(offset).limit(limit)).all()
                items = [serialize_model_call(row) for row in rows]
        return {"kind": kind, "offset": offset, "limit": limit, "items": items, "has_more": len(items) == limit}

    def history_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        target_thread_id = int(arguments.get("thread_id") or context.work_thread_id)
        if target_thread_id != context.work_thread_id:
            with self.database.sessions() as session:
                agent = session.get(AgentProfile, context.owner_agent_id)
                if not agent or not agent.is_controller:
                    raise ToolRegistryError("只有总控Agent可以读取其他工作线程历史")
        raw_tool_call_id = arguments.get("tool_call_id")
        if raw_tool_call_id is not None and str(raw_tool_call_id).strip():
            if str(arguments.get("kind") or "") != "tool_calls":
                raise ToolRegistryError("tool_call_id只能与kind=tool_calls一起使用")
            tool_call_text = str(raw_tool_call_id).strip()
            if tool_call_text.startswith("tool_call:"):
                tool_call_text = tool_call_text.split(":", 1)[1]
            try:
                tool_call_id = int(tool_call_text)
            except ValueError as exc:
                raise ToolRegistryError(
                    "tool_call_id必须是正整数或tool_call:<正整数>"
                ) from exc
            result_offset = min(
                10_000_000, max(0, int(arguments.get("result_offset") or 0))
            )
            result_limit = min(
                12_000, max(1_000, int(arguments.get("result_limit") or 8_000))
            )
            with self.database.sessions() as session:
                row = session.get(ToolCall, tool_call_id)
                if not row or row.work_thread_id != target_thread_id:
                    raise ToolRegistryError("指定工具调用不存在或不属于目标工作线程")
                result_json = row.result_json or ""
                result_chunk = result_json[
                    result_offset : result_offset + result_limit
                ]
                next_offset = result_offset + len(result_chunk)
                return {
                    "kind": "tool_calls",
                    "tool_call_id": row.id,
                    "tool_name": row.tool_name,
                    "status": row.status,
                    "arguments": stored_json(row.arguments_json, {}),
                    "error": row.error,
                    "result_offset": result_offset,
                    "result_limit": result_limit,
                    "result_chars": len(result_json),
                    "result_chunk": result_chunk,
                    "has_more": next_offset < len(result_json),
                    "next_result_offset": (
                        next_offset if next_offset < len(result_json) else None
                    ),
                    "instruction": (
                        "result_chunk是原始JSON文本片段；has_more为true时保持"
                        "tool_call_id和result_limit不变，用next_result_offset继续。"
                    ),
                }
        return self.history(
            target_thread_id,
            str(arguments.get("kind") or ""),
            arguments.get("offset", 0),
            arguments.get("limit", 10),
        )

    def list_recent_work_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        with self.database.sessions() as session:
            agent = session.get(AgentProfile, context.owner_agent_id)
            if not agent or not agent.is_controller:
                raise ToolRegistryError("只有总控Agent可以查看公司历史工作")
            limit = min(50, max(1, int(arguments.get("limit") or 10)))
            status = str(arguments.get("status") or "").strip()
            statement = select(WorkThread).where(
                WorkThread.id != context.work_thread_id,
                WorkThread.parent_thread_id.is_(None),
            )
            if status:
                statement = statement.where(WorkThread.status == status)
            statement = statement.order_by(
                WorkThread.updated_at.desc(), WorkThread.id.desc()
            ).limit(limit)
            rows = session.scalars(statement).all()
            items = []
            for row in rows:
                project = (
                    session.get(WorkProject, row.project_id)
                    if row.project_id
                    else None
                )
                summary = session.scalar(
                    select(WorkContextSummary)
                    .where(WorkContextSummary.work_thread_id == row.id)
                    .order_by(WorkContextSummary.version.desc())
                    .limit(1)
                )
                summary_value = stored_json(summary.summary_json, {}) if summary else {}
                items.append(
                    {
                        "thread_id": row.id,
                        "project_id": row.project_id,
                        "project_status": project.status if project else None,
                        "generation_number": row.generation_number,
                        "title": row.title,
                        "objective": clip(row.objective, 2000),
                        "status": row.status,
                        "result_preview": clip(row.result_text, 2500),
                        "state_summary": clip(summary_value.get("state_summary"), 1500),
                        "progress": summary_value.get("progress"),
                        "updated_at": row.updated_at.isoformat(),
                    }
                )
        return {
            "items": items,
            "instruction": "选择目标thread_id后，用read_work_history(thread_id=该ID, kind=...)读取原始账本。新任务不要反复读取自己的历史来猜旧任务。",
        }
