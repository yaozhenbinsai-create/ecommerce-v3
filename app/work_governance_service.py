from __future__ import annotations

from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import json
import re
import threading
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .database import Database
from .models import (
    AgentProfile,
    Evidence,
    ResearchReport,
    RuntimeJobState,
    WorkGovernanceIssue,
    WorkRun,
    WorkThread,
)
from .tool_runtime import ToolContext
from .research_quality_service import ResearchQualityService
from .work_runtime import (
    OPEN_RUN_STATUSES,
    TERMINAL_THREAD_STATUSES,
    WorkConflict,
    WorkNotFound,
    WorkRuntime,
    json_text,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def stored_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def serialize_governance_issue(issue: WorkGovernanceIssue) -> dict[str, Any]:
    return {
        "id": issue.id,
        "issue_key": issue.issue_key,
        "issue_type": issue.issue_type,
        "status": issue.status,
        "severity": issue.severity,
        "work_thread_id": issue.work_thread_id,
        "related_thread_id": issue.related_thread_id,
        "reviewer_agent_id": issue.reviewer_agent_id,
        "rework_thread_id": issue.rework_thread_id,
        "detected_by": issue.detected_by,
        "summary": issue.summary,
        "details": stored_json(issue.details_json, {}),
        "resolution": stored_json(issue.resolution_json, None),
        "review_decision": issue.review_decision,
        "detected_at": as_utc(issue.detected_at).isoformat(),
        "resolved_at": as_utc(issue.resolved_at).isoformat() if issue.resolved_at else None,
        "created_at": as_utc(issue.created_at).isoformat(),
        "updated_at": as_utc(issue.updated_at).isoformat(),
    }


class WorkGovernanceService:
    CONTROLLER_TOOLS = {
        "list_recent_work",
        "inspect_company_work",
        "list_governance_issues",
        "review_work_quality",
        "resolve_governance_issue",
    }

    def __init__(
        self,
        database: Database,
        work_runtime: WorkRuntime,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.database = database
        self.work_runtime = work_runtime
        self._scan_lock = threading.Lock()
        self.configure(settings or {})

    def configure(self, settings: dict[str, Any]) -> None:
        config = settings.get("work_governance") or {}
        self.enabled = bool(config.get("enabled", True))
        self.scan_interval_seconds = max(10, int(config.get("scan_interval_seconds", 60)))
        self.duplicate_similarity_threshold = min(
            0.99, max(0.50, float(config.get("duplicate_similarity_threshold", 0.82)))
        )
        self.stalled_minutes = max(1, int(config.get("stalled_minutes", 30)))
        self.quality_review_enabled = bool(config.get("quality_review_enabled", True))

    @staticmethod
    def _controller(session: Any, agent_id: int) -> AgentProfile:
        agent = session.get(AgentProfile, agent_id)
        if not agent:
            raise WorkNotFound("Agent不存在")
        if not agent.is_controller:
            raise WorkConflict("只有总控Agent可以处理公司治理")
        return agent

    @staticmethod
    def _root_ids(threads: list[WorkThread]) -> dict[int, int]:
        parents = {thread.id: thread.parent_thread_id for thread in threads}
        roots: dict[int, int] = {}
        for thread_id in parents:
            current = thread_id
            seen: set[int] = set()
            while parents.get(current) and current not in seen:
                seen.add(current)
                current = int(parents[current])
            roots[thread_id] = current
        return roots

    @staticmethod
    def _normalized_text(thread: WorkThread) -> str:
        criteria = stored_json(thread.success_criteria_json, [])
        raw = " ".join([thread.title, thread.objective, *criteria]).lower()
        return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", raw)

    @staticmethod
    def _shingles(text: str) -> set[str]:
        if len(text) < 3:
            return {text} if text else set()
        return {text[index : index + 3] for index in range(len(text) - 2)}

    @classmethod
    def _similarity(cls, left: WorkThread, right: WorkThread) -> float:
        left_text = cls._normalized_text(left)
        right_text = cls._normalized_text(right)
        if min(len(left_text), len(right_text)) < 12:
            return 0.0
        sequence = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
        left_set = cls._shingles(left_text)
        right_set = cls._shingles(right_text)
        union = left_set | right_set
        jaccard = len(left_set & right_set) / len(union) if union else 0.0
        return round(max(sequence, jaccard), 4)

    @staticmethod
    def _upsert_issue(
        session: Any,
        *,
        issue_key: str,
        issue_type: str,
        work_thread_id: int,
        summary: str,
        details: dict[str, Any],
        severity: str = "warning",
        related_thread_id: int | None = None,
        detected_by: str = "automatic",
    ) -> tuple[WorkGovernanceIssue, bool]:
        row = session.scalar(
            select(WorkGovernanceIssue).where(
                WorkGovernanceIssue.issue_key == issue_key
            )
        )
        if row:
            resolution = stored_json(row.resolution_json, {})
            if (
                row.status == "resolved"
                and resolution.get("reason") == "automatic_condition_cleared"
            ):
                row.status = "open"
                row.resolved_at = None
                row.resolution_json = None
                row.detected_at = utc_now()
            if row.status == "open":
                row.summary = summary
                row.details_json = json_text(details)
                row.severity = severity
                row.updated_at = utc_now()
            return row, False
        row = WorkGovernanceIssue(
            issue_key=issue_key,
            issue_type=issue_type,
            status="open",
            severity=severity,
            work_thread_id=work_thread_id,
            related_thread_id=related_thread_id,
            summary=summary,
            details_json=json_text(details),
            detected_by=detected_by,
        )
        session.add(row)
        session.flush()
        return row, True

    @staticmethod
    def _criterion_requires_evidence(criterion: str) -> bool:
        text = criterion.lower()
        mentions_evidence = any(
            token in text for token in ("证据", "来源", "source", "evidence")
        )
        negative_only = any(
            token in text
            for token in ("不得保存", "禁止保存", "无需保存", "不需要保存", "不得提供", "禁止提供")
        )
        return mentions_evidence and not negative_only

    @staticmethod
    def _resolve_missing_open_issues(
        session: Any, issue_type: str, active_keys: set[str]
    ) -> int:
        rows = session.scalars(
            select(WorkGovernanceIssue).where(
                WorkGovernanceIssue.issue_type == issue_type,
                WorkGovernanceIssue.status == "open",
            )
        ).all()
        resolved = 0
        for row in rows:
            if row.issue_key in active_keys:
                continue
            row.status = "resolved"
            row.resolved_at = utc_now()
            row.resolution_json = json_text(
                {"reason": "automatic_condition_cleared", "note": "自动复查时条件已消失"}
            )
            resolved += 1
        return resolved

    def scan(self, *, detected_by: str = "automatic") -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "created": 0, "resolved": 0, "open": 0}
        if not self._scan_lock.acquire(blocking=False):
            with self.database.sessions() as session:
                open_count = int(
                    session.scalar(
                        select(func.count(WorkGovernanceIssue.id)).where(
                            WorkGovernanceIssue.status == "open"
                        )
                    )
                    or 0
                )
            return {
                "enabled": True,
                "skipped": "scan_already_running",
                "created": 0,
                "resolved": 0,
                "open": open_count,
            }
        try:
            return self._scan_locked(detected_by=detected_by)
        finally:
            self._scan_lock.release()

    def _scan_locked(self, *, detected_by: str) -> dict[str, Any]:
        now = utc_now()
        created = 0
        resolved = 0
        counts = {"duplicate": 0, "stalled": 0, "quality_review": 0}
        with self.database.sessions() as session:
            threads = session.scalars(select(WorkThread).order_by(WorkThread.id)).all()
            roots = self._root_ids(threads)
            active = [
                thread for thread in threads if thread.status not in TERMINAL_THREAD_STATUSES
            ]
            duplicate_keys: set[str] = set()
            for index, left in enumerate(active):
                for right in active[index + 1 :]:
                    if roots.get(left.id) == roots.get(right.id):
                        continue
                    score = self._similarity(left, right)
                    if score < self.duplicate_similarity_threshold:
                        continue
                    first, second = sorted((left.id, right.id))
                    key = f"duplicate:{first}:{second}"
                    duplicate_keys.add(key)
                    _, was_created = self._upsert_issue(
                        session,
                        issue_key=key,
                        issue_type="duplicate",
                        work_thread_id=second,
                        related_thread_id=first,
                        summary=f"线程 {first} 与 {second} 可能重复，等待总控判断",
                        details={
                            "similarity": score,
                            "threshold": self.duplicate_similarity_threshold,
                            "suggestion": "比较目标与产出后决定保留、协调或分别继续；系统不会自动合并或取消。",
                        },
                        detected_by=detected_by,
                    )
                    created += int(was_created)
                    counts["duplicate"] += 1
            resolved += self._resolve_missing_open_issues(
                session, "duplicate", duplicate_keys
            )

            latest_runs: dict[int, WorkRun] = {}
            for run in session.scalars(select(WorkRun).order_by(WorkRun.id.desc())).all():
                latest_runs.setdefault(run.work_thread_id, run)
            stalled_keys: set[str] = set()
            cutoff = now - timedelta(minutes=self.stalled_minutes)
            for thread in active:
                run = latest_runs.get(thread.id)
                if not run or run.status not in OPEN_RUN_STATUSES or run.status == "waiting_user":
                    continue
                last_activity = max(
                    value
                    for value in (
                        as_utc(run.updated_at),
                        as_utc(run.heartbeat_at),
                        as_utc(thread.updated_at),
                    )
                    if value is not None
                )
                overdue_wait = (
                    run.status in {"waiting_retry", "waiting_schedule"}
                    and run.next_wake_at is not None
                    and as_utc(run.next_wake_at) <= cutoff
                )
                is_stalled = last_activity <= cutoff or overdue_wait
                if not is_stalled:
                    continue
                key = f"stalled:{thread.id}:{run.id}"
                stalled_keys.add(key)
                _, was_created = self._upsert_issue(
                    session,
                    issue_key=key,
                    issue_type="stalled",
                    work_thread_id=thread.id,
                    summary=f"线程 {thread.id} 已超过 {self.stalled_minutes} 分钟没有有效推进",
                    details={
                        "run_id": run.id,
                        "run_status": run.status,
                        "last_activity_at": last_activity.isoformat(),
                        "next_wake_at": as_utc(run.next_wake_at).isoformat() if run.next_wake_at else None,
                        "stalled_minutes": self.stalled_minutes,
                        "suggestion": "检查错误、等待条件、租约与负责人；治理扫描不会直接抢占或重启。",
                    },
                    detected_by=detected_by,
                )
                created += int(was_created)
                counts["stalled"] += 1
            resolved += self._resolve_missing_open_issues(
                session, "stalled", stalled_keys
            )

            if self.quality_review_enabled:
                baseline_row = session.get(
                    RuntimeJobState, "governance_quality_baseline_thread_id"
                )
                try:
                    quality_baseline = int(baseline_row.value) if baseline_row else 0
                except (TypeError, ValueError):
                    quality_baseline = 0
                completed = [
                    thread
                    for thread in threads
                    if thread.status == "completed" and thread.id > quality_baseline
                ]
                quality_keys: set[str] = set()
                for thread in completed:
                    key = f"quality_review:{thread.id}"
                    checkpoint = stored_json(
                        latest_runs.get(thread.id).checkpoint_json
                        if latest_runs.get(thread.id)
                        else None,
                        {},
                    )
                    last_action = checkpoint.get("last_action") or {}
                    criteria = stored_json(thread.success_criteria_json, [])
                    evidence_count = int(
                        session.scalar(
                            select(func.count(Evidence.id)).where(
                                Evidence.work_thread_id == thread.id
                            )
                        )
                        or 0
                    )
                    evidence_rows = session.scalars(
                        select(Evidence)
                        .where(Evidence.work_thread_id == thread.id)
                        .order_by(Evidence.id)
                        .limit(20)
                    ).all()
                    research_quality = ResearchQualityService.summarize_rows(
                        evidence_rows
                    )
                    latest_report = session.scalar(
                        select(ResearchReport)
                        .where(ResearchReport.work_thread_id == thread.id)
                        .order_by(ResearchReport.version.desc())
                        .limit(1)
                    )
                    flags: list[str] = []
                    if not (thread.result_text or "").strip():
                        flags.append("结果为空")
                    criteria_evidence = last_action.get("criteria_evidence") or []
                    if criteria and (
                        last_action.get("criteria_met") is not True
                        or len(criteria_evidence) != len(criteria)
                    ):
                        flags.append("成功标准自证不完整")
                    if (
                        any(self._criterion_requires_evidence(item) for item in criteria)
                        and evidence_count == 0
                    ):
                        flags.append("要求证据但证据库为空")
                    if not flags:
                        # Normal completion is not a governance issue.  The
                        # durable report and ledgers remain available for an
                        # explicit audit without filling the dashboard with
                        # automatic "waiting for review" noise.
                        continue
                    quality_keys.add(key)
                    _, was_created = self._upsert_issue(
                        session,
                        issue_key=key,
                        issue_type="quality_review",
                        work_thread_id=thread.id,
                        summary=(
                            f"线程 {thread.id} 存在质量风险：{'；'.join(flags)}"
                            if flags
                            else f"线程 {thread.id} 存在质量风险"
                        ),
                        details={
                            "flags": flags,
                            "success_criteria": criteria,
                            "criteria_evidence": criteria_evidence,
                            "evidence_count": evidence_count,
                            "research_quality": research_quality,
                            "evidence": [
                                {
                                    "id": evidence.id,
                                    "source_title": evidence.source_title,
                                    "source_url": evidence.source_url,
                                    "claim": evidence.claim,
                                    "confidence": evidence.confidence,
                                    "stance": evidence.stance,
                                    "source_type": evidence.source_type,
                                    "source_publisher": evidence.source_publisher,
                                    "claim_key": evidence.claim_key,
                                    "review_status": evidence.review_status,
                                }
                                for evidence in evidence_rows
                            ],
                            "report": (
                                {
                                    "id": latest_report.id,
                                    "version": latest_report.version,
                                    "preview": latest_report.markdown_content[:4000],
                                }
                                if latest_report
                                else None
                            ),
                            "result_preview": (thread.result_text or "")[:2000],
                        },
                        severity="warning",
                        detected_by=detected_by,
                    )
                    created += int(was_created)
                    counts["quality_review"] += 1
                resolved += self._resolve_missing_open_issues(
                    session, "quality_review", quality_keys
                )
            session.commit()
            open_count = int(
                session.scalar(
                    select(func.count(WorkGovernanceIssue.id)).where(
                        WorkGovernanceIssue.status == "open"
                    )
                )
                or 0
            )
        return {
            "enabled": True,
            "created": created,
            "resolved": resolved,
            "open": open_count,
            "detected": counts,
            "scanned_at": now.isoformat(),
            "detected_by": detected_by,
        }

    def list_issues(
        self,
        *,
        status: str | None = None,
        issue_type: str | None = None,
        limit: int = 100,
    ) -> list[WorkGovernanceIssue]:
        with self.database.sessions() as session:
            statement = select(WorkGovernanceIssue)
            if status:
                statement = statement.where(WorkGovernanceIssue.status == status)
            if issue_type:
                statement = statement.where(
                    WorkGovernanceIssue.issue_type == issue_type
                )
            return session.scalars(
                statement.order_by(
                    WorkGovernanceIssue.status,
                    WorkGovernanceIssue.updated_at.desc(),
                    WorkGovernanceIssue.id.desc(),
                ).limit(max(1, min(500, limit)))
            ).all()

    def review_quality(
        self,
        issue_id: int,
        *,
        reviewer_agent_id: int,
        decision: str,
        notes: str,
        target_agent_id: int | None = None,
    ) -> WorkGovernanceIssue:
        if decision not in {"pass", "rework", "reject"}:
            raise WorkConflict("质量复核决定必须是 pass、rework 或 reject")
        notes = notes.strip()
        if not notes:
            raise WorkConflict("质量复核必须填写可审计说明")
        with self.database.sessions() as session:
            self._controller(session, reviewer_agent_id)
            issue = session.get(WorkGovernanceIssue, issue_id)
            if not issue:
                raise WorkNotFound("治理记录不存在")
            if issue.issue_type != "quality_review":
                raise WorkConflict("该治理记录不是质量复核")
            if issue.status != "open":
                raise WorkConflict("该质量复核已经处理")
            original = session.get(WorkThread, issue.work_thread_id)
            if not original:
                raise WorkNotFound("原工作线程不存在")
            rework_thread: WorkThread | None = None
            if decision == "rework":
                owner_id = target_agent_id or original.owner_agent_id
                if not session.get(AgentProfile, owner_id):
                    raise WorkNotFound("返工负责人不存在")
                criteria = stored_json(original.success_criteria_json, [])
                constraints = stored_json(original.constraints_json, [])
                rework_thread = WorkThread(
                    owner_agent_id=owner_id,
                    project_id=original.project_id,
                    generation_number=original.generation_number,
                    parent_thread_id=original.id,
                    title=f"返工：{original.title}"[:200],
                    objective=(
                        f"对工作线程 {original.id} 的结果进行返工。\n"
                        f"原目标：{original.objective}\n复核意见：{notes}"
                    ),
                    success_criteria_json=json_text(criteria),
                    constraints_json=json_text(
                        [*constraints, f"保留原线程 {original.id} 的历史，不覆盖原结果", f"返工复核意见：{notes}"]
                    ),
                    status="queued",
                    priority=original.priority,
                )
                session.add(rework_thread)
                session.flush()
                run = WorkRun(
                    work_thread_id=rework_thread.id,
                    owner_agent_id=owner_id,
                    trigger_kind="governance_rework",
                    status="queued",
                    checkpoint_json=json_text(
                        {
                            "state_summary": "总控质量复核要求返工",
                            "progress": 0,
                            "governance_issue_id": issue.id,
                            "original_thread_id": original.id,
                            "review_notes": notes,
                        }
                    ),
                )
                session.add(run)
                session.flush()
                WorkRuntime._event(
                    session,
                    rework_thread.id,
                    "thread_created",
                    actor_id=f"controller:{reviewer_agent_id}",
                    payload={"owner_agent_id": owner_id, "priority": original.priority, "governance_issue_id": issue.id},
                )
                WorkRuntime._event(
                    session,
                    rework_thread.id,
                    "run_created",
                    run_id=run.id,
                    actor_id=f"controller:{reviewer_agent_id}",
                    payload={"trigger_kind": "governance_rework", "original_thread_id": original.id},
                )
                issue.rework_thread_id = rework_thread.id
                issue.status = "rework_created"
            else:
                issue.status = "resolved"
            issue.reviewer_agent_id = reviewer_agent_id
            issue.review_decision = decision
            issue.resolution_json = json_text(
                {
                    "decision": decision,
                    "notes": notes,
                    "target_agent_id": target_agent_id,
                    "original_thread_unchanged": True,
                    "rework_thread_id": rework_thread.id if rework_thread else None,
                }
            )
            issue.resolved_at = utc_now()
            issue.updated_at = utc_now()
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise WorkConflict("该质量复核已被其他操作处理") from exc
            session.refresh(issue)
            return issue

    def resolve_issue(
        self, issue_id: int, *, reviewer_agent_id: int, note: str
    ) -> WorkGovernanceIssue:
        note = note.strip()
        if not note:
            raise WorkConflict("处理治理记录必须填写说明")
        with self.database.sessions() as session:
            self._controller(session, reviewer_agent_id)
            issue = session.get(WorkGovernanceIssue, issue_id)
            if not issue:
                raise WorkNotFound("治理记录不存在")
            if issue.issue_type == "quality_review":
                raise WorkConflict("质量复核请使用 pass、rework 或 reject")
            if issue.status != "open":
                raise WorkConflict("治理记录已经处理")
            issue.status = "resolved"
            issue.reviewer_agent_id = reviewer_agent_id
            issue.resolved_at = utc_now()
            issue.resolution_json = json_text(
                {"reason": "controller_resolved", "note": note}
            )
            session.commit()
            session.refresh(issue)
            return issue

    def scan_tool(self, context: ToolContext, _: dict[str, Any]) -> dict[str, Any]:
        with self.database.sessions() as session:
            self._controller(session, context.owner_agent_id)
        return self.scan(detected_by=f"controller:{context.owner_agent_id}")

    def list_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        with self.database.sessions() as session:
            self._controller(session, context.owner_agent_id)
        rows = self.list_issues(
            status=arguments.get("status"),
            issue_type=arguments.get("issue_type"),
            limit=int(arguments.get("limit", 50)),
        )
        return {"count": len(rows), "items": [serialize_governance_issue(row) for row in rows]}

    def review_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        row = self.review_quality(
            int(arguments["issue_id"]),
            reviewer_agent_id=context.owner_agent_id,
            decision=str(arguments["decision"]),
            notes=str(arguments["notes"]),
            target_agent_id=arguments.get("target_agent_id"),
        )
        return serialize_governance_issue(row)

    def resolve_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        row = self.resolve_issue(
            int(arguments["issue_id"]),
            reviewer_agent_id=context.owner_agent_id,
            note=str(arguments["note"]),
        )
        return serialize_governance_issue(row)
