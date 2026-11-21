from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from .database import Database
from .models import AgentProfile, Evidence, WorkCollaboration, WorkEvent, WorkRun, WorkThread
from .tool_runtime import ToolContext
from .work_runtime import OPEN_RUN_STATUSES, TERMINAL_THREAD_STATUSES, WorkConflict, WorkNotFound


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stored_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def is_market_discovery_objective(value: Any) -> bool:
    text = str(value or "").lower()
    # Delegated objectives are often shortened to one side of the original
    # request (for example “持续热点研究” or “普通人兴趣观察”).  Requiring one
    # keyword from two separate groups silently dropped the discovery contract
    # on those children.  Any explicit discovery/market marker is sufficient;
    # verification wording is handled separately below.
    return any(
        marker in text
        for marker in (
            "热点", "热门", "话题", "趋势", "市场", "市场洞察",
            "生活", "消费", "爱好", "兴趣", "社交", "娱乐", "文化",
            "小众", "普通人", "产品设计", "market", "trend", "consumer",
            "social", "culture",
        )
    )


MARKET_DISCOVERY_SUCCESS_CRITERIA = (
    "发现并用中文整理真正有趣、具体且可用于后续判断的市场线索或洞察；数量由内容价值决定，不得凑数",
    "搜索标题和摘要只用于选择候选；最终保留的候选应读取可访问的代表正文。无法完整核验时标注不确定性，"
    "不得仅因缺少群体规模、持续时长、具体地点、第一人称、图片或独立双重来源而丢弃有趣线索",
)


def is_market_verification_objective(value: Any) -> bool:
    text = str(value or "").lower()
    return any(
        marker in text
        for marker in (
            "事实核验",
            "独立核验",
            "交叉验证",
            "审计",
            "查证",
            "fact check",
            "verification",
            "audit",
        )
    )


def normalize_market_success_criteria(
    objective: Any, success_criteria: list[Any]
) -> list[str]:
    """Keep delegated discovery aligned with the user's exploratory goal.

    The controller may narrow the subject in ``objective`` but may not silently
    turn ordinary discovery into a proof audit through stricter child criteria.
    Explicit verification tasks retain their supplied criteria.
    """
    normalized = [str(item).strip()[:2000] for item in success_criteria if str(item).strip()]
    if not is_market_discovery_objective(objective) or is_market_verification_objective(
        objective
    ):
        return normalized[:50]
    return list(MARKET_DISCOVERY_SUCCESS_CRITERIA)


class CollaborationService:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _parent_temporal_contract(
        session: Any, parent: WorkThread
    ) -> dict[str, Any] | None:
        parent_run = session.scalar(
            select(WorkRun)
            .where(WorkRun.work_thread_id == parent.id)
            .order_by(WorkRun.id.desc())
        )
        checkpoint = stored_json(parent_run.checkpoint_json, {}) if parent_run else {}
        contract = (
            checkpoint.get("temporal_contract")
            if isinstance(checkpoint, dict)
            else None
        )
        if not isinstance(contract, dict):
            return None
        inherited = json.loads(json.dumps(contract, ensure_ascii=False))
        inherited["inherited_from_thread_id"] = parent.id
        return inherited

    @staticmethod
    def _validate_values(values: dict[str, Any]) -> dict[str, Any]:
        title = str(values.get("title") or "").strip()
        objective = str(values.get("objective") or "").strip()
        if not title or not objective:
            raise WorkConflict("协作标题和目标不能为空")
        success = values.get("success_criteria") or []
        constraints = values.get("constraints") or []
        if not isinstance(success, list) or not isinstance(constraints, list):
            raise WorkConflict("成功标准和限制必须是数组")
        raw_evidence_ids = values.get("evidence_ids") or []
        if not isinstance(raw_evidence_ids, list):
            raise WorkConflict("委派证据ID必须是数组")
        evidence_ids = {
            int(item) for item in raw_evidence_ids[:50] if str(item).isdigit() and int(item) > 0
        }
        # "证据ID 11-16" is itself an explicit delegation scope. Convert only
        # that exact notation into IDs; ordinary numbers in the objective are
        # never interpreted as authorization.
        for start_text, end_text in re.findall(
            r"证据\s*ID\s*(\d+)\s*[-—至]\s*(\d+)", objective, flags=re.I
        ):
            start, end = int(start_text), int(end_text)
            if 0 < start <= end and end - start <= 49:
                evidence_ids.update(range(start, end + 1))
        normalized_success = normalize_market_success_criteria(objective, success)
        normalized_constraints = [str(item)[:2000] for item in constraints[:50]]
        if is_market_discovery_objective(objective):
            value_constraint = "不得设置机械数量、评分或分类配额；按内容是否真正有趣和可行动自主判断"
            if value_constraint not in normalized_constraints:
                normalized_constraints.append(value_constraint)
            exploration_constraint = (
                "这是市场探索而不是论文式证据审计；阅读全文用于理解选中的候选，"
                "证据不足时保留为已标注不确定性的线索，不得自行增加用户未要求的"
                "规模、时空、第一人称、图像或双重来源门槛"
            )
            if exploration_constraint not in normalized_constraints:
                normalized_constraints.append(exploration_constraint)
        return {
            "title": title[:200],
            "objective": objective[:20000],
            "success_criteria": normalized_success,
            "constraints": normalized_constraints[:50],
            "priority": max(-100, min(100, int(values.get("priority") or 0))),
            "wait_for_result": bool(values.get("wait_for_result", True)),
            "reason": (str(values.get("reason"))[:4000] if values.get("reason") else None),
            "evidence_ids": sorted(evidence_ids)[:50],
        }

    @staticmethod
    def _delegated_evidence_snapshot(
        session: Any, parent: WorkThread, evidence_ids: list[int]
    ) -> list[dict[str, Any]]:
        if not evidence_ids or not parent.project_id:
            return []
        project_thread_ids = select(WorkThread.id).where(
            WorkThread.project_id == parent.project_id
        )
        rows = session.scalars(
            select(Evidence)
            .where(
                Evidence.id.in_(evidence_ids),
                Evidence.work_thread_id.in_(project_thread_ids),
            )
            .order_by(Evidence.id)
        ).all()
        return [
            {
                "id": row.id,
                "work_thread_id": row.work_thread_id,
                "source_url": row.source_url,
                "source_title": row.source_title,
                "published_at": row.published_at,
                "claim": row.claim,
                "excerpt": row.excerpt,
                "confidence": row.confidence,
                "stance": row.stance,
                "source_type": row.source_type,
                "source_publisher": row.source_publisher,
                "claim_key": row.claim_key,
                "verification_notes": row.verification_notes,
            }
            for row in rows
        ]

    @staticmethod
    def _request_key(parent_thread_id: int, requester_agent_id: int, values: dict[str, Any]) -> str:
        explicit = str(values.get("request_key") or "").strip()
        if explicit:
            return sha256(f"explicit:{parent_thread_id}:{requester_agent_id}:{explicit}".encode()).hexdigest()
        stable = {
            "parent_thread_id": parent_thread_id,
            "requester_agent_id": requester_agent_id,
            "target_agent_id": values.get("target_agent_id"),
            "title": values["title"],
            "objective": values["objective"],
            "success_criteria": values["success_criteria"],
            "constraints": values["constraints"],
        }
        return sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _event(session: Any, thread_id: int, event_type: str, actor_id: int | None, payload: dict[str, Any]) -> None:
        session.add(
            WorkEvent(
                work_thread_id=thread_id,
                event_type=event_type,
                actor_kind="agent" if actor_id else "runtime",
                actor_id=str(actor_id) if actor_id else "collaboration-service",
                payload_json=json_text(payload),
            )
        )

    def list_agents_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            raise WorkConflict("list_company_agents不接受参数")
        with self.database.sessions() as session:
            agents = session.scalars(select(AgentProfile).order_by(AgentProfile.is_controller.desc(), AgentProfile.id)).all()
            running = {
                row[0] for row in session.execute(
                    select(WorkRun.owner_agent_id).where(WorkRun.status == "running")
                ).all()
            }
            return {
                "agents": [
                    {
                        "id": agent.id,
                        "name": agent.name,
                        "is_controller": agent.is_controller,
                        "role_responsibilities": agent.role_responsibilities,
                        "work_content": agent.work_content,
                        "public_status": "running" if agent.id in running else "available",
                    }
                    for agent in agents if agent.id != context.owner_agent_id
                ]
            }

    def delegate_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        target_agent_id = int(arguments.get("target_agent_id") or 0)
        return self.create_direct(
            parent_thread_id=context.work_thread_id,
            requester_agent_id=context.owner_agent_id,
            target_agent_id=target_agent_id,
            values=arguments,
        )

    def request_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        target = arguments.get("target_agent_id")
        return self.create_request(
            parent_thread_id=context.work_thread_id,
            requester_agent_id=context.owner_agent_id,
            target_agent_id=int(target) if target else None,
            values=arguments,
        )

    def transfer_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        thread_id = int(arguments.get("thread_id") or 0)
        target_agent_id = int(arguments.get("target_agent_id") or 0)
        row = self.transfer_thread(
            thread_id,
            actor_agent_id=context.owner_agent_id,
            target_agent_id=target_agent_id,
            reason=str(arguments.get("reason") or "Agent自主转移工作"),
        )
        return {"thread_id": row.id, "owner_agent_id": row.owner_agent_id, "status": row.status}

    def update_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        collaboration_id = int(arguments.get("collaboration_id") or 0)
        row = self.update_collaboration(
            collaboration_id,
            actor_agent_id=context.owner_agent_id,
            values=arguments,
        )
        return serialize_collaboration(row)

    def report_progress(
        self,
        child_thread_id: int,
        actor_agent_id: int,
        message: str,
        progress: float | None,
    ) -> dict[str, Any]:
        clean_message = str(message or "").strip()
        if not clean_message:
            raise WorkConflict("阶段成果不能为空")
        with self.database.sessions() as session:
            row = session.scalar(
                select(WorkCollaboration).where(
                    WorkCollaboration.child_thread_id == child_thread_id
                )
            )
            child = session.get(WorkThread, child_thread_id)
            if not row or not child:
                raise WorkNotFound("当前工作不是协作子任务")
            if child.owner_agent_id != actor_agent_id:
                raise WorkConflict("只有当前子任务负责人可以回传阶段成果")
            if row.status != "assigned" or child.status in TERMINAL_THREAD_STATUSES:
                raise WorkConflict("当前协作不能继续回传阶段成果")
            now = utc_now()
            update_payload = {
                "collaboration_id": row.id,
                "child_thread_id": child.id,
                "actor_agent_id": actor_agent_id,
                "progress": progress,
                "message": clean_message[:12000],
                "created_at": now.isoformat(),
            }
            row.result_text = update_payload["message"]
            row.updated_at = now
            parent = session.get(WorkThread, row.parent_thread_id)
            child.summary = update_payload["message"]
            child.updated_at = now
            if parent:
                parent.summary = (
                    f"员工阶段成果（协作 {row.id}）：{update_payload['message']}"
                )[:20000]
                parent.updated_at = now
            parent_run = session.scalar(
                select(WorkRun).where(
                    WorkRun.work_thread_id == row.parent_thread_id,
                    WorkRun.status.in_({"waiting_dependency", "waiting_schedule"}),
                )
            )
            parent_woken = False
            if parent and parent_run:
                checkpoint = stored_json(parent_run.checkpoint_json, {})
                if checkpoint.get("waiting_collaboration_id") == row.id:
                    updates = list(
                        checkpoint.get("collaboration_progress_updates") or []
                    )
                    updates.append(update_payload)
                    checkpoint["collaboration_progress_updates"] = updates[-20:]
                    checkpoint["latest_collaboration_progress"] = update_payload
                    parent_run.checkpoint_json = json_text(checkpoint)
                    parent_run.status = "queued"
                    parent_run.next_wake_at = now
                    parent_run.last_error = None
                    parent_run.updated_at = now
                    parent.status = "queued"
                    parent.next_wake_at = now
                    parent.updated_at = now
                    parent_woken = True
            self._event(
                session,
                row.parent_thread_id,
                "collaboration_progress",
                actor_agent_id,
                {**update_payload, "parent_woken": parent_woken},
            )
            self._event(
                session,
                child_thread_id,
                "progress_reported",
                actor_agent_id,
                {**update_payload, "parent_woken": parent_woken},
            )
            session.commit()
            return {
                **update_payload,
                "parent_thread_id": row.parent_thread_id,
                "parent_woken": parent_woken,
            }

    def resume_child_after_progress(
        self, collaboration_id: int, parent_agent_id: int
    ) -> bool:
        with self.database.sessions() as session:
            row = session.get(WorkCollaboration, collaboration_id)
            if not row or row.requester_agent_id != parent_agent_id:
                return False
            if row.status != "assigned" or not row.child_thread_id:
                return False
            run = session.scalar(
                select(WorkRun).where(
                    WorkRun.work_thread_id == row.child_thread_id,
                    WorkRun.status.in_({"waiting_dependency", "waiting_schedule"}),
                )
            )
            child = session.get(WorkThread, row.child_thread_id)
            if not run or not child:
                return False
            checkpoint = stored_json(run.checkpoint_json, {})
            if checkpoint.get("waiting_parent_progress_collaboration_id") != row.id:
                return False
            checkpoint.pop("waiting_parent_progress_collaboration_id", None)
            checkpoint["parent_progress_acknowledged_at"] = utc_now().isoformat()
            run.checkpoint_json = json_text(checkpoint)
            run.status = "queued"
            run.next_wake_at = utc_now()
            run.last_error = None
            run.updated_at = utc_now()
            child.status = "queued"
            child.next_wake_at = run.next_wake_at
            child.updated_at = run.updated_at
            self._event(
                session,
                child.id,
                "progress_acknowledged",
                parent_agent_id,
                {"collaboration_id": row.id, "parent_thread_id": row.parent_thread_id},
            )
            session.commit()
            return True

    def transfer_thread(
        self,
        thread_id: int,
        *,
        actor_agent_id: int,
        target_agent_id: int,
        reason: str,
    ) -> WorkThread:
        with self.database.sessions() as session:
            thread = session.get(WorkThread, thread_id)
            actor = session.get(AgentProfile, actor_agent_id)
            target = session.get(AgentProfile, target_agent_id)
            if not thread or not actor or not target:
                raise WorkNotFound("工作线程或Agent不存在")
            if thread.status in TERMINAL_THREAD_STATUSES:
                raise WorkConflict("已结束工作不能转移")
            if thread.owner_agent_id == target_agent_id:
                return thread
            running = session.scalar(
                select(WorkRun.id).where(
                    WorkRun.work_thread_id == thread_id, WorkRun.status == "running"
                )
            )
            if running:
                raise WorkConflict("运行中的工作不能强制抢占；请等待Checkpoint让出后再转移")
            previous_owner = thread.owner_agent_id
            thread.owner_agent_id = target_agent_id
            open_runs = session.scalars(
                select(WorkRun).where(
                    WorkRun.work_thread_id == thread_id,
                    WorkRun.status.in_(OPEN_RUN_STATUSES),
                )
            ).all()
            for run in open_runs:
                run.owner_agent_id = target_agent_id
            collaboration = session.scalar(
                select(WorkCollaboration).where(WorkCollaboration.child_thread_id == thread_id)
            )
            if collaboration:
                collaboration.target_agent_id = target_agent_id
                collaboration.updated_at = utc_now()
            self._event(
                session,
                thread_id,
                "thread_transferred",
                actor_agent_id,
                {
                    "previous_owner_agent_id": previous_owner,
                    "target_agent_id": target_agent_id,
                    "reason": reason[:4000],
                },
            )
            session.commit()
            session.refresh(thread)
            return thread

    def update_collaboration(
        self,
        collaboration_id: int,
        *,
        actor_agent_id: int,
        values: dict[str, Any],
    ) -> WorkCollaboration:
        with self.database.sessions() as session:
            row = session.get(WorkCollaboration, collaboration_id)
            actor = session.get(AgentProfile, actor_agent_id)
            if not row or not actor:
                raise WorkNotFound("协作记录或操作者不存在")
            if row.status in {"completed", "failed", "rejected", "cancelled"}:
                raise WorkConflict("已结束协作不能修改")
            child = session.get(WorkThread, row.child_thread_id) if row.child_thread_id else None
            if child:
                running = session.scalar(
                    select(WorkRun.id).where(
                        WorkRun.work_thread_id == child.id,
                        WorkRun.status == "running",
                    )
                )
                if running:
                    raise WorkConflict("运行中的协作不能修改；请等待Checkpoint让出")
            changes: dict[str, Any] = {}
            text_fields = {"title": 200, "objective": 20000, "reason": 4000}
            for name, limit in text_fields.items():
                if name in values and values[name] is not None:
                    value = str(values[name]).strip()[:limit]
                    if name != "reason" and not value:
                        raise WorkConflict("标题和目标不能为空")
                    setattr(row, name, value or None)
                    if child and name in {"title", "objective"}:
                        setattr(child, name, value)
                    changes[name] = value
            for name, column in (("success_criteria", "success_criteria_json"), ("constraints", "constraints_json")):
                if name in values and values[name] is not None:
                    raw = values[name]
                    if not isinstance(raw, list):
                        raise WorkConflict(f"{name}必须是数组")
                    objective = str(
                        values.get("objective")
                        or row.objective
                        or (child.objective if child else "")
                    )
                    normalized = [str(item)[:2000] for item in raw[:50]]
                    if name == "success_criteria":
                        normalized = normalize_market_success_criteria(
                            objective, normalized
                        )
                    setattr(row, column, json_text(normalized))
                    if child:
                        setattr(child, column, json_text(normalized))
                    changes[name] = normalized
            if "priority" in values and values["priority"] is not None:
                priority = max(-100, min(100, int(values["priority"])))
                row.priority = priority
                if child:
                    child.priority = priority
                changes["priority"] = priority
            if "wait_for_result" in values and values["wait_for_result"] is not None:
                row.wait_for_result = bool(values["wait_for_result"])
                changes["wait_for_result"] = row.wait_for_result
            if not changes:
                raise WorkConflict("没有可修改的协作字段")
            row.updated_at = utc_now()
            self._event(
                session,
                row.parent_thread_id,
                "collaboration_updated",
                actor_agent_id,
                {"collaboration_id": row.id, "changes": changes},
            )
            session.commit()
            session.refresh(row)
            return row

    def create_direct(self, *, parent_thread_id: int, requester_agent_id: int, target_agent_id: int, values: dict[str, Any]) -> dict[str, Any]:
        clean = self._validate_values(values)
        if target_agent_id <= 0 or target_agent_id == requester_agent_id:
            raise WorkConflict("直接委派必须选择另一名Agent")
        request_key = self._request_key(parent_thread_id, requester_agent_id, {**clean, "target_agent_id": target_agent_id, "request_key": values.get("request_key")})
        with self.database.sessions() as session:
            parent = session.get(WorkThread, parent_thread_id)
            requester = session.get(AgentProfile, requester_agent_id)
            target = session.get(AgentProfile, target_agent_id)
            if not parent or not requester or not target:
                raise WorkNotFound("父工作或协作Agent不存在")
            if parent.owner_agent_id != requester_agent_id:
                raise WorkConflict("只有当前工作负责人可以直接委派")
            if parent.status in TERMINAL_THREAD_STATUSES:
                raise WorkConflict("已结束工作不能继续委派")
            existing = session.scalar(select(WorkCollaboration).where(WorkCollaboration.parent_thread_id == parent_thread_id, WorkCollaboration.request_key == request_key))
            if existing:
                return {**serialize_collaboration(existing), "reused": True}
            return self._create_assigned(session, parent, requester_agent_id, target_agent_id, clean, request_key, channel="direct")

    def create_request(self, *, parent_thread_id: int, requester_agent_id: int, target_agent_id: int | None, values: dict[str, Any]) -> dict[str, Any]:
        clean = self._validate_values(values)
        request_key = self._request_key(parent_thread_id, requester_agent_id, {**clean, "target_agent_id": target_agent_id, "request_key": values.get("request_key")})
        with self.database.sessions() as session:
            parent = session.get(WorkThread, parent_thread_id)
            requester = session.get(AgentProfile, requester_agent_id)
            target = session.get(AgentProfile, target_agent_id) if target_agent_id else None
            if not parent or not requester or (target_agent_id and not target):
                raise WorkNotFound("父工作或协作Agent不存在")
            if parent.owner_agent_id != requester_agent_id:
                raise WorkConflict("只有当前工作负责人可以申请协作")
            if parent.status in TERMINAL_THREAD_STATUSES:
                raise WorkConflict("已结束工作不能申请协作")
            if target_agent_id == requester_agent_id:
                raise WorkConflict("协作目标不能是申请者自己")
            existing = session.scalar(select(WorkCollaboration).where(WorkCollaboration.parent_thread_id == parent_thread_id, WorkCollaboration.request_key == request_key))
            if existing:
                return {**serialize_collaboration(existing), "reused": True}
            row = WorkCollaboration(
                parent_thread_id=parent_thread_id, requester_agent_id=requester_agent_id,
                target_agent_id=target_agent_id, channel="request", status="pending_controller",
                request_key=request_key, title=clean["title"], objective=clean["objective"],
                success_criteria_json=json_text(clean["success_criteria"]), constraints_json=json_text(clean["constraints"]),
                priority=clean["priority"], wait_for_result=clean["wait_for_result"], reason=clean["reason"],
            )
            session.add(row)
            session.flush()
            self._event(session, parent_thread_id, "collaboration_requested", requester_agent_id, {"collaboration_id": row.id, "target_agent_id": target_agent_id})
            session.commit()
            session.refresh(row)
            return {**serialize_collaboration(row), "reused": False}

    def _create_assigned(self, session: Any, parent: WorkThread, requester_agent_id: int, target_agent_id: int, clean: dict[str, Any], request_key: str, *, channel: str, decided_by_agent_id: int | None = None) -> dict[str, Any]:
        requester = session.get(AgentProfile, requester_agent_id)
        if requester and requester.is_controller and parent.parent_thread_id is None:
            # The root controller owns orchestration, not execution.  A model may
            # accidentally send wait_for_result=false; accepting it leaves the
            # controller running administrative turns while the employee works.
            # Root delegation therefore always transfers execution ownership
            # until the collaboration emits progress or a terminal event.
            clean = dict(clean)
            clean["wait_for_result"] = True
        evidence_snapshot = self._delegated_evidence_snapshot(
            session, parent, clean.get("evidence_ids") or []
        )
        child = WorkThread(
            owner_agent_id=target_agent_id, project_id=parent.project_id,
            generation_number=parent.generation_number, parent_thread_id=parent.id,
            title=clean["title"], objective=clean["objective"],
            success_criteria_json=json_text(clean["success_criteria"]), constraints_json=json_text(clean["constraints"]),
            priority=clean["priority"], status="queued",
        )
        session.add(child)
        session.flush()
        child_checkpoint = {
            "delegation_context": {
                "parent_thread_id": parent.id,
                "evidence_ids": clean.get("evidence_ids") or [],
                "evidence_snapshot": evidence_snapshot,
                "missing_evidence_ids": sorted(
                    set(clean.get("evidence_ids") or [])
                    - {item["id"] for item in evidence_snapshot}
                ),
            }
        }
        temporal_contract = self._parent_temporal_contract(session, parent)
        if temporal_contract:
            child_checkpoint["temporal_contract"] = temporal_contract
        run = WorkRun(
            work_thread_id=child.id,
            owner_agent_id=target_agent_id,
            trigger_kind="delegation",
            status="queued",
            checkpoint_json=json_text(child_checkpoint),
        )
        session.add(run)
        session.flush()
        row = WorkCollaboration(
            parent_thread_id=parent.id, child_thread_id=child.id,
            requester_agent_id=requester_agent_id, target_agent_id=target_agent_id,
            decided_by_agent_id=decided_by_agent_id, channel=channel, status="assigned",
            request_key=request_key, title=clean["title"], objective=clean["objective"],
            success_criteria_json=json_text(clean["success_criteria"]), constraints_json=json_text(clean["constraints"]),
            priority=clean["priority"], wait_for_result=clean["wait_for_result"], reason=clean["reason"],
            decided_at=utc_now() if decided_by_agent_id else None,
        )
        session.add(row)
        session.flush()
        self._event(session, parent.id, "collaboration_assigned", requester_agent_id, {"collaboration_id": row.id, "child_thread_id": child.id, "target_agent_id": target_agent_id, "channel": channel})
        self._event(session, child.id, "thread_created", requester_agent_id, {"owner_agent_id": target_agent_id, "parent_thread_id": parent.id, "collaboration_id": row.id})
        self._event(session, child.id, "run_created", requester_agent_id, {"trigger_kind": "delegation", "collaboration_id": row.id})
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(select(WorkCollaboration).where(WorkCollaboration.parent_thread_id == parent.id, WorkCollaboration.request_key == request_key))
            if existing:
                return {**serialize_collaboration(existing), "reused": True}
            raise
        session.refresh(row)
        return {**serialize_collaboration(row), "reused": False}

    def decide(self, collaboration_id: int, *, controller_agent_id: int, approved: bool, target_agent_id: int | None, reason: str | None) -> WorkCollaboration:
        with self.database.sessions() as session:
            controller = session.get(AgentProfile, controller_agent_id)
            row = session.get(WorkCollaboration, collaboration_id)
            if not controller or not controller.is_controller:
                raise WorkConflict("只有总控Agent可以处理协作申请")
            if not row:
                raise WorkNotFound("协作申请不存在")
            if row.status != "pending_controller":
                raise WorkConflict("协作申请已处理")
            assigned_agent_id = target_agent_id or row.target_agent_id
            if approved and (
                not assigned_agent_id
                or assigned_agent_id == row.requester_agent_id
                or not session.get(AgentProfile, assigned_agent_id)
            ):
                raise WorkConflict("批准协作必须指定有效的其他Agent")
            reservation = session.execute(
                update(WorkCollaboration)
                .where(
                    WorkCollaboration.id == collaboration_id,
                    WorkCollaboration.status == "pending_controller",
                )
                .values(status="deciding")
            )
            if reservation.rowcount != 1:
                session.rollback()
                raise WorkConflict("协作申请已处理")
            session.flush()
            session.refresh(row)
            if not approved:
                row.status = "rejected"
                row.decided_by_agent_id = controller_agent_id
                row.decided_at = utc_now()
                row.reason = reason or row.reason
                self._wake_parent_if_waiting(
                    session, row, "rejected", reason or "总控拒绝了协作申请"
                )
                self._event(session, row.parent_thread_id, "collaboration_rejected", controller_agent_id, {"collaboration_id": row.id, "reason": reason})
                session.commit(); session.refresh(row); return row
            assert assigned_agent_id is not None
            parent = session.get(WorkThread, row.parent_thread_id)
            assert parent is not None
            clean = self._validate_values(
                {
                    "title": row.title,
                    "objective": row.objective,
                    "success_criteria": stored_json(row.success_criteria_json, []),
                    "constraints": stored_json(row.constraints_json, []),
                    "priority": row.priority,
                    "wait_for_result": row.wait_for_result,
                    "reason": reason or row.reason,
                }
            )
            evidence_snapshot = self._delegated_evidence_snapshot(
                session, parent, clean.get("evidence_ids") or []
            )
            child = WorkThread(owner_agent_id=assigned_agent_id, project_id=parent.project_id, generation_number=parent.generation_number, parent_thread_id=parent.id, title=row.title, objective=row.objective, success_criteria_json=row.success_criteria_json, constraints_json=row.constraints_json, priority=row.priority, status="queued")
            session.add(child); session.flush()
            child_checkpoint = {
                "delegation_context": {
                    "parent_thread_id": parent.id,
                    "evidence_ids": clean.get("evidence_ids") or [],
                    "evidence_snapshot": evidence_snapshot,
                    "missing_evidence_ids": sorted(
                        set(clean.get("evidence_ids") or [])
                        - {item["id"] for item in evidence_snapshot}
                    ),
                }
            }
            temporal_contract = self._parent_temporal_contract(session, parent)
            if temporal_contract:
                child_checkpoint["temporal_contract"] = temporal_contract
            run = WorkRun(
                work_thread_id=child.id,
                owner_agent_id=assigned_agent_id,
                trigger_kind="delegation_approved",
                status="queued",
                checkpoint_json=json_text(child_checkpoint),
            )
            session.add(run); session.flush()
            row.child_thread_id=child.id; row.target_agent_id=assigned_agent_id; row.decided_by_agent_id=controller_agent_id; row.status="assigned"; row.decided_at=utc_now(); row.reason=clean["reason"]
            self._event(session, parent.id, "collaboration_approved", controller_agent_id, {"collaboration_id": row.id, "child_thread_id": child.id, "target_agent_id": assigned_agent_id})
            self._event(session, child.id, "thread_created", controller_agent_id, {"owner_agent_id": assigned_agent_id, "parent_thread_id": parent.id, "collaboration_id": row.id})
            self._event(session, child.id, "run_created", controller_agent_id, {"trigger_kind": "delegation_approved", "collaboration_id": row.id})
            session.commit(); session.refresh(row); return row

    def sync_child_completion(self, child_thread_id: int) -> None:
        with self.database.sessions() as session:
            row = session.scalar(select(WorkCollaboration).where(WorkCollaboration.child_thread_id == child_thread_id))
            child = session.get(WorkThread, child_thread_id)
            if not row or not child or child.status not in TERMINAL_THREAD_STATUSES or row.status in {"completed", "failed", "rejected", "cancelled"}:
                return
            row.status = "completed" if child.status == "completed" else "failed"
            row.result_text = child.result_text
            row.completed_at = utc_now()
            parent = session.get(WorkThread, row.parent_thread_id)
            self._wake_parent_if_waiting(session, row, row.status, row.result_text)
            self._event(session, parent.id if parent else child.id, "collaboration_completed", child.owner_agent_id, {"collaboration_id": row.id, "child_thread_id": child.id, "status": row.status})
            session.commit()

    def reconcile_tree(self, root_thread_id: int) -> int:
        """Repair terminal child/collaboration pairs after crashes or old runs."""
        with self.database.sessions() as session:
            root = session.get(WorkThread, root_thread_id)
            if not root:
                raise WorkNotFound("工作线程不存在")
            tree_ids = [root.id]
            frontier = [root.id]
            while frontier:
                children = session.scalars(
                    select(WorkThread.id).where(
                        WorkThread.parent_thread_id.in_(frontier)
                    )
                ).all()
                frontier = [item for item in children if item not in tree_ids]
                tree_ids.extend(frontier)
            rows = session.scalars(
                select(WorkCollaboration).where(
                    WorkCollaboration.parent_thread_id.in_(tree_ids),
                    WorkCollaboration.status.in_(
                        {"assigned", "pending_controller", "deciding"}
                    ),
                )
            ).all()
            repaired = 0
            for row in rows:
                child = (
                    session.get(WorkThread, row.child_thread_id)
                    if row.child_thread_id
                    else None
                )
                if child and child.status == "waiting_schedule":
                    open_run = session.scalar(
                        select(WorkRun).where(
                            WorkRun.work_thread_id == child.id,
                            WorkRun.status == "waiting_schedule",
                        )
                    )
                    checkpoint = (
                        stored_json(open_run.checkpoint_json, {})
                        if open_run
                        else {}
                    )
                    last_action = checkpoint.get("last_action") or {}
                    if open_run and last_action.get("type") == "report_progress":
                        open_run.status = "waiting_dependency"
                        open_run.next_wake_at = None
                        open_run.last_error = "阶段成果已回传，等待总控Agent处理"
                        checkpoint["next_wake_at"] = None
                        checkpoint["thread_status"] = "waiting_dependency"
                        open_run.checkpoint_json = json_text(checkpoint)
                        child.status = "waiting_dependency"
                        child.next_wake_at = None
                        child.updated_at = utc_now()
                        self._event(
                            session,
                            child.id,
                            "collaboration_wait_migrated",
                            child.owner_agent_id,
                            {
                                "collaboration_id": row.id,
                                "from_status": "waiting_schedule",
                                "to_status": "waiting_dependency",
                            },
                        )
                        repaired += 1
                if not child or child.status not in TERMINAL_THREAD_STATUSES:
                    continue
                row.status = "completed" if child.status == "completed" else "failed"
                row.result_text = child.result_text or child.summary
                row.completed_at = row.completed_at or utc_now()
                row.updated_at = utc_now()
                self._wake_parent_if_waiting(
                    session, row, row.status, row.result_text
                )
                self._event(
                    session,
                    row.parent_thread_id,
                    "collaboration_reconciled",
                    child.owner_agent_id,
                    {
                        "collaboration_id": row.id,
                        "child_thread_id": child.id,
                        "status": row.status,
                    },
                )
                repaired += 1
            session.commit()
            return repaired

    def _wake_parent_if_waiting(
        self,
        session: Any,
        row: WorkCollaboration,
        status: str,
        result_text: str | None,
    ) -> None:
        parent = session.get(WorkThread, row.parent_thread_id)
        if not parent or parent.status not in {"waiting_dependency", "waiting_schedule"}:
            return
        open_run = session.scalar(
            select(WorkRun).where(
                WorkRun.work_thread_id == parent.id,
                WorkRun.status.in_({"waiting_dependency", "waiting_schedule"}),
            )
        )
        if not open_run:
            return
        checkpoint = stored_json(open_run.checkpoint_json, {})
        if checkpoint.get("waiting_collaboration_id") != row.id:
            return
        results = list(checkpoint.get("collaboration_results") or [])
        results.append(
            {
                "collaboration_id": row.id,
                "child_thread_id": row.child_thread_id,
                "status": status,
                "result_text": result_text,
            }
        )
        checkpoint["collaboration_results"] = results[-10:]
        checkpoint["waiting_collaboration_id"] = None
        open_run.checkpoint_json = json_text(checkpoint)
        open_run.status = "queued"
        open_run.next_wake_at = utc_now()
        open_run.last_error = None
        parent.status = "queued"
        parent.next_wake_at = utc_now()

    def list(self, *, parent_thread_id: int | None = None, status: str | None = None) -> list[WorkCollaboration]:
        with self.database.sessions() as session:
            statement = select(WorkCollaboration)
            if parent_thread_id is not None: statement = statement.where(WorkCollaboration.parent_thread_id == parent_thread_id)
            if status: statement = statement.where(WorkCollaboration.status == status)
            return session.scalars(statement.order_by(WorkCollaboration.id.desc())).all()


def serialize_collaboration(row: WorkCollaboration) -> dict[str, Any]:
    return {
        "id": row.id, "parent_thread_id": row.parent_thread_id, "child_thread_id": row.child_thread_id,
        "requester_agent_id": row.requester_agent_id, "target_agent_id": row.target_agent_id,
        "decided_by_agent_id": row.decided_by_agent_id, "channel": row.channel, "status": row.status,
        "request_key": row.request_key, "title": row.title, "objective": row.objective,
        "success_criteria": stored_json(row.success_criteria_json, []), "constraints": stored_json(row.constraints_json, []),
        "priority": row.priority, "wait_for_result": row.wait_for_result, "reason": row.reason,
        "result_text": row.result_text, "created_at": row.created_at.isoformat(),
        "decided_at": row.decided_at.isoformat() if row.decided_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "updated_at": row.updated_at.isoformat(),
    }
