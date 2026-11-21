from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any
from uuid import uuid4

from sqlalchemy import case, or_, select, update
from sqlalchemy.exc import IntegrityError

from .database import Database
from .models import (
    AgentProfile,
    ResearchCandidate,
    ToolCall,
    WorkCollaboration,
    WorkEvent,
    WorkProject,
    WorkRun,
    WorkThread,
)


OPEN_RUN_STATUSES = {
    "queued",
    "running",
    "waiting_retry",
    "waiting_schedule",
    "waiting_dependency",
    "waiting_user",
}
RUNNABLE_STATUSES = {"queued", "waiting_retry", "waiting_schedule"}
PAUSE_STATUSES = {
    "queued",
    "waiting_retry",
    "waiting_schedule",
    "waiting_dependency",
    "waiting_user",
}
TERMINAL_THREAD_STATUSES = {"completed", "failed", "cancelled", "rolled_over"}
TERMINAL_PROJECT_STATUSES = {"completed", "failed", "cancelled"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class WorkRuntimeError(RuntimeError):
    pass


class WorkNotFound(WorkRuntimeError):
    pass


class WorkConflict(WorkRuntimeError):
    pass


class WorkLeaseLost(WorkConflict):
    pass


class WorkRuntime:
    def __init__(
        self,
        database: Database,
        *,
        lease_seconds: int = 120,
        instance_id: str | None = None,
    ) -> None:
        self.database = database
        self.lease_seconds = max(10, int(lease_seconds))
        self.instance_id = instance_id or f"runtime-{uuid4().hex}"

    def _lease_deadline(self, now: datetime) -> datetime:
        return now + timedelta(seconds=self.lease_seconds)

    @staticmethod
    def _event(
        session: Any,
        thread_id: int,
        event_type: str,
        *,
        run_id: int | None = None,
        actor_id: str | None = None,
        execution_token: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            WorkEvent(
                work_thread_id=thread_id,
                work_run_id=run_id,
                event_type=event_type,
                actor_id=actor_id,
                execution_token=execution_token,
                payload_json=json_text(payload or {}),
            )
        )

    def create_thread(
        self,
        *,
        owner_agent_id: int,
        title: str,
        objective: str,
        success_criteria: list[str] | None = None,
        constraints: list[str] | None = None,
        priority: int = 0,
        parent_thread_id: int | None = None,
        continuation_of_thread_id: int | None = None,
        project_id: int | None = None,
        generation_number: int | None = None,
    ) -> WorkThread:
        title = title.strip()
        objective = objective.strip()
        if not title or not objective:
            raise WorkConflict("工作标题和目标不能为空")
        with self.database.sessions() as session:
            if not session.get(AgentProfile, owner_agent_id):
                raise WorkNotFound("负责Agent不存在")
            if parent_thread_id and continuation_of_thread_id:
                raise WorkConflict("续作必须创建为新的根线程，不能同时指定上级线程")
            parent = session.get(WorkThread, parent_thread_id) if parent_thread_id else None
            if parent_thread_id and not parent:
                raise WorkNotFound("上级工作线程不存在")
            continuation_source = None
            if continuation_of_thread_id:
                continuation_source = session.get(WorkThread, continuation_of_thread_id)
                if not continuation_source:
                    raise WorkNotFound("被续作的工作线程不存在")
                if continuation_source.parent_thread_id is not None:
                    raise WorkConflict("续作来源必须是项目根线程")
                if continuation_source.status not in TERMINAL_THREAD_STATUSES:
                    raise WorkConflict("旧任务尚未结束，不能并行创建续作")
                active_successor = session.scalar(
                    select(WorkThread.id).where(
                        WorkThread.continuation_of_thread_id == continuation_source.id,
                        ~WorkThread.status.in_(TERMINAL_THREAD_STATUSES),
                    )
                )
                if active_successor:
                    raise WorkConflict(f"该任务已有未结束的续作线程 #{active_successor}")
            inherited_project_id = (
                parent.project_id
                if parent
                else continuation_source.project_id
                if continuation_source
                else None
            )
            inherited_generation = (
                parent.generation_number
                if parent
                else continuation_source.generation_number + 1
                if continuation_source
                else None
            )
            if project_id and inherited_project_id and project_id != inherited_project_id:
                raise WorkConflict("子任务或续作线程必须归属原项目")
            project = session.get(WorkProject, project_id or inherited_project_id) if (project_id or inherited_project_id) else None
            if (project_id or inherited_project_id) and not project:
                raise WorkNotFound("工作项目不存在")
            if project is None:
                project = WorkProject(
                    owner_agent_id=owner_agent_id,
                    title=title,
                    objective=objective,
                    success_criteria_json=json_text(success_criteria or []),
                    constraints_json=json_text(constraints or []),
                    status="running",
                    generation_count=1,
                )
                session.add(project)
                session.flush()
            resolved_generation = max(
                1, int(generation_number or inherited_generation or 1)
            )
            thread = WorkThread(
                owner_agent_id=owner_agent_id,
                project_id=project.id,
                generation_number=resolved_generation,
                parent_thread_id=parent_thread_id,
                continuation_of_thread_id=continuation_of_thread_id,
                title=title,
                objective=objective,
                success_criteria_json=json_text(success_criteria or []),
                constraints_json=json_text(constraints or []),
                priority=priority,
                status="queued",
            )
            session.add(thread)
            session.flush()
            if parent_thread_id is None:
                project.current_root_thread_id = thread.id
                project.generation_count = max(
                    int(project.generation_count or 1), resolved_generation
                )
                project.status = "running"
                project.finished_at = None
                project.updated_at = utc_now()
            if continuation_source:
                continuation_source.updated_at = utc_now()
                self._event(
                    session,
                    continuation_source.id,
                    "thread_continued",
                    actor_id=self.instance_id,
                    payload={"successor_thread_id": thread.id},
                )
            self._event(
                session,
                thread.id,
                "thread_created",
                actor_id=self.instance_id,
                payload={
                    "owner_agent_id": owner_agent_id,
                    "priority": priority,
                    "continuation_of_thread_id": continuation_of_thread_id,
                    "project_id": project.id,
                    "generation_number": resolved_generation,
                },
            )
            session.commit()
            session.refresh(thread)
            return thread

    def create_run(
        self,
        thread_id: int,
        *,
        trigger_kind: str = "manual",
        checkpoint: dict[str, Any] | None = None,
        next_wake_at: datetime | None = None,
    ) -> WorkRun:
        now = utc_now()
        next_wake_at = normalize_utc(next_wake_at)
        initial_status = (
            "waiting_schedule"
            if next_wake_at is not None and next_wake_at > now
            else "queued"
        )
        with self.database.sessions() as session:
            thread = session.get(WorkThread, thread_id)
            if not thread:
                raise WorkNotFound("工作线程不存在")
            if thread.status in TERMINAL_THREAD_STATUSES:
                raise WorkConflict("已结束的工作线程不能创建运行")
            run = WorkRun(
                work_thread_id=thread.id,
                owner_agent_id=thread.owner_agent_id,
                trigger_kind=trigger_kind,
                status=initial_status,
                checkpoint_json=json_text(checkpoint) if checkpoint is not None else None,
                next_wake_at=next_wake_at,
            )
            session.add(run)
            try:
                session.flush()
            except IntegrityError as exc:
                session.rollback()
                raise WorkConflict("工作线程已有未结束的运行") from exc
            thread.status = initial_status
            thread.next_wake_at = next_wake_at
            thread.updated_at = utc_now()
            self._event(
                session,
                thread.id,
                "run_created",
                run_id=run.id,
                actor_id=self.instance_id,
                payload={"trigger_kind": trigger_kind},
            )
            session.commit()
            session.refresh(run)
            return run

    def reopen_thread(
        self,
        thread_id: int,
        *,
        trigger_kind: str = "context_recovery",
        checkpoint: dict[str, Any] | None = None,
    ) -> WorkRun:
        """Continue a logical thread with a new Run without creating a successor thread."""
        now = utc_now()
        with self.database.sessions() as session:
            thread = session.get(WorkThread, thread_id)
            if not thread:
                raise WorkNotFound("工作线程不存在")
            if thread.parent_thread_id is not None:
                raise WorkConflict("只有项目根线程可以执行项目级恢复")
            open_run = session.scalar(
                select(WorkRun)
                .where(
                    WorkRun.work_thread_id == thread.id,
                    WorkRun.status.in_(OPEN_RUN_STATUSES),
                )
                .order_by(WorkRun.id.desc())
            )
            if open_run:
                return open_run
            if thread.status not in TERMINAL_THREAD_STATUSES:
                raise WorkConflict("工作线程尚未结束且没有活动Run，不能按终态恢复")
            run = WorkRun(
                work_thread_id=thread.id,
                owner_agent_id=thread.owner_agent_id,
                trigger_kind=trigger_kind,
                status="queued",
                checkpoint_json=json_text(checkpoint or {}),
                next_wake_at=now,
            )
            session.add(run)
            session.flush()
            previous_status = thread.status
            thread.status = "queued"
            thread.result_text = None
            thread.finished_at = None
            thread.next_wake_at = now
            thread.updated_at = now
            project = (
                session.get(WorkProject, thread.project_id)
                if thread.project_id
                else None
            )
            if project:
                project.status = "running"
                project.current_root_thread_id = thread.id
                project.result_text = None
                project.finished_at = None
                project.updated_at = now
            self._event(
                session,
                thread.id,
                "thread_reopened",
                run_id=run.id,
                actor_id=self.instance_id,
                payload={
                    "previous_status": previous_status,
                    "trigger_kind": trigger_kind,
                },
            )
            self._event(
                session,
                thread.id,
                "run_created",
                run_id=run.id,
                actor_id=self.instance_id,
                payload={"trigger_kind": trigger_kind},
            )
            session.commit()
            session.refresh(run)
            return run

    def claim_next(self, *, agent_id: int | None = None) -> WorkRun | None:
        reclaimed = self._reclaim_expired(agent_id=agent_id)
        if reclaimed:
            return reclaimed
        for _ in range(5):
            now = utc_now()
            with self.database.sessions() as session:
                statement = (
                    select(WorkRun)
                    .join(WorkThread, WorkThread.id == WorkRun.work_thread_id)
                    .where(
                        WorkRun.status.in_(RUNNABLE_STATUSES),
                        or_(WorkRun.next_wake_at.is_(None), WorkRun.next_wake_at <= now),
                        ~WorkThread.status.in_(TERMINAL_THREAD_STATUSES),
                    )
                    .order_by(
                        WorkThread.priority.desc(),
                        case(
                            (WorkRun.status == "waiting_retry", 0),
                            (WorkRun.status == "queued", 1),
                            else_=2,
                        ),
                        WorkRun.next_wake_at,
                        WorkRun.created_at,
                        WorkRun.id,
                    )
                    .limit(25)
                )
                if agent_id is not None:
                    statement = statement.where(WorkRun.owner_agent_id == agent_id)
                candidates = session.scalars(statement).all()
                if not candidates:
                    return None
                for candidate in candidates:
                    active = session.scalar(
                        select(WorkRun.id).where(
                            WorkRun.owner_agent_id == candidate.owner_agent_id,
                            WorkRun.status == "running",
                        )
                    )
                    if active:
                        continue
                    token = candidate.execution_token + 1
                    result = session.execute(
                        update(WorkRun)
                        .where(
                            WorkRun.id == candidate.id,
                            WorkRun.status == candidate.status,
                            WorkRun.execution_token == candidate.execution_token,
                        )
                        .values(
                            status="running",
                            lease_owner=self.instance_id,
                            lease_expires_at=self._lease_deadline(now),
                            heartbeat_at=now,
                            execution_token=token,
                            started_at=candidate.started_at or now,
                            next_wake_at=None,
                            last_error=None,
                            updated_at=now,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    if result.rowcount != 1:
                        session.rollback()
                        break
                    thread = session.get(WorkThread, candidate.work_thread_id)
                    assert thread is not None
                    thread.status = "running"
                    thread.next_wake_at = None
                    thread.updated_at = now
                    self._event(
                        session,
                        thread.id,
                        "run_claimed",
                        run_id=candidate.id,
                        actor_id=self.instance_id,
                        execution_token=token,
                    )
                    try:
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                        break
                    session.expire_all()
                    claimed = session.get(WorkRun, candidate.id)
                    assert claimed is not None
                    return claimed
                else:
                    return None
        return None

    def _reclaim_expired(self, *, agent_id: int | None = None) -> WorkRun | None:
        now = utc_now()
        with self.database.sessions() as session:
            statement = (
                select(WorkRun)
                .join(WorkThread, WorkThread.id == WorkRun.work_thread_id)
                .where(
                    WorkRun.status == "running",
                    or_(
                        WorkRun.lease_expires_at.is_(None),
                        WorkRun.lease_expires_at <= now,
                    ),
                    ~WorkThread.status.in_(TERMINAL_THREAD_STATUSES),
                )
                .order_by(WorkThread.priority.desc(), WorkRun.created_at, WorkRun.id)
            )
            if agent_id is not None:
                statement = statement.where(WorkRun.owner_agent_id == agent_id)
            candidate = session.scalar(statement.limit(1))
            if not candidate:
                return None
            previous_owner = candidate.lease_owner
            token = candidate.execution_token + 1
            result = session.execute(
                update(WorkRun)
                .where(
                    WorkRun.id == candidate.id,
                    WorkRun.status == "running",
                    WorkRun.execution_token == candidate.execution_token,
                    or_(
                        WorkRun.lease_expires_at.is_(None),
                        WorkRun.lease_expires_at <= now,
                    ),
                )
                .values(
                    lease_owner=self.instance_id,
                    lease_expires_at=self._lease_deadline(now),
                    heartbeat_at=now,
                    execution_token=token,
                    last_error=None,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                session.rollback()
                return None
            self._event(
                session,
                candidate.work_thread_id,
                "run_reclaimed",
                run_id=candidate.id,
                actor_id=self.instance_id,
                execution_token=token,
                payload={"previous_owner": previous_owner},
            )
            session.commit()
            session.expire_all()
            reclaimed = session.get(WorkRun, candidate.id)
            assert reclaimed is not None
            return reclaimed

    def renew(self, run_id: int, execution_token: int) -> WorkRun:
        now = utc_now()
        with self.database.sessions() as session:
            run = self._owned_run(session, run_id, execution_token, now)
            run.heartbeat_at = now
            run.lease_expires_at = self._lease_deadline(now)
            run.updated_at = now
            self._event(
                session,
                run.work_thread_id,
                "lease_renewed",
                run_id=run.id,
                actor_id=self.instance_id,
                execution_token=execution_token,
            )
            session.commit()
            session.refresh(run)
            return run

    def checkpoint(
        self, run_id: int, execution_token: int, checkpoint: dict[str, Any]
    ) -> WorkRun:
        now = utc_now()
        with self.database.sessions() as session:
            run = self._owned_run(session, run_id, execution_token, now)
            run.checkpoint_json = json_text(checkpoint)
            run.updated_at = now
            self._event(
                session,
                run.work_thread_id,
                "checkpoint_saved",
                run_id=run.id,
                actor_id=self.instance_id,
                execution_token=execution_token,
            )
            session.commit()
            session.refresh(run)
            return run

    def pause(
        self,
        run_id: int,
        execution_token: int,
        *,
        status: str,
        next_wake_at: datetime | None = None,
        error: str | None = None,
    ) -> WorkRun:
        if status not in PAUSE_STATUSES:
            raise WorkConflict("无效的暂停状态")
        next_wake_at = normalize_utc(next_wake_at)
        if status in {"waiting_retry", "waiting_schedule"} and not next_wake_at:
            raise WorkConflict("等待重试或定时唤醒必须设置时间")
        if status == "waiting_dependency" and next_wake_at is not None:
            raise WorkConflict("依赖等待由事件唤醒，不能设置定时唤醒")
        now = utc_now()
        with self.database.sessions() as session:
            run = self._owned_run(session, run_id, execution_token, now)
            run.status = status
            run.next_wake_at = next_wake_at
            run.last_error = error
            run.lease_owner = None
            run.lease_expires_at = None
            run.heartbeat_at = None
            run.updated_at = now
            thread = session.get(WorkThread, run.work_thread_id)
            assert thread is not None
            thread.status = status
            thread.next_wake_at = next_wake_at
            thread.updated_at = now
            self._event(
                session,
                run.work_thread_id,
                "run_paused",
                run_id=run.id,
                actor_id=self.instance_id,
                execution_token=execution_token,
                payload={"status": status, "error": error},
            )
            session.commit()
            session.refresh(run)
            return run

    def resume(self, run_id: int) -> WorkRun:
        now = utc_now()
        with self.database.sessions() as session:
            run = session.get(WorkRun, run_id)
            if not run:
                raise WorkNotFound("工作运行不存在")
            if run.status not in PAUSE_STATUSES - {"queued"}:
                raise WorkConflict("当前运行不处于可恢复状态")
            run.status = "queued"
            run.next_wake_at = now
            run.updated_at = now
            thread = session.get(WorkThread, run.work_thread_id)
            assert thread is not None
            thread.status = "queued"
            thread.next_wake_at = now
            thread.updated_at = now
            self._event(
                session,
                run.work_thread_id,
                "run_resumed",
                run_id=run.id,
                actor_id=self.instance_id,
                execution_token=run.execution_token,
            )
            session.commit()
            session.refresh(run)
            return run

    def complete(self, run_id: int, execution_token: int, result_text: str) -> WorkRun:
        return self._finish(run_id, execution_token, "completed", result_text, None)

    def roll_over(
        self,
        run_id: int,
        execution_token: int,
        reason: str,
        *,
        result_text: str | None = None,
    ) -> WorkRun:
        return self._finish(
            run_id, execution_token, "rolled_over", result_text, reason
        )

    def fail(
        self,
        run_id: int,
        execution_token: int,
        error: str,
        *,
        result_text: str | None = None,
    ) -> WorkRun:
        return self._finish(run_id, execution_token, "failed", result_text, error)

    def cancel_all(self, reason: str) -> dict[str, int]:
        clean_reason = str(reason or "用户要求停止所有工作").strip()[:10000]
        now = utc_now()
        with self.database.sessions() as session:
            runs = session.scalars(
                select(WorkRun).where(WorkRun.status.in_(OPEN_RUN_STATUSES))
            ).all()
            affected_thread_ids = {run.work_thread_id for run in runs}
            threads = session.scalars(
                select(WorkThread).where(
                    ~WorkThread.status.in_(TERMINAL_THREAD_STATUSES)
                )
            ).all()
            affected_thread_ids.update(thread.id for thread in threads)
            for run in runs:
                run.status = "cancelled"
                run.execution_token = int(run.execution_token or 0) + 1
                run.last_error = clean_reason
                run.next_wake_at = None
                run.lease_owner = None
                run.lease_expires_at = None
                run.heartbeat_at = None
                run.finished_at = now
                run.updated_at = now
            for thread in threads:
                thread.status = "cancelled"
                thread.next_wake_at = None
                thread.finished_at = now
                thread.updated_at = now
                if not thread.result_text:
                    thread.result_text = (
                        "工作已按用户要求停止；Checkpoint、阶段成果、证据和工具账本均已保留。"
                    )
                self._event(
                    session,
                    thread.id,
                    "work_cancelled",
                    actor_id=self.instance_id,
                    payload={"reason": clean_reason},
                )
            for candidate in session.scalars(
                select(ResearchCandidate).where(
                    ResearchCandidate.work_thread_id.in_(affected_thread_ids),
                    ResearchCandidate.status.in_(["queued", "reading"]),
                    ResearchCandidate.read_tool_call_id.is_(None),
                )
            ).all():
                candidate.status = "deferred"
                candidate.disposition_reason = "任务已取消，本轮未开始正文读取。"
            collaborations = session.scalars(
                select(WorkCollaboration).where(
                    WorkCollaboration.status.in_(
                        {"assigned", "pending_controller", "deciding"}
                    )
                )
            ).all()
            for row in collaborations:
                row.status = "cancelled"
                row.result_text = row.result_text or clean_reason
                row.completed_at = now
                row.updated_at = now
            tool_calls = session.scalars(
                select(ToolCall).where(
                    ToolCall.status.in_({"pending", "running", "waiting_approval"})
                )
            ).all()
            for call in tool_calls:
                call.status = "cancelled"
                call.error = clean_reason
                call.finished_at = now
                call.updated_at = now
            projects = session.scalars(
                select(WorkProject).where(
                    ~WorkProject.status.in_(TERMINAL_PROJECT_STATUSES)
                )
            ).all()
            for project in projects:
                project.status = "cancelled"
                project.summary = clean_reason
                project.finished_at = now
                project.updated_at = now
            session.commit()
            return {
                "runs": len(runs),
                "threads": len(threads),
                "collaborations": len(collaborations),
                "tool_calls": len(tool_calls),
                "projects": len(projects),
            }

    def _finish(
        self,
        run_id: int,
        execution_token: int,
        status: str,
        result_text: str | None,
        error: str | None,
    ) -> WorkRun:
        now = utc_now()
        with self.database.sessions() as session:
            run = self._owned_run(session, run_id, execution_token, now)
            run.status = status
            run.result_text = result_text
            run.last_error = error
            run.lease_owner = None
            run.lease_expires_at = None
            run.heartbeat_at = None
            run.finished_at = now
            run.updated_at = now
            thread = session.get(WorkThread, run.work_thread_id)
            assert thread is not None
            thread.status = status
            thread.result_text = result_text
            thread.finished_at = now
            thread.next_wake_at = None
            thread.updated_at = now
            if status in {"completed", "failed", "cancelled"}:
                for candidate in session.scalars(
                    select(ResearchCandidate).where(
                        ResearchCandidate.work_thread_id == run.work_thread_id,
                        ResearchCandidate.status.in_(["queued", "reading"]),
                        ResearchCandidate.read_tool_call_id.is_(None),
                    )
                ).all():
                    candidate.status = "deferred"
                    candidate.disposition_reason = "任务已结束，本轮未开始正文读取。"
            self._event(
                session,
                run.work_thread_id,
                f"run_{status}",
                run_id=run.id,
                actor_id=self.instance_id,
                execution_token=execution_token,
                payload={"error": error},
            )
            session.commit()
            session.refresh(run)
            return run

    def _owned_run(
        self, session: Any, run_id: int, execution_token: int, now: datetime
    ) -> WorkRun:
        run = session.get(WorkRun, run_id)
        if not run:
            raise WorkNotFound("工作运行不存在")
        expires_at = run.lease_expires_at
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if (
            run.status != "running"
            or run.lease_owner != self.instance_id
            or run.execution_token != execution_token
            or not expires_at
            or expires_at <= now
        ):
            raise WorkLeaseLost("执行租约已失效，拒绝写入迟到结果")
        return run

    def get_thread(self, thread_id: int) -> WorkThread:
        with self.database.sessions() as session:
            thread = session.get(WorkThread, thread_id)
            if not thread:
                raise WorkNotFound("工作线程不存在")
            return thread

    def list_threads(
        self, *, owner_agent_id: int | None = None, status: str | None = None
    ) -> list[WorkThread]:
        with self.database.sessions() as session:
            statement = select(WorkThread)
            if owner_agent_id is not None:
                statement = statement.where(WorkThread.owner_agent_id == owner_agent_id)
            if status:
                statement = statement.where(WorkThread.status == status)
            return session.scalars(
                statement.order_by(WorkThread.priority.desc(), WorkThread.created_at)
            ).all()

    def list_events(self, thread_id: int) -> list[WorkEvent]:
        with self.database.sessions() as session:
            if not session.get(WorkThread, thread_id):
                raise WorkNotFound("工作线程不存在")
            return session.scalars(
                select(WorkEvent)
                .where(WorkEvent.work_thread_id == thread_id)
                .order_by(WorkEvent.id)
            ).all()

    def list_runs(self, thread_id: int) -> list[WorkRun]:
        with self.database.sessions() as session:
            if not session.get(WorkThread, thread_id):
                raise WorkNotFound("工作线程不存在")
            return session.scalars(
                select(WorkRun)
                .where(WorkRun.work_thread_id == thread_id)
                .order_by(WorkRun.id)
            ).all()
