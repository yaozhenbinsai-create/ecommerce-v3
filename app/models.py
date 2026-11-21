from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelEntry(Base):
    __tablename__ = "model_entries"
    __table_args__ = (
        UniqueConstraint("connection_key", "model_type", name="uq_connection_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    connection_key: Mapped[str] = mapped_column(String(80), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    access_method: Mapped[str] = mapped_column(String(30), nullable=False)
    protocol: Mapped[str] = mapped_column(String(20), nullable=False)
    model_type: Mapped[str] = mapped_column(String(10), nullable=False)
    model_identifier: Mapped[str | None] = mapped_column(String(120))
    base_url: Mapped[str | None] = mapped_column(String(500))
    context_window: Mapped[int | None] = mapped_column(Integer)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer)
    credential_env: Mapped[str | None] = mapped_column(String(100))
    image_generation_endpoint: Mapped[str | None] = mapped_column(String(500))
    image_edit_endpoint: Mapped[str | None] = mapped_column(String(500))
    image_generation: Mapped[bool] = mapped_column(Boolean, default=False)
    image_edit: Mapped[bool] = mapped_column(Boolean, default=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    health_status: Mapped[str] = mapped_column(String(20), default="untested")
    health_message: Mapped[str | None] = mapped_column(Text)
    health_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    circuit_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class RoutingPlan(Base):
    __tablename__ = "routing_plans"
    __table_args__ = (
        UniqueConstraint("plan_type", "name", name="uq_routing_plan_type_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    plan_type: Mapped[str] = mapped_column(String(10), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    items: Mapped[list["RoutingPlanModel"]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="RoutingPlanModel.position",
    )


class RoutingPlanModel(Base):
    __tablename__ = "routing_plan_models"
    __table_args__ = (
        UniqueConstraint("plan_id", "model_id", name="uq_routing_plan_model"),
        UniqueConstraint("plan_id", "position", name="uq_routing_plan_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("routing_plans.id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model_entries.id", ondelete="RESTRICT"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    plan: Mapped[RoutingPlan] = relationship(back_populates="items")
    model: Mapped[ModelEntry] = relationship()


class AgentProfile(Base):
    __tablename__ = "agent_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    role_responsibilities: Mapped[str] = mapped_column(Text, default="")
    work_content: Mapped[str] = mapped_column(Text, default="")
    personality: Mapped[str] = mapped_column(Text, default="")
    preferences: Mapped[str] = mapped_column(Text, default="")
    is_controller: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    needs_ai_organization: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT")
    )
    chat_plan_id: Mapped[int | None] = mapped_column(
        ForeignKey("routing_plans.id", ondelete="RESTRICT")
    )
    image_plan_id: Mapped[int | None] = mapped_column(
        ForeignKey("routing_plans.id", ondelete="RESTRICT")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    parent: Mapped["AgentProfile | None"] = relationship(
        remote_side="AgentProfile.id", foreign_keys=[parent_id]
    )
    chat_plan: Mapped[RoutingPlan | None] = relationship(foreign_keys=[chat_plan_id])
    image_plan: Mapped[RoutingPlan | None] = relationship(foreign_keys=[image_plan_id])


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    sender_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    sender_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT")
    )
    recipient_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_internal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="completed", nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(120))
    work_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PendingChatTask(Base):
    __tablename__ = "pending_chat_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    user_message_id: Mapped[int] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="RESTRICT"), nullable=False
    )
    request_text: Mapped[str] = mapped_column(Text, nullable=False)
    task_kind: Mapped[str] = mapped_column(String(30), default="chat", nullable=False)
    parent_task_id: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30), default="waiting_retry")
    next_retry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    checkpoint_json: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    result_message_id: Mapped[int | None] = mapped_column(Integer)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AgentRuntimeState(Base):
    __tablename__ = "agent_runtime_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    main_session_key: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="waiting", nullable=False)
    active_task_id: Mapped[int | None] = mapped_column(Integer)
    active_work_run_id: Mapped[int | None] = mapped_column(Integer)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkProject(Base):
    __tablename__ = "work_projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    success_criteria_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    constraints_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="running", nullable=False)
    current_root_thread_id: Mapped[int | None] = mapped_column(Integer, index=True)
    generation_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    stagnant_generations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_marker_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    result_text: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkThread(Base):
    __tablename__ = "work_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_projects.id", ondelete="RESTRICT"), index=True
    )
    generation_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    parent_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="RESTRICT")
    )
    continuation_of_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="RESTRICT"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    success_criteria_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    constraints_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="queued", nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    result_text: Mapped[str | None] = mapped_column(Text)
    next_wake_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkRun(Base):
    __tablename__ = "work_runs"
    __table_args__ = (
        Index(
            "uq_work_runs_open_thread",
            "work_thread_id",
            unique=True,
            sqlite_where=text(
                "status IN ('queued','running','waiting_retry',"
                "'waiting_schedule','waiting_dependency','waiting_user')"
            ),
        ),
        Index(
            "uq_work_runs_running_agent",
            "owner_agent_id",
            unique=True,
            sqlite_where=text("status = 'running'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    trigger_kind: Mapped[str] = mapped_column(String(30), default="manual", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="queued", nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(120))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_token: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    checkpoint_json: Mapped[str | None] = mapped_column(Text)
    result_text: Mapped[str | None] = mapped_column(Text)
    last_error: Mapped[str | None] = mapped_column(Text)
    next_wake_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkEvent(Base):
    __tablename__ = "work_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    work_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_runs.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_kind: Mapped[str] = mapped_column(String(30), default="runtime", nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(120))
    execution_token: Mapped[int | None] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class WorkContextSummary(Base):
    __tablename__ = "work_context_summaries"
    __table_args__ = (
        UniqueConstraint("work_thread_id", "version", name="uq_work_context_summary_version"),
        UniqueConstraint("work_thread_id", "source_hash", name="uq_work_context_summary_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    summary_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    through_tool_call_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    through_event_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class WorkModelCall(Base):
    __tablename__ = "work_model_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    work_run_id: Mapped[int] = mapped_column(
        ForeignKey("work_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    execution_token: Mapped[int] = mapped_column(Integer, nullable=False)
    purpose: Mapped[str] = mapped_column(String(40), default="decision", nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    model_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_entries.id", ondelete="SET NULL"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="completed", nullable=False)
    error_kind: Mapped[str | None] = mapped_column(String(40))
    error: Mapped[str | None] = mapped_column(Text)
    prompt_hash: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    prompt_chars: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    usage_source: Mapped[str] = mapped_column(String(30), default="estimated", nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class WorkCollaboration(Base):
    __tablename__ = "work_collaborations"
    __table_args__ = (
        UniqueConstraint(
            "parent_thread_id", "request_key", name="uq_collaboration_parent_request"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    child_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="SET NULL"), unique=True, index=True
    )
    requester_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    target_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), index=True
    )
    decided_by_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT")
    )
    channel: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    request_key: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    success_criteria_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    constraints_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    wait_for_result: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    result_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkGovernanceIssue(Base):
    __tablename__ = "work_governance_issues"
    __table_args__ = (
        UniqueConstraint("issue_key", name="uq_work_governance_issue_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    issue_key: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    issue_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(30), default="open", nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), default="info", nullable=False)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    related_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="SET NULL"), index=True
    )
    reviewer_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), index=True
    )
    rework_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="SET NULL"), unique=True, index=True
    )
    detected_by: Mapped[str] = mapped_column(String(30), default="automatic", nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    resolution_json: Mapped[str | None] = mapped_column(Text)
    review_decision: Mapped[str | None] = mapped_column(String(20))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AgentToolPermission(Base):
    __tablename__ = "agent_tool_permissions"
    __table_args__ = (
        UniqueConstraint("agent_id", "tool_name", name="uq_agent_tool_permission"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)
    permission: Mapped[str] = mapped_column(String(20), default="deny", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (
        UniqueConstraint("action_key", name="uq_tool_call_action_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    work_run_id: Mapped[int] = mapped_column(
        ForeignKey("work_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    step_id: Mapped[str] = mapped_column(String(120), nullable=False)
    action_key: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)
    arguments_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    permission: Mapped[str] = mapped_column(String(20), nullable=False)
    retry_safety: Mapped[str] = mapped_column(String(20), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    execution_token: Mapped[int] = mapped_column(Integer, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class Evidence(Base):
    __tablename__ = "evidence"
    __table_args__ = (
        UniqueConstraint(
            "work_thread_id", "content_hash", name="uq_evidence_thread_content"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_title: Mapped[str] = mapped_column(String(1000), nullable=False)
    published_at: Mapped[str | None] = mapped_column(String(80))
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str | None] = mapped_column(String(200))
    metric_value: Mapped[str | None] = mapped_column(String(200))
    metric_unit: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)
    stance: Mapped[str] = mapped_column(String(20), default="supports", nullable=False)
    source_type: Mapped[str] = mapped_column(String(30), default="unknown", nullable=False)
    source_publisher: Mapped[str | None] = mapped_column(String(300))
    claim_key: Mapped[str | None] = mapped_column(String(200), index=True)
    verification_notes: Mapped[str | None] = mapped_column(Text)
    value_analysis_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    value_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    review_status: Mapped[str] = mapped_column(
        String(30), default="unreviewed", nullable=False
    )
    reviewer_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), index=True
    )
    review_notes: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ResearchCandidate(Base):
    __tablename__ = "research_candidates"
    __table_args__ = (
        UniqueConstraint(
            "work_thread_id", "canonical_url", name="uq_research_candidate_thread_url"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    snippet: Mapped[str] = mapped_column(Text, default="", nullable=False)
    sources_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    # Candidate processing is a durable queue.  Search results are never
    # treated as insights; every URL advances through this state machine.
    status: Mapped[str] = mapped_column(
        String(30), default="queued", nullable=False, index=True
    )
    discovery_tool_call_id: Mapped[int | None] = mapped_column(Integer, index=True)
    read_tool_call_id: Mapped[int | None] = mapped_column(Integer, index=True)
    source_rank: Mapped[int | None] = mapped_column(Integer)
    content_kind: Mapped[str | None] = mapped_column(String(40))
    source_text: Mapped[str | None] = mapped_column(Text)
    content_chars: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    disposition_reason: Mapped[str | None] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class MarketInsightRecord(Base):
    """Durable synthesized research output, independent from checkpoint limits."""

    __tablename__ = "market_insights"
    __table_args__ = (
        UniqueConstraint(
            "work_thread_id", "content_hash", name="uq_market_insight_thread_hash"
        ),
        Index("ix_market_insights_project_created", "project_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_projects.id", ondelete="CASCADE"), index=True
    )
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    category: Mapped[str | None] = mapped_column(String(120), index=True)
    source_refs_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    insight_json: Mapped[str] = mapped_column(Text, nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class BusinessHotspot(Base):
    __tablename__ = "business_hotspots"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_business_hotspot_fingerprint"),
        Index("ix_business_hotspots_owner_stage", "owner_agent_id", "business_stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    lifecycle_status: Mapped[str] = mapped_column(String(30), default="emerging", nullable=False, index=True)
    research_status: Mapped[str] = mapped_column(String(30), default="discovered", nullable=False, index=True)
    business_stage: Mapped[str] = mapped_column(String(40), default="unprocessed", nullable=False, index=True)
    automation_status: Mapped[str] = mapped_column(String(30), default="completed", nullable=False, index=True)
    next_action: Mapped[str | None] = mapped_column(Text)
    missing_information_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    active_work_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="SET NULL"), index=True
    )
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    next_review_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class BusinessContentSource(Base):
    __tablename__ = "business_content_sources"
    __table_args__ = (
        UniqueConstraint("hotspot_id", "source_url_hash", name="uq_business_source_hotspot_url"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hotspot_id: Mapped[int] = mapped_column(ForeignKey("business_hotspots.id", ondelete="CASCADE"), nullable=False, index=True)
    owner_agent_id: Mapped[int] = mapped_column(ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True)
    evidence_id: Mapped[int | None] = mapped_column(ForeignKey("evidence.id", ondelete="SET NULL"), index=True)
    market_insight_id: Mapped[int | None] = mapped_column(ForeignKey("market_insights.id", ondelete="SET NULL"), index=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_url_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    publisher: Mapped[str | None] = mapped_column(String(300))
    source_type: Mapped[str] = mapped_column(String(40), default="unknown", nullable=False)
    published_at: Mapped[str | None] = mapped_column(String(80))
    excerpt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    access_status: Mapped[str] = mapped_column(String(30), default="available", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class BusinessContentFact(Base):
    __tablename__ = "business_content_facts"
    __table_args__ = (
        UniqueConstraint("hotspot_id", "fingerprint", name="uq_business_fact_hotspot_fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hotspot_id: Mapped[int] = mapped_column(ForeignKey("business_hotspots.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("business_content_sources.id", ondelete="SET NULL"), index=True)
    owner_agent_id: Mapped[int] = mapped_column(ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    fact_type: Mapped[str] = mapped_column(String(50), default="market_observation", nullable=False, index=True)
    stance: Mapped[str] = mapped_column(String(20), default="supports", nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)
    uncertainty: Mapped[str | None] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class BusinessContentTheme(Base):
    __tablename__ = "business_content_themes"
    __table_args__ = (UniqueConstraint("hotspot_id", "fingerprint", name="uq_business_theme_hotspot_fingerprint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hotspot_id: Mapped[int] = mapped_column(ForeignKey("business_hotspots.id", ondelete="CASCADE"), nullable=False, index=True)
    owner_agent_id: Mapped[int] = mapped_column(ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    core_tension: Mapped[str | None] = mapped_column(Text)
    emotions_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    people_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    scenes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    signals_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="active", nullable=False, index=True)
    next_action: Mapped[str | None] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ExpressionProposition(Base):
    __tablename__ = "expression_propositions"
    __table_args__ = (UniqueConstraint("theme_id", "fingerprint", name="uq_expression_theme_fingerprint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    theme_id: Mapped[int] = mapped_column(ForeignKey("business_content_themes.id", ondelete="CASCADE"), nullable=False, index=True)
    owner_agent_id: Mapped[int] = mapped_column(ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    core_expression: Mapped[str] = mapped_column(Text, nullable=False)
    target_emotion: Mapped[str | None] = mapped_column(Text)
    audience_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    scenes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    purchase_reason: Mapped[str | None] = mapped_column(Text)
    exploration_angles_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    theme_form: Mapped[str] = mapped_column(String(100), default="概念", nullable=False)
    creative_seeds_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    avoid_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    risks_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="draft", nullable=False, index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ProductCapability(Base):
    __tablename__ = "product_capabilities"
    __table_args__ = (UniqueConstraint("capability_key", "version", name="uq_product_capability_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    capability_key: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    product_type: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(10), default="CNY", nullable=False)
    cost_min: Mapped[int | None] = mapped_column(Integer)
    cost_max: Mapped[int | None] = mapped_column(Integer)
    retail_min: Mapped[int | None] = mapped_column(Integer)
    retail_max: Mapped[int | None] = mapped_column(Integer)
    minimum_order_quantity: Mapped[int | None] = mapped_column(Integer)
    lead_time_days: Mapped[int | None] = mapped_column(Integer)
    materials_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    processes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    packaging_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    constraints_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    quality_requirements_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ProductComponent(Base):
    __tablename__ = "product_components"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_capability_id: Mapped[int] = mapped_column(ForeignKey("product_capabilities.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    sizes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    materials_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    supported_pattern_modes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    supports_ab_sides: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    constraints_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ProductionRule(Base):
    __tablename__ = "production_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    scope: Mapped[str] = mapped_column(String(40), default="global", nullable=False, index=True)
    product_capability_id: Mapped[int | None] = mapped_column(ForeignKey("product_capabilities.id", ondelete="CASCADE"), index=True)
    rule_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    rule_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ProductFitAssessment(Base):
    __tablename__ = "product_fit_assessments"
    __table_args__ = (UniqueConstraint("proposition_id", "product_capability_id", name="uq_product_fit_pair"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    proposition_id: Mapped[int] = mapped_column(ForeignKey("expression_propositions.id", ondelete="CASCADE"), nullable=False, index=True)
    product_capability_id: Mapped[int] = mapped_column(ForeignKey("product_capabilities.id", ondelete="RESTRICT"), nullable=False, index=True)
    owner_agent_id: Mapped[int] = mapped_column(ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True)
    conclusion: Mapped[str] = mapped_column(String(30), default="needs_validation", nullable=False, index=True)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    target_audience_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    scenes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    risks_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    missing_information_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ProductOpportunity(Base):
    __tablename__ = "product_opportunities"
    __table_args__ = (UniqueConstraint("fit_assessment_id", name="uq_product_opportunity_fit"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fit_assessment_id: Mapped[int] = mapped_column(ForeignKey("product_fit_assessments.id", ondelete="RESTRICT"), nullable=False, index=True)
    owner_agent_id: Mapped[int] = mapped_column(ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="candidate", nullable=False, index=True)
    core_expression: Mapped[str] = mapped_column(Text, nullable=False)
    purchase_reason: Mapped[str | None] = mapped_column(Text)
    commercial_positioning: Mapped[str | None] = mapped_column(Text)
    risks_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    missing_information_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    next_action: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ProductDesignBrief(Base):
    __tablename__ = "product_design_briefs"
    __table_args__ = (UniqueConstraint("product_opportunity_id", name="uq_design_brief_opportunity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_opportunity_id: Mapped[int] = mapped_column(ForeignKey("product_opportunities.id", ondelete="RESTRICT"), nullable=False, index=True)
    owner_agent_id: Mapped[int] = mapped_column(ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True)
    product_capability_id: Mapped[int] = mapped_column(ForeignKey("product_capabilities.id", ondelete="RESTRICT"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(30), default="ready_for_design", nullable=False, index=True)
    brief_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    source_snapshot_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class DesignJob(Base):
    __tablename__ = "design_jobs"
    __table_args__ = (
        UniqueConstraint("product_design_brief_id", name="uq_design_job_brief"),
        Index("ix_design_jobs_owner_stage", "owner_agent_id", "stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_design_brief_id: Mapped[int] = mapped_column(
        ForeignKey("product_design_briefs.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    product_opportunity_id: Mapped[int] = mapped_column(
        ForeignKey("product_opportunities.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    stage: Mapped[str] = mapped_column(String(50), default="intake", nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(30), default="not_started", nullable=False, index=True)
    automation_status: Mapped[str] = mapped_column(String(30), default="queued", nullable=False, index=True)
    active_work_thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_threads.id", ondelete="SET NULL"), index=True
    )
    next_action: Mapped[str | None] = mapped_column(Text)
    last_error: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class DesignInputSnapshot(Base):
    __tablename__ = "design_input_snapshots"
    __table_args__ = (UniqueConstraint("design_job_id", "version", name="uq_design_input_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    design_job_id: Mapped[int] = mapped_column(ForeignKey("design_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    snapshot_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DesignRoute(Base):
    __tablename__ = "design_routes"
    __table_args__ = (UniqueConstraint("design_job_id", "route_number", "version", name="uq_design_route_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    design_job_id: Mapped[int] = mapped_column(ForeignKey("design_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    route_number: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="draft", nullable=False, index=True)
    route_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class DesignAsset(Base):
    __tablename__ = "design_assets"
    __table_args__ = (UniqueConstraint("design_job_id", "asset_key", "version", name="uq_design_asset_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    design_job_id: Mapped[int] = mapped_column(ForeignKey("design_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_key: Mapped[str] = mapped_column(String(120), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="generated", nullable=False, index=True)
    source_asset_id: Mapped[int | None] = mapped_column(ForeignKey("design_assets.id", ondelete="SET NULL"))
    uri: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    specification_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class DesignReview(Base):
    __tablename__ = "design_reviews"
    __table_args__ = (Index("ix_design_reviews_job_stage", "design_job_id", "stage"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    design_job_id: Mapped[int] = mapped_column(ForeignKey("design_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    target_type: Mapped[str] = mapped_column(String(40), nullable=False)
    target_id: Mapped[int | None] = mapped_column(Integer)
    reviewer_agent_id: Mapped[int | None] = mapped_column(ForeignKey("agent_profiles.id", ondelete="SET NULL"), index=True)
    decision: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    review_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    work_thread_id: Mapped[int | None] = mapped_column(ForeignKey("work_threads.id", ondelete="SET NULL"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DesignComponentSpec(Base):
    __tablename__ = "design_component_specs"
    __table_args__ = (UniqueConstraint("design_job_id", "component_id", "version", name="uq_design_component_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    design_job_id: Mapped[int] = mapped_column(ForeignKey("design_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    component_id: Mapped[int] = mapped_column(ForeignKey("product_components.id", ondelete="RESTRICT"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    specification_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class BusinessObjectEvent(Base):
    __tablename__ = "business_object_events"
    __table_args__ = (Index("ix_business_events_object", "object_type", "object_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    object_type: Mapped[str] = mapped_column(String(50), nullable=False)
    object_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    actor_kind: Mapped[str] = mapped_column(String(20), default="system", nullable=False)
    actor_agent_id: Mapped[int | None] = mapped_column(ForeignKey("agent_profiles.id", ondelete="SET NULL"), index=True)
    reason: Mapped[str | None] = mapped_column(Text)
    before_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    after_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    work_thread_id: Mapped[int | None] = mapped_column(ForeignKey("work_threads.id", ondelete="SET NULL"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class ResearchProviderHealth(Base):
    __tablename__ = "research_provider_health"

    provider: Mapped[str] = mapped_column(String(80), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="unknown", nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    empty_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_latency_ms: Mapped[int | None] = mapped_column(Integer)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ResearchReport(Base):
    __tablename__ = "research_reports"
    __table_args__ = (
        UniqueConstraint("work_thread_id", "version", name="uq_report_thread_version"),
        UniqueConstraint("work_thread_id", "source_hash", name="uq_report_thread_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    work_thread_id: Mapped[int] = mapped_column(
        ForeignKey("work_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    executive_summary: Mapped[str] = mapped_column(Text, nullable=False)
    findings_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    recommendations_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    caveats_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    evidence_snapshot_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    markdown_content: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    through_message_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="CASCADE")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    vector_json: Mapped[str] = mapped_column(Text, nullable=False)
    vector_backend: Mapped[str] = mapped_column(String(40), default="local_hash_v1")
    source_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryOperationLog(Base):
    __tablename__ = "memory_operation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int | None] = mapped_column(Integer)
    agent_id: Mapped[int | None] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(20), nullable=False)
    before_content: Mapped[str | None] = mapped_column(Text)
    after_content: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemoryMaintenanceState(Base):
    __tablename__ = "memory_maintenance_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agent_profiles.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    last_message_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RuntimeJobState(Base):
    __tablename__ = "runtime_job_states"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
