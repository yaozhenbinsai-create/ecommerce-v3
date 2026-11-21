from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CustomProtocol = Literal[
    "openai",
    "openai_responses",
    "openai_images",
    "custom_image",
    "claude",
    "claude_openai",
]


class CustomModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    protocol: CustomProtocol
    model_type: Literal["chat", "image"]
    model_identifier: str = Field(min_length=1, max_length=120)
    base_url: str = Field(min_length=1, max_length=500)
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    api_key: str = Field(min_length=1, max_length=10000)
    image_generation_endpoint: str | None = Field(default=None, max_length=500)
    image_edit_endpoint: str | None = Field(default=None, max_length=500)
    image_generation: bool = False
    image_edit: bool = False
    is_enabled: bool = True

    @field_validator("name", "model_identifier", "base_url", "api_key")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value

    @model_validator(mode="after")
    def validate_token_limits(self):
        if (
            self.context_window
            and self.max_output_tokens
            and self.max_output_tokens >= self.context_window
        ):
            raise ValueError("最大输出Token必须小于上下文窗口")
        if self.model_type == "chat" and self.protocol in {
            "openai_images",
            "custom_image",
        }:
            raise ValueError("图片接口格式不能用于对话模型")
        if self.model_type == "image" and self.protocol not in {
            "openai_images",
            "openai_responses",
            "custom_image",
        }:
            raise ValueError("图片模型请选择 Images API、Responses 图片工具或自定义图片接口")
        if self.protocol == "custom_image" and not self.image_generation_endpoint:
            raise ValueError("自定义图片接口必须填写图片生成接口路径")
        return self

    @field_validator("image_generation_endpoint", "image_edit_endpoint")
    @classmethod
    def strip_optional_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ModelUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120)
    protocol: CustomProtocol | None = None
    model_type: Literal["chat", "image"] | None = None
    model_identifier: str | None = Field(default=None, max_length=120)
    base_url: str | None = Field(default=None, max_length=500)
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    api_key: str | None = Field(default=None, max_length=10000)
    image_generation_endpoint: str | None = Field(default=None, max_length=500)
    image_edit_endpoint: str | None = Field(default=None, max_length=500)
    image_generation: bool | None = None
    image_edit: bool | None = None
    is_enabled: bool | None = None

    @field_validator(
        "name",
        "model_identifier",
        "base_url",
        "api_key",
        "image_generation_endpoint",
        "image_edit_endpoint",
    )
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class ModelConnectionProbe(BaseModel):
    model_id: int | None = Field(default=None, gt=0)
    protocol: CustomProtocol
    model_type: Literal["chat", "image"] | None = None
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str | None = None
    model_identifier: str | None = Field(default=None, max_length=120)
    image_generation_endpoint: str | None = Field(default=None, max_length=500)
    image_edit_endpoint: str | None = Field(default=None, max_length=500)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("必须是 http:// 或 https:// 地址")
        return value

    @field_validator(
        "api_key",
        "model_identifier",
        "image_generation_endpoint",
        "image_edit_endpoint",
    )
    @classmethod
    def strip_probe_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ConversationSettings(BaseModel):
    unknown_context_window: int = Field(gt=0)
    default_max_output_tokens: int = Field(gt=0)
    reserve_ratio: float = Field(gt=0, le=0.5)
    reserve_tokens_floor: int = Field(ge=0)
    reserve_tokens_ceiling: int = Field(gt=0)
    auto_compaction_enabled: bool = True
    compaction_trigger_ratio: float = Field(default=0.75, ge=0.5, le=0.95)
    keep_recent_tokens: int = Field(default=12000, gt=0)
    summary_max_tokens: int = Field(default=3000, gt=0)

    @model_validator(mode="after")
    def validate_conversation_limits(self):
        if self.default_max_output_tokens >= self.unknown_context_window:
            raise ValueError("默认最大输出必须小于默认上下文窗口")
        if self.reserve_tokens_floor > self.reserve_tokens_ceiling:
            raise ValueError("最小预留不能大于最大预留")
        if (
            self.default_max_output_tokens + self.reserve_tokens_floor
            >= self.unknown_context_window
        ):
            raise ValueError("默认输出和最小预留之和必须小于默认上下文窗口")
        return self


class MemorySettings(BaseModel):
    enabled: bool = True
    vector_dimensions: int = Field(default=256, ge=64, le=2048)
    top_k: int = Field(default=5, ge=1, le=20)
    minimum_score: float = Field(default=0.18, ge=0, le=1)
    daily_hour: int = Field(default=3, ge=0, le=23)
    soft_delete_days: int = Field(default=30, ge=1, le=3650)


class RetrySettings(BaseModel):
    request_timeout_seconds: int = Field(default=300, ge=10, le=3600)
    attempts_per_model: int = Field(default=3, ge=1, le=10)
    retry_delay_minutes: int = Field(default=1, ge=1, le=10080)


class ProcessSettings(BaseModel):
    enabled: bool = True
    check_interval_seconds: int = Field(default=60, ge=10, le=3600)
    warning_mb: int = Field(default=1024, ge=128)
    restart_mb: int = Field(default=1536, ge=256)
    consecutive_breaches: int = Field(default=3, ge=1, le=20)
    max_uptime_hours: int = Field(default=24, ge=1, le=720)

    @model_validator(mode="after")
    def validate_process_limits(self):
        if self.restart_mb <= self.warning_mb:
            raise ValueError("重启阈值必须大于警告阈值")
        return self


class AgentRuntimeSettings(BaseModel):
    heartbeat_enabled: bool = True
    heartbeat_interval_minutes: int = Field(default=30, ge=1, le=1440)
    queue_scan_seconds: int = Field(default=2, ge=1, le=60)
    discussion_max_rounds: int = Field(default=6, ge=1, le=20)
    discussion_max_minutes: int = Field(default=60, ge=1, le=1440)


class WorkContextSettings(BaseModel):
    enabled: bool = True
    max_prompt_chars: int = Field(default=16000, ge=12000, le=200000)
    checkpoint_max_chars: int = Field(default=12000, ge=4000, le=80000)
    recent_tool_calls: int = Field(default=2, ge=1, le=20)
    recent_result_chars: int = Field(default=1200, ge=500, le=20000)
    summary_interval_tool_calls: int = Field(default=5, ge=1, le=100)
    evidence_limit: int = Field(default=20, ge=1, le=100)
    failure_limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_work_context_limits(self):
        if self.checkpoint_max_chars >= self.max_prompt_chars:
            raise ValueError("Checkpoint上限必须小于工作提示总上限")
        return self


class WorkBudgetSettings(BaseModel):
    max_model_tokens_per_tree: int = Field(default=500000, ge=1000, le=100000000)
    max_active_minutes_per_tree: int = Field(default=240, ge=1, le=10080)


class WorkGovernanceSettings(BaseModel):
    enabled: bool = True
    scan_interval_seconds: int = Field(default=60, ge=10, le=3600)
    duplicate_similarity_threshold: float = Field(default=0.82, ge=0.5, le=0.99)
    stalled_minutes: int = Field(default=30, ge=1, le=10080)
    quality_review_enabled: bool = True


class WorkspaceSettings(BaseModel):
    max_file_bytes: int = Field(default=20_000_000, ge=100_000, le=100_000_000)
    max_text_chars: int = Field(default=30_000, ge=2_000, le=200_000)
    max_results: int = Field(default=100, ge=10, le=500)
    max_pdf_pages_per_call: int = Field(default=20, ge=1, le=100)
    max_pdf_search_pages: int = Field(default=200, ge=1, le=1000)
    max_table_scan_rows: int = Field(default=20_000, ge=100, le=200_000)
    max_table_rows_per_call: int = Field(default=200, ge=10, le=2_000)
    max_table_columns: int = Field(default=100, ge=5, le=300)
    max_table_groups: int = Field(default=200, ge=10, le=2_000)
    max_table_cell_chars: int = Field(default=2_000, ge=100, le=20_000)
    max_workbook_sheets: int = Field(default=50, ge=1, le=200)
    max_xlsx_uncompressed_bytes: int = Field(
        default=100_000_000, ge=5_000_000, le=500_000_000
    )


class TabularFilter(BaseModel):
    column: str = Field(min_length=1, max_length=200)
    operator: Literal[
        "eq",
        "ne",
        "contains",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "is_empty",
        "not_empty",
    ]
    value: Any = None


class TabularMetric(BaseModel):
    operation: Literal[
        "count_rows",
        "count_nonempty",
        "sum",
        "average",
        "min",
        "max",
        "distinct_count",
    ]
    column: str | None = Field(default=None, max_length=200)
    alias: str | None = Field(default=None, max_length=100)


class TabularInspectRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    sheet: str | None = Field(default=None, max_length=200)
    sample_rows: int = Field(default=50, ge=1, le=200)


class TabularReadRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    sheet: str | None = Field(default=None, max_length=200)
    columns: list[str] | None = Field(default=None, max_length=300)
    filters: list[TabularFilter] = Field(default_factory=list, max_length=10)
    sort_by: str | None = Field(default=None, max_length=200)
    sort_order: Literal["asc", "desc"] = "asc"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=2_000)


class TabularSummaryRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    sheet: str | None = Field(default=None, max_length=200)
    group_by: list[str] = Field(default_factory=list, max_length=2)
    filters: list[TabularFilter] = Field(default_factory=list, max_length=10)
    metrics: list[TabularMetric] = Field(min_length=1, max_length=20)


class BrowserSettings(BaseModel):
    navigation_timeout_ms: int = Field(default=30_000, ge=3_000, le=120_000)
    max_page_chars: int = Field(default=30_000, ge=2_000, le=100_000)
    max_elements: int = Field(default=120, ge=10, le=300)
    max_response_bytes: int = Field(default=5_000_000, ge=100_000, le=20_000_000)
    max_sessions: int = Field(default=4, ge=1, le=16)
    idle_timeout_seconds: int = Field(default=600, ge=30, le=3600)


class MemoryUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def strip_memory_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("记忆内容不能为空")
        return value


class RoutingPlanSave(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    plan_type: Literal["chat", "image"]
    model_ids: list[int] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def strip_plan_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value

    @field_validator("model_ids")
    @classmethod
    def unique_models(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("模型不能重复")
        return value


class AgentSave(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role_responsibilities: str = Field(min_length=1, max_length=4000)
    work_content: str = Field(min_length=1, max_length=4000)
    personality: str = Field(min_length=1, max_length=2000)
    preferences: str = Field(min_length=1, max_length=2000)
    chat_plan_id: int | None = None
    image_plan_id: int | None = None
    ai_organized: bool = False

    @field_validator(
        "name", "role_responsibilities", "work_content", "personality", "preferences"
    )
    @classmethod
    def strip_agent_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class AgentOrganize(BaseModel):
    role_responsibilities: str = Field(min_length=1, max_length=4000)
    work_content: str = Field(min_length=1, max_length=4000)
    personality: str = Field(min_length=1, max_length=2000)
    preferences: str = Field(min_length=1, max_length=2000)
    chat_plan_id: int | None = None

    @field_validator(
        "role_responsibilities", "work_content", "personality", "preferences"
    )
    @classmethod
    def strip_organize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class ChatMessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20000)

    @field_validator("content")
    @classmethod
    def strip_chat_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("消息不能为空")
        return value


class ControllerWorkCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    title: str | None = Field(default=None, max_length=200)
    success_criteria: list[str] = Field(default_factory=list, max_length=50)
    constraints: list[str] = Field(default_factory=list, max_length=50)
    priority: int = Field(default=0, ge=-100, le=100)
    continuation_of_thread_id: int | None = Field(default=None, gt=0)

    @field_validator("content", "title")
    @classmethod
    def strip_controller_work_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class WorkThreadCreate(BaseModel):
    owner_agent_id: int = Field(gt=0)
    parent_thread_id: int | None = Field(default=None, gt=0)
    continuation_of_thread_id: int | None = Field(default=None, gt=0)
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=20000)
    success_criteria: list[str] = Field(default_factory=list, max_length=50)
    constraints: list[str] = Field(default_factory=list, max_length=50)
    priority: int = Field(default=0, ge=-100, le=100)

    @field_validator("title", "objective")
    @classmethod
    def strip_work_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class WorkRunCreate(BaseModel):
    trigger_kind: str = Field(default="manual", min_length=1, max_length=30)
    checkpoint: dict[str, Any] | None = None
    next_wake_at: datetime | None = None


class WorkRunToken(BaseModel):
    execution_token: int = Field(ge=1)


class WorkRunCheckpoint(WorkRunToken):
    checkpoint: dict[str, Any]


class WorkRunPause(WorkRunToken):
    status: Literal[
        "queued",
        "waiting_retry",
        "waiting_schedule",
        "waiting_dependency",
        "waiting_user",
    ]
    next_wake_at: datetime | None = None
    error: str | None = Field(default=None, max_length=10000)


class WorkRunComplete(WorkRunToken):
    result_text: str = Field(min_length=1, max_length=100000)


class WorkRunFail(WorkRunToken):
    error: str = Field(min_length=1, max_length=10000)


class WorkStopAll(BaseModel):
    reason: str = Field(default="用户要求停止所有工作", min_length=1, max_length=10000)


class BusinessDeleteAll(BaseModel):
    confirmation: str = Field(min_length=1, max_length=100)


class WorkDataCleanup(BaseModel):
    scope: Literal["governance", "history"]
    confirmation: str = Field(min_length=1, max_length=100)


class AgentToolPermissionSave(BaseModel):
    permission: Literal["allow", "approval", "deny"]


class ToolCallApproval(BaseModel):
    approved: bool
    reason: str | None = Field(default=None, max_length=4000)


class WorkSchedulerSettings(BaseModel):
    auto_execute_enabled: bool = True
    queue_scan_seconds: int = Field(default=2, ge=1, le=60)
    max_concurrent_agents: int = Field(default=4, ge=1, le=16)


class CollaborationDecision(BaseModel):
    approved: bool
    target_agent_id: int | None = Field(default=None, gt=0)
    reason: str | None = Field(default=None, max_length=4000)


class GovernanceQualityReview(BaseModel):
    reviewer_agent_id: int = Field(gt=0)
    decision: Literal["pass", "rework", "reject"]
    notes: str = Field(min_length=1, max_length=10000)
    target_agent_id: int | None = Field(default=None, gt=0)


class GovernanceIssueResolution(BaseModel):
    reviewer_agent_id: int = Field(gt=0)
    note: str = Field(min_length=1, max_length=10000)


class EvidenceReview(BaseModel):
    reviewer_agent_id: int = Field(gt=0)
    status: Literal["verified", "rejected", "needs_followup"]
    notes: str = Field(min_length=1, max_length=10000)


class BusinessInsightIngest(BaseModel):
    insight_ids: list[int] = Field(min_length=1, max_length=200)


class BusinessHotspotIntervention(BaseModel):
    actor_agent_id: int | None = Field(default=None, gt=0)
    reason: str = Field(min_length=1, max_length=10000)
    updates: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def strip_business_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class ProductComponentSave(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    quantity: int = Field(default=1, ge=1, le=1000)
    sizes: list[str] = Field(default_factory=list, max_length=200)
    materials: list[str] = Field(default_factory=list, max_length=200)
    supported_pattern_modes: list[str] = Field(default_factory=list, max_length=50)
    supports_ab_sides: bool = False
    constraints: list[str] = Field(default_factory=list, max_length=200)
    position: int = Field(default=0, ge=0, le=1000)

    @field_validator("name")
    @classmethod
    def strip_component_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class ProductCapabilitySave(BaseModel):
    capability_key: str | None = Field(default=None, max_length=120)
    name: str = Field(min_length=1, max_length=200)
    product_type: str = Field(min_length=1, max_length=120)
    is_enabled: bool = True
    currency: str = Field(default="CNY", min_length=1, max_length=10)
    cost_min: int | None = Field(default=None, ge=0)
    cost_max: int | None = Field(default=None, ge=0)
    retail_min: int | None = Field(default=None, ge=0)
    retail_max: int | None = Field(default=None, ge=0)
    minimum_order_quantity: int | None = Field(default=None, ge=1)
    lead_time_days: int | None = Field(default=None, ge=0)
    materials: list[str] = Field(default_factory=list, max_length=200)
    processes: list[str] = Field(default_factory=list, max_length=200)
    packaging: list[str] = Field(default_factory=list, max_length=200)
    constraints: list[str] = Field(default_factory=list, max_length=200)
    quality_requirements: list[str] = Field(default_factory=list, max_length=200)
    components: list[ProductComponentSave] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_product_ranges(self) -> "ProductCapabilitySave":
        for low, high, label in (
            (self.cost_min, self.cost_max, "成本"),
            (self.retail_min, self.retail_max, "零售价"),
        ):
            if low is not None and high is not None and low > high:
                raise ValueError(f"{label}最低值不能高于最高值")
        return self


class ProductCapabilityVersionSave(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    product_type: str | None = Field(default=None, min_length=1, max_length=120)
    is_enabled: bool | None = None
    currency: str | None = Field(default=None, min_length=1, max_length=10)
    cost_min: int | None = Field(default=None, ge=0)
    cost_max: int | None = Field(default=None, ge=0)
    retail_min: int | None = Field(default=None, ge=0)
    retail_max: int | None = Field(default=None, ge=0)
    minimum_order_quantity: int | None = Field(default=None, ge=1)
    lead_time_days: int | None = Field(default=None, ge=0)
    materials: list[str] | None = Field(default=None, max_length=200)
    processes: list[str] | None = Field(default=None, max_length=200)
    packaging: list[str] | None = Field(default=None, max_length=200)
    constraints: list[str] | None = Field(default=None, max_length=200)
    quality_requirements: list[str] | None = Field(default=None, max_length=200)
    components: list[ProductComponentSave] | None = Field(default=None, max_length=100)


class ExpressionPropositionSave(BaseModel):
    owner_agent_id: int | None = Field(default=None, gt=0)
    title: str = Field(min_length=1, max_length=500)
    core_expression: str = Field(min_length=1, max_length=10000)
    target_emotion: str | None = Field(default=None, max_length=5000)
    audience: list[str] = Field(default_factory=list, max_length=100)
    scenes: list[str] = Field(default_factory=list, max_length=100)
    purchase_reason: str | None = Field(default=None, max_length=5000)
    exploration_angles: list[str] = Field(default_factory=list, max_length=100)
    theme_form: str = Field(default="概念", min_length=1, max_length=100)
    creative_seeds: list[dict[str, str]] = Field(default_factory=list, max_length=50)
    avoid: list[str] = Field(default_factory=list, max_length=100)
    risks: list[str] = Field(default_factory=list, max_length=100)
    status: Literal["draft", "reviewing", "qualified", "watching", "rejected"] = "qualified"
    actor_kind: Literal["agent", "system", "user"] = "agent"


class PropositionCreativeContextSave(BaseModel):
    theme_form: str = Field(min_length=1, max_length=100)
    creative_seeds: list[dict[str, str]] = Field(min_length=1, max_length=50)
    actor_kind: Literal["agent", "system", "user"] = "agent"


class ProductFitAssessmentSave(BaseModel):
    proposition_id: int = Field(gt=0)
    product_capability_id: int = Field(gt=0)
    owner_agent_id: int | None = Field(default=None, gt=0)
    conclusion: Literal["fit", "possible", "not_fit", "needs_validation"]
    rationale: str = Field(min_length=1, max_length=10000)
    target_audience: list[str] = Field(default_factory=list, max_length=100)
    scenes: list[str] = Field(default_factory=list, max_length=100)
    risks: list[str] = Field(default_factory=list, max_length=100)
    missing_information: list[str] = Field(default_factory=list, max_length=100)
    actor_kind: Literal["agent", "system", "user"] = "agent"


class PendingProductFitRun(BaseModel):
    query: str | None = Field(default=None, max_length=200)
    proposition_ids: list[int] = Field(default_factory=list, max_length=200)


class ProductionRuleSave(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    scope: Literal["global", "product", "supplier"] = "global"
    product_capability_id: int | None = Field(default=None, gt=0)
    rule_type: str = Field(min_length=1, max_length=80)
    rule: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True


class DesignJobClaim(BaseModel):
    opportunity_id: int = Field(gt=0)
    owner_agent_id: int = Field(gt=0)


class DesignJobTransition(BaseModel):
    stage: Literal[
        "intake", "brief_ready", "references_learning", "routes_generating",
        "routes_reviewing", "master_generating", "master_reviewing",
        "master_approved", "renders_generating", "renders_reviewing",
        "production_files_generating", "production_files_reviewing",
        "design_completed", "rework_required", "waiting_retry", "deferred",
        "discarded", "failed",
    ]
    status: str | None = Field(default=None, max_length=30)
    automation_status: str | None = Field(default=None, max_length=30)
    next_action: str | None = Field(default=None, max_length=10000)
    last_error: str | None = Field(default=None, max_length=10000)
    reason: str = Field(min_length=1, max_length=10000)
    actor_kind: Literal["agent", "system", "user"] = "agent"
    actor_agent_id: int | None = Field(default=None, gt=0)
