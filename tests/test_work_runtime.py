from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import app.work_executor as work_executor_module
from app.database import Database
from app.business_service import BusinessService
from app.collaboration_service import CollaborationService
from app.env_store import EnvStore
from app.main import create_app, normalize_legacy_temporal_text
from app.models import AgentProfile, Evidence, MarketInsightRecord, ResearchCandidate, ResearchReport, ToolCall, WorkCollaboration, WorkContextSummary, WorkEvent, WorkGovernanceIssue, WorkModelCall, WorkProject, WorkRun, WorkThread
from app.tool_runtime import ToolContext, ToolRegistry, ToolSpec
from app.work_executor import (
    WorkExecutor,
    WorkExecutorError,
    MarketInsight,
    WorkStatePatch,
    compact_tool_result,
    governed_progress,
    has_continuation_intent,
    merge_work_state,
    parse_decision,
    requested_duration_minutes,
    normalized_discovery_candidates,
    discovery_source_candidates,
    salient_research_excerpt,
    ORDINARY_PARTICIPATION_LENSES,
)
from app.work_context_service import WorkContextService, compact_work_state
from app.work_budget_service import ModelUsage, WorkBudgetService, derive_model_circuit_states, select_model_routes
from app.report_service import ReportService
from app.work_runtime import WorkLeaseLost, WorkRuntime
from app.work_governance_service import WorkGovernanceService


def test_continuation_intent_requires_affirmative_language():
    assert has_continuation_intent("继续之前未完成的任务") is True
    assert has_continuation_intent("重新开始完成之前未完成工作") is True
    assert has_continuation_intent("全新市场研究任务") is False
    assert has_continuation_intent("不要使用之前项目内容，重新做") is False


def test_decision_parser_locally_normalizes_unambiguous_wait_and_empty_batch():
    wait = parse_decision('{"action":{"type":"wait","message":"等待硬截止"}}')
    assert wait.action is not None
    assert wait.action.wait_status == "waiting_schedule"

    state_only = parse_decision(
        '{"state_summary":"已完成状态核验","state_update":{"phase":"等待硬截止收口"},"actions":[]}'
    )
    assert state_only.action is not None
    assert state_only.action.type == "plan_update"

    both = parse_decision(
        '{"action":{"type":"wait","message":"等待"},"actions":[{"type":"tool_call","step_id":"x","tool_name":"web_search","arguments":{}}]}'
    )
    assert both.action is not None
    assert both.action.type == "wait"

    mixed = parse_decision(
        '{"actions":[{"type":"plan_update","message":"换路"},{"type":"tool_call","step_id":"x","tool_name":"web_search","arguments":{}}]}'
    )
    assert mixed.action is not None
    assert mixed.action.type == "plan_update"


def test_business_backlog_gates_new_research_and_exposes_open_association_rules(
    tmp_path: Path,
):
    database = Database(f"sqlite:///{tmp_path / 'business-gate.db'}")
    database.create_tables()
    with database.sessions() as session:
        worker = AgentProfile(name="市场Agent", is_controller=False)
        session.add(worker)
        session.flush()
        thread = WorkThread(
            owner_agent_id=worker.id,
            title="开放市场研究",
            objective="从社会事件和生活讨论形成表达主题与产品机会",
        )
        session.add(thread)
        session.flush()
        session.add_all(
            [
                Evidence(
                    work_thread_id=thread.id,
                    owner_agent_id=worker.id,
                    source_url="https://example.com/coast",
                    source_title="海边生活讨论",
                    published_at="2026-08-20",
                    claim="人们分享在海边放慢生活节奏。",
                    excerpt="海岸空间成为休息入口。",
                    confidence="high",
                    source_type="independent_secondary",
                    source_publisher="example.com",
                    content_hash="coast-a",
                ),
                Evidence(
                    work_thread_id=thread.id,
                    owner_agent_id=worker.id,
                    source_url="https://example.org/coast",
                    source_title="海岸休息行为观察",
                    published_at="2026-08-21",
                    claim="另一来源记录了相同的海岸休息行为。",
                    excerpt="普通人把海边作为短暂离开压力的空间。",
                    confidence="high",
                    source_type="independent_secondary",
                    source_publisher="example.org",
                    content_hash="coast-b",
                ),
            ]
        )
        insight = MarketInsightRecord(
            work_thread_id=thread.id,
            owner_agent_id=worker.id,
            title="海边生活讨论",
            category="生活文化",
            source_refs_json='["https://example.com/coast", "https://example.org/coast"]',
            insight_json=json.dumps(
                {
                    "title": "海边生活讨论",
                    "signal": "人们分享在海边放慢生活节奏。",
                    "phenomenon": "海岸空间成为离开日常压力的想象入口。",
                    "why_interesting": "工作节奏与休息愿望之间存在张力。",
                    "source_refs": [
                        "https://example.com/coast",
                        "https://example.org/coast",
                    ],
                    "creative_theme_directions": [
                        {
                            "title": "暂时离开日常",
                            "inspiration_path": "从海边讨论联想到度假与自由。",
                            "human_meaning": "允许自己短暂离开效率要求。",
                            "emotions": ["自由", "松弛"],
                            "symbols": ["海浪", "沙滩"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            search_text="海边 生活 度假",
            content_hash="b" * 64,
        )
        session.add(insight)
        session.commit()
        worker_id, thread_id, insight_id = worker.id, thread.id, insight.id
    BusinessService(database).ingest_market_insight(insight_id)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    executor = app.state.work_executor
    executor.set_tool_permission(worker_id, "web_search", "allow")
    runtime = app.state.work_runtime
    run = runtime.create_run(thread_id)
    with database.sessions() as session:
        thread = session.get(WorkThread, thread_id)
        assert thread is not None
    checkpoint: dict[str, Any] = {}
    prompt = executor._build_prompt(
        thread, run, checkpoint, executor.list_tool_permissions(worker_id)
    )

    assert checkpoint["offered_tool_names"] == [
        "get_business_hotspot",
        "list_business_hotspots",
        "save_expression_proposition",
    ]
    assert '"stage": "expression_proposition"' in prompt
    assert "开放联想规则" in prompt
    assert "creative_theme_directions" in prompt
    assert "和平、反战、度假" in prompt

    executor.decision_provider = lambda *_: work_decision(
        {
            "type": "fail",
            "message": "搜索工具暂时不可用，因此工作失败",
        }
    )
    executor.max_steps_per_run = 1
    claimed = runtime.claim_next(agent_id=worker_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)
    assert result.outcome == "yielded"
    assert result.run.status == "queued"
    saved_checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert saved_checkpoint["business_stage_failure_rejections"] == 1
    assert "禁止将任务标记为失败" in saved_checkpoint["action_required"]


class NoopCodex:
    def close(self):
        return None


def test_governance_scan_is_idempotent_and_clears_stalled_issue(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="governance-scan")
    first = runtime.create_thread(
        owner_agent_id=agent_id,
        title="收集主要应用用户数据",
        objective="搜索并核验注册用户超过五千万的应用，整理可靠来源",
    )
    second = runtime.create_thread(
        owner_agent_id=agent_id,
        title="收集主要应用用户数据副本",
        objective="搜索并核验注册用户超过五千万的应用，整理可靠来源",
    )
    runtime.create_run(first.id)
    runtime.create_run(second.id)
    with database.sessions() as session:
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        run = session.scalar(select(WorkRun).where(WorkRun.work_thread_id == first.id))
        assert run is not None
        run.updated_at = old
        thread = session.get(WorkThread, first.id)
        assert thread is not None
        thread.updated_at = old
        session.commit()

    service = WorkGovernanceService(
        database,
        runtime,
        {
            "work_governance": {
                "duplicate_similarity_threshold": 0.7,
                "stalled_minutes": 30,
            }
        },
    )
    initial = service.scan()
    repeated = service.scan()
    assert initial["created"] == 2
    assert repeated["created"] == 0
    with database.sessions() as session:
        assert session.query(WorkGovernanceIssue).count() == 2
        run = session.scalar(select(WorkRun).where(WorkRun.work_thread_id == first.id))
        assert run is not None
        run.updated_at = datetime.now(timezone.utc)
        thread = session.get(WorkThread, first.id)
        assert thread is not None
        thread.updated_at = datetime.now(timezone.utc)
        session.commit()
    cleared = service.scan()
    assert cleared["resolved"] == 1
    assert len(service.list_issues(status="open", issue_type="stalled")) == 0
    with database.sessions() as session:
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        run = session.scalar(select(WorkRun).where(WorkRun.work_thread_id == first.id))
        thread = session.get(WorkThread, first.id)
        assert run is not None and thread is not None
        run.updated_at = old
        thread.updated_at = old
        session.commit()
    service.scan()
    assert len(service.list_issues(status="open", issue_type="stalled")) == 1


def test_governance_rework_preserves_original_and_is_single_use(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'governance-rework.db'}")
    database.create_tables()
    with database.sessions() as session:
        controller = AgentProfile(name="总控", is_controller=True)
        worker = AgentProfile(name="市场", is_controller=False)
        session.add_all([controller, worker])
        session.commit()
        controller_id = controller.id
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="governance-rework")
    original = runtime.create_thread(
        owner_agent_id=worker_id,
        title="市场报告",
        objective="形成市场报告",
        success_criteria=["引用两条可靠来源"],
        constraints=["不能编造"],
    )
    runtime.create_run(original.id)
    claimed = runtime.claim_next(agent_id=worker_id)
    assert claimed is not None
    runtime.complete(claimed.id, claimed.execution_token, "第一版结果")
    service = WorkGovernanceService(database, runtime, {})
    service.scan()
    issue = service.list_issues(status="open", issue_type="quality_review")[0]

    reviewed = service.review_quality(
        issue.id,
        reviewer_agent_id=controller_id,
        decision="rework",
        notes="来源数量不足，请补证后重新提交",
    )
    assert reviewed.status == "rework_created"
    assert reviewed.rework_thread_id is not None
    with database.sessions() as session:
        old = session.get(WorkThread, original.id)
        rework = session.get(WorkThread, reviewed.rework_thread_id)
        assert old is not None and old.status == "completed"
        assert old.result_text == "第一版结果"
        assert rework is not None and rework.parent_thread_id == original.id
        assert rework.status == "queued"
        run = session.scalar(select(WorkRun).where(WorkRun.work_thread_id == rework.id))
        assert run is not None and run.trigger_kind == "governance_rework"
    with pytest.raises(Exception, match="已经处理"):
        service.review_quality(
            issue.id,
            reviewer_agent_id=controller_id,
            decision="rework",
            notes="重复点击",
        )


def test_governance_tools_are_controller_only(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'governance-permission.db'}")
    database.create_tables()
    with database.sessions() as session:
        controller = AgentProfile(name="总控", is_controller=True)
        worker = AgentProfile(name="员工", is_controller=False)
        session.add_all([controller, worker])
        session.commit()
        controller_id, worker_id = controller.id, worker.id
    runtime = WorkRuntime(database)
    registry = ToolRegistry()
    for name in ("inspect_company_work", "list_governance_issues", "review_work_quality", "resolve_governance_issue"):
        registry.register(ToolSpec(name=name, description=name, input_schema={"type": "object"}, handler=lambda *_: {}))
    executor = WorkExecutor(database, runtime, registry, lambda *_: "{}")
    assert executor.list_tool_permissions(controller_id)["inspect_company_work"] == "allow"
    assert executor.list_tool_permissions(controller_id)["read_web_page"] == "allow"
    assert executor.list_tool_permissions(worker_id)["inspect_company_work"] == "deny"
    assert executor.list_tool_permissions(worker_id)["read_web_page"] == "deny"


def test_governance_evidence_risk_ignores_negative_constraint():
    assert WorkGovernanceService._criterion_requires_evidence("逐项提供可靠证据")
    assert not WorkGovernanceService._criterion_requires_evidence("不得保存未经核验的证据")


def test_work_decision_accepts_empty_optional_collaboration_id():
    decision = parse_decision(
        json.dumps(
            {
                "state_summary": "准备查询同事",
                "progress": 0.1,
                "plan": [],
                "action": {
                    "type": "tool_call",
                    "step_id": "list-agents",
                    "tool_name": "list_company_agents",
                    "arguments": {},
                    "collaboration_id": "",
                },
            },
            ensure_ascii=False,
        )
    )
    assert decision.action.collaboration_id is None
    waiting = parse_decision(
        json.dumps(
            {
                "state_summary": "等待协作",
                "progress": 0.5,
                "plan": [],
                "action": {
                    "type": "wait_collaboration",
                    "collaboration_id": "",
                },
            }
        )
    )
    assert waiting.action.collaboration_id is None


def test_work_decision_accepts_trailing_explanation_after_valid_json():
    decision = parse_decision(
        work_decision({"type": "complete", "message": "任务完成"})
        + "\n以上是本轮决定。"
    )

    assert decision.action is not None
    assert decision.action.type == "complete"
    assert decision.action.message == "任务完成"


def test_work_decision_ignores_extra_json_after_valid_decision():
    decision = parse_decision(
        "```json\n"
        + work_decision({"type": "complete", "message": "采用首个决定"})
        + "\n```\n"
        + json.dumps({"note": "额外说明"}, ensure_ascii=False)
    )

    assert decision.action is not None
    assert decision.action.message == "采用首个决定"


def test_work_decision_normalizes_direct_and_flattened_action_shapes():
    direct = parse_decision(
        json.dumps(
            {
                "type": "tool_call",
                "step_id": "search",
                "tool_name": "web_search",
                "arguments": {"query": "生活热点"},
            },
            ensure_ascii=False,
        )
    )
    assert direct.action is not None
    assert direct.action.tool_name == "web_search"

    flattened = parse_decision(
        json.dumps(
            {
                "decision_summary": "继续阅读全文",
                "consumed_tool_call_ids": [88],
                "type": "tool_call",
                "step_id": "read",
                "tool_name": "read_web_page",
                "arguments": {"url": "https://example.com/story"},
            },
            ensure_ascii=False,
        )
    )
    assert flattened.decision_summary == "继续阅读全文"
    assert flattened.consumed_tool_call_ids == [88]
    assert flattened.action is not None
    assert flattened.action.tool_name == "read_web_page"


def test_missing_collaboration_id_is_replanned_without_retry_pause(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="collab-replan")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="先委派再等待",
        objective="组织员工完成任务",
    )
    runtime.create_run(thread.id)
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {"type": "wait_collaboration", "message": "等待员工结果"}
            )
        assert "不能用wait_collaboration代替委派" in prompt
        assert '"outbound_collaborations": []' in prompt
        return work_decision({"type": "complete", "message": "已重新规划"})

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, max_steps_per_run=3
    )
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert len(prompts) == 2
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["invalid_collaboration_waits"] == 1


def test_executor_repairs_one_malformed_decision_without_repeating_actions(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="format-repair")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="格式修复", objective="完成一次决定"
    )
    runtime.create_run(thread.id)
    responses = iter(
        [
            '{"state_summary":"截断","progress":0.5,"action":',
            work_decision({"type": "complete", "message": "修复后完成"}),
        ]
    )
    prompts: list[str] = []

    def provider(thread, run, prompt):
        prompts.append(prompt)
        return next(responses)

    executor = WorkExecutor(database, runtime, ToolRegistry(), provider)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)
    assert result.outcome == "completed"
    assert result.run.result_text == "修复后完成"
    assert len(prompts) == 2
    assert "上一条响应不是可解析的完整JSON" in prompts[1]
    assert result.run.checkpoint_json
    assert json.loads(result.run.checkpoint_json)["decision_format_repairs"] == 1


def legacy_format_repair_preserves_search_material_consumption_and_deep_read(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="material-repair")
    registry = ToolRegistry()
    candidate_url = "https://example.com/capybara-plush"
    executed: list[str] = []

    registry.register(
        ToolSpec(
            name="web_search",
            description="搜索公开网页",
            input_schema={"type": "object"},
            handler=lambda *_: executed.append("web_search")
            or {
                "query": "巨型水豚玩偶",
                "results": [
                    {
                        "title": "普通人分享巨型水豚玩偶",
                        "url": candidate_url,
                        "snippet": "购买者正在发布合影和讨论摆放方式",
                    }
                ],
            },
        )
    )
    registry.register(
        ToolSpec(
            name="read_web_page",
            description="读取公开网页正文",
            input_schema={"type": "object"},
            handler=lambda *_: executed.append("read_web_page")
            or {
                "url": candidate_url,
                "resolved_url": candidate_url,
                "title": "巨型水豚玩偶成为合影道具",
                "text": "购买者把巨型水豚玩偶带回家后拍摄合影，并讨论如何搬运、摆放和送礼。" * 30,
            },
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "search",
                    "tool_name": "web_search",
                    "arguments": {"query": "巨型水豚玩偶"},
                    "purpose": "寻找普通人参与线索",
                    "criterion_id": "目标",
                    "expected_delta": "得到可阅读全文的候选",
                    "stop_condition": "出现具体参与候选",
                }
            )
        if len(prompts) == 2:
            assert candidate_url in prompt
            assert "隔离的‘搜索结果提纯’工作包" in prompt
            assert len(prompt) < 20_000
            return '{"decision_summary":"选择候选","state_update":'
        if len(prompts) == 3:
            assert "不能退化成只有action" in prompt
            assert "consumed_tool_call_ids必须包含上述ID" in prompt
            assert candidate_url in prompt
            matched = re.search(r"材料ID \[(\d+)\]", prompt)
            assert matched is not None
            search_call_id = int(matched.group(1))
            return json.dumps(
                {
                    "decision_summary": "候选包含具体购买和分享行为，选择阅读全文",
                    "state_update": {
                        "current_judgment": "巨型水豚玩偶可能形成生活消费谈资",
                        "current_strategy": "读取代表正文判断参与方式",
                        "market_candidates": [
                            {
                                "candidate_ref": candidate_url,
                                "title": "巨型水豚玩偶",
                                "lens": "生活消费",
                                "decision": "deep_read",
                                "reason": "出现购买、合影和讨论行为",
                                "source_url": candidate_url,
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "找到可阅读全文的普通人参与线索",
                        "strategy_change_required": False,
                    },
                    "consumed_tool_call_ids": [search_call_id],
                    "action": {
                        "type": "tool_call",
                        "step_id": "read-candidate",
                        "tool_name": "read_web_page",
                        "arguments": {"url": candidate_url},
                        "purpose": "核对具体参与行为",
                        "criterion_id": "目标",
                        "expected_delta": "获得完整正文",
                        "stop_condition": "足以判断是否保留",
                    },
                },
                ensure_ascii=False,
            )
        assert "当前全文" in prompt
        matched = re.search(r"必须把 (\d+) 放入consumed_tool_call_ids", prompt)
        assert matched is not None
        page_call_id = int(matched.group(1))
        return json.dumps(
            {
                "decision_summary": "正文确认购买后合影与讨论行为，保留线索",
                "state_update": {
                    "current_judgment": "巨型玩偶正在成为家庭合影和送礼谈资",
                    "market_candidates": [
                        {
                            "candidate_ref": candidate_url,
                            "title": "巨型水豚玩偶",
                            "lens": "生活消费",
                            "decision": "keep",
                            "reason": "正文描述购买、合影和摆放讨论",
                            "source_url": candidate_url,
                        }
                    ],
                    "market_insights": [
                        {
                            "title": "巨型玩偶成为家庭社交道具",
                            "category": "生活消费",
                            "signal": "购买者分享合影并讨论搬运和摆放",
                            "phenomenon": "大体积可爱玩偶从商品变成社交谈资",
                            "why_interesting": "夸张尺寸带来反差和可模仿内容",
                            "source_refs": [f"tool_call:{page_call_id}", candidate_url],
                        }
                    ],
                },
                "observation_assessment": {
                    "information_gain": "useful",
                    "expectation_met": True,
                    "finding": "正文补足具体参与行为",
                    "strategy_change_required": False,
                },
                "consumed_tool_call_ids": [page_call_id],
                "action": {"type": "plan_update"},
            },
            ensure_ascii=False,
        )

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=3
    )
    executor.set_tool_permission(agent_id, "web_search", "allow")
    executor.set_tool_permission(agent_id, "read_web_page", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="搜索结果格式修复",
        objective="寻找普通人参与的有趣消费现象",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "yielded"
    assert executed == ["web_search", "read_web_page"]
    assert len(prompts) == 4
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["decision_format_repairs"] == 1
    assert "pending_research_material_ids" not in checkpoint
    assert "rejected_plan_updates" not in checkpoint
    assert checkpoint["work_state"]["market_insights"][0]["title"] == (
        "巨型玩偶成为家庭社交道具"
    )


def legacy_direct_action_during_search_refinement_reprompts_without_retry_pause(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="direct-action")
    registry = ToolRegistry()
    candidate_url = "https://example.com/weak-candidate"
    registry.register(
        ToolSpec(
            name="web_search",
            description="搜索公开网页",
            input_schema={"type": "object"},
            handler=lambda *_: {
                "query": "普通人生活热点",
                "results": [
                    {
                        "title": "泛化标题",
                        "url": candidate_url,
                        "snippet": "没有具体事件或参与行为",
                    }
                ],
            },
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "search",
                    "tool_name": "web_search",
                    "arguments": {"query": "普通人生活热点"},
                }
            )
        if len(prompts) == 2:
            assert "搜索结果提纯" in prompt
            # This is the exact provider shape that previously caused a
            # one-minute waiting_retry loop.
            return '{"type":"plan_update"}'
        if len(prompts) == 3:
            assert "搜索结果提纯" in prompt
            assert "本次不完整决定未写入" in prompt
            matched = re.search(r"把 (\d+) 写入consumed_tool_call_ids", prompt)
            assert matched is not None
            material_id = int(matched.group(1))
            return json.dumps(
                {
                    "decision_summary": "候选过于空泛，本轮放弃并更换来源",
                    "state_update": {
                        "current_judgment": "本次候选没有具体事件或参与行为",
                        "current_strategy": "下一轮改用事件型来源",
                        "strategy_change_reason": "泛化结果缺少业务增量",
                        "rejected_paths": [f"{candidate_url}：内容过于空泛"],
                    },
                    "observation_assessment": {
                        "information_gain": "low",
                        "expectation_met": False,
                        "finding": "当前来源无法提供具体生活现象",
                        "strategy_change_required": True,
                    },
                    "consumed_tool_call_ids": [material_id],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        return work_decision({"type": "complete", "message": "已完成换路判断"})

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=3
    )
    executor.set_tool_permission(agent_id, "web_search", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="搜索提纯单动作兼容",
        objective="寻找具体的普通人生活热点",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert len(prompts) == 4
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["unconsumed_discovery_decisions"] == 1
    assert "pending_research_material_ids" not in checkpoint
    with database.sessions() as session:
        pauses = session.scalars(
            select(WorkEvent).where(
                WorkEvent.work_thread_id == thread.id,
                WorkEvent.event_type == "run_paused",
            )
        ).all()
    assert pauses == []


def test_format_repair_keeps_registered_tool_enum(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="format-tool-enum")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="格式修复工具约束",
        objective="继续恢复任务时，格式修复不能创造账本读取工具",
    )
    runtime.create_run(thread.id)
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="read_work_history",
            description="读取原始工作账本",
            input_schema={"type": "object"},
            handler=lambda *_: {"items": []},
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return (
                '{"state_summary":"准备读取完整结果","action":'
                '{"type":"tool_call","step_id":"recover","tool_name":'
                '"read_tool_ledger_result"'
            )
        if len(prompts) == 2:
            assert "tool_name只能从这个枚举中选择" in prompt
            assert '"read_work_history"' in prompt
            assert "不能创造名称" in prompt
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "recover",
                    "tool_name": "read_work_history",
                    "arguments": {"kind": "tool_calls", "limit": 1},
                }
            )
        assert "read_work_history" in prompt
        return work_decision(
            {"type": "complete", "message": "已使用注册工具完成恢复"},
            progress=1,
        )

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=3
    )
    executor.set_tool_permission(agent_id, "read_work_history", "allow")
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert result.steps_executed == 2
    calls = executor.list_tool_calls(thread.id)
    assert [call.tool_name for call in calls] == ["read_work_history"]
    assert json.loads(result.run.checkpoint_json)["decision_format_repairs"] == 1


def test_work_context_summary_is_versioned_and_preserves_original_ledgers(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="context-summary")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="长任务上下文",
        objective="保留目标、失败方法、证据与继续位置",
        success_criteria=["结论有证据"],
        constraints=["不得删除原始记录"],
    )
    run = runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    with database.sessions() as session:
        call = ToolCall(
            work_thread_id=thread.id,
            work_run_id=run.id,
            owner_agent_id=agent_id,
            step_id="failed-search",
            action_key="c" * 64,
            tool_name="web_search",
            arguments_json=json.dumps({"query": "first method"}),
            status="failed",
            permission="allow",
            retry_safety="safe",
            execution_token=1,
            error="provider unavailable",
        )
        session.add(call)
        session.add(
            Evidence(
                work_thread_id=thread.id,
                owner_agent_id=agent_id,
                source_url="https://example.com/official",
                source_title="Official",
                claim="已确认事实",
                excerpt="Evidence text",
                confidence="high",
                stance="supports",
                content_hash="d" * 64,
            )
        )
        session.commit()
        original_tool_count = session.query(ToolCall).count()
        original_event_count = session.query(WorkEvent).count()
        original_evidence_count = session.query(Evidence).count()

    service = WorkContextService(
        database,
        {"work_context": {"summary_interval_tool_calls": 1}},
    )
    checkpoint = {
        "state_summary": "已经尝试搜索，下一步核对官方来源",
        "progress": 0.4,
        "plan_version": 2,
        "plan": [
            {"id": "a", "title": "搜索", "status": "completed"},
            {"id": "b", "title": "核验", "status": "in_progress"},
        ],
        "last_action": {"type": "tool_call", "tool_name": "web_search"},
    }
    first = service.ensure_summary(thread.id, checkpoint, force=True)
    assert first is not None
    assert first.version == 1
    payload = json.loads(first.summary_json)
    assert payload["objective"] == "保留目标、失败方法、证据与继续位置"
    assert payload["failed_methods"][0]["error"] == "provider unavailable"
    assert payload["evidence_refs"][0]["claim"] == "已确认事实"
    assert payload["continuation"]["last_action"]["tool_name"] == "web_search"

    checkpoint["state_summary"] = "进入第二阶段"
    second = service.ensure_summary(thread.id, checkpoint, force=True)
    assert second is not None
    assert second.version == 2
    with database.sessions() as session:
        assert session.query(WorkContextSummary).count() == 2
        assert session.query(ToolCall).count() == original_tool_count
        assert session.query(WorkEvent).count() == original_event_count
        assert session.query(Evidence).count() == original_evidence_count


def test_work_prompt_is_bounded_and_history_tool_recovers_original_records(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="context-budget")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="预算上下文", objective="测试长断点裁剪"
    )
    run = runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    with database.sessions() as session:
        for index in range(7):
            session.add(
                ToolCall(
                    work_thread_id=thread.id,
                    work_run_id=run.id,
                    owner_agent_id=agent_id,
                    step_id=f"lookup-{index}",
                    action_key=f"{index:064x}",
                    tool_name="lookup",
                    arguments_json=json.dumps({"index": index}),
                    status="completed",
                    permission="allow",
                    retry_safety="safe",
                    execution_token=1,
                    result_json=json.dumps({"raw": "x" * 5000}),
                )
            )
        session.commit()
    service = WorkContextService(
        database,
        {
            "work_context": {
                "max_prompt_chars": 12000,
                "checkpoint_max_chars": 4000,
                "recent_tool_calls": 3,
                "recent_result_chars": 500,
                "summary_interval_tool_calls": 1,
            }
        },
    )
    checkpoint = {
        "state_summary": "长任务",
        "progress": 0.5,
        "plan": [],
        "recent_tool_results": [{"raw": "y" * 30000}],
        "oversized_extra": "z" * 30000,
    }
    context = service.prompt_context(thread.id, checkpoint)
    encoded = json.dumps(context, ensure_ascii=False)
    assert len(encoded) < 12000
    assert len(context["recent_tool_calls"]) == 3
    assert "recent_tool_results" not in context["current_checkpoint"]
    history = service.history_tool(
        ToolContext(thread.id, run.id, agent_id, 1),
        {"kind": "tool_calls", "offset": 0, "limit": 7},
    )
    assert len(history["items"]) == 7
    assert history["items"][0]["result"]["raw"] == "x" * 5000
    tool_call_id = history["items"][0]["id"]
    chunks: list[str] = []
    result_offset = 0
    while True:
        page = service.history_tool(
            ToolContext(thread.id, run.id, agent_id, 1),
            {
                "kind": "tool_calls",
                "tool_call_id": f"tool_call:{tool_call_id}",
                "result_offset": result_offset,
                "result_limit": 1000,
            },
        )
        chunks.append(page["result_chunk"])
        if not page["has_more"]:
            break
        result_offset = page["next_result_offset"]
    assert json.loads("".join(chunks))["raw"] == "x" * 5000

    compacted = compact_tool_result({"raw": "z" * 20000}, max_chars=100)
    assert compacted["truncated_for_checkpoint"] is True
    assert "read_work_history" in compacted["note"]
    assert "tool_call_id=<数字>" in compacted["note"]
    assert "创造新工具名" in compacted["note"]


def test_large_summary_is_folded_to_hard_prompt_budget(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="hard-context-budget")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="巨大摘要",
        objective="o" * 20000,
        success_criteria=["c" * 1000 for _ in range(20)],
    )
    service = WorkContextService(
        database,
        {
            "work_context": {
                "max_prompt_chars": 12000,
                "checkpoint_max_chars": 4000,
                "summary_interval_tool_calls": 1,
            }
        },
    )
    service.ensure_summary(
        thread.id,
        {
            "state_summary": "s" * 20000,
            "plan": [
                {"id": str(i), "title": "p" * 500, "status": "completed"}
                for i in range(50)
            ],
        },
        force=True,
    )
    context = service.prompt_context(
        thread.id, {"state_summary": "s" * 20000, "oversized": "z" * 30000}
    )
    assert len(json.dumps(context, ensure_ascii=False)) <= 12000
    assert context["context_budget"]["actual_chars"] == len(
        json.dumps(context, ensure_ascii=False)
    )
    assert context["context_budget_truncated"] is True


def test_work_context_settings_api_and_tool_registration(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'context-api.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8")
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        defaults = client.get("/api/settings/work-context")
        assert defaults.status_code == 200
        payload = defaults.json()
        payload.update({"max_prompt_chars": 50000, "checkpoint_max_chars": 20000})
        saved = client.put("/api/settings/work-context", json=payload)
        assert saved.status_code == 200
        assert saved.json()["max_prompt_chars"] == 50000
        tools = {item["name"] for item in client.get("/api/tools").json()}
        assert "read_work_history" in tools
        budget_defaults = client.get("/api/settings/work-budget")
        assert budget_defaults.status_code == 200
        budget_payload = budget_defaults.json()
        budget_payload.update(
            {"max_model_tokens_per_tree": 250000, "max_active_minutes_per_tree": 120}
        )
        budget_saved = client.put("/api/settings/work-budget", json=budget_payload)
        assert budget_saved.status_code == 200
        assert budget_saved.json() == budget_payload
        history_tool = next(
            item for item in client.get("/api/tools").json()
            if item["name"] == "read_work_history"
        )
        assert "model_calls" in history_tool["input_schema"]["properties"]["kind"]["enum"]


def test_work_budget_threshold_is_shared_but_triggers_review_not_terminal(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="work-budget")
    parent = runtime.create_thread(
        owner_agent_id=agent_id, title="父任务", objective="验证共享模型预算"
    )
    child = runtime.create_thread(
        owner_agent_id=agent_id,
        title="子任务",
        objective="不能绕过父任务预算",
        parent_thread_id=parent.id,
        priority=10,
    )
    parent_run = runtime.create_run(parent.id)
    child_run = runtime.create_run(child.id)
    service = WorkBudgetService(
        database,
        {"work_budget": {"max_model_tokens_per_tree": 1000, "max_active_minutes_per_tree": 60}},
    )
    service.record_model_call(
        thread_id=parent.id,
        run_id=parent_run.id,
        owner_agent_id=agent_id,
        execution_token=1,
        purpose="decision",
        attempt_number=1,
        model_id=None,
        model_name="测试模型",
        status="completed",
        prompt="prompt",
        output="result",
        usage=ModelUsage(700, 300, 1000, "provider"),
        duration_ms=500,
    )
    called = False

    def provider(*_):
        nonlocal called
        called = True
        return work_decision({"type": "complete", "message": "审查后正常完成"}, progress=1)

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, budget_service=service
    )
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None and claimed.id == child_run.id
    result = executor.execute(claimed.id, claimed.execution_token)
    assert result.outcome == "completed"
    assert result.run.status == "completed"
    assert called is True
    usage = service.usage(child.id)
    assert usage["thread_ids"] == [parent.id, child.id]
    assert usage["model_tokens"] == 1000
    assert usage["active_ms"] == 500
    assert usage["review_required"] is True
    assert usage["review_reasons"] == ["model_tokens"]
    assert usage["exceeded"] is False


def test_failed_estimated_attempts_are_audited_but_do_not_exhaust_budget(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="budget-estimates")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="失败重试账本", objective="区分真实用量与失败估算"
    )
    run = runtime.create_run(thread.id)
    service = WorkBudgetService(
        database,
        {
            "work_budget": {
                "max_model_tokens_per_tree": 1000,
                "max_active_minutes_per_tree": 60,
            }
        },
    )
    service.record_model_call(
        thread_id=thread.id,
        run_id=run.id,
        owner_agent_id=agent_id,
        execution_token=1,
        purpose="decision",
        attempt_number=1,
        model_id=None,
        model_name="测试模型",
        status="failed",
        prompt="network attempt",
        usage=ModelUsage(1200, 0, 1200, "estimated"),
        error_kind="transient",
        error="502",
        duration_ms=10,
    )
    service.record_model_call(
        thread_id=thread.id,
        run_id=run.id,
        owner_agent_id=agent_id,
        execution_token=1,
        purpose="decision",
        attempt_number=2,
        model_id=None,
        model_name="测试模型",
        status="completed",
        prompt="successful attempt",
        output="ok",
        usage=ModelUsage(80, 20, 100, "provider"),
        duration_ms=10,
    )

    usage = service.usage(thread.id)

    assert usage["model_tokens"] == 100
    assert usage["provider_reported_tokens"] == 100
    assert usage["estimated_failed_attempt_tokens"] == 1200
    assert usage["attempted_model_tokens"] == 1300
    assert usage["completed_model_calls"] == 1
    assert usage["failed_model_calls"] == 1
    assert usage["failed_usage_unknown_count"] == 1
    assert usage["unique_failed_prompt_count"] == 1
    assert usage["duplicate_retry_attempts"] == 0
    assert usage["exceeded"] is False


def test_same_failed_prompt_is_estimated_once_across_network_retries(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="retry-dedup")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="重试去重", objective="相同提示只估算一次"
    )
    run = runtime.create_run(thread.id)
    service = WorkBudgetService(database, {})
    for attempt in range(1, 4):
        service.record_model_call(
            thread_id=thread.id,
            run_id=run.id,
            owner_agent_id=agent_id,
            execution_token=1,
            purpose="decision",
            attempt_number=attempt,
            model_id=None,
            model_name="测试模型",
            status="failed",
            prompt="exact same prompt",
            usage=ModelUsage(9000, 0, 9000, "estimated"),
            error_kind="transient",
            error="503",
            duration_ms=10,
        )

    usage = service.usage(thread.id)

    assert usage["estimated_failed_attempt_tokens"] == 9000
    assert usage["estimated_failed_attempt_tokens_raw"] == 27000
    assert usage["failed_usage_unknown_count"] == 3
    assert usage["unique_failed_prompt_count"] == 1
    assert usage["duplicate_retry_attempts"] == 2
    assert usage["model_tokens"] == 0
    with database.sessions() as session:
        for row in session.scalars(
            select(WorkModelCall).where(WorkModelCall.work_thread_id == thread.id)
        ).all():
            row.prompt_hash = ""
        session.commit()
    legacy_usage = service.usage(thread.id)
    assert legacy_usage["estimated_failed_attempt_tokens"] == 9000
    assert legacy_usage["duplicate_retry_attempts"] == 2


def test_event_driven_model_circuit_uses_timed_probe_not_other_model_success():
    failed = [
        SimpleNamespace(
            model_id=1,
            status="failed",
            error_kind="transient",
            error="503",
        )
        for _ in range(3)
    ]
    open_state = derive_model_circuit_states(failed, [1, 2])
    assert open_state[1]["state"] == "open"
    assert open_state[1]["probe_ready"] is False

    alternatives = [
        SimpleNamespace(model_id=2, status="completed", error_kind=None, error=None)
        for _ in range(5)
    ]
    ready = derive_model_circuit_states(failed + alternatives, [1, 2])
    assert ready[1]["state"] == "open"
    assert ready[1]["probe_ready"] is False
    assert ready[1]["attempt_limit"] == 1

    probe_time = datetime.now(timezone.utc)
    timed_failures = [
        SimpleNamespace(
            model_id=1,
            status="failed",
            error_kind="transient",
            error="503",
            finished_at=probe_time - timedelta(minutes=6),
        )
        for _ in range(3)
    ]
    timed_ready = derive_model_circuit_states(
        timed_failures + alternatives,
        [1, 2],
        probe_cooldown_seconds=300,
        now=probe_time,
    )
    assert timed_ready[1]["probe_ready"] is True

    first_probe = SimpleNamespace(
        model_id=1,
        status="completed",
        error_kind=None,
        error=None,
        finished_at=probe_time,
    )
    half_open = derive_model_circuit_states(
        timed_failures + alternatives + [first_probe], [1, 2]
    )
    assert half_open[1]["state"] == "half_open"
    assert half_open[1]["successful_probes"] == 1

    healthy = derive_model_circuit_states(
        timed_failures + alternatives + [first_probe, first_probe], [1, 2]
    )
    assert healthy[1]["state"] == "healthy"
    assert healthy[1]["consecutive_transient_failures"] == 0

    reset_at = datetime.now(timezone.utc)
    historical_failures = [
        SimpleNamespace(
            model_id=1,
            status="failed",
            error_kind="transient",
            error="503",
            finished_at=reset_at - timedelta(minutes=1),
        )
        for _ in range(3)
    ]
    manually_recovered = derive_model_circuit_states(
        historical_failures,
        [1],
        reset_at_by_model={1: reset_at},
    )
    assert manually_recovered[1]["state"] == "healthy"
    assert manually_recovered[1]["probe_ready"] is False


def test_open_primary_is_probed_again_after_cooldown_without_backup_success():
    failed_at = datetime.now(timezone.utc) - timedelta(minutes=6)
    failures = [
        SimpleNamespace(
            model_id=1,
            status="failed",
            error_kind="transient",
            error="503",
            finished_at=failed_at,
        )
        for _ in range(3)
    ]

    states = derive_model_circuit_states(
        failures,
        [1, 2],
        probe_cooldown_seconds=300,
    )
    routed, attempts = select_model_routes(
        [SimpleNamespace(id=1), SimpleNamespace(id=2)],
        states,
        attempts_per_model=3,
    )

    assert states[1]["state"] == "open"
    assert states[1]["probe_ready"] is True
    # The configured primary is due for a single recovery probe and therefore
    # precedes the healthy backup for this decision.
    assert [model.id for model in routed] == [1, 2]
    assert attempts[1] == 1


def test_repeated_balance_auth_failures_open_a_timed_probe_circuit():
    failed_at = datetime.now(timezone.utc) - timedelta(minutes=6)
    failures = [
        SimpleNamespace(
            model_id=2,
            status="failed",
            error_kind="auth",
            error="INSUFFICIENT_BALANCE",
            finished_at=failed_at,
        )
        for _ in range(3)
    ]

    states = derive_model_circuit_states(
        failures,
        [2],
        probe_cooldown_seconds=300,
    )

    assert states[2]["state"] == "open"
    assert states[2]["probe_ready"] is True


def test_all_open_model_routes_probe_every_configured_provider_before_waiting():
    models = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    states = {
        model.id: {
            "state": "open",
            "probe_ready": False,
        }
        for model in models
    }

    routed, attempts = select_model_routes(
        models,
        states,
        attempts_per_model=3,
    )

    assert [model.id for model in routed] == [1, 2]
    assert attempts == {1: 1, 2: 1}


def test_resource_review_does_not_create_successor_thread(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="project-review")
    root = runtime.create_thread(
        owner_agent_id=agent_id,
        title="连续项目",
        objective="达到效率阈值后仍在同一线程完成",
    )
    run = runtime.create_run(root.id)
    service = WorkBudgetService(
        database,
        {
            "work_budget": {
                "max_model_tokens_per_tree": 1000,
                "max_active_minutes_per_tree": 60,
            }
        },
    )
    service.record_model_call(
        thread_id=root.id,
        run_id=run.id,
        owner_agent_id=agent_id,
        execution_token=1,
        purpose="decision",
        attempt_number=1,
        model_id=None,
        model_name="预算测试模型",
        status="completed",
        prompt="prompt",
        output="result",
        usage=ModelUsage(700, 300, 1000, "provider"),
        duration_ms=10,
    )

    def provider(*_):
        return work_decision(
            {"type": "complete", "message": "效率审查后完成"}, progress=1
        )

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, budget_service=service
    )
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    with database.sessions() as session:
        threads = session.scalars(
            select(WorkThread).where(WorkThread.project_id == root.project_id)
        ).all()
        assert [item.id for item in threads] == [root.id]
        project = session.get(WorkProject, root.project_id)
        assert project is not None
        assert project.generation_count == 1


def test_reopen_thread_continues_same_identity_with_new_run(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="same-thread")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="连续工作",
        objective="上下文整理后保持同一线程",
    )
    runtime.create_run(thread.id, checkpoint={"state_summary": "阶段一"})
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    runtime.roll_over(
        claimed.id,
        claimed.execution_token,
        "旧机制资源收口",
        result_text="阶段成果",
    )

    reopened = runtime.reopen_thread(
        thread.id,
        checkpoint={
            "state_summary": "阶段一",
            "work_state": {"phase": "继续阶段"},
        },
    )

    assert reopened.work_thread_id == thread.id
    assert reopened.status == "queued"
    assert runtime.get_thread(thread.id).status == "queued"
    assert len(runtime.list_runs(thread.id)) == 2
    with database.sessions() as session:
        project = session.get(WorkProject, thread.project_id)
        assert project is not None
        assert project.current_root_thread_id == thread.id
        assert project.status == "running"


def test_legacy_blocked_project_recovery_reuses_current_root_thread(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'legacy-recovery.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        controller_id = client.get("/api/agents").json()["controller"]["id"]
        thread = client.post(
            "/api/work/threads",
            json={
                "owner_agent_id": controller_id,
                "title": "旧机制项目",
                "objective": "从已有成果继续",
                "success_criteria": ["形成最终成果"],
            },
        ).json()
        client.post(
            f"/api/work/threads/{thread['id']}/runs",
            json={"checkpoint": {"state_summary": "已有阶段成果"}},
        )
        claimed = app.state.work_runtime.claim_next(agent_id=controller_id)
        assert claimed is not None
        app.state.work_runtime.roll_over(
            claimed.id,
            claimed.execution_token,
            "旧三代上限",
            result_text="已有阶段成果",
        )
        with database.sessions() as session:
            project = session.get(WorkProject, thread["project_id"])
            assert project is not None
            project.status = "blocked_safety"
            session.commit()

        recovered = client.post(
            f"/api/work/projects/{thread['project_id']}/recover-context"
        )

        assert recovered.status_code == 200
        assert recovered.json()["run"]["work_thread_id"] == thread["id"]
        detail = client.get(f"/api/work/threads/{thread['id']}").json()
        assert detail["status"] == "queued"
        assert len(detail["runs"]) == 2
        assert detail["work_state"]["phase"] == "恢复并提纯旧项目状态"
        project = client.get(f"/api/work/projects/{thread['project_id']}").json()
        assert project["status"] == "running"
        assert len(project["threads"]) == 1


def test_semantic_duplicate_url_is_reused_and_recorded(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="semantic-reuse")
    registry = ToolRegistry()
    handler_calls = 0

    def read_page(_, arguments):
        nonlocal handler_calls
        handler_calls += 1
        return {"url": arguments["url"], "text": "same page"}

    registry.register(
        ToolSpec("read_web_page", "读取网页", {"type": "object"}, read_page)
    )
    decisions = iter(
        [
            work_decision(
                {
                    "type": "tool_call",
                    "step_id": "read-1",
                    "tool_name": "read_web_page",
                    "arguments": {"url": "https://example.com/news/?utm_source=a"},
                }
            ),
            json.dumps(
                {
                    "state_update": {
                        "current_judgment": "页面内容已经读取并完成当前判断",
                        "verified_facts": ["同一页面内容 tool_call:1"],
                    },
                    "consumed_tool_call_ids": [1],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            ),
            work_decision(
                {
                    "type": "tool_call",
                    "step_id": "read-2",
                    "tool_name": "read_web_page",
                    "arguments": {"url": "https://EXAMPLE.com/news"},
                }
            ),
        ]
    )
    executor = WorkExecutor(
        database,
        runtime,
        registry,
        lambda *_: next(decisions),
        max_steps_per_run=3,
        default_tool_permissions={"read_web_page": "allow"},
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="语义去重", objective="不重复读取同一页面"
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "yielded"
    assert handler_calls == 1
    assert len(executor.list_tool_calls(thread.id)) == 1
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["duplicate_actions_prevented"] == 1


def test_work_history_reads_model_call_ledger(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="model-history")
    thread = runtime.create_thread(owner_agent_id=agent_id, title="模型账本", objective="审计")
    run = runtime.create_run(thread.id)
    service = WorkBudgetService(database, {})
    service.record_model_call(
        thread_id=thread.id,
        run_id=run.id,
        owner_agent_id=agent_id,
        execution_token=1,
        purpose="decision",
        attempt_number=2,
        model_id=None,
        model_name="备用模型",
        status="completed",
        prompt="hello",
        output="world",
        usage=ModelUsage(5, 3, 8, "provider"),
        duration_ms=25,
    )
    history = WorkContextService(database, {}).history(thread.id, "model_calls")
    assert history["items"][0]["model_name"] == "备用模型"
    assert history["items"][0]["usage_source"] == "provider"
    assert history["items"][0]["total_tokens"] == 8


def test_controller_can_locate_and_read_previous_work_but_worker_cannot(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        worker = AgentProfile(name="普通员工")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="previous-work")
    previous = runtime.create_thread(
        owner_agent_id=controller_id,
        title="之前未完成的研究",
        objective="寻找热点并形成报告",
    )
    previous_run = runtime.create_run(previous.id)
    claimed = runtime.claim_next(agent_id=controller_id)
    assert claimed is not None and claimed.id == previous_run.id
    runtime.fail(claimed.id, claimed.execution_token, "预算不足", result_text="已有两条证据")
    current = runtime.create_thread(
        owner_agent_id=controller_id,
        title="继续旧工作",
        objective="恢复之前未完成工作",
    )
    current_run = runtime.create_run(current.id)
    service = WorkContextService(database, {})
    controller_context = ToolContext(current.id, current_run.id, controller_id, 1)
    listed = service.list_recent_work_tool(controller_context, {"status": "failed"})
    assert listed["items"][0]["thread_id"] == previous.id
    assert listed["items"][0]["result_preview"] == "已有两条证据"
    history = service.history_tool(
        controller_context,
        {"thread_id": previous.id, "kind": "events", "limit": 10},
    )
    assert history["items"]
    with pytest.raises(Exception, match="只有总控Agent"):
        service.history_tool(
            ToolContext(current.id, current_run.id, worker_id, 1),
            {"thread_id": previous.id, "kind": "events"},
        )


def test_controller_prompt_routes_tools_and_stays_compact(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        session.commit()
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8")
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    runtime = app.state.work_runtime
    thread = runtime.create_thread(
        owner_agent_id=controller_id,
        title="恢复任务",
        objective="重新开始完成之前未完成工作",
    )
    run = runtime.create_run(thread.id)
    executor = app.state.work_executor
    prompt = executor._build_prompt(
        thread,
        run,
        {},
        executor.list_tool_permissions(controller_id),
    )
    assert "先调用list_recent_work定位旧thread_id" in prompt
    assert '"list_workspace_files"' not in prompt
    assert '"list_recent_work"' in prompt
    assert "本轮唯一可调用工具" in prompt
    assert "最终成果一律使用自然简体中文" in prompt
    assert "禁止直接倾倒外文标题、摘要或正文" in prompt
    assert len(prompt) < 18000


def test_fresh_controller_project_does_not_offer_history_or_research_tools(
    tmp_path: Path,
):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        session.commit()
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    runtime = app.state.work_runtime
    thread = runtime.create_thread(
        owner_agent_id=controller_id,
        title="全新市场热点任务",
        objective="研究今天普通人参与的生活、兴趣和娱乐热点",
    )
    run = runtime.create_run(thread.id)
    checkpoint: dict[str, Any] = {}
    prompt = app.state.work_executor._build_prompt(
        thread,
        run,
        checkpoint,
        app.state.work_executor.list_tool_permissions(controller_id),
    )

    assert checkpoint["offered_tool_names"] == [
        "delegate_work",
        "list_company_agents",
    ]
    assert "这是独立的新工作上下文" in prompt
    assert '"read_work_history"' not in prompt
    assert '"web_search"' not in prompt
    assert '"read_web_page"' not in prompt


def test_root_controller_delegation_always_waits_even_if_model_disables_it(
    tmp_path: Path,
):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        worker = AgentProfile(
            name="市场Agent",
            role_responsibilities="执行市场研究",
            work_content="搜索、分析和提炼",
            personality="客观",
            preferences="中文交付",
        )
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="forced-handoff")
    parent = runtime.create_thread(
        owner_agent_id=controller_id,
        title="总控委派",
        objective="让市场Agent执行热点研究",
    )
    service = CollaborationService(database)

    assigned = service.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={
            "title": "市场研究",
            "objective": "搜索并提炼普通人参与的热点",
            "success_criteria": ["形成中文洞察"],
            "constraints": [],
            "wait_for_result": False,
            "request_key": "forced-root-handoff",
        },
    )

    assert assigned["wait_for_result"] is True


def test_tool_count_threshold_triggers_review_without_stopping_work(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="tool-budget")
    registry = ToolRegistry()
    registry.register(ToolSpec("lookup", "查询", {}, lambda *_: {"ok": True}))
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="预算测试", objective="避免无限工具循环"
    )
    runtime.create_run(thread.id)
    decisions = iter(
        [
            work_decision(
                {
                    "type": "tool_call",
                    "step_id": "lookup-1",
                    "tool_name": "lookup",
                    "arguments": {"query": "first"},
                }
            ),
            work_decision(
                {
                    "type": "tool_call",
                    "step_id": "lookup-2",
                    "tool_name": "lookup",
                    "arguments": {"query": "second"},
                }
            ),
            work_decision(
                {"type": "complete", "message": "审查工具价值后完成"},
                progress=1,
            ),
        ]
    )
    prompts: list[str] = []

    def provider(thread, run, prompt):
        prompts.append(prompt)
        return next(decisions)

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_tool_calls_per_thread=1,
        default_tool_permissions={"lookup": "allow"},
    )
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)
    assert result.outcome == "completed"
    assert result.run.status == "completed"
    assert '"review_required": true' in prompts[1]
    with database.sessions() as session:
        assert session.query(ToolCall).filter_by(work_thread_id=thread.id).count() == 2


def test_tool_review_threshold_is_shared_without_blocking_sibling_threads(tmp_path: Path):
    database, parent_agent_id = make_database(tmp_path)
    with database.sessions() as session:
        child_agent = AgentProfile(name="子Agent")
        session.add(child_agent)
        session.commit()
        child_agent_id = child_agent.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="tree-budget")
    parent = runtime.create_thread(
        owner_agent_id=parent_agent_id, title="项目根任务", objective="共享预算"
    )
    child_one = runtime.create_thread(
        owner_agent_id=child_agent_id,
        parent_thread_id=parent.id,
        title="子任务一",
        objective="先使用预算",
    )
    child_two = runtime.create_thread(
        owner_agent_id=child_agent_id,
        parent_thread_id=parent.id,
        title="子任务二",
        objective="不能重置预算",
    )
    runtime.create_run(child_one.id)
    runtime.create_run(child_two.id)
    registry = ToolRegistry()
    registry.register(ToolSpec("lookup", "查询", {}, lambda *_: {"ok": True}))

    def provider(thread, run, prompt):
        return work_decision(
            {
                "type": "tool_call",
                "step_id": f"lookup-{thread.id}",
                "tool_name": "lookup",
                "arguments": {"thread_id": thread.id},
            }
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=1,
        max_tool_calls_per_thread=1,
        default_tool_permissions={"lookup": "allow"},
    )
    first = runtime.claim_next(agent_id=child_agent_id)
    assert first is not None and first.work_thread_id == child_one.id
    assert executor.execute(first.id, first.execution_token).outcome == "yielded"
    resumed_first = runtime.claim_next(agent_id=child_agent_id)
    assert resumed_first is not None and resumed_first.work_thread_id == child_one.id
    runtime.fail(resumed_first.id, resumed_first.execution_token, "测试中结束首个子任务")
    second = runtime.claim_next(agent_id=child_agent_id)
    assert second is not None and second.work_thread_id == child_two.id
    continued = executor.execute(second.id, second.execution_token)
    assert continued.outcome == "yielded"
    assert continued.run.status == "queued"
    with database.sessions() as session:
        assert session.query(ToolCall).count() == 2


def make_database(tmp_path: Path) -> tuple[Database, int]:
    database = Database(f"sqlite:///{tmp_path / 'work-runtime.db'}")
    database.create_tables()
    with database.sessions() as session:
        agent = AgentProfile(
            name="市场Agent",
            role_responsibilities="市场研究",
            work_content="收集并分析信息",
            personality="严谨",
            preferences="优先可靠来源",
        )
        session.add(agent)
        session.commit()
        return database, agent.id


def test_work_runtime_api_records_checkpoint_and_completion(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'api.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("work_runtime:\n  lease_seconds: 60\n", encoding="utf-8")
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        controller = client.get("/api/agents").json()["controller"]
        thread_response = client.post(
            "/api/work/threads",
            json={
                "owner_agent_id": controller["id"],
                "title": "注册用户规模研究",
                "objective": "寻找注册用户超过5000万的App",
                "success_criteria": ["每项结论有来源"],
                "constraints": ["不能把下载量当成注册用户"],
                "priority": 10,
            },
        )
        assert thread_response.status_code == 201
        thread = thread_response.json()
        run_response = client.post(
            f"/api/work/threads/{thread['id']}/runs",
            json={"trigger_kind": "user", "checkpoint": {"stage": "new"}},
        )
        assert run_response.status_code == 201
        run_id = run_response.json()["id"]

        claimed = client.post(
            "/api/work/runs/claim", params={"agent_id": controller["id"]}
        ).json()["run"]
        assert claimed["id"] == run_id
        assert claimed["status"] == "running"
        assert claimed["execution_token"] == 1

        checkpoint = client.post(
            f"/api/work/runs/{run_id}/checkpoint",
            json={
                "execution_token": claimed["execution_token"],
                "checkpoint": {"completed_steps": ["定义口径"]},
            },
        )
        assert checkpoint.status_code == 200
        assert checkpoint.json()["checkpoint"]["completed_steps"] == ["定义口径"]

        heartbeat = client.post(
            f"/api/work/runs/{run_id}/heartbeat",
            json={"execution_token": claimed["execution_token"]},
        )
        assert heartbeat.status_code == 200
        assert heartbeat.json()["heartbeat_at"]

        completed = client.post(
            f"/api/work/runs/{run_id}/complete",
            json={
                "execution_token": claimed["execution_token"],
                "result_text": "研究完成",
            },
        )
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"
        detail = client.get(f"/api/work/threads/{thread['id']}").json()
        assert detail["status"] == "completed"
        assert detail["result_text"] == "研究完成"
        assert detail["runs"][0]["status"] == "completed"
        events = client.get(f"/api/work/threads/{thread['id']}/events").json()
        assert [item["event_type"] for item in events] == [
            "thread_created",
            "run_created",
            "run_claimed",
            "checkpoint_saved",
            "lease_renewed",
            "run_completed",
            "report_generated",
        ]


def test_completed_work_generates_versioned_auditable_report(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'reports.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8")
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        controller = client.get("/api/agents").json()["controller"]
        thread = client.post(
            "/api/work/threads",
            json={
                "owner_agent_id": controller["id"],
                "title": "注册用户研究",
                "objective": "核对用户规模",
                "success_criteria": ["每项结论有来源"],
                "constraints": ["不使用下载量"],
            },
        ).json()
        run = client.post(
            f"/api/work/threads/{thread['id']}/runs", json={"trigger_kind": "test"}
        ).json()
        claimed = client.post(
            "/api/work/runs/claim", params={"agent_id": controller["id"]}
        ).json()["run"]
        with database.sessions() as session:
            session.add(
                Evidence(
                    work_thread_id=thread["id"],
                    owner_agent_id=controller["id"],
                    source_url="https://example.com/official",
                    source_title="Official disclosure",
                    published_at="2026-08-01",
                    claim="Example App拥有6000万注册用户",
                    excerpt="The service has 60 million registered users.",
                    metric_name="registered users",
                    metric_value="60",
                    metric_unit="million",
                    confidence="high",
                    stance="supports",
                    content_hash="a" * 64,
                )
            )
            session.commit()

        completed = client.post(
            f"/api/work/runs/{run['id']}/complete",
            json={
                "execution_token": claimed["execution_token"],
                "result_text": "确认Example App超过5000万注册用户。",
            },
        )
        assert completed.status_code == 200
        report_meta = completed.json()["report"]
        assert report_meta["version"] == 1
        assert report_meta["evidence_count"] == 1

        latest = client.get(
            f"/api/work/threads/{thread['id']}/reports/latest"
        ).json()
        assert "Example App拥有6000万注册用户" in latest["markdown_content"]
        assert latest["evidence_snapshot"][0]["source_url"] == "https://example.com/official"
        reused = client.post(f"/api/work/threads/{thread['id']}/reports").json()
        assert reused["reused"] is True
        assert reused["id"] == report_meta["id"]

        html = client.get(
            f"/api/work/threads/{thread['id']}/reports/{report_meta['id']}/download",
            params={"format": "html"},
        )
        assert html.status_code == 200
        assert "text/html" in html.headers["content-type"]
        assert "Official disclosure" in html.text
        assert "attachment;" in html.headers["content-disposition"]
        preview = client.get(
            f"/api/work/threads/{thread['id']}/reports/{report_meta['id']}/preview"
        )
        assert preview.status_code == 200
        assert "attachment" not in preview.headers.get("content-disposition", "")
        assert "执行成果" in preview.text
        markdown = client.get(
            f"/api/work/threads/{thread['id']}/reports/{report_meta['id']}/download",
            params={"format": "md"},
        )
        assert "text/markdown" in markdown.headers["content-type"]
        assert "## 执行成果" in markdown.text
        assert latest["generated_at"].endswith("+00:00")
        dashboard_item = next(
            item for item in client.get("/api/work/dashboard").json()["items"]
            if item["id"] == thread["id"]
        )
        assert dashboard_item["report"]["id"] == report_meta["id"]
        with database.sessions() as session:
            session.add(
                Evidence(
                    work_thread_id=thread["id"],
                    owner_agent_id=controller["id"],
                    source_url="https://example.com/context",
                    source_title="Second disclosure",
                    claim="第二来源提供补充背景",
                    excerpt="Additional context for the registered user metric.",
                    confidence="medium",
                    stance="context",
                    content_hash="b" * 64,
                )
            )
            session.commit()
        second_version = client.post(
            f"/api/work/threads/{thread['id']}/reports"
        ).json()
        assert second_version["reused"] is False
        assert second_version["version"] == 2
        assert second_version["evidence_count"] == 2
        original = client.get(
            f"/api/work/threads/{thread['id']}/reports/{report_meta['id']}"
        ).json()
        assert original["version"] == 1
        assert original["evidence_count"] == 1
        with database.sessions() as session:
            assert session.query(ResearchReport).count() == 2


def test_parent_report_aggregates_results_and_evidence_without_raw_tool_ledger(
    tmp_path: Path,
):
    database = Database(f"sqlite:///{tmp_path / 'delivery.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        controller = client.get("/api/agents").json()["controller"]
        with database.sessions() as session:
            employee = AgentProfile(
                name="市场分析Agent",
                role_responsibilities="市场研究",
                work_content="搜索、核验并回传证据",
                personality="严谨",
                preferences="保留工具依据",
            )
            session.add(employee)
            session.commit()
            employee_id = employee.id
        parent = client.post(
            "/api/work/threads",
            json={
                "owner_agent_id": controller["id"],
                "title": "总控汇总市场成果",
                "objective": "委派员工核验市场数据并汇总",
            },
        ).json()
        child = client.post(
            "/api/work/threads",
            json={
                "owner_agent_id": employee_id,
                "parent_thread_id": parent["id"],
                "title": "员工核验公开结果",
                "objective": "调用工具并保存来源",
            },
        ).json()
        with database.sessions() as session:
            parent_row = session.get(WorkThread, parent["id"])
            child_row = session.get(WorkThread, child["id"])
            assert parent_row is not None and child_row is not None
            parent_row.status = "completed"
            parent_row.result_text = "总控确认员工已经完成核验。"
            parent_row.finished_at = datetime.now(timezone.utc)
            child_row.status = "completed"
            child_row.result_text = "员工确认Example App拥有6000万注册用户。"
            child_row.finished_at = datetime.now(timezone.utc)
            run = WorkRun(
                work_thread_id=child_row.id,
                owner_agent_id=employee_id,
                trigger_kind="delegation",
                status="completed",
                execution_token=1,
                result_text=child_row.result_text,
                finished_at=datetime.now(timezone.utc),
            )
            session.add(run)
            session.flush()
            session.add(
                ToolCall(
                    work_thread_id=child_row.id,
                    work_run_id=run.id,
                    owner_agent_id=employee_id,
                    step_id="verify-users",
                    action_key="d" * 64,
                    tool_name="read_web_page",
                    arguments_json=json.dumps(
                        {"url": "https://example.com/official"}, ensure_ascii=False
                    ),
                    status="completed",
                    permission="allow",
                    retry_safety="safe",
                    attempt_count=1,
                    execution_token=1,
                    result_json=json.dumps(
                        {
                            "title": "Official disclosure",
                            "text": "The service has 60 million registered users.",
                        },
                        ensure_ascii=False,
                    ),
                    finished_at=datetime.now(timezone.utc),
                )
            )
            session.add(
                Evidence(
                    work_thread_id=child_row.id,
                    owner_agent_id=employee_id,
                    source_url="https://example.com/official",
                    source_title="Official disclosure",
                    claim="Example App拥有6000万注册用户",
                    excerpt="The service has 60 million registered users.",
                    confidence="high",
                    stance="supports",
                    content_hash="e" * 64,
                )
            )
            session.commit()

        response = client.post(f"/api/work/threads/{parent['id']}/reports")
        assert response.status_code == 201
        report = response.json()
        assert report["agent_result_count"] == 2
        assert report["tool_result_count"] == 0
        assert report["evidence_count"] == 1
        assert "员工确认Example App拥有6000万注册用户" in report["markdown_content"]
        assert "read_web_page" not in report["markdown_content"]
        assert not any(item.get("kind") == "tool_result" for item in report["findings"])
        assert "https://example.com/official" in report["markdown_content"]
        output_dir = tmp_path / "outputs" / "reports" / f"work-{parent['id']}"
        markdown_files = list(output_dir.glob("*.md"))
        html_files = list(output_dir.glob("*.html"))
        assert len(markdown_files) == 1
        assert len(html_files) == 1
        assert "员工确认Example App拥有6000万注册用户" in markdown_files[
            0
        ].read_text(encoding="utf-8")
        assert "Official disclosure" in html_files[0].read_text(encoding="utf-8")


def test_direct_delegation_waits_for_child_and_resumes_parent(tmp_path: Path):
    database, parent_agent_id = make_database(tmp_path)
    with database.sessions() as session:
        child_agent = AgentProfile(
            name="研究协作Agent",
            role_responsibilities="协助研究",
            work_content="完成子任务",
            personality="可靠",
            preferences="及时回传",
        )
        session.add(child_agent)
        session.commit()
        child_agent_id = child_agent.id

    runtime = WorkRuntime(database, lease_seconds=60, instance_id="collab-runtime")
    collaboration = CollaborationService(database)
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "delegate_work",
            "委派",
            {},
            collaboration.delegate_tool,
            risk_level="write",
        )
    )
    parent = runtime.create_thread(
        owner_agent_id=parent_agent_id,
        title="父任务",
        objective="委派子任务并汇总",
    )
    runtime.create_run(parent.id)
    calls = {"parent": 0}

    def provider(thread, run, prompt):
        if thread.parent_thread_id:
            return work_decision(
                {
                    "type": "complete",
                    "message": "子Agent结果：核验完成",
                    "criteria_met": True,
                    "criteria_evidence": ["已返回明确结论：核验完成"],
                },
                progress=1,
            )
        calls["parent"] += 1
        if calls["parent"] == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "delegate",
                    "tool_name": "delegate_work",
                    "arguments": {
                        "target_agent_id": child_agent_id,
                        "title": "核验子任务",
                        "objective": "独立核验一个结论",
                        "success_criteria": ["返回明确结论"],
                        "wait_for_result": True,
                        "request_key": "verify-once",
                    },
                },
                progress=0.2,
            )
        if calls["parent"] == 2:
            row = collaboration.list(parent_thread_id=parent.id)[0]
            return work_decision(
                {
                    "type": "wait_collaboration",
                    "collaboration_id": row.id,
                    "message": "等待协作结果",
                },
                progress=0.4,
            )
        assert "子Agent结果：核验完成" in prompt
        return work_decision(
            {"type": "complete", "message": "父任务已汇总协作结果"},
            progress=1,
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=4,
        terminal_handler=lambda thread_id, _: collaboration.sync_child_completion(thread_id),
        default_tool_permissions={"delegate_work": "allow"},
    )
    parent_claim = runtime.claim_next(agent_id=parent_agent_id)
    assert parent_claim is not None
    waiting = executor.execute(parent_claim.id, parent_claim.execution_token)
    assert waiting.outcome == "waiting_collaboration"
    rows = collaboration.list(parent_thread_id=parent.id)
    assert len(rows) == 1
    assert rows[0].status == "assigned"

    child_claim = runtime.claim_next(agent_id=child_agent_id)
    assert child_claim is not None
    child_done = executor.execute(child_claim.id, child_claim.execution_token)
    assert child_done.outcome == "completed"
    resumed = runtime.claim_next(agent_id=parent_agent_id)
    assert resumed is not None and resumed.id == parent_claim.id
    final = executor.execute(resumed.id, resumed.execution_token)
    assert final.outcome == "completed"
    assert (final.run.result_text or "").startswith("父任务已汇总协作结果")
    assert "子Agent结果：核验完成" in (final.run.result_text or "")
    with database.sessions() as session:
        stored = session.get(WorkCollaboration, rows[0].id)
        assert stored is not None
        assert stored.status == "completed"
        assert stored.result_text == "子Agent结果：核验完成"
        assert session.query(WorkThread).filter_by(parent_thread_id=parent.id).count() == 1


def test_child_progress_event_hands_execution_to_parent_and_back(tmp_path: Path):
    database, parent_agent_id = make_database(tmp_path)
    with database.sessions() as session:
        child_agent = AgentProfile(name="阶段研究Agent")
        session.add(child_agent)
        session.commit()
        child_agent_id = child_agent.id

    runtime = WorkRuntime(database, lease_seconds=60, instance_id="progress-handoff")
    collaboration = CollaborationService(database)
    parent = runtime.create_thread(
        owner_agent_id=parent_agent_id,
        title="总控任务",
        objective="接收阶段发现并继续调度",
    )
    runtime.create_run(parent.id)
    assigned = collaboration.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=parent_agent_id,
        target_agent_id=child_agent_id,
        values={"title": "持续研究", "objective": "先回传阶段发现，再完成研究"},
    )
    calls = {"parent": 0, "child": 0}

    def provider(thread, run, prompt):
        if thread.id == assigned["child_thread_id"]:
            calls["child"] += 1
            assert '"collaboration": {"id":' in prompt
            if calls["child"] == 1:
                return json.dumps(
                    {
                        "state_summary": "已经找到关键候选，需总控确认研究方向",
                        "progress": 0.45,
                        "action": {
                            "type": "report_progress",
                            "message": "阶段发现：候选A有一手来源，建议继续核验。",
                        },
                    },
                    ensure_ascii=False,
                )
            return work_decision(
                {"type": "complete", "message": "子任务最终完成：候选A已核验"},
                progress=1,
            )
        calls["parent"] += 1
        if calls["parent"] == 2:
            assert "阶段发现：候选A有一手来源" in prompt
        if calls["parent"] <= 2:
            return work_decision(
                {
                    "type": "wait_collaboration",
                    "collaboration_id": assigned["id"],
                    "message": "已处理当前阶段成果，继续等待子Agent最终结果",
                }
            )
        assert "子任务最终完成：候选A已核验" in prompt
        return work_decision({"type": "complete", "message": "总控已汇总最终结果"})

    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        provider,
        progress_handler=collaboration.report_progress,
        progress_resume_handler=collaboration.resume_child_after_progress,
        terminal_handler=lambda thread_id, _: collaboration.sync_child_completion(thread_id),
    )

    parent_claim = runtime.claim_next(agent_id=parent_agent_id)
    assert parent_claim is not None
    first_wait = executor.execute(parent_claim.id, parent_claim.execution_token)
    assert first_wait.outcome == "waiting_collaboration"

    child_claim = runtime.claim_next(agent_id=child_agent_id)
    assert child_claim is not None
    progress = executor.execute(child_claim.id, child_claim.execution_token)
    assert progress.outcome == "progress_reported"
    assert progress.run.status == "waiting_dependency"

    parent_claim = runtime.claim_next(agent_id=parent_agent_id)
    assert parent_claim is not None
    handed_back = executor.execute(parent_claim.id, parent_claim.execution_token)
    assert handed_back.outcome == "waiting_collaboration"

    with database.sessions() as session:
        queued_child = session.scalar(
            select(WorkRun).where(
                WorkRun.work_thread_id == assigned["child_thread_id"]
            )
        )
        assert queued_child is not None and queued_child.status == "queued"
        assert queued_child.last_error is None
    child_claim = runtime.claim_next(agent_id=child_agent_id)
    assert child_claim is not None
    child_done = executor.execute(child_claim.id, child_claim.execution_token)
    assert child_done.outcome == "completed"

    parent_claim = runtime.claim_next(agent_id=parent_agent_id)
    assert parent_claim is not None
    parent_done = executor.execute(parent_claim.id, parent_claim.execution_token)
    assert parent_done.outcome == "completed"
    assert calls == {"parent": 3, "child": 2}
    with database.sessions() as session:
        event_types = session.scalars(
            select(WorkEvent.event_type).where(
                WorkEvent.work_thread_id.in_([parent.id, assigned["child_thread_id"]])
            )
        ).all()
        assert "collaboration_progress" in event_types
        assert "progress_reported" in event_types
        assert "progress_acknowledged" in event_types


def test_collaboration_is_idempotent_and_running_work_cannot_be_seized(tmp_path: Path):
    database, owner_id = make_database(tmp_path)
    with database.sessions() as session:
        second = AgentProfile(name="第二Agent")
        third = AgentProfile(name="第三Agent")
        session.add_all([second, third])
        session.commit()
        second_id, third_id = second.id, third.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="transfer-runtime")
    service = CollaborationService(database)
    parent = runtime.create_thread(owner_agent_id=owner_id, title="父任务", objective="协作")
    values = {
        "title": "幂等子任务",
        "objective": "只创建一次",
        "request_key": "stable-key",
    }
    first = service.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=owner_id,
        target_agent_id=second_id,
        values=values,
    )
    second_call = service.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=owner_id,
        target_agent_id=second_id,
        values=values,
    )
    assert first["id"] == second_call["id"]
    assert second_call["reused"] is True
    child_id = first["child_thread_id"]
    claimed = runtime.claim_next(agent_id=second_id)
    assert claimed is not None and claimed.work_thread_id == child_id
    with pytest.raises(Exception, match="运行中的工作不能强制抢占"):
        service.transfer_thread(
            child_id,
            actor_agent_id=third_id,
            target_agent_id=third_id,
            reason="尝试运行中抢占",
        )
    runtime.pause(
        claimed.id,
        claimed.execution_token,
        status="queued",
        error="主动让出",
    )
    transferred = service.transfer_thread(
        child_id,
        actor_agent_id=third_id,
        target_agent_id=third_id,
        reason="让出后由同事接手",
    )
    assert transferred.owner_agent_id == third_id
    next_claim = runtime.claim_next(agent_id=third_id)
    assert next_claim is not None and next_claim.work_thread_id == child_id


def test_terminal_collaboration_result_does_not_put_parent_to_sleep(tmp_path: Path):
    database, parent_agent_id = make_database(tmp_path)
    with database.sessions() as session:
        child_agent = AgentProfile(name="协作Agent")
        session.add(child_agent)
        session.commit()
        child_agent_id = child_agent.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="early-result")
    service = CollaborationService(database)
    parent = runtime.create_thread(
        owner_agent_id=parent_agent_id, title="父任务", objective="接收已完成结果"
    )
    runtime.create_run(parent.id)
    collaboration = service.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=parent_agent_id,
        target_agent_id=child_agent_id,
        values={"title": "快速子任务", "objective": "先于父任务完成"},
    )
    child_claim = runtime.claim_next(agent_id=child_agent_id)
    assert child_claim is not None
    runtime.complete(child_claim.id, child_claim.execution_token, "提前完成的结果")
    service.sync_child_completion(collaboration["child_thread_id"])

    calls = 0

    def provider(thread, run, prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            return work_decision(
                {
                    "type": "wait_collaboration",
                    "collaboration_id": collaboration["id"],
                }
            )
        assert "提前完成的结果" in prompt
        return work_decision({"type": "complete", "message": "父任务正常完成"})

    executor = WorkExecutor(database, runtime, ToolRegistry(), provider)
    parent_claim = runtime.claim_next(agent_id=parent_agent_id)
    assert parent_claim is not None
    result = executor.execute(parent_claim.id, parent_claim.execution_token)
    assert result.outcome == "completed"
    assert calls == 2


def test_controller_collaboration_queue_approval_and_default_tools(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'collaboration-api.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("work_runtime:\n  auto_execute: false\n", encoding="utf-8")
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        controller = client.get("/api/agents").json()["controller"]
        with database.sessions() as session:
            requester = AgentProfile(name="市场Agent", work_content="市场研究")
            target = AgentProfile(name="分析Agent", work_content="信息核验")
            session.add_all([requester, target])
            session.commit()
            requester_id, target_id = requester.id, target.id
        parent = app.state.work_runtime.create_thread(
            owner_agent_id=requester_id,
            title="市场研究",
            objective="请求额外分析支持",
        )
        requested = app.state.collaboration_service.create_request(
            parent_thread_id=parent.id,
            requester_agent_id=requester_id,
            target_agent_id=None,
            values={
                "title": "核验市场数据",
                "objective": "独立核验关键数字",
                "reason": "需要第二视角",
            },
        )

        dashboard = client.get("/api/work/dashboard").json()
        assert dashboard["pending_collaborations"][0]["id"] == requested["id"]
        assert dashboard["pending_collaborations"][0]["requester_agent_name"] == "市场Agent"
        assert {item["name"] for item in dashboard["available_agents"]} >= {
            "总控Agent",
            "市场Agent",
            "分析Agent",
        }

        approved = client.post(
            f"/api/work/collaborations/{requested['id']}/decision",
            json={
                "approved": True,
                "target_agent_id": target_id,
                "reason": "总控批准",
            },
        )
        assert approved.status_code == 200
        approved_body = approved.json()
        assert approved_body["status"] == "assigned"
        assert approved_body["child_thread_id"]
        duplicate = client.post(
            f"/api/work/collaborations/{requested['id']}/decision",
            json={"approved": True, "target_agent_id": target_id},
        )
        assert duplicate.status_code == 409
        with database.sessions() as session:
            children = session.query(WorkThread).filter_by(parent_thread_id=parent.id).all()
            assert len(children) == 1

        tools = {
            item["name"]: item["permission"]
            for item in client.get(f"/api/agents/{requester_id}/tools").json()
        }
        assert tools["delegate_work"] == "allow"
        assert tools["request_collaboration"] == "allow"
        denied = client.put(
            f"/api/agents/{requester_id}/tools/delegate_work",
            json={"permission": "deny"},
        )
        assert denied.status_code == 200
        tools_after = {
            item["name"]: item["permission"]
            for item in client.get(f"/api/agents/{requester_id}/tools").json()
        }
        assert tools_after["delegate_work"] == "deny"
        assert controller["is_controller"] is True


def test_waiting_retry_does_not_overlap_new_task_for_same_agent(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="server-a")

    thread_a = runtime.create_thread(
        owner_agent_id=agent_id,
        title="任务A",
        objective="先执行后等待重试",
    )
    run_a = runtime.create_run(thread_a.id)
    claimed_a = runtime.claim_next(agent_id=agent_id)
    assert claimed_a and claimed_a.id == run_a.id
    runtime.pause(
        run_a.id,
        claimed_a.execution_token,
        status="waiting_retry",
        next_wake_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        error="临时故障",
    )

    thread_b = runtime.create_thread(
        owner_agent_id=agent_id,
        title="任务B",
        objective="等待期间的新任务",
    )
    run_b = runtime.create_run(thread_b.id)
    claimed_b = runtime.claim_next(agent_id=agent_id)
    assert claimed_b and claimed_b.id == run_b.id

    with database.sessions() as session:
        stored_a = session.get(WorkRun, run_a.id)
        assert stored_a is not None
        stored_a.next_wake_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    assert runtime.claim_next(agent_id=agent_id) is None
    runtime.complete(run_b.id, claimed_b.execution_token, "B完成")
    resumed_a = runtime.claim_next(agent_id=agent_id)
    assert resumed_a and resumed_a.id == run_a.id
    assert resumed_a.execution_token == 2


def test_expired_lease_is_reclaimed_and_stale_writer_is_rejected(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    old_runtime = WorkRuntime(database, lease_seconds=60, instance_id="old-server")
    new_runtime = WorkRuntime(database, lease_seconds=60, instance_id="new-server")
    thread = old_runtime.create_thread(
        owner_agent_id=agent_id,
        title="断线恢复",
        objective="恢复同一个Run",
    )
    run = old_runtime.create_run(thread.id, checkpoint={"step": 1})
    first_claim = old_runtime.claim_next(agent_id=agent_id)
    assert first_claim and first_claim.execution_token == 1
    assert new_runtime.claim_next(agent_id=agent_id) is None

    with database.sessions() as session:
        stored = session.get(WorkRun, run.id)
        assert stored is not None
        stored.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    reclaimed = new_runtime.claim_next(agent_id=agent_id)
    assert reclaimed and reclaimed.id == run.id
    assert reclaimed.execution_token == 2
    assert reclaimed.lease_owner == "new-server"
    with pytest.raises(WorkLeaseLost):
        old_runtime.checkpoint(run.id, first_claim.execution_token, {"stale": True})
    finished = new_runtime.complete(run.id, reclaimed.execution_token, "恢复完成")
    assert finished.status == "completed"
    with database.sessions() as session:
        event_types = [
            item.event_type
            for item in session.query(WorkEvent)
            .filter_by(work_thread_id=thread.id)
            .order_by(WorkEvent.id)
        ]
    assert "run_reclaimed" in event_types


def test_two_runtime_instances_cannot_run_same_agent_concurrently(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    setup_runtime = WorkRuntime(database, instance_id="setup")
    for suffix in ("A", "B"):
        thread = setup_runtime.create_thread(
            owner_agent_id=agent_id,
            title=f"并发任务{suffix}",
            objective="验证数据库串行容量",
        )
        setup_runtime.create_run(thread.id)

    runtime_a = WorkRuntime(database, lease_seconds=60, instance_id="server-a")
    runtime_b = WorkRuntime(database, lease_seconds=60, instance_id="server-b")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda runtime: runtime.claim_next(agent_id=agent_id),
                (runtime_a, runtime_b),
            )
        )
    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    with database.sessions() as session:
        running = session.query(WorkRun).filter_by(
            owner_agent_id=agent_id, status="running"
        ).all()
        assert len(running) == 1


def work_decision(action: dict, *, progress: float = 0.2) -> str:
    return json.dumps(
        {
            "state_summary": "正在执行测试工作",
            "progress": progress,
            "plan": [
                {
                    "id": "research",
                    "title": "调用工具并整理结果",
                    "status": "in_progress" if progress < 1 else "completed",
                }
            ],
            "action": action,
        },
        ensure_ascii=False,
    )


def test_employee_business_tools_are_not_hidden_by_task_keywords(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="tool-autonomy")
    registry = ToolRegistry()
    calls: list[dict] = []

    def inspect_table(_, arguments):
        calls.append(arguments)
        return {"columns": ["name", "signal"], "rows": 12}

    registry.register(
        ToolSpec(
            name="inspect_table",
            description="检查结构化业务记录",
            input_schema={"type": "object"},
            handler=inspect_table,
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            assert '"inspect_table"' in prompt
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "inspect-records",
                    "tool_name": "inspect_table",
                    "arguments": {"dataset": "current"},
                    "purpose": "了解现有业务记录结构",
                    "criterion_id": "understand-input",
                    "expected_delta": "确认可分析字段和记录规模",
                    "stop_condition": "取得字段清单",
                }
            )
        return work_decision(
            {"type": "complete", "message": "已确认现有业务记录结构"},
            progress=1,
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=2,
        default_tool_permissions={"inspect_table": "allow"},
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="判断下一项工作",
        objective="判断当前材料中哪项业务记录最值得关注",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert calls == [{"dataset": "current"}]


def test_cognitive_state_merges_by_stable_id_and_survives_compaction():
    initial = merge_work_state(
        {},
        WorkStatePatch.model_validate(
            {
                "current_strategy": "先验证用户行为假设",
                "working_hypotheses": [
                    {
                        "hypothesis_id": "behavior-shift",
                        "statement": "用户偏好正在转移",
                        "next_test": "检查真实行为记录",
                    }
                ],
                "strategy_assessments": [
                    {
                        "strategy_id": "public-sources",
                        "approach": "先检查公开来源",
                    }
                ],
            }
        ),
    )
    updated = merge_work_state(
        initial,
        WorkStatePatch.model_validate(
            {
                "strategy_change_reason": "公开来源只能提供表面信号",
                "working_hypotheses": [
                    {
                        "hypothesis_id": "behavior-shift",
                        "statement": "用户偏好正在转移",
                        "status": "supported",
                        "evidence_refs": ["tool_call:9"],
                    }
                ],
                "strategy_assessments": [
                    {
                        "strategy_id": "public-sources",
                        "approach": "先检查公开来源",
                        "outcome": "mixed",
                        "evidence_refs": ["tool_call:9"],
                        "lesson": "需要行为数据继续验证",
                    }
                ],
            }
        ),
    )
    compact = compact_work_state(updated)

    assert len(updated["working_hypotheses"]) == 1
    assert updated["working_hypotheses"][0]["status"] == "supported"
    assert compact["current_strategy"] == "先验证用户行为假设"
    assert compact["strategy_change_reason"] == "公开来源只能提供表面信号"
    assert compact["strategy_assessments"][0]["outcome"] == "mixed"


def legacy_agent_can_enter_market_full_text_mode_without_keyword_routing(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="dynamic-market")
    registry = ToolRegistry()
    source_url = "https://example.com/interesting-scene"
    registry.register(
        ToolSpec(
            "web_search",
            "发现公开页面",
            {"type": "object"},
            lambda *_: {
                "results": [
                    {
                        "title": "An unusual gathering",
                        "url": source_url,
                        "snippet": "People are joining a new kind of gathering",
                    }
                ]
            },
        )
    )
    registry.register(
        ToolSpec(
            "read_web_page",
            "阅读全文",
            {"type": "object"},
            lambda *_: {
                "url": source_url,
                "title": "An unusual gathering",
                "text": "DYNAMIC_MARKET_FULL_BODY 普通人正在参与一种反常的新聚会。" * 40,
            },
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            assert '"web_search"' in prompt
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "discover",
                    "tool_name": "web_search",
                    "arguments": {"query": "unusual things people are doing"},
                    "purpose": "寻找可能值得创作的真实现象",
                    "criterion_id": "find-material",
                    "expected_delta": "得到可阅读全文的候选来源",
                    "stop_condition": "找到至少一个具体现象候选",
                }
            )
        if len(prompts) == 2:
            assert "People are joining" in prompt
            return json.dumps(
                {
                    "state_update": {
                        "current_strategy": "选择反常聚会候选阅读全文",
                        "market_candidates": [
                            {
                                "candidate_ref": source_url,
                                "title": "An unusual gathering",
                                "lens": "社交关系",
                                "decision": "deep_read",
                                "reason": "可能包含普通人真实参与的反常现象",
                                "source_url": source_url,
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "召回一个需要正文核验的反常聚会候选",
                    },
                    "consumed_tool_call_ids": [1],
                    "action": {
                        "type": "tool_call",
                        "step_id": "read-selected",
                        "tool_name": "read_web_page",
                        "arguments": {"url": source_url},
                        "purpose": "核验具体参与行为和现象",
                        "criterion_id": "find-material",
                        "expected_delta": "获得代表页面全文",
                        "stop_condition": "能判断是否值得形成创作洞察",
                    },
                },
                ensure_ascii=False,
            )
        if len(prompts) == 3:
            assert "DYNAMIC_MARKET_FULL_BODY" in prompt
            assert '"market_insights"' in prompt
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "正文确认普通人参与一种反常的新聚会",
                        "market_insights": [
                            {
                                "title": "反常聚会成为新的参与场景",
                                "signal": "普通人主动参加并讨论新形式聚会",
                                "phenomenon": "社交活动从标准形式转向非典型现场",
                                "why_interesting": "它提供了可继续挖掘的行为反差",
                                "source_refs": [source_url, "tool_call:2"],
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "decisive",
                        "expectation_met": True,
                        "finding": "全文确认了具体现象与参与行为",
                    },
                    "consumed_tool_call_ids": [2],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        return work_decision(
            {"type": "complete", "message": "已形成一条有来源的中文创作洞察"},
            progress=1,
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=5,
        default_tool_permissions={
            "web_search": "allow",
            "read_web_page": "allow",
        },
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="寻找内容灵感",
        objective="给我找点有意思的东西作为创作材料",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["work_state"]["market_insights"][0]["title"] == (
        "反常聚会成为新的参与场景"
    )


def test_repeated_low_information_requires_a_real_strategy_change(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="strategy-change")
    registry = ToolRegistry()
    calls: list[str] = []

    def lookup(_, arguments):
        calls.append(arguments["route"])
        return {"finding": "没有新增有效信息", "route": arguments["route"]}

    registry.register(
        ToolSpec(
            name="lookup",
            description="测试一个信息路线",
            input_schema={"type": "object"},
            handler=lookup,
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        call_number = len(prompts)
        if call_number == 1:
            return work_decision(
                {
                    "type": "tool_call",
                        "step_id": "lookup-1",
                        "tool_name": "lookup",
                        "arguments": {"route": "same-source-1"},
                        "purpose": "测试第一条信息路线",
                        "criterion_id": "find-signal",
                        "expected_delta": "取得一条可判断的新信息",
                        "stop_condition": "能判断路线是否有效",
                }
            )
        if call_number == 2:
            return json.dumps(
                {
                    "observation_assessment": {
                        "information_gain": "low",
                        "expectation_met": False,
                        "finding": "第一条路线没有新增信息",
                    },
                    "action": {
                        "type": "tool_call",
                            "step_id": "lookup-2",
                            "tool_name": "lookup",
                            "arguments": {"route": "same-source-2"},
                            "purpose": "复查同类信息路线",
                            "criterion_id": "find-signal",
                            "expected_delta": "确认同类来源是否仍有增量",
                            "stop_condition": "能够决定保留或换路",
                    },
                },
                ensure_ascii=False,
            )
        if call_number == 3:
            return json.dumps(
                {
                    "observation_assessment": {
                        "information_gain": "none",
                        "expectation_met": False,
                        "finding": "同类来源仍然没有信息增量",
                    },
                    "action": {
                        "type": "tool_call",
                            "step_id": "lookup-rejected",
                            "tool_name": "lookup",
                            "arguments": {"route": "same-source-3"},
                            "purpose": "继续同类路线",
                            "criterion_id": "find-signal",
                            "expected_delta": "寻找新的有效信息",
                            "stop_condition": "取得增量或确认路线无效",
                    },
                },
                ensure_ascii=False,
            )
        assert "不能只替换近义词" in prompt
        return json.dumps(
            {
                "state_update": {
                    "current_strategy": "改用行为数据验证，而不是继续查同类来源",
                    "strategy_change_reason": "连续两次同类来源没有信息增量",
                    "strategy_assessments": [
                        {
                            "strategy_id": "same-source",
                            "approach": "连续查询同类公开来源",
                            "outcome": "low_yield",
                            "lesson": "来源同质化，无法验证目标",
                            "next_change": "改用行为数据",
                        }
                    ],
                },
                "action": {
                    "type": "tool_call",
                        "step_id": "lookup-different",
                        "tool_name": "lookup",
                        "arguments": {"route": "behavior-data"},
                        "purpose": "用不同类型的行为数据验证假设",
                        "criterion_id": "find-signal",
                        "expected_delta": "获得区别于公开来源的行为证据",
                        "stop_condition": "能够支持或否定假设",
                },
            },
            ensure_ascii=False,
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=4,
        default_tool_permissions={"lookup": "allow"},
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="寻找有效工作路线",
        objective="自主选择方法并根据结果调整策略",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "yielded"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert calls == ["same-source-1", "same-source-2", "behavior-data"], json.dumps(
        checkpoint, ensure_ascii=False, sort_keys=True
    )
    assert checkpoint["work_state"]["current_strategy"].startswith("改用行为数据")
    assert not checkpoint.get("strategy_change_required")


def test_executor_runs_independent_tool_batch_before_reentering_model(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="batch-executor")
    registry = ToolRegistry()
    handler_calls: list[int] = []

    def lookup(_, arguments):
        value = int(arguments["value"])
        handler_calls.append(value)
        return {"value": value, "finding": f"证据-{value}"}

    registry.register(
        ToolSpec(
            name="lookup",
            description="相互独立的只读查询",
            input_schema={"type": "object", "required": ["value"]},
            handler=lookup,
        )
    )
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return json.dumps(
                {
                    "state_summary": "已规划三个独立查询",
                    "actions": [
                        {
                            "type": "tool_call",
                            "step_id": f"lookup-{value}",
                            "tool_name": "lookup",
                            "arguments": {"value": value},
                        }
                        for value in (1, 2, 3)
                    ],
                },
                ensure_ascii=False,
            )
        assert "工作上下文" in prompt
        assert all(f"证据-{value}" in prompt for value in (1, 2, 3))
        return work_decision(
            {"type": "complete", "message": "三个结果已统一提纯"}, progress=1
        )

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=6
    )
    executor.set_tool_permission(agent_id, "lookup", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="批量研究",
        objective="一次规划多个独立查询后统一分析",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert result.steps_executed == 4
    assert provider_calls == 2
    assert handler_calls == [1, 2, 3]
    assert len(executor.list_tool_calls(thread.id)) == 3


def test_executor_calls_allowed_tool_then_completes(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="executor")
    registry = ToolRegistry()
    handler_calls: list[dict] = []

    def lookup(_, arguments):
        handler_calls.append(arguments)
        return {"items": ["可靠来源A", "可靠来源B"]}

    registry.register(
        ToolSpec(
            name="lookup",
            description="只读查询",
            input_schema={"type": "object"},
            handler=lookup,
        )
    )
    provider_calls = 0

    def provider(_, __, ___):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "research",
                    "tool_name": "lookup",
                    "arguments": {"query": "目标市场"},
                }
            )
        return work_decision(
            {"type": "complete", "message": "已根据工具结果完成研究"},
            progress=1,
        )

    executor = WorkExecutor(database, runtime, registry, provider, max_steps_per_run=4)
    executor.set_tool_permission(agent_id, "lookup", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="自主循环",
        objective="自行调用工具后完成",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)
    assert result.outcome == "completed"
    assert result.steps_executed == 2
    assert handler_calls == [{"query": "目标市场"}]
    calls = executor.list_tool_calls(thread.id)
    assert len(calls) == 1
    assert calls[0].status == "completed"
    assert calls[0].attempt_count == 1


def test_executor_stops_repeated_plan_updates_and_requires_action(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="plan-gate")
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 2:
            assert "行动提示" in prompt
        return json.dumps(
            {
                "state_summary": "已完成反思，下一步应调用工具",
                "state_update": {
                    "phase": "准备执行",
                    "next_actions": ["调用具体工具核验关键事实"],
                },
                "action": {
                    "type": "plan_update",
                    "message": "下一轮开始执行",
                },
            },
            ensure_ascii=False,
        )

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, max_steps_per_run=6
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="计划空转保护",
        objective="反思后必须产生具体行动",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "waiting_action_replan"
    assert result.run.status == "waiting_retry"
    assert provider_calls == 3
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["consecutive_plan_updates"] == 1
    assert checkpoint["rejected_plan_updates"] == 2
    assert "必须转化为具体" in checkpoint["action_required"]
    assert checkpoint["work_state_version"] == 1
    assert result.run.next_wake_at is not None
    next_wake = result.run.next_wake_at
    if next_wake.tzinfo is None:
        next_wake = next_wake.replace(tzinfo=timezone.utc)
    assert next_wake <= datetime.now(timezone.utc) + timedelta(minutes=2)


def test_material_candidate_disposition_is_saved_even_when_action_is_plan_update(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="material-replan")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="综合市场研究",
        objective="从各种渠道发现普通人正在参与的市场现象",
    )
    runtime.create_run(
        thread.id,
        checkpoint={
            "action_required": "失败候选必须先完成处置",
            "work_state": {
                "market_candidates": [
                    {
                        "candidate_ref": "broken-page",
                        "title": "失效活动页面",
                        "lens": "小众社群",
                        "decision": "deep_read",
                        "source_url": "https://example.com/missing",
                    }
                ]
            },
        },
    )
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return json.dumps(
                {
                    "state_update": {
                        "market_candidates": [
                            {
                                "candidate_ref": "broken-page",
                                "title": "失效活动页面",
                                "lens": "小众社群",
                                "decision": "reject",
                                "reason": "代表页面返回404，候选不可继续深读",
                                "source_url": "https://example.com/missing",
                            }
                        ],
                        "rejected_paths": ["失效活动页面返回404"],
                    },
                    "action": {
                        "type": "plan_update",
                        "message": "淘汰失效候选并切换路线",
                    },
                },
                ensure_ascii=False,
            )
        assert "行动提示" in prompt
        return work_decision({"type": "complete", "message": "按当前编辑判断收口"})

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, max_steps_per_run=4
    )
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert provider_calls == 2
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["work_state"]["market_candidates"][0]["decision"] == "reject"
    assert "rejected_plan_updates" not in checkpoint


def test_failed_market_tool_result_does_not_become_unsynthesized_content(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="failed-market-read")
    registry = ToolRegistry()

    def missing_page(_, __):
        raise RuntimeError("网页返回HTTP 404")

    registry.register(
        ToolSpec(
            name="read_web_page",
            description="读取网页",
            input_schema={"type": "object"},
            handler=missing_page,
        )
    )
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "read-candidate",
                    "tool_name": "read_web_page",
                    "arguments": {"url": "https://example.com/missing"},
                }
            )
        assert "网页返回HTTP 404" in prompt
        return work_decision({"type": "complete", "message": "AI判断该候选无效并收口"})

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=4
    )
    executor.set_tool_permission(agent_id, "read_web_page", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="综合市场热点分析",
        objective="从各种渠道寻找生活、爱好和小众话题",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert "unsynthesized_tool_results" not in checkpoint
    assert "market_synthesis_required" not in checkpoint


def test_rich_market_page_is_preserved_and_consumed_by_exact_id(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="rich-material")
    registry = ToolRegistry()
    source_url = "https://example.com/reset-to-real"
    page_text = (
        "Navigation\n" * 80
        + "Reset to Real研究调查了4051名18至35岁参与者。\n"
        + "79%的年轻人计划参加更多线下活动，拼图比赛参与量翻倍。\n"
        + "咖啡加跑步活动增长233%，动漫与Cosplay派对增长82%。\n"
        + "方法：2025年7月在美国和英国收集样本，并比较平台活动数据。"
    )
    registry.register(
        ToolSpec(
            name="read_web_page",
            description="读取网页",
            input_schema={"type": "object"},
            handler=lambda *_: {
                "url": source_url,
                "resolved_url": source_url,
                "title": "Reset to Real",
                "text": page_text,
            },
        )
    )
    calls = 0

    def provider(_, __, prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "read-reset",
                    "tool_name": "read_web_page",
                    "arguments": {"url": source_url},
                }
            )
        assert "4051" in prompt and "增长233%" in prompt
        return json.dumps(
            {
                "state_update": {
                    "criteria_status": [
                        {
                            "criterion_id": "final-insight",
                            "status": "unverified",
                            "note": "尚无有效洞察，仍需形成最终成果。",
                        }
                    ],
                    "market_insights": [
                        {
                            "title": "低压力真实社交正在替代精心表演",
                            "category": "社交关系",
                            "signal": "年轻人增加线下活动参与",
                            "phenomenon": "真实、偶发和可参与的活动受到偏爱",
                            "why_interesting": "线上展示疲劳正在改变线下体验设计",
                            "source_refs": [source_url, "tool_call:1"],
                        }
                    ]
                },
                "consumed_tool_call_ids": [1],
                "action": {"type": "complete", "message": "已整理可用洞察"},
            },
            ensure_ascii=False,
        )

    executor = WorkExecutor(database, runtime, registry, provider, max_steps_per_run=4)
    executor.set_tool_permission(agent_id, "read_web_page", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="生活社交热点研究",
        objective="研究普通人参与的社交和生活市场热点",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint.get("pending_research_material_ids") is None
    assert checkpoint["research_materials"][0]["consumed"] is True
    assert "低压力真实社交" in (result.run.result_text or "")
    assert checkpoint["runtime_criteria_facts"]["durable_market_insight_count"] == 1
    assert checkpoint["runtime_criteria_facts"]["sourced_market_insight_count"] == 1
    criterion = checkpoint["work_state"]["criteria_status"][0]
    assert criterion["status"] == "partially_met"
    assert "运行时账本已保存1条洞察" in criterion["note"]
    assert criterion["evidence_refs"] == ["insight:1"]


def test_successful_rich_page_cannot_be_rejected_as_unreadable(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="source-conflict")
    registry = ToolRegistry()
    source_url = "https://example.com/participation-study"
    registry.register(
        ToolSpec(
            name="read_web_page",
            description="读取网页",
            input_schema={"type": "object"},
            handler=lambda *_: {
                "url": source_url,
                "title": "参与研究",
                "text": "完整研究正文。" + "普通人参与活动并讨论真实体验。" * 80,
            },
        )
    )
    calls = 0

    def provider(_, __, prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "read-study",
                    "tool_name": "read_web_page",
                    "arguments": {"url": source_url},
                }
            )
        if calls == 2:
            return json.dumps(
                {
                    "state_update": {
                        "market_candidates": [
                            {
                                "candidate_ref": source_url,
                                "title": "参与研究",
                                "lens": "社交关系",
                                "decision": "reject",
                                "reason": "页面读取失败，没有正文",
                                "source_url": source_url,
                            }
                        ]
                    },
                    "consumed_tool_call_ids": [1],
                    "action": {"type": "complete", "message": "没有可用内容"},
                },
                ensure_ascii=False,
            )
        assert "不能按‘无法读取’淘汰" in prompt or "尚未进入工作判断" in prompt
        return json.dumps(
            {
                "state_update": {
                    "market_insights": [
                        {
                            "title": "真实体验参与",
                            "signal": "普通人参与活动并公开讨论",
                            "phenomenon": "参与式体验正在获得注意",
                            "why_interesting": "它连接了社交需求和内容表达",
                            "source_refs": [source_url, "tool_call:1"],
                        }
                    ]
                },
                "consumed_tool_call_ids": [1],
                "action": {"type": "complete", "message": "已按正文重新判断"},
            },
            ensure_ascii=False,
        )

    executor = WorkExecutor(database, runtime, registry, provider, max_steps_per_run=5)
    executor.set_tool_permission(agent_id, "read_web_page", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="社交市场热点",
        objective="寻找普通人参与和讨论的社交市场热点",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["source_disposition_conflicts"] == 1
    assert checkpoint["work_state"]["market_candidates"][0]["decision"] == "deep_read"
    assert "真实体验参与" in (result.run.result_text or "")


def test_market_delegation_rewrites_controller_invented_numeric_quota(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        worker = AgentProfile(name="市场员工")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, instance_id="market-quota")
    parent = runtime.create_thread(
        owner_agent_id=controller_id,
        title="市场总控",
        objective="整理普通人的生活、社交和娱乐市场热点",
    )
    runtime.create_run(parent.id)
    collaboration = CollaborationService(database)
    assigned = collaboration.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={
            "title": "生活热点研究",
            "objective": "研究普通人生活、社交与娱乐市场洞察",
            "success_criteria": [
                "产出至少5条具体市场洞察",
                "每条说明现象与产品启示",
            ],
        },
    )
    child = runtime.get_thread(assigned["child_thread_id"])
    criteria = json.loads(child.success_criteria_json)
    constraints = json.loads(child.constraints_json)

    assert all("至少5条" not in item for item in criteria)
    assert any("数量由内容价值决定" in item for item in criteria)
    assert any("不得仅因缺少群体规模" in item for item in criteria)
    assert any("不得设置机械数量" in item for item in constraints)
    assert any("不是论文式证据审计" in item for item in constraints)


def test_salient_excerpt_skips_navigation_and_keeps_participation_data():
    text = "\n".join([f"Menu {index}" for index in range(200)]) + (
        "\n调查覆盖4051名参与者，79%计划参加更多活动，咖啡加跑步增长233%。"
    )
    excerpt = salient_research_excerpt(text, max_chars=500)

    assert "4051" in excerpt
    assert "增长233%" in excerpt


def test_context_budget_exposes_one_atomic_pending_research_material(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="material-context")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="市场材料压缩",
        objective="研究普通人生活消费和社交热点",
    )
    materials = [
        {
            "tool_call_id": index,
            "tool_name": "read_web_page",
            "kind": "page",
            "title": f"来源{index}",
            "source_url": f"https://example.com/{index}",
            "content_chars": 5000,
            "has_substantial_content": True,
            "content_excerpt": (f"关键参与数据-{index}-" + "正文" * 1800),
        }
        for index in range(1, 7)
    ]
    checkpoint = {
        "state_summary": "旧摘要" * 10000,
        "work_state": {
            "verified_facts": ["旧事实" * 1000 for _ in range(30)],
            "market_insights": [],
        },
        "research_materials": materials,
        "pending_research_material_ids": list(range(1, 7)),
    }
    runtime.create_run(thread.id, checkpoint=checkpoint)
    service = WorkContextService(
        database,
        {"work_context": {"max_prompt_chars": 12000, "checkpoint_max_chars": 4000}},
    )

    context = service.prompt_context(thread.id, checkpoint)
    pending = context["current_checkpoint"]["pending_research_materials"]

    assert [item["tool_call_id"] for item in pending] == [1]
    assert "关键参与数据-1" in pending[0]["content_excerpt"]


def legacy_search_results_are_read_and_judged_one_article_at_a_time(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="atomic-research")
    registry = ToolRegistry()
    page_one = "https://example.com/ordinary-life"
    page_two = "https://example.com/hobby-scene"
    opened: list[str] = []

    registry.register(
        ToolSpec(
            name="web_search",
            description="搜索网页",
            input_schema={"type": "object"},
            handler=lambda *_: {
                "query": "全球普通人新生活趋势",
                "results": [
                    {"title": "生活现场", "url": page_one, "snippet": "标题摘要一"},
                    {"title": "兴趣现场", "url": page_two, "snippet": "标题摘要二"},
                ],
            },
        )
    )

    def read_page(_, arguments):
        url = arguments["url"]
        opened.append(url)
        marker = "ARTICLE_ONE_FULL_BODY" if url == page_one else "ARTICLE_TWO_FULL_BODY"
        return {
            "url": url,
            "resolved_url": url,
            "title": "全球现场",
            "text": marker + " 普通人参加、讨论并分享真实体验。" * 80,
        }

    registry.register(
        ToolSpec(
            name="read_web_page",
            description="阅读全文",
            input_schema={"type": "object"},
            handler=read_page,
        )
    )
    model_prompts: list[str] = []

    def provider(_, __, prompt):
        model_prompts.append(prompt)
        call_number = len(model_prompts)
        if call_number == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "search-once",
                    "tool_name": "web_search",
                    "arguments": {"query": "全球普通人新生活趋势"},
                }
            )
        if call_number == 2:
            assert "标题摘要一" in prompt
            assert "标题摘要二" in prompt
            assert "ARTICLE_ONE_FULL_BODY" not in prompt
            assert "ARTICLE_TWO_FULL_BODY" not in prompt
            return json.dumps(
                {
                    "state_update": {
                        "current_strategy": "先阅读生活现场，再根据正文价值决定是否阅读兴趣现场",
                        "working_hypotheses": [
                            {
                                "hypothesis_id": "ordinary-participation",
                                "statement": "普通人正在偏好可参与的真实线下体验",
                                "next_test": "阅读代表页面核验参与行为",
                            }
                        ],
                        "market_candidates": [
                            {
                                "candidate_ref": page_one,
                                "title": "生活现场",
                                "lens": "生活消费",
                                "decision": "deep_read",
                                "reason": "先核验普通人参与证据",
                                "source_url": page_one,
                            },
                            {
                                "candidate_ref": page_two,
                                "title": "兴趣现场",
                                "lens": "兴趣爱好",
                                "decision": "pending",
                                "reason": "作为后续不同视角候选",
                                "source_url": page_two,
                            },
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "搜索召回了两个可区分的参与场景",
                    },
                    "consumed_tool_call_ids": [1],
                    "action": {
                        "type": "tool_call",
                        "step_id": "read-selected-one",
                        "tool_name": "read_web_page",
                        "arguments": {"url": page_one},
                        "purpose": "核验生活现场是否包含具体参与行为",
                        "criterion_id": "article-understanding",
                        "expected_delta": "获得第一篇代表页面正文",
                        "stop_condition": "能够保留或淘汰该候选",
                    },
                },
                ensure_ascii=False,
            )
        if call_number == 3:
            assert "ARTICLE_ONE_FULL_BODY" in prompt
            assert "ARTICLE_TWO_FULL_BODY" not in prompt
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "第一篇内容普通，缺少可继续挖掘的具体反常现象。",
                        "rejected_paths": [f"{page_one}：只有一般参与描述"],
                    },
                    "observation_assessment": {
                        "information_gain": "low",
                        "expectation_met": False,
                        "finding": "正文只有一般参与描述，未验证反常现象",
                    },
                    "consumed_tool_call_ids": [2],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        if call_number == 4:
            assert "ARTICLE_ONE_FULL_BODY" not in prompt
            assert "ARTICLE_TWO_FULL_BODY" not in prompt
            return json.dumps(
                {
                    "state_update": {
                        "current_strategy": "转向兴趣社群页面，寻找具体行为与讨论信号",
                        "strategy_change_reason": "生活类代表页信息增量低，改用不同场景验证假设",
                    },
                    "action": {
                        "type": "tool_call",
                        "step_id": "read-selected-two",
                        "tool_name": "read_web_page",
                        "arguments": {"url": page_two},
                        "purpose": "核验兴趣活动中的真实社交参与",
                        "criterion_id": "article-understanding",
                        "expected_delta": "获得第二篇不同场景的完整正文",
                        "stop_condition": "能够形成或否定参与型现象",
                    },
                },
                ensure_ascii=False,
            )
        if call_number == 5:
            assert "ARTICLE_TWO_FULL_BODY" in prompt
            assert "ARTICLE_ONE_FULL_BODY" not in prompt
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "第二篇呈现了普通人把兴趣活动变成真实社交场景。",
                        "market_insights": [
                            {
                                "title": "兴趣活动正在替代精心设计的社交",
                                "signal": "普通人参加活动并公开讨论真实体验",
                                "phenomenon": "低包装、可参与的兴趣现场获得偏好",
                                "why_interesting": "它把爱好消费与真实社交需求连接起来",
                                "source_refs": [page_two, "tool_call:3"],
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "正文提供了兴趣活动进入真实社交的参与信号",
                    },
                    "consumed_tool_call_ids": [3],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        assert "ARTICLE_ONE_FULL_BODY" not in prompt
        assert "ARTICLE_TWO_FULL_BODY" not in prompt
        return work_decision(
            {"type": "complete", "message": "已逐篇阅读全文并完成中文提炼"},
            progress=1,
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=8,
        default_tool_permissions={
            "web_search": "allow",
            "read_web_page": "allow",
        },
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="全球生活兴趣热点研究",
        objective="从全球来源寻找普通人参与的生活、兴趣、社交和娱乐现象，用中文提炼",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert opened == [page_one, page_two]
    first_article_chars = len(model_prompts[2])
    second_article_chars = len(model_prompts[4])
    assert abs(first_article_chars - second_article_chars) / max(
        first_article_chars, second_article_chars
    ) < 0.2
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["research_cycle_count"] == 1
    assert not checkpoint.get("pending_research_material_ids")
    assert len(checkpoint["work_state"]["market_insights"]) == 1
    assert all(
        "content_excerpt" not in item and "candidates" not in item
        for item in checkpoint["research_materials"]
        if item.get("consumed")
    )


def test_context_service_attaches_full_current_article_without_previous_article(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="full-article-packet")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="逐篇阅读全文",
        objective="从全球来源阅读并用中文提炼普通人的生活现象",
    )
    run = runtime.create_run(thread.id)
    with database.sessions() as session:
        for index, marker in ((1, "TAIL_OF_ARTICLE_ONE"), (2, "TAIL_OF_ARTICLE_TWO")):
            session.add(
                ToolCall(
                    work_thread_id=thread.id,
                    work_run_id=run.id,
                    owner_agent_id=agent_id,
                    step_id=f"read-{index}",
                    action_key=f"{index:064x}",
                    tool_name="read_web_page",
                    arguments_json=json.dumps(
                        {"url": f"https://example.com/{index}"}
                    ),
                    status="completed",
                    permission="allow",
                    retry_safety="safe",
                    execution_token=1,
                    result_json=json.dumps(
                        {
                            "url": f"https://example.com/{index}",
                            "title": f"Article {index}",
                            "text": ("正文段落。" * 1200) + marker,
                        }
                    ),
                )
            )
        session.commit()
    materials = [
        {
            "tool_call_id": index,
            "tool_name": "read_web_page",
            "kind": "page",
            "title": f"Article {index}",
            "source_url": f"https://example.com/{index}",
            "content_chars": 6000,
            "has_substantial_content": True,
            "content_excerpt": f"这只是第{index}篇的短摘录",
        }
        for index in (1, 2)
    ]
    service = WorkContextService(
        database,
        {
            "work_context": {
                "max_prompt_chars": 12000,
                "active_input_chars": 20000,
            }
        },
    )
    first_checkpoint = {
        "work_state": {"market_insights": []},
        "research_materials": materials,
        "pending_research_material_ids": [1, 2],
        "context_runtime": {
            "active_unit": {
                "unit_id": "unit-1",
                "purpose": "judge_article",
                "dependency_tool_call_ids": [1],
            }
        },
    }
    first_context = service.prompt_context(thread.id, first_checkpoint)
    assert "TAIL_OF_ARTICLE_ONE" in first_context["active_input"]["text"]
    assert "TAIL_OF_ARTICLE_TWO" not in json.dumps(first_context, ensure_ascii=False)

    second_checkpoint = {
        **first_checkpoint,
        "work_state": {
            "current_judgment": "第一篇已经提炼为有界中文结论",
            "market_insights": [],
        },
        "pending_research_material_ids": [2],
        "context_runtime": {
            "active_unit": {
                "unit_id": "unit-2",
                "purpose": "judge_article",
                "dependency_tool_call_ids": [2],
            }
        },
    }
    second_context = service.prompt_context(thread.id, second_checkpoint)
    second_encoded = json.dumps(second_context, ensure_ascii=False)
    assert "TAIL_OF_ARTICLE_TWO" in second_context["active_input"]["text"]
    assert "TAIL_OF_ARTICLE_ONE" not in second_encoded
    assert "第一篇已经提炼为有界中文结论" in second_encoded


def test_administrative_tool_result_does_not_trigger_market_synthesis_gate(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="market-admin-tool")
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="list_company_agents",
            description="列出员工",
            input_schema={"type": "object"},
            handler=lambda *_: {"agents": []},
        )
    )
    provider_calls = 0

    def provider(_, __, ___):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "list-agents",
                    "tool_name": "list_company_agents",
                    "arguments": {},
                }
            )
        return work_decision({"type": "complete", "message": "AI按当前信息完成判断"})

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=4
    )
    executor.set_tool_permission(agent_id, "list_company_agents", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="综合市场分析",
        objective="从各种渠道研究市场热点",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert "unsynthesized_tool_results" not in checkpoint


def test_executor_stops_history_scanning_until_real_progress(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="history-gate")
    registry = ToolRegistry()
    executed_offsets: list[int] = []

    def history(_, arguments):
        executed_offsets.append(int(arguments["offset"]))
        return {"items": [], "has_more": False}

    registry.register(
        ToolSpec(
            name="read_work_history",
            description="读取审计历史",
            input_schema={"type": "object"},
            handler=history,
        )
    )
    calls = 0

    def provider(_, __, ___):
        nonlocal calls
        calls += 1
        if calls <= 4:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": f"history-{calls}",
                    "tool_name": "read_work_history",
                    "arguments": {
                        "kind": "tool_calls",
                        "offset": (calls - 1) * 50,
                    },
                    "purpose": "检查历史",
                    "criterion_id": "history",
                    "expected_delta": "发现遗漏记录",
                    "stop_condition": "账本没有更多记录",
                }
            )
        return work_decision({"type": "complete", "message": "停止扫描"})

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=6
    )
    executor.set_tool_permission(agent_id, "read_work_history", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="历史扫描保护",
        objective="不得用翻账本代替工作",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "tool_decision_rejected"
    assert executed_offsets == []
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["rejected_unknown_tool_decisions_total"] >= 1
    assert checkpoint["last_rejected_tool_decision"]["unknown_tools"] == [
        "read_work_history"
    ]


def test_duration_contract_blocks_early_completion_and_governs_progress(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="duration-gate")
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="web_search",
            description="搜索公开网页",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda *_: {"items": ["已取得一项可审计结果"]},
        )
    )
    provider_calls = 0

    def provider(_, __, ___):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "initial-research",
                    "tool_name": "web_search",
                    "arguments": {"query": "持续研究"},
                }
            )
        return json.dumps(
            {
                "plan": [
                    {
                        "id": "research",
                        "title": "初步研究已完成",
                        "status": "completed",
                    }
                ],
                "state_update": {
                    "criteria_status": [
                            {
                                "criterion_id": "duration",
                                "status": "met",
                                "note": "已产生首项可审计研究活动",
                            }
                    ]
                },
                "progress": 1,
                "action": {
                    "type": "complete",
                    "message": "快速搜索结束",
                    "criteria_met": True,
                    "criteria_evidence": ["已搜索，但时长不足"],
                },
            },
            ensure_ascii=False,
        )

    executor = WorkExecutor(database, runtime, registry, provider)
    executor.set_tool_permission(agent_id, "web_search", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="持续研究",
        objective="持续进行5小时互联网研究",
        success_criteria=["持续执行约5小时并分阶段研究"],
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "waiting_temporal_contract"
    assert result.run.status == "waiting_schedule"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["progress"] < 1
    assert checkpoint["temporal_contract"]["required_minutes"] == 300
    assert checkpoint["temporal_contract"]["contract_version"] == 3
    assert (
        checkpoint["temporal_contract"]["completion_basis"]
        == "fixed_duration_then_finalize"
    )
    assert checkpoint["temporal_contract"]["deadline_at"] == checkpoint["temporal_contract"]["earliest_finish_at"]
    assert "required_meaningful_windows" not in checkpoint["temporal_contract"]
    assert "window_spacing_minutes" not in checkpoint["temporal_contract"]


def test_duration_contract_retries_recoverable_failures_before_deadline(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="duration-retry")
    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        lambda *_: work_decision(
            {
                "type": "fail",
                "message": "当前运行受工具不可用和协作失败双重阻塞",
            }
        ),
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="持续市场运营",
        objective="持续24小时开展市场分析",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "waiting_recoverable_failure"
    assert result.run.status == "waiting_retry"
    assert result.run.next_wake_at is not None
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["recoverable_failure_count"] == 1
    with database.sessions() as session:
        saved_thread = session.get(WorkThread, thread.id)
        assert saved_thread is not None
        assert saved_thread.status == "waiting_retry"


def test_duration_deadline_enters_forced_finalization_before_completion(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="hard-deadline")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="有界研究",
        objective="持续进行1分钟市场研究",
        success_criteria=["尽可能找到真正有趣且可核验的内容"],
    )
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "verified_facts": ["当前工具目录未提供通用网页搜索工具。"]
            },
            "temporal_contract": {
                "started_at": started.isoformat(),
                "meaningful_activities": [
                    {
                        "at": started.isoformat(),
                        "tool_call_id": 1,
                        "tool_name": "web_search",
                    }
                ],
            },
            "work_state": {
                "current_judgment": "已发现一项有明显反差的跨文化传播事件。",
                "verified_facts": ["公开原文已经保存。"],
                "open_questions": ["第二个独立来源尚未取得。"],
            },
        },
    )

    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        assert "研究截止时间已到" in prompt
        return work_decision(
            {
                "type": "complete",
                "message": "已根据持久成果完成收尾；第二个独立来源尚未取得。",
                "criteria_met": False,
                "criteria_evidence": ["已交付当前成果并说明来源限制"],
            }
        )

    executor = WorkExecutor(database, runtime, ToolRegistry(), provider)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization"
    assert result.run.status == "completed"
    assert len(prompts) == 0
    assert result.run.result_text
    assert "第二个独立来源尚未取得" in (result.run.result_text or "")


def test_root_with_completed_child_finalizes_locally_at_hard_deadline(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="root-deadline")
    root = runtime.create_thread(
        owner_agent_id=agent_id,
        title="1分钟总控任务",
        objective="持续1分钟组织员工研究并在到点后收口",
    )
    child = runtime.create_thread(
        owner_agent_id=agent_id,
        parent_thread_id=root.id,
        title="员工研究任务",
        objective="完成市场研究",
    )
    with database.sessions() as session:
        saved_child = session.get(WorkThread, child.id)
        saved_child.status = "completed"
        saved_child.result_text = "员工已提交持久业务成果。"
        saved_child.finished_at = datetime.now(timezone.utc)
        session.commit()
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    runtime.create_run(
        root.id,
        checkpoint={
            "temporal_contract": {
                "required_minutes": 1,
                "started_at": started.isoformat(),
                "earliest_finish_at": (started + timedelta(minutes=1)).isoformat(),
                "deadline_at": (started + timedelta(minutes=1)).isoformat(),
                "meaningful_activities": [{"tool_call_id": 1}],
            }
        },
    )
    provider_calls = 0

    def provider(*_):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("根任务到达硬截止且子任务已结束后不得再次调用模型")

    executor = WorkExecutor(database, runtime, ToolRegistry(), provider)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization"
    assert result.run.status == "completed"
    assert provider_calls == 0
    with database.sessions() as session:
        saved_root = session.get(WorkThread, root.id)
        assert saved_root.status == "completed"


def test_deadline_complete_missing_acceptance_fields_uses_runtime_facts(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="deadline-facts")
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="事实账本收尾",
        objective="持续进行1分钟研究",
        success_criteria=["交付当前研究成果"],
    )
    runtime.create_run(
        thread.id,
        checkpoint={
            "temporal_contract": {
                "required_minutes": 1,
                "started_at": started.isoformat(),
                "earliest_finish_at": (started + timedelta(minutes=1)).isoformat(),
                "deadline_at": (started + timedelta(minutes=1)).isoformat(),
                "meaningful_activities": [{"tool_call_id": 1}],
            },
            "work_state": {"verified_facts": ["已保存一项可交付成果。"]},
        },
    )
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        assert "runtime_facts" in prompt
        return work_decision(
            {
                "type": "complete",
                "message": "已根据持久事实账本完成收尾交付。",
            }
        )

    executor = WorkExecutor(database, runtime, ToolRegistry(), provider)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization"
    assert provider_calls == 0
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["deadline_finalization"]["fallback_used"] is True
    assert checkpoint["runtime_criteria_facts"]


def test_runtime_facts_tolerate_legacy_string_candidate_sources(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="legacy-source-shape")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="兼容旧候选来源",
        objective="统计候选来源角色时不得被旧字符串数据中断",
    )
    with database.sessions() as session:
        session.add(
            ResearchCandidate(
                work_thread_id=thread.id,
                owner_agent_id=agent_id,
                canonical_url="https://example.com/legacy",
                title="旧候选",
                status="queued",
                sources_json=json.dumps(["legacy-search"], ensure_ascii=False),
            )
        )
        session.commit()

    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "")
    facts = executor._runtime_criteria_facts(thread, {})

    assert facts["candidate_source_role_counts"] == {"general_discovery": 1}


def test_final_result_does_not_repeat_agent_preview_or_existing_insight():
    title = "低门槛怀旧模仿正在扩散"
    checkpoint = {
        "tree_execution_summary": {
            "thread_count": 2,
            "network_discovery_count": 1,
            "completed_tool_counts": {"web_search": 1, "read_web_page": 2},
        },
        "agent_execution_results": [
            {
                "title": "市场研究",
                "result_preview": "这是一段会与最终交付重复的完整员工成果。",
            }
        ],
        "selected_market_insights": [
            {
                "title": title,
                "signal": "用户开始大量模仿。",
                "source_refs": ["https://example.com/story"],
            }
        ],
    }
    model_result = (
        "主要成果\n\n1. 怀旧模仿成为新的低门槛社交语言\n"
        "用户开始大量模仿。\n"
        "来源：https://example.com/story"
    )

    with_summary = WorkExecutor._append_tree_execution_summary(
        model_result, checkpoint
    )
    final_result = WorkExecutor._append_market_insights(with_summary, checkpoint)

    assert title not in final_result
    assert "会与最终交付重复的完整员工成果" not in final_result
    assert "已合并 1 份员工Agent成果" in final_result
    assert "## 热门线索与市场洞察" not in final_result


def test_executor_crossing_deadline_bounds_forbidden_finalization_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="cross-deadline")
    registry = ToolRegistry()
    tool_runs = 0

    def lookup(*_):
        nonlocal tool_runs
        tool_runs += 1
        return {"value": "不得在收尾执行"}

    registry.register(
        ToolSpec(
            name="lookup",
            description="测试查询",
            input_schema={"type": "object"},
            handler=lookup,
        )
    )
    started = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)
    clock = {"now": started}
    monkeypatch.setattr(work_executor_module, "utc_now", lambda: clock["now"])
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="市场热点跨截止收尾",
        objective="持续进行1分钟市场热点研究",
        success_criteria=["形成当前成果交付"],
    )
    initial_run = runtime.create_run(
        thread.id,
        checkpoint={
            "temporal_contract": {
                "required_minutes": 1,
                "started_at": started.isoformat(),
                "earliest_finish_at": (started + timedelta(minutes=1)).isoformat(),
                "deadline_at": (started + timedelta(minutes=1)).isoformat(),
                "meaningful_activities": [{"tool_call_id": 1}],
            },
            "work_state": {
                "current_judgment": "已经形成一项可保留成果。",
                "next_actions": ["继续搜索"],
            },
        },
    )
    with database.sessions() as session:
        session.add(
            ToolCall(
                work_thread_id=thread.id,
                work_run_id=initial_run.id,
                owner_agent_id=agent_id,
                step_id="prior-research",
                action_key="cross-deadline-prior-research",
                tool_name="web_search",
                arguments_json=json.dumps({"query": "已完成的研究"}),
                status="completed",
                permission="allow",
                retry_safety="safe",
                execution_token=initial_run.execution_token,
                result_json=json.dumps({"items": []}),
            )
        )
        session.commit()
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            assert "最终收尾工作包" not in prompt
            clock["now"] = started + timedelta(minutes=2)
            return json.dumps(
                {
                    "state_update": {
                        "verified_facts": ["截止前最后一项事实已经归约。"]
                    },
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        assert "最终收尾工作包" in prompt
        assert "本轮没有任何工具" in prompt
        return work_decision(
            {
                "type": "tool_call",
                "step_id": f"forbidden-{provider_calls}",
                "tool_name": "lookup",
                "arguments": {},
            }
        )

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=20
    )
    executor.set_tool_permission(agent_id, "lookup", "allow")
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization", (
        result.run.last_error,
        json.loads(result.run.checkpoint_json or "{}"),
    )
    assert result.run.status == "completed"
    assert provider_calls == 1
    assert tool_runs == 0
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["deadline_finalization"]["attempts"] == 0
    assert checkpoint["deadline_finalization"]["fallback_used"] is True
    assert checkpoint["work_state"]["next_actions"] == []


def test_deadline_model_failures_use_local_result_without_retry_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="deadline-errors")
    started = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        work_executor_module,
        "utc_now",
        lambda: started + timedelta(minutes=2),
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="错误时确定性交付",
        objective="持续进行1分钟研究",
    )
    runtime.create_run(
        thread.id,
        checkpoint={
            "temporal_contract": {
                "required_minutes": 1,
                "started_at": started.isoformat(),
                "earliest_finish_at": (started + timedelta(minutes=1)).isoformat(),
                "deadline_at": (started + timedelta(minutes=1)).isoformat(),
                "meaningful_activities": [{"tool_call_id": 9}],
            },
            "work_state": {"verified_facts": ["已经保存当前阶段成果。"]},
        },
    )
    provider_calls = 0

    def provider(*_):
        nonlocal provider_calls
        provider_calls += 1
        raise TimeoutError("模拟模型超时")

    executor = WorkExecutor(database, runtime, ToolRegistry(), provider)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization"
    assert provider_calls == 0
    assert result.run.status == "completed"
    assert result.run.next_wake_at is None
    assert "已经保存当前阶段成果" in (result.run.result_text or "")


@pytest.mark.parametrize(
    ("objective", "expected_minutes"),
    [
        ("持续进行半小时市场研究", 30),
        ("开展半个小时热点分析", 30),
        ("执行三十分钟公开信息收集", 30),
        ("研究两小时消费趋势", 120),
        ("持续进行0.5小时市场研究", 30),
    ],
)
def test_chinese_duration_forms_are_recognized(
    tmp_path: Path, objective: str, expected_minutes: int
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="duration-forms")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="有界任务",
        objective=objective,
    )
    assert requested_duration_minutes(thread) == expected_minutes


def test_governed_progress_never_moves_backwards():
    checkpoint = {
        "progress": 0.38,
        "work_state": {
            "criteria_status": [
                {"criterion_id": "one", "status": "partially_met"},
                {"criterion_id": "two", "status": "unverified"},
            ]
        },
    }
    assert governed_progress(checkpoint, 0.25) == 0.38


def test_root_deadline_closes_waiting_child_and_does_not_loop(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        worker = AgentProfile(name="截止测试员工")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="deadline-tree")
    collaboration = CollaborationService(database)
    root = runtime.create_thread(
        owner_agent_id=controller_id,
        title="半小时热点任务",
        objective="持续进行半小时热点研究",
    )
    started = datetime.now(timezone.utc) - timedelta(minutes=31)
    runtime.create_run(
        root.id,
        checkpoint={
            "temporal_contract": {
                "required_minutes": 30,
                "started_at": started.isoformat(),
                "deadline_at": (started + timedelta(minutes=30)).isoformat(),
                "earliest_finish_at": (started + timedelta(minutes=30)).isoformat(),
                "meaningful_activities": [],
            }
        },
    )
    assigned = collaboration.create_direct(
        parent_thread_id=root.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={"title": "热点收集", "objective": "整理当前公开热点"},
    )
    with database.sessions() as session:
        child = session.get(WorkThread, assigned["child_thread_id"])
        child_run = session.scalar(
            select(WorkRun).where(WorkRun.work_thread_id == child.id)
        )
        assert child is not None and child_run is not None
        child.status = "waiting_dependency"
        child_run.status = "waiting_dependency"
        child_run.checkpoint_json = json.dumps(
            {
                "work_state": {
                    "current_judgment": "已收集一条具有代际反差的趋势线索。",
                    "market_insights": [
                        {
                            "title": "代际行为反转",
                            "signal": "年轻群体行为接近老一代",
                            "phenomenon": "身份与消费方式正在重新组合",
                            "why_interesting": "打破固有代际标签",
                        }
                    ],
                }
            },
            ensure_ascii=False,
        )
        session.add(
            ToolCall(
                work_thread_id=child.id,
                work_run_id=child_run.id,
                owner_agent_id=worker_id,
                step_id="public-search",
                action_key="deadline-tree-public-search",
                tool_name="web_search",
                arguments_json=json.dumps({"query": "公开热门讨论"}),
                status="completed",
                permission="allow",
                retry_safety="safe",
                execution_token=child_run.execution_token,
                result_json=json.dumps(
                    {"items": [{"title": "代际行为反转讨论"}]},
                    ensure_ascii=False,
                ),
            )
        )
        session.commit()

    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        lambda *_: work_decision(
            {"type": "complete", "message": "已整理截止时的全部Agent成果"}
        ),
    )
    claimed = runtime.claim_next(agent_id=controller_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization"
    assert "代际行为反转" in (result.run.result_text or "")
    assert "web_search 1次" in (result.run.result_text or "")
    assert "当前工具目录未提供通用网页搜索工具" not in (
        result.run.result_text or ""
    )
    with database.sessions() as session:
        child = session.get(WorkThread, assigned["child_thread_id"])
        collaboration_row = session.get(WorkCollaboration, assigned["id"])
        assert child is not None and child.status == "completed"
        assert collaboration_row is not None and collaboration_row.status == "completed"
        assert "代际行为反转" in (child.result_text or "")


def test_deadline_child_completion_closes_collaboration_and_wakes_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        worker = AgentProfile(name="截止回写员工")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="deadline-handoff")
    collaboration = CollaborationService(database)
    started = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        work_executor_module,
        "utc_now",
        lambda: started + timedelta(minutes=2),
    )
    root = runtime.create_thread(
        owner_agent_id=controller_id,
        title="截止协作闭环",
        objective="持续进行1分钟市场热点研究",
    )
    root_run = runtime.create_run(
        root.id,
        checkpoint={
            "temporal_contract": {
                "required_minutes": 1,
                "started_at": started.isoformat(),
                "earliest_finish_at": (started + timedelta(minutes=1)).isoformat(),
                "deadline_at": (started + timedelta(minutes=1)).isoformat(),
                "meaningful_activities": [],
            }
        },
    )
    assigned = collaboration.create_direct(
        parent_thread_id=root.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={
            "title": "员工热点研究",
            "objective": "持续进行1分钟市场热点研究并回写现有成果",
            "wait_for_result": True,
        },
    )
    with database.sessions() as session:
        parent = session.get(WorkThread, root.id)
        persisted_root_run = session.get(WorkRun, root_run.id)
        assert parent is not None and persisted_root_run is not None
        parent.status = "waiting_dependency"
        persisted_root_run.status = "waiting_dependency"
        persisted_root_run.checkpoint_json = json.dumps(
            {
                "temporal_contract": {
                    "required_minutes": 1,
                    "started_at": started.isoformat(),
                    "earliest_finish_at": (
                        started + timedelta(minutes=1)
                    ).isoformat(),
                    "deadline_at": (started + timedelta(minutes=1)).isoformat(),
                    "meaningful_activities": [],
                },
                "waiting_collaboration_id": assigned["id"],
            },
            ensure_ascii=False,
        )
        child_run = session.scalar(
            select(WorkRun).where(
                WorkRun.work_thread_id == assigned["child_thread_id"]
            )
        )
        assert child_run is not None
        child_run.checkpoint_json = json.dumps(
            {
                "temporal_contract": {
                    "required_minutes": 1,
                    "started_at": started.isoformat(),
                    "earliest_finish_at": (
                        started + timedelta(minutes=1)
                    ).isoformat(),
                    "deadline_at": (started + timedelta(minutes=1)).isoformat(),
                    "meaningful_activities": [],
                },
                "work_state": {"verified_facts": ["员工已保存阶段成果。"]},
            },
            ensure_ascii=False,
        )
        session.commit()

    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        lambda *_: work_decision(
            {"type": "complete", "message": "员工已整理并回写截止成果。"}
        ),
        terminal_handler=lambda thread_id, _: collaboration.sync_child_completion(
            thread_id
        ),
    )
    claimed = runtime.claim_next(agent_id=worker_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization"
    with database.sessions() as session:
        row = session.get(WorkCollaboration, assigned["id"])
        parent = session.get(WorkThread, root.id)
        persisted_root_run = session.get(WorkRun, root_run.id)
        assert row is not None and row.status == "completed"
        assert parent is not None and parent.status == "queued"
        assert persisted_root_run is not None and persisted_root_run.status == "queued"
        checkpoint = json.loads(persisted_root_run.checkpoint_json or "{}")
        assert checkpoint["collaboration_results"][0]["child_thread_id"] == assigned[
            "child_thread_id"
        ]
        assert checkpoint["collaboration_results"][0]["result_text"]


def test_hotspot_completion_uses_editorial_judgment_without_score_gate(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="editorial-quality")
    registry = ToolRegistry()
    source_url = "https://example.com/nostalgic-internet-culture"
    registry.register(
        ToolSpec(
            name="read_web_page",
            description="读取网页全文",
            input_schema={"type": "object"},
            handler=lambda *_: {
                "url": source_url,
                "title": "怀旧互联网文化回潮",
                "text": "公开讨论正文。" + "用户集中分享旧梗、早期社交体验并复刻旧界面。" * 80,
            },
        )
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="发现有趣热点",
        objective="寻找真正有趣而不是凑分的热点",
    )
    runtime.create_run(thread.id)
    provider_calls = 0

    def provider(*_):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "read-nostalgia",
                    "tool_name": "read_web_page",
                    "arguments": {"url": source_url},
                }
            )
        return json.dumps(
            {
                "state_update": {
                    "market_insights": [
                        {
                            "title": "怀旧互联网文化回潮",
                            "category": "文化",
                            "signal": "旧互联网符号重新进入公开讨论",
                            "phenomenon": "年轻人借过去的网络语言寻找共同记忆",
                            "why_interesting": "怀旧正在从复刻视觉扩展为身份表达",
                            "drivers": ["平台内容趋同带来的疲劳"],
                            "possible_waves": ["旧界面美学和低清影像再流行"],
                            "adjacent_searches": ["寻找2010年代社群符号"],
                            "implications": ["可发展怀旧主题内容系列"],
                            "discussion_signals": ["用户分享旧梗和早期社交体验"],
                            "participant_groups": ["普通内容创作者", "怀旧社群"],
                            "participation_behaviors": ["分享旧梗", "复刻旧界面"],
                            "participation_evidence": ["公开帖子出现集中回忆和二创"],
                            "ordinary_people_relevance": "普通人可以通过内容创作和社群讨论直接参与。",
                            "source_refs": [source_url, "tool_call:1"],
                            "confidence": "medium",
                        }
                    ]
                },
                "consumed_tool_call_ids": [1],
                "action": {
                    "type": "complete",
                    "message": "已根据意外性和跨圈层传播完成主编判断",
                },
            },
            ensure_ascii=False,
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
    )
    executor.set_tool_permission(agent_id, "read_web_page", "allow")
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert result.run.status == "completed"
    assert "怀旧互联网文化回潮" in (result.run.result_text or "")
    assert "潜在浪潮" in (result.run.result_text or "")
    assert "参与行为：分享旧梗；复刻旧界面" in (result.run.result_text or "")
    assert "普通人关联" in (result.run.result_text or "")
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert "hot_content_quality" not in checkpoint


def test_normal_controller_completion_includes_child_agent_outputs(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        worker = AgentProfile(name="成果回流员工")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="tree-merge")
    collaboration = CollaborationService(database)
    root = runtime.create_thread(
        owner_agent_id=controller_id,
        title="市场分析",
        objective="整理公开市场热点",
    )
    runtime.create_run(root.id)
    assigned = collaboration.create_direct(
        parent_thread_id=root.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={"title": "公众讨论搜索", "objective": "搜索并分析热门话题"},
    )
    with database.sessions() as session:
        child = session.get(WorkThread, assigned["child_thread_id"])
        child_run = session.scalar(
            select(WorkRun).where(WorkRun.work_thread_id == child.id)
        )
        row = session.get(WorkCollaboration, assigned["id"])
        assert child is not None and child_run is not None and row is not None
        child_run.checkpoint_json = json.dumps(
            {
                "work_state": {
                    "market_insights": [
                        {
                            "title": "真实感消费上升",
                            "signal": "公开讨论偏好不完美的真实体验",
                            "phenomenon": "消费者开始反感模板化精致",
                            "why_interesting": "可能改变内容与产品审美",
                        }
                    ]
                }
            },
            ensure_ascii=False,
        )
        child_run.status = "completed"
        child_run.result_text = "已完成公众讨论搜索并形成真实感消费洞察。"
        child_run.finished_at = datetime.now(timezone.utc)
        child.status = "completed"
        child.result_text = child_run.result_text
        child.finished_at = child_run.finished_at
        row.status = "completed"
        row.result_text = child_run.result_text
        row.completed_at = child_run.finished_at
        session.add(
            ToolCall(
                work_thread_id=child.id,
                work_run_id=child_run.id,
                owner_agent_id=worker_id,
                step_id="trend-search",
                action_key="normal-tree-trend-search",
                tool_name="web_search",
                arguments_json=json.dumps({"query": "真实感 消费 讨论"}),
                status="completed",
                permission="allow",
                retry_safety="safe",
                execution_token=child_run.execution_token,
                result_json=json.dumps({"items": ["真实感消费"]}, ensure_ascii=False),
            )
        )
        session.commit()

    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        lambda *_: work_decision({"type": "complete", "message": "总控整理完成"}),
    )
    claimed = runtime.claim_next(agent_id=controller_id)
    assert claimed is not None
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert "真实感消费上升" in (result.run.result_text or "")
    assert "公众讨论搜索" in (result.run.result_text or "")
    assert "web_search 1次" in (result.run.result_text or "")


def test_delegated_agent_inherits_root_hard_deadline(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        worker = AgentProfile(name="热点编辑Agent")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, instance_id="deadline-delegation")
    collaboration = CollaborationService(database)
    root = runtime.create_thread(
        owner_agent_id=controller_id,
        title="5小时热点任务",
        objective="持续进行5小时热点研究",
    )
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(hours=5)
    runtime.create_run(
        root.id,
        checkpoint={
            "temporal_contract": {
                "contract_version": 3,
                "completion_basis": "fixed_duration_then_finalize",
                "required_minutes": 300,
                "started_at": started.isoformat(),
                "earliest_finish_at": deadline.isoformat(),
                "deadline_at": deadline.isoformat(),
                "meaningful_activities": [],
            }
        },
    )

    assigned = collaboration.create_direct(
        parent_thread_id=root.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={"title": "员工热点研究", "objective": "寻找并核验有趣事件"},
    )

    with database.sessions() as session:
        child_run = session.scalar(
            select(WorkRun).where(
                WorkRun.work_thread_id == assigned["child_thread_id"]
            )
        )
        assert child_run is not None
        checkpoint = json.loads(child_run.checkpoint_json or "{}")
    inherited = checkpoint["temporal_contract"]
    assert inherited["deadline_at"] == deadline.isoformat()
    assert inherited["inherited_from_thread_id"] == root.id


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        (
            "研究至少5小时，并形成至少5个间隔约60分钟的有效研究窗口",
            "研究至少5小时，并在完整持续时间内推进实际研究并产生可审计成果",
        ),
        (
            "至少形成5个间隔约60分钟的有效研究窗口",
            "在完整持续时间内推进实际研究并产生可审计成果",
        ),
        (
            "研究期间保持多个间隔开的有效窗口",
            "研究期间保持持续产生可审计的有效研究活动",
        ),
        (
            "持续时长契约与5个有效研究活动要求均远未满足",
            "持续时长契约与可审计的有效研究活动要求均远未满足",
        ),
        (
            "窗口2：恢复候选；有效研究活动数量仍不足",
            "持续研究阶段：恢复候选；有效研究成果质量仍不足",
        ),
    ],
)
def test_legacy_fixed_window_text_is_normalized(legacy: str, expected: str):
    assert normalize_legacy_temporal_text(legacy) == expected


def test_scheduled_wait_is_rejected_while_immediate_research_remains(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="work-density")
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return json.dumps(
                {
                    "state_update": {
                        "next_actions": ["立即检索并核验生活类原文候选"]
                    },
                    "action": {
                        "type": "wait",
                        "wait_status": "waiting_schedule",
                        "next_wake_at": (
                            datetime.now(timezone.utc) + timedelta(hours=1)
                        ).isoformat(),
                        "message": "等待下一个研究窗口",
                    },
                },
                ensure_ascii=False,
            )
        assert "计划等待被拒绝" in prompt
        return work_decision({"type": "fail", "message": "测试结束"})

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, max_steps_per_run=3
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="五小时热点研究",
        objective="持续进行5小时生活、运动、艺术、动漫、小众热点研究",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "failed"
    assert len(prompts) == 2
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["scheduled_wait_rejections"] == 1
    assert "不能通过空等消耗时长" in checkpoint["last_scheduled_wait_error"]


def test_browser_page_inspection_resets_discovery_rhythm_guard(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="rhythm-reset")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="艺术热点研究",
        objective="研究近期艺术热点",
    )
    run = runtime.create_run(thread.id)
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "")

    with database.sessions() as session:
        for index in range(8):
            session.add(
                ToolCall(
                    work_thread_id=thread.id,
                    work_run_id=run.id,
                    owner_agent_id=agent_id,
                    step_id=f"discover-{index}",
                    action_key=f"{index + 1:064x}",
                    tool_name="web_search",
                    arguments_json=json.dumps({"query": f"art {index}"}),
                    status="completed",
                    permission="allow",
                    retry_safety="safe",
                    execution_token=0,
                    result_json=json.dumps({"results": [index]}),
                )
            )
        session.flush()
        assert executor._research_rhythm(session, thread.id)["guard_active"] is True
        session.add(
            ToolCall(
                work_thread_id=thread.id,
                work_run_id=run.id,
                owner_agent_id=agent_id,
                step_id="inspect-page",
                action_key=f"{99:064x}",
                tool_name="browser_open",
                arguments_json=json.dumps({"url": "https://example.com/art"}),
                status="completed",
                permission="allow",
                retry_safety="safe",
                execution_token=0,
                result_json=json.dumps({"text": "verified article"}),
            )
        )
        session.flush()
        rhythm = executor._research_rhythm(session, thread.id)

    assert rhythm["guard_active"] is False
    assert rhythm["discovery_calls_without_page_inspection"] == 0


def test_internal_discovery_guard_cannot_become_waiting_user(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="guard-wait")
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "wait",
                    "wait_status": "waiting_user",
                    "message": "请总控解除发现保护并提供发现权限",
                }
            )
        assert "等待用户被拒绝" in prompt
        return work_decision({"type": "fail", "message": "测试结束"})

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, max_steps_per_run=3
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="自主热点研究",
        objective="自主研究近期热点",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "failed"
    assert len(prompts) == 2
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["research_rhythm_wait_rejections"] == 1
    assert "不能要求用户或总控手动解除" in checkpoint[
        "last_research_rhythm_wait_error"
    ]


def test_unattended_work_cannot_become_waiting_user(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="unattended-wait")
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "wait",
                    "wait_status": "waiting_user",
                    "message": "请项目管理员核查业务资格后再继续",
                }
            )
        assert "这是无人任务" in prompt
        return work_decision(
            {
                "type": "complete",
                "message": "已根据持久业务状态自行完成核查",
                "criteria_met": True,
            }
        )

    executor = WorkExecutor(
        database, runtime, ToolRegistry(), provider, max_steps_per_run=3
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="无人产品适配",
        objective="自主完成业务记录",
        constraints=["这是无人执行任务，不等待人工确认"],
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert len(prompts) == 2
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["unattended_wait_rejections"] == 1


def test_startup_recovers_legacy_discovery_guard_user_wait(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="guard-recovery")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="待恢复研究",
        objective="自主研究公开热点",
    )
    run = runtime.create_run(thread.id)
    with database.sessions() as session:
        thread_row = session.get(WorkThread, thread.id)
        run_row = session.get(WorkRun, run.id)
        assert thread_row is not None and run_row is not None
        thread_row.status = "waiting_user"
        run_row.status = "waiting_user"
        run_row.last_error = "请总控解除发现保护后继续"
        run_row.checkpoint_json = json.dumps({"work_state": {"next_actions": []}})
        session.commit()
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )

    with TestClient(app):
        pass

    with database.sessions() as session:
        recovered_thread = session.get(WorkThread, thread.id)
        recovered_run = session.get(WorkRun, run.id)
        assert recovered_thread is not None and recovered_run is not None
        checkpoint = json.loads(recovered_run.checkpoint_json or "{}")
        assert recovered_thread.status == "queued"
        assert recovered_run.status == "queued"
        assert recovered_run.next_wake_at is None
        assert "不需要用户输入" in recovered_run.last_error
        assert "自主执行新的有界候选发现" in checkpoint["action_required"]


def test_startup_recovers_old_action_replan_wait_and_drops_failed_synthesis(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="replan-recovery")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="待恢复市场研究",
        objective="继续研究公开市场热点",
    )
    run = runtime.create_run(thread.id)
    with database.sessions() as session:
        thread_row = session.get(WorkThread, thread.id)
        run_row = session.get(WorkRun, run.id)
        assert thread_row is not None and run_row is not None
        thread_row.status = "waiting_retry"
        run_row.status = "waiting_retry"
        run_row.next_wake_at = datetime.now(timezone.utc) + timedelta(minutes=30)
        run_row.last_error = "反思尚未转化为具体行动；已保留工作状态，稍后在同一任务中重试。"
        run_row.checkpoint_json = json.dumps(
            {
                "market_synthesis_required": "失败结果需要提炼",
                "action_required": "失败结果需要提炼",
                "unsynthesized_tool_results": [
                    {
                        "tool_call_id": 99,
                        "tool_name": "read_web_page",
                        "status": "failed",
                        "result": {"error": "网页返回HTTP 404"},
                    }
                ],
            },
            ensure_ascii=False,
        )
        session.commit()
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )

    with TestClient(app):
        pass

    with database.sessions() as session:
        recovered_thread = session.get(WorkThread, thread.id)
        recovered_run = session.get(WorkRun, run.id)
        assert recovered_thread is not None and recovered_run is not None
        checkpoint = json.loads(recovered_run.checkpoint_json or "{}")
        assert recovered_thread.status == "queued"
        assert recovered_run.status == "queued"
        assert recovered_run.next_wake_at is None
        assert "状态更新与行动约束冲突" in recovered_run.last_error
        assert "unsynthesized_tool_results" not in checkpoint
        assert "market_synthesis_required" not in checkpoint


def test_market_discovery_batch_is_an_editorial_choice_not_an_executor_gate(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="research-batch")
    registry = ToolRegistry()
    executed: list[str] = []
    registry.register(
        ToolSpec(
            name="web_search",
            description="搜索公开网页",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda _, arguments: executed.append(arguments["query"])
            or {"items": [arguments["query"]]},
        )
    )
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "life-search",
                    "tool_name": "web_search",
                    "arguments": {"query": "生活热点"},
                }
            )
        if provider_calls == 2:
            assert "不是评分、最低数量或完成门槛" in prompt
            return json.dumps(
                {
                    "state_update": {
                        "market_lens_reviews": [
                            {
                                "lens": "生活消费",
                                "status": "no_viable_signal",
                                "note": "首轮结果没有值得保留的具体线索",
                            }
                        ]
                    },
                    "action": {
                        "type": "tool_call",
                        "step_id": "social-search",
                        "tool_name": "web_search",
                        "arguments": {"query": "朋友聚会线下社交流行活动"},
                    },
                },
                ensure_ascii=False,
            )
        assert "以普通人参与为主" in prompt
        return work_decision({"type": "fail", "message": "测试结束"})

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=6
    )
    executor.set_tool_permission(agent_id, "web_search", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="多分类热点研究",
        objective="研究生活、运动、艺术、动漫、小众热点",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "failed"
    assert provider_calls == 3
    assert executed == [
        "生活热点",
        "朋友聚会线下社交流行活动",
    ]
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert "rejected_single_discovery_actions" not in checkpoint


def test_duration_activities_are_aggregated_from_child_agent_tree(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="tree-windows")
    started = datetime.now(timezone.utc) - timedelta(minutes=150)
    parent = runtime.create_thread(
        owner_agent_id=agent_id,
        title="跨Agent持续研究",
        objective="持续进行2小时互联网研究",
    )
    child = runtime.create_thread(
        owner_agent_id=agent_id,
        parent_thread_id=parent.id,
        title="员工研究",
        objective="核验公开来源",
    )
    root_run = runtime.create_run(
        parent.id,
        checkpoint={
            "temporal_contract": {
                "started_at": started.isoformat(),
                "meaningful_windows": [],  # legacy checkpoint is migrated in place
                "required_meaningful_windows": 2,
                "window_spacing_minutes": 60,
            }
        },
    )
    with database.sessions() as session:
        child_row = session.get(WorkThread, child.id)
        assert child_row is not None
        child_row.status = "completed"
        child_row.finished_at = datetime.now(timezone.utc)
        child_run = WorkRun(
            work_thread_id=child.id,
            owner_agent_id=agent_id,
            trigger_kind="delegation",
            status="completed",
            execution_token=1,
            finished_at=datetime.now(timezone.utc),
        )
        session.add(child_run)
        session.flush()
        for index, finished in enumerate(
            (started + timedelta(minutes=10), started + timedelta(minutes=80)),
            start=1,
        ):
            session.add(
                ToolCall(
                    work_thread_id=child.id,
                    work_run_id=child_run.id,
                    owner_agent_id=agent_id,
                    step_id=f"research-{index}",
                    action_key=f"{index:064x}",
                    tool_name="read_web_page",
                    arguments_json=json.dumps({"url": f"https://example.com/{index}"}),
                    status="completed",
                    permission="allow",
                    retry_safety="safe",
                    execution_token=1,
                    result_json=json.dumps({"text": f"new fact {index}"}),
                    finished_at=finished,
                )
            )
        session.add(
            ToolCall(
                work_thread_id=child.id,
                work_run_id=child_run.id,
                owner_agent_id=agent_id,
                step_id="administrative-check",
                action_key="f" * 64,
                tool_name="list_company_agents",
                arguments_json="{}",
                status="completed",
                permission="allow",
                retry_safety="safe",
                execution_token=1,
                result_json=json.dumps({"agents": []}),
                finished_at=started + timedelta(minutes=90),
            )
        )
        session.commit()

    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        lambda *_: work_decision({"type": "complete", "message": "研究完成"}),
    )
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None and claimed.id == root_run.id
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed_with_local_finalization"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    activities = checkpoint["temporal_contract"]["meaningful_activities"]
    assert len(activities) == 2
    assert {item["source_thread_id"] for item in activities} == {child.id}
    assert checkpoint["temporal_contract"]["activity_scope"] == "work_tree"
    assert "meaningful_windows" not in checkpoint["temporal_contract"]
    assert "required_meaningful_windows" not in checkpoint["temporal_contract"]
    assert "window_spacing_minutes" not in checkpoint["temporal_contract"]


def test_root_cannot_complete_while_child_agent_is_open(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        worker = AgentProfile(name="未完成员工")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="tree-gate")
    collaboration = CollaborationService(database)
    parent = runtime.create_thread(
        owner_agent_id=controller_id,
        title="总控工作",
        objective="等待员工完成后再交付",
    )
    runtime.create_run(parent.id)
    assigned = collaboration.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={"title": "员工工作", "objective": "完成实际子任务"},
    )
    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        lambda *_: work_decision({"type": "complete", "message": "过早完成"}),
        progress_resume_handler=collaboration.resume_child_after_progress,
    )
    claimed = runtime.claim_next(agent_id=controller_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "waiting_tree_dependencies"
    assert result.run.status == "waiting_dependency"
    assert result.run.next_wake_at is None
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["waiting_collaboration_id"] == assigned["id"]
    assert checkpoint["open_tree_dependencies"]["open_thread_ids"] == [
        assigned["child_thread_id"]
    ]


def test_market_editorial_status_uses_insights_not_evidence_scores(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="hot-quality")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="生活、运动、艺术、动漫和小众热点",
        objective="寻找大家正在讨论的具体热点话题",
    )
    run = runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "market_insights": [
                    {
                        "title": f"{category}热门线索",
                        "category": category,
                        "signal": f"{category}公开讨论正在增长",
                        "phenomenon": f"{category}偏好出现变化",
                        "why_interesting": "存在反差和延伸空间",
                    }
                    for category in ("生活", "运动", "艺术", "动漫")
                ]
            }
        },
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")
    with database.sessions() as session:
        status = executor._market_editorial_status(session, thread)
    assert status is not None
    assert status["missing_categories"] == ["小众社群"]
    with database.sessions() as session:
        run_row = session.get(WorkRun, run.id)
        assert run_row is not None
        checkpoint = json.loads(run_row.checkpoint_json or "{}")
        checkpoint["work_state"]["market_insights"].append(
            {
                "title": "小众社群热门线索",
                "category": "小众",
                "signal": "小众社群出现集中讨论",
                "phenomenon": "边缘兴趣正在向外扩散",
                "why_interesting": "可能形成跨圈层传播",
            }
        )
        run_row.checkpoint_json = json.dumps(checkpoint, ensure_ascii=False)
        session.commit()
        ready = executor._market_editorial_status(session, thread)
    assert ready is not None and ready["missing_categories"] == []


def test_broad_market_research_keeps_lenses_open_instead_of_using_fixed_quota(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="ordinary-lenses")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="综合市场热点分析",
        objective="从各种渠道寻找新闻、话题、小众等等并整理为产品资料",
    )
    runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "market_insights": [
                    {
                        "title": "新模型发布",
                        "category": "科技",
                        "signal": "开发者讨论模型参数",
                        "phenomenon": "模型竞争继续",
                        "why_interesting": "技术能力提升",
                    },
                    {
                        "title": "模型推理升级",
                        "category": "人工智能",
                        "signal": "技术社区讨论推理性能",
                        "phenomenon": "算力继续增长",
                        "why_interesting": "开发者关注",
                    },
                ]
            }
        },
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")

    assert executor._hot_content_categories(thread) == []
    with database.sessions() as session:
        status = executor._market_editorial_status(session, thread)
    assert status is not None
    assert status["technology_dominant"] is True
    assert status["ordinary_participation_count"] == 0
    assert status["search_lenses"] == []
    assert status["topic_switch_required"] is True
    assert status["saturated_lenses"] == ["科技生活"]


def test_two_insights_in_one_ordinary_lens_require_a_free_topic_switch(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="topic-switch")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="开放市场热点",
        objective="广泛寻找普通生活和文化内容",
    )
    runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "market_insights": [
                    {"title": "社区旧物兑换", "category": "生活消费"},
                    {"title": "社区上门回收", "category": "生活消费"},
                ]
            }
        },
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")

    with database.sessions() as session:
        status = executor._market_editorial_status(session, thread)

    assert status is not None
    assert status["topic_switch_required"] is True
    assert status["saturated_lenses"] == ["生活消费"]
    assert "兴趣爱好" in status["alternative_lenses"]


def test_market_search_execution_is_not_the_same_as_editorial_completion(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="market-editorial-status")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="综合市场热点分析",
        objective="从各种渠道寻找生活、爱好、社交、娱乐和小众热门内容",
    )
    terminal_reviews = [
        {
            "lens": lens,
            "status": "no_viable_signal",
            "note": "已搜索并判断本轮没有值得保留的具体线索",
        }
        for lens in ("生活消费", "兴趣爱好", "社交关系", "小众社群")
    ]
    run = runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "market_lens_reviews": [
                    *terminal_reviews,
                    {
                        "lens": "娱乐文化",
                        "status": "search_invalid",
                        "note": "结果返回日本日历，与查询意图无关",
                    },
                ]
            }
        },
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")

    with database.sessions() as session:
        status = executor._market_editorial_status(session, thread)
    assert status is not None
    assert status["search_lenses"] == ["娱乐文化"]
    assert status["lens_statuses"]["娱乐文化"] == "search_invalid"

    with database.sessions() as session:
        run_row = session.get(WorkRun, run.id)
        checkpoint = json.loads(run_row.checkpoint_json or "{}")
        checkpoint["work_state"]["market_lens_reviews"][-1] = {
            "lens": "娱乐文化",
            "status": "no_viable_signal",
            "note": "已换词重查，仍无具体可用线索",
        }
        run_row.checkpoint_json = json.dumps(checkpoint, ensure_ascii=False)
        session.commit()
        status = executor._market_editorial_status(session, thread)
    assert status is not None and status["search_lenses"] == []


def test_heavy_news_mix_requires_next_general_discovery_from_ordinary_life(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="ordinary-content-balance")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="开放市场热点",
        objective="广泛寻找近期公开热点和普通人参与行为",
    )
    runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "market_insights": [
                    {
                        "title": "战争推动城市防空",
                        "category": "公共议题",
                        "signal": "导弹袭击和政府军事声明",
                        "phenomenon": "战争与城市安全",
                    },
                    {
                        "title": "机器人挑战体育纪录",
                        "category": "科技生活",
                        "signal": "机器人技术竞赛",
                        "phenomenon": "宏观技术竞争",
                    },
                ]
            }
        },
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")

    with database.sessions() as session:
        status = executor._market_editorial_status(session, thread)

    assert status is not None
    assert status["ordinary_priority_required"] is True
    assert status["heavy_news_count"] >= 2
    assert status["ordinary_participation_count"] == 0
    assert status["ordinary_priority_lenses"] == list(ORDINARY_PARTICIPATION_LENSES)


def test_market_candidate_must_be_dispositioned_before_coverage_is_ready(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="market-candidate-status")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="综合市场热点分析",
        objective="广泛研究普通人参与的市场话题",
    )
    runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "market_lens_reviews": [
                    {"lens": lens, "status": "no_viable_signal", "note": "已处置"}
                    for lens in (
                        "生活消费",
                        "兴趣爱好",
                        "社交关系",
                        "娱乐文化",
                        "小众社群",
                    )
                ],
                "market_candidates": [
                    {
                        "candidate_ref": "hobby-maxxing",
                        "title": "Hobby-maxxing取代doomscrolling",
                        "lens": "兴趣爱好",
                        "decision": "deep_read",
                        "source_url": "https://example.com/hobby-maxxing",
                    }
                ],
            }
        },
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")

    with database.sessions() as session:
        status = executor._market_editorial_status(session, thread)
    assert status is not None
    assert status["unresolved_candidates"][0]["decision"] == "deep_read"


def test_market_insights_semantically_merge_and_keep_richer_sources():
    first = {
        "title": "病毒式购物正在放大真实性与隐私矛盾",
        "category": "生活消费",
        "signal": "短视频病毒式购物让消费者一边追随真实体验一边担心隐私",
        "phenomenon": "购物发现机制从搜索迁移到社交内容",
        "why_interesting": "购买与身份表达正在合流",
        "source_refs": ["tool_call:1"],
    }
    duplicate = {
        "title": "病毒购物背后的真实性和隐私拉扯",
        "category": "生活消费",
        "signal": "短视频病毒式购物让消费者一边追随真实体验一边担心个人隐私",
        "phenomenon": "购物发现机制从主动搜索迁移到社交内容推荐",
        "why_interesting": "它把消费、身份表达与平台信任连在一起",
        "source_refs": ["https://example.com/viral-shopping"],
    }
    state = merge_work_state(
        {"market_insights": [first]},
        WorkStatePatch.model_validate({"market_insights": [duplicate]}),
    )

    assert len(state["market_insights"]) == 1
    merged = state["market_insights"][0]
    assert "https://example.com/viral-shopping" in merged["source_refs"]
    assert "平台信任" in merged["why_interesting"]


def test_deadline_market_report_keeps_concrete_source_urls():
    checkpoint = {
        "work_state": {
            "market_insights": [
                {
                    "title": "线下手作重新成为社交入口",
                    "signal": "陶艺和编织活动增加",
                    "phenomenon": "年轻人用共同制作建立弱关系",
                    "why_interesting": "爱好开始承担第三空间功能",
                    "source_refs": ["https://example.com/social-crafts"],
                }
            ]
        }
    }
    text = WorkExecutor._deadline_result_text(
        checkpoint,
        {"required_minutes": 60, "has_meaningful_activity": True},
    )
    assert "https://example.com/social-crafts" in text


def test_deadline_market_report_never_dumps_untranslated_source_prose():
    source_url = "https://example.com/global-source"
    checkpoint = {
        "work_state": {"market_insights": []},
        "tree_research_materials": [
            {
                "tool_call_id": 88,
                "title": "A foreign language headline",
                "source_url": source_url,
                "content_excerpt": "This entire English article must not leak into delivery.",
                "candidates": [
                    {
                        "title": "2025年（令和7年）カレンダー",
                        "snippet": "この日本語の検索要約も成果ではありません。",
                        "url": "https://example.com/noisy-result",
                    }
                ],
            }
        ],
    }

    text = WorkExecutor._deadline_result_text(
        checkpoint,
        {"required_minutes": 60, "has_meaningful_activity": True},
    )

    assert source_url in text
    assert "https://example.com/noisy-result" not in text
    assert "不计为正式成果" in text
    assert "This entire English article" not in text
    assert "この日本語" not in text
    assert "A foreign language headline" not in text


def test_controller_hotspot_insights_do_not_require_item_by_item_review(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        session.commit()
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="review-gate")
    thread = runtime.create_thread(
        owner_agent_id=controller_id,
        title="生活、运动、艺术、动漫和小众热点",
        objective="寻找大家正在讨论的具体热点话题",
    )
    categories = ("生活", "运动", "艺术", "动漫", "小众")
    runtime.create_run(
        thread.id,
        checkpoint={
            "work_state": {
                "market_insights": [
                    {
                        "title": f"{category}趋势",
                        "category": category,
                        "signal": f"{category}讨论线索",
                        "phenomenon": f"{category}行为变化",
                        "why_interesting": "具备进一步分析价值",
                    }
                    for category in categories
                ]
            }
        },
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")
    with database.sessions() as session:
        for index, category in enumerate(categories, start=1):
            session.add(
                Evidence(
                    work_thread_id=thread.id,
                    owner_agent_id=controller_id,
                    source_url=f"https://example.com/review/{index}",
                    source_title=f"来源{index}",
                    claim=f"具体事件{index}",
                    excerpt=f"细节{index}",
                    confidence="medium",
                    stance="supports",
                    content_hash=f"{index:064x}",
                    value_score=12,
                    value_analysis_json=json.dumps({"category": category}),
                    review_status="verified" if index < 5 else "unreviewed",
                )
            )
        session.commit()
        status = executor._market_editorial_status(session, thread)
        assert status is not None
        assert "require_controller_review" not in status
        assert status["insight_count"] == 5
def test_delegation_copies_only_explicit_evidence_snapshot(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        worker = AgentProfile(
            name="员工Agent",
            role_responsibilities="核验证据",
            work_content="读取已授权材料",
            personality="严谨",
            preferences="只读",
        )
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="evidence-snapshot")
    parent = runtime.create_thread(
        owner_agent_id=controller_id,
        title="父任务",
        objective="委派核验",
    )
    runtime.create_run(parent.id)
    with database.sessions() as session:
        first = Evidence(
            work_thread_id=parent.id,
            owner_agent_id=controller_id,
            source_url="https://example.com/allowed",
            source_title="允许证据",
            claim="允许员工核验",
            excerpt="允许内容",
            content_hash="a" * 64,
        )
        second = Evidence(
            work_thread_id=parent.id,
            owner_agent_id=controller_id,
            source_url="https://example.com/private",
            source_title="未授权证据",
            claim="不得自动复制",
            excerpt="未授权内容",
            content_hash="b" * 64,
        )
        session.add_all([first, second])
        session.commit()
        allowed_id = first.id
        denied_id = second.id
    collaboration = CollaborationService(database)

    assigned = collaboration.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={
            "title": "精确证据核验",
            "objective": "只核验明确授权的证据",
            "evidence_ids": [allowed_id],
        },
    )

    with database.sessions() as session:
        run = session.scalar(
            select(WorkRun).where(
                WorkRun.work_thread_id == assigned["child_thread_id"]
            )
        )
        assert run is not None
        checkpoint = json.loads(run.checkpoint_json or "{}")
    snapshot = checkpoint["delegation_context"]["evidence_snapshot"]
    assert [item["id"] for item in snapshot] == [allowed_id]
    assert denied_id not in checkpoint["delegation_context"]["evidence_ids"]


def test_cancel_all_preserves_checkpoints_and_closes_open_work(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="cancel-all")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="待停止任务",
        objective="保存现场后停止",
    )
    runtime.create_run(thread.id, checkpoint={"kept": True})
    with database.sessions() as session:
        project = session.get(WorkProject, thread.project_id)
        assert project is not None
        project.status = "blocked_external"
        session.add(
            ResearchCandidate(
                work_thread_id=thread.id,
                owner_agent_id=agent_id,
                canonical_url="https://example.com/unread",
                status="queued",
            )
        )
        session.commit()

    result = runtime.cancel_all("用户要求停止")

    assert result["runs"] == 1
    assert result["threads"] == 1
    stopped = runtime.get_thread(thread.id)
    assert stopped.status == "cancelled"
    with database.sessions() as session:
        run = session.scalar(select(WorkRun).where(WorkRun.work_thread_id == thread.id))
        project = session.get(WorkProject, thread.project_id)
        assert run is not None and json.loads(run.checkpoint_json or "{}")["kept"] is True
        assert run.status == "cancelled"
        assert project is not None and project.status == "cancelled"
        candidate = session.scalar(
            select(ResearchCandidate).where(
                ResearchCandidate.work_thread_id == thread.id
            )
        )
        assert candidate is not None and candidate.status == "deferred"
        assert "任务已取消" in str(candidate.disposition_reason)


def test_completed_run_archives_unread_candidate_queue(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="terminal-candidates")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="候选终态",
        objective="完成时归档未读候选",
    )
    runtime.create_run(thread.id)
    with database.sessions() as session:
        session.add(
            ResearchCandidate(
                work_thread_id=thread.id,
                owner_agent_id=agent_id,
                canonical_url="https://example.com/waiting",
                status="reading",
            )
        )
        session.commit()
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    runtime.complete(claimed.id, claimed.execution_token, "完成")

    with database.sessions() as session:
        candidate = session.scalar(
            select(ResearchCandidate).where(
                ResearchCandidate.work_thread_id == thread.id
            )
        )
        assert candidate is not None and candidate.status == "deferred"
        assert "任务已结束" in str(candidate.disposition_reason)


def test_active_work_can_generate_visible_stage_report(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="stage-report")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="长时研究",
        objective="持续研究并分阶段交付",
    )
    runtime.create_run(thread.id)
    with database.sessions() as session:
        stored = session.get(WorkThread, thread.id)
        assert stored is not None
        stored.summary = "第一阶段已核验两条来源，任务仍在继续。"
        session.commit()

    report, reused = ReportService(
        database, tmp_path / "outputs"
    ).generate(thread.id, allow_incomplete=True)

    assert reused is False
    assert report.title.endswith("阶段成果")
    assert "第一阶段已核验" in report.executive_summary
    assert "不代表任务完成" in report.markdown_content


def test_child_waiting_on_own_collaboration_becomes_progress_handoff(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        worker = AgentProfile(
            name="回传员工",
            role_responsibilities="研究",
            work_content="阶段回传",
            personality="严谨",
            preferences="事件交接",
        )
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="child-handoff")
    parent = runtime.create_thread(
        owner_agent_id=controller_id,
        title="总控任务",
        objective="等待员工阶段成果",
    )
    parent_run = runtime.create_run(parent.id)
    collaboration = CollaborationService(database)
    assigned = collaboration.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={"title": "员工研究", "objective": "形成阶段成果"},
    )
    parent_claim = runtime.claim_next(agent_id=controller_id)
    assert parent_claim is not None and parent_claim.id == parent_run.id
    runtime.checkpoint(
        parent_claim.id,
        parent_claim.execution_token,
        {"waiting_collaboration_id": assigned["id"]},
    )
    runtime.pause(
        parent_claim.id,
        parent_claim.execution_token,
        status="waiting_schedule",
        next_wake_at=datetime.now(timezone.utc) + timedelta(days=1),
        error="等待协作",
    )

    def provider(_, __, ___):
        return work_decision(
            {
                "type": "wait_collaboration",
                "collaboration_id": assigned["id"],
                "message": "阶段证据已经形成，请总控审查。",
            },
            progress=0.5,
        )

    executor = WorkExecutor(
        database,
        runtime,
        ToolRegistry(),
        provider,
        progress_handler=collaboration.report_progress,
    )
    child_claim = runtime.claim_next(agent_id=worker_id)
    assert child_claim is not None

    result = executor.execute(child_claim.id, child_claim.execution_token)

    assert result.outcome == "progress_reported"
    assert result.run.status == "waiting_dependency"
    with database.sessions() as session:
        row = session.get(WorkCollaboration, assigned["id"])
        assert row is not None and "阶段证据已经形成" in (row.result_text or "")
        stored_parent_run = session.get(WorkRun, parent_run.id)
        assert stored_parent_run is not None and stored_parent_run.status == "queued"


def test_reconcile_tree_migrates_legacy_progress_wait_to_dependency(tmp_path: Path):
    database, controller_id = make_database(tmp_path)
    with database.sessions() as session:
        controller = session.get(AgentProfile, controller_id)
        assert controller is not None
        controller.is_controller = True
        worker = AgentProfile(name="旧任务员工", role_responsibilities="研究")
        session.add(worker)
        session.commit()
        worker_id = worker.id
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="legacy-progress")
    parent = runtime.create_thread(
        owner_agent_id=controller_id,
        title="旧总控任务",
        objective="接收旧阶段成果",
    )
    collaboration = CollaborationService(database)
    assigned = collaboration.create_direct(
        parent_thread_id=parent.id,
        requester_agent_id=controller_id,
        target_agent_id=worker_id,
        values={"title": "旧员工任务", "objective": "回传成果"},
    )
    with database.sessions() as session:
        child = session.get(WorkThread, assigned["child_thread_id"])
        run = session.scalar(
            select(WorkRun).where(WorkRun.work_thread_id == child.id)
        )
        assert child is not None and run is not None
        child.status = "waiting_schedule"
        child.next_wake_at = datetime.now(timezone.utc) + timedelta(days=365)
        run.status = "waiting_schedule"
        run.next_wake_at = child.next_wake_at
        run.checkpoint_json = json.dumps(
            {
                "last_action": {"type": "report_progress"},
                "next_wake_at": child.next_wake_at.isoformat(),
                "thread_status": "waiting_schedule",
            }
        )
        session.commit()

    assert collaboration.reconcile_tree(parent.id) == 1

    with database.sessions() as session:
        child = session.get(WorkThread, assigned["child_thread_id"])
        run = session.scalar(
            select(WorkRun).where(WorkRun.work_thread_id == child.id)
        )
        assert child is not None and child.status == "waiting_dependency"
        assert child.next_wake_at is None
        assert run is not None and run.status == "waiting_dependency"
        assert run.next_wake_at is None


def test_complete_with_unmet_success_criteria_keeps_working(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="criteria")
    registry = ToolRegistry()

    def provider(_, __, prompt):
        assert "必须一致" in prompt
        return work_decision(
            {
                "type": "complete",
                "message": "只找到搜索次数，不能证明用户数",
                "criteria_met": False,
                "criteria_evidence": ["未满足：来源只披露每月搜索次数"],
            },
            progress=1,
        )

    executor = WorkExecutor(database, runtime, registry, provider)
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="验收门",
        objective="寻找月度用户数",
        success_criteria=["找到官方月度用户数证据"],
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "yielded"
    assert runtime.get_thread(thread.id).status == "queued"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert "必须调整计划并继续工作" in checkpoint["completion_validation_error"]


def test_executor_reuses_tool_result_after_yield(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="executor")
    registry = ToolRegistry()
    handler_count = 0

    def lookup(_, __):
        nonlocal handler_count
        handler_count += 1
        return {"value": 42}

    registry.register(
        ToolSpec(
            name="lookup",
            description="可安全重试的只读工具",
            input_schema={"type": "object"},
            handler=lookup,
        )
    )

    def provider(_, __, ___):
        return work_decision(
            {
                "type": "tool_call",
                "step_id": "research",
                "tool_name": "lookup",
                "arguments": {"same": True},
            }
        )

    executor = WorkExecutor(database, runtime, registry, provider, max_steps_per_run=1)
    executor.set_tool_permission(agent_id, "lookup", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="幂等测试",
        objective="重复决定不能重复调用工具",
    )
    runtime.create_run(thread.id)

    first = runtime.claim_next(agent_id=agent_id)
    assert first is not None
    assert executor.execute(first.id, first.execution_token).outcome == "yielded"
    second = runtime.claim_next(agent_id=agent_id)
    assert second is not None and second.execution_token == 2
    assert executor.execute(second.id, second.execution_token).outcome == "yielded"

    assert handler_count == 1
    calls = executor.list_tool_calls(thread.id)
    assert len(calls) == 1
    assert calls[0].attempt_count == 1


def test_executor_pauses_for_tool_approval_and_resumes(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="executor")
    registry = ToolRegistry()
    published: list[str] = []

    def publish(_, arguments):
        published.append(arguments["text"])
        return {"published": True}

    registry.register(
        ToolSpec(
            name="publish",
            description="模拟对外发布",
            input_schema={"type": "object", "required": ["text"]},
            handler=publish,
            risk_level="destructive",
            retry_safety="verify",
            requires_approval=False,
        )
    )

    def provider(_, __, ___):
        return work_decision(
            {
                "type": "tool_call",
                "step_id": "research",
                "tool_name": "publish",
                "arguments": {"text": "待发布内容"},
            }
        )

    executor = WorkExecutor(database, runtime, registry, provider, max_steps_per_run=1)
    executor.set_tool_permission(agent_id, "publish", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="审批测试",
        objective="高风险工具必须批准",
    )
    runtime.create_run(thread.id)
    first = runtime.claim_next(agent_id=agent_id)
    assert first is not None
    waiting = executor.execute(first.id, first.execution_token)
    assert waiting.outcome == "waiting_approval"
    assert waiting.run.status == "waiting_user"
    assert published == []

    call = executor.list_tool_calls(thread.id)[0]
    assert call.status == "waiting_approval"
    executor.resolve_approval(call.id, approved=True, reason="允许本次执行")
    resumed = runtime.claim_next(agent_id=agent_id)
    assert resumed is not None and resumed.execution_token == 2
    assert executor.execute(resumed.id, resumed.execution_token).outcome == "yielded"
    assert published == ["待发布内容"]
    assert executor.list_tool_calls(thread.id)[0].status == "completed"


def test_executor_returns_denied_tool_result_to_agent_for_replanning(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="executor")
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="restricted",
            description="未授权工具",
            input_schema={"type": "object"},
            handler=lambda *_: {"should_not_run": True},
        )
    )
    decisions = 0

    def provider(_, __, prompt):
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "restricted-step",
                    "tool_name": "restricted",
                    "arguments": {},
                }
            )
        assert "Agent没有该工具权限" in prompt
        return work_decision({"type": "complete", "message": "已改用安全方案"})

    executor = WorkExecutor(database, runtime, registry, provider, max_steps_per_run=3)
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="权限拒绝后重规划",
        objective="工具不可用时自行调整",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert result.steps_executed == 1
    calls = executor.list_tool_calls(thread.id)
    assert calls == []
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["rejected_unknown_tool_decisions_total"] == 1


def test_executor_replans_unknown_tool_without_pausing_run(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="unknown-tool")
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="read_work_history",
            description="读取原始工作账本",
            input_schema={"type": "object"},
            handler=lambda *_: {"items": []},
        )
    )
    registry.register(
        ToolSpec(
            name="describe_tools",
            description="查询工具定义",
            input_schema={"type": "object"},
            handler=lambda *_: {"tools": []},
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "recover-ledger",
                    "tool_name": "read_tool_ledger_result",
                    "arguments": {"tool_call_id": "tool_call:700"},
                }
            )
        assert "尚未被接受" in prompt
        assert "工具不属于当前角色和工作状态可用范围：read_tool_ledger_result" in prompt
        assert "tool_name只能从这个枚举中选择" in prompt
        assert "read_work_history" in prompt
        return work_decision(
            {"type": "complete", "message": "已从真实工具目录重新规划"},
            progress=1,
        )

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=3
    )
    executor.set_tool_permission(agent_id, "read_work_history", "allow")
    executor.set_tool_permission(agent_id, "describe_tools", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="不存在工具后恢复",
        objective="继续恢复工作时，工具选择错误仍在同一任务内修复",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert result.run.status == "completed"
    assert result.steps_executed == 1
    assert executor.list_tool_calls(thread.id) == []
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["invalid_tool_selection_count"] == 0
    assert checkpoint["rejected_unknown_tool_decisions_total"] == 1
    assert checkpoint["decision_tool_repairs"] == 1
    assert "invalid_tool_selections_total" not in checkpoint
    assert "last_tool_selection_error" not in checkpoint
    assert checkpoint["last_rejected_tool_decision"]["unknown_tools"] == [
        "read_tool_ledger_result"
    ]
    assert "tool_call:700" in checkpoint["last_rejected_tool_decision"][
        "response_preview"
    ]


def test_unknown_tool_rejects_entire_batch_before_any_tool_runs(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="unknown-batch")
    registry = ToolRegistry()
    handler_calls = 0

    def lookup(_, __):
        nonlocal handler_calls
        handler_calls += 1
        return {"value": 42}

    registry.register(
        ToolSpec(
            name="lookup",
            description="已注册查询工具",
            input_schema={"type": "object"},
            handler=lookup,
        )
    )
    provider_calls = 0

    def provider(_, __, prompt):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return json.dumps(
                {
                    "state_summary": "准备批量查询",
                    "actions": [
                        {
                            "type": "tool_call",
                            "step_id": "valid-first",
                            "tool_name": "lookup",
                            "arguments": {},
                        },
                        {
                            "type": "tool_call",
                            "step_id": "invented-second",
                            "tool_name": "invented_lookup",
                            "arguments": {},
                        },
                    ],
                },
                ensure_ascii=False,
            )
        assert "invented_lookup" in prompt
        return work_decision(
            {"type": "complete", "message": "无等价能力，已调整路线"},
            progress=1,
        )

    executor = WorkExecutor(
        database, runtime, registry, provider, max_steps_per_run=3
    )
    executor.set_tool_permission(agent_id, "lookup", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="整批动作前置校验",
        objective="包含未知工具的批次不得部分执行",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    assert claimed is not None

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert result.steps_executed == 1
    assert handler_calls == 0
    assert executor.list_tool_calls(thread.id) == []


def test_executor_api_uses_agent_model_and_registered_tool(tmp_path: Path):
    class DecisionCodex(NoopCodex):
        def __init__(self):
            self.work_calls = 0
            self.session_keys = []

        def inspect_account(self):
            return {
                "account": {"type": "chatgpt", "planType": "pro"},
                "providerCapabilities": {"imageGeneration": False},
                "models": [],
            }

        def run_text_turn(self, prompt, model=None, session_key=None):
            assert "持久工作线程" in prompt
            assert session_key and ":work:" in session_key
            self.session_keys.append(session_key)
            self.work_calls += 1
            if self.work_calls == 1:
                return work_decision(
                    {
                        "type": "tool_call",
                        "step_id": "research",
                        "tool_name": "current_time",
                        "arguments": {},
                    }
                )
            return work_decision(
                {"type": "complete", "message": "已读取工具并完成"}, progress=1
            )

    database = Database(f"sqlite:///{tmp_path / 'executor-api.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "conversation:\n  model_retry_pause_seconds: 0\n"
        "work_runtime:\n  lease_seconds: 60\n  max_steps_per_run: 4\n",
        encoding="utf-8",
    )
    codex = DecisionCodex()
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=codex,
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        chat_model = next(
            item for item in connected["models"] if item["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "自主工作模型",
                "plan_type": "chat",
                "model_ids": [chat_model["id"]],
            },
        ).json()
        controller = client.get("/api/agents").json()["controller"]
        client.put(
            f"/api/agents/{controller['id']}",
            json={
                "name": "总控Agent",
                "role_responsibilities": "执行工作",
                "work_content": "自主选择工具",
                "personality": "严谨",
                "preferences": "保存证据",
                "chat_plan_id": plan["id"],
                "image_plan_id": None,
                "ai_organized": True,
            },
        )
        permission = client.put(
            f"/api/agents/{controller['id']}/tools/current_time",
            json={"permission": "allow"},
        )
        assert permission.status_code == 200
        thread = client.post(
            "/api/work/threads",
            json={
                "owner_agent_id": controller["id"],
                "title": "API自主执行",
                "objective": "调用时间工具后完成",
            },
        ).json()
        client.post(
            f"/api/work/threads/{thread['id']}/runs",
            json={"trigger_kind": "user"},
        )
        executed = client.post(
            "/api/work/executor/run-next",
            params={"agent_id": controller["id"]},
        )
        assert executed.status_code == 200
        assert executed.json()["status"] == "completed"
        assert executed.json()["steps_executed"] == 2
        calls = client.get(
            f"/api/work/threads/{thread['id']}/tool-calls"
        ).json()
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "current_time"
        assert calls[0]["status"] == "completed"
        model_calls = client.get(
            f"/api/work/threads/{thread['id']}/model-calls"
        ).json()
        assert len(model_calls) == 2
        assert all(item["usage_source"] == "estimated" for item in model_calls)
        detail = client.get(f"/api/work/threads/{thread['id']}").json()
        assert detail["work_budget"]["model_calls"] == 2
        assert detail["work_budget"]["model_tokens"] > 0
        assert len(codex.session_keys) == 2
        assert all(":packet:" in item for item in codex.session_keys)
        assert codex.session_keys[0] != codex.session_keys[1]
        dashboard_item = next(
            item for item in client.get("/api/work/dashboard").json()["items"]
            if item["id"] == thread["id"]
        )
        assert dashboard_item["work_budget"]["model_calls"] == 2
        evidence = client.get(
            f"/api/work/threads/{thread['id']}/evidence"
        )
        assert evidence.status_code == 200
        assert evidence.json() == []
        registered = {item["name"] for item in client.get("/api/tools").json()}
        assert {
            "web_search",
            "discover_site_pages",
            "read_web_page",
            "save_evidence",
            "list_evidence",
        } <= registered


def test_background_scheduler_completes_work_without_manual_execute(tmp_path: Path):
    class BackgroundCodex(NoopCodex):
        def inspect_account(self):
            return {
                "account": {"type": "chatgpt", "planType": "pro"},
                "providerCapabilities": {"imageGeneration": False},
                "models": [],
            }

        def run_text_turn(self, prompt, model=None, session_key=None):
            return work_decision(
                {"type": "complete", "message": "后台自动完成"}, progress=1
            )

    database = Database(f"sqlite:///{tmp_path / 'background.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "conversation:\n  model_retry_pause_seconds: 0\n"
        "work_runtime:\n"
        "  lease_seconds: 30\n"
        "  max_steps_per_run: 3\n"
        "  auto_execute_enabled: true\n"
        "  queue_scan_seconds: 1\n",
        encoding="utf-8",
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=BackgroundCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        model = next(item for item in connected["models"] if item["model_type"] == "chat")
        plan = client.post(
            "/api/plans",
            json={"name": "后台执行", "plan_type": "chat", "model_ids": [model["id"]]},
        ).json()
        controller = client.get("/api/agents").json()["controller"]
        client.put(
            f"/api/agents/{controller['id']}",
            json={
                "name": "总控Agent",
                "role_responsibilities": "后台工作",
                "work_content": "自动领取任务",
                "personality": "可靠",
                "preferences": "保存进度",
                "chat_plan_id": plan["id"],
                "image_plan_id": None,
                "ai_organized": True,
            },
        )
        thread = client.post(
            "/api/work/threads",
            json={
                "owner_agent_id": controller["id"],
                "title": "后台任务",
                "objective": "无需手动执行API即可完成",
            },
        ).json()
        client.post(
            f"/api/work/threads/{thread['id']}/runs",
            json={"trigger_kind": "background_test"},
        )

        deadline = time.monotonic() + 8
        item = None
        while time.monotonic() < deadline:
            dashboard = client.get("/api/work/dashboard").json()
            item = next(
                value for value in dashboard["items"] if value["id"] == thread["id"]
            )
            if item["status"] == "completed":
                break
            time.sleep(0.1)

        assert item is not None
        assert item["status"] == "completed"
        assert item["run"]["status"] == "completed"
        assert item["run"]["result_text"] == "后台自动完成"
        assert item["progress"] == 1
        assert item["tool_call_count"] == 0
        assert client.get("/api/settings/work-scheduler").json() == {
            "auto_execute_enabled": True,
            "queue_scan_seconds": 1,
            "max_concurrent_agents": 4,
        }


def test_background_scheduler_runs_different_agents_concurrently(tmp_path: Path):
    class ConcurrentCodex(NoopCodex):
        def __init__(self):
            self.lock = threading.Lock()
            self.release = threading.Event()
            self.active = 0
            self.peak = 0

        def inspect_account(self):
            return {
                "account": {"type": "chatgpt", "planType": "pro"},
                "providerCapabilities": {"imageGeneration": False},
                "models": [],
            }

        def run_text_turn(self, prompt, model=None, session_key=None):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active >= 2:
                    self.release.set()
            try:
                assert self.release.wait(timeout=5), "第二个Agent没有并发进入模型调用"
                time.sleep(0.05)
                return work_decision(
                    {"type": "complete", "message": "并发任务完成"}, progress=1
                )
            finally:
                with self.lock:
                    self.active -= 1

    codex = ConcurrentCodex()
    database = Database(f"sqlite:///{tmp_path / 'concurrent-background.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "conversation:\n  model_retry_pause_seconds: 0\n"
        "work_runtime:\n"
        "  lease_seconds: 30\n"
        "  max_steps_per_run: 2\n"
        "  auto_execute_enabled: true\n"
        "  queue_scan_seconds: 1\n"
        "  max_concurrent_agents: 2\n",
        encoding="utf-8",
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=codex,
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        model = next(
            item for item in connected["models"] if item["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={"name": "并发执行", "plan_type": "chat", "model_ids": [model["id"]]},
        ).json()
        base_agent = {
            "role_responsibilities": "独立执行工作",
            "work_content": "完成自己的任务",
            "personality": "可靠",
            "preferences": "及时交付",
            "chat_plan_id": plan["id"],
            "image_plan_id": None,
            "ai_organized": True,
        }
        controller = client.get("/api/agents").json()["controller"]
        client.put(
            f"/api/agents/{controller['id']}",
            json={**base_agent, "name": "总控Agent"},
        )
        worker = client.post(
            "/api/agents", json={**base_agent, "name": "市场Agent"}
        ).json()

        thread_ids = []
        for agent_id, title in (
            (controller["id"], "总控并发任务"),
            (worker["id"], "市场并发任务"),
        ):
            thread = client.post(
                "/api/work/threads",
                json={
                    "owner_agent_id": agent_id,
                    "title": title,
                    "objective": "验证不同Agent能够同时执行",
                },
            ).json()
            thread_ids.append(thread["id"])
            client.post(
                f"/api/work/threads/{thread['id']}/runs",
                json={"trigger_kind": "concurrency_test"},
            )

        deadline = time.monotonic() + 10
        statuses = {}
        while time.monotonic() < deadline:
            dashboard = client.get("/api/work/dashboard").json()
            statuses = {
                item["id"]: item["status"]
                for item in dashboard["items"]
                if item["id"] in thread_ids
            }
            if statuses and all(value == "completed" for value in statuses.values()):
                break
            time.sleep(0.1)

        assert all(statuses.get(thread_id) == "completed" for thread_id in thread_ids)
        assert codex.peak == 2


def test_discovery_candidate_contract_accepts_public_trend_candidates():
    result = {
        "candidates": [
            {
                "title": "Public discussion",
                "url": "https://example.com/topic",
                "snippet": "people are discussing it",
            }
        ]
    }
    assert normalized_discovery_candidates(result)[0]["title"] == "Public discussion"
    queue = discovery_source_candidates(result, 17)
    assert queue == [
        {
            "url": "https://example.com/topic",
            "canonical_url": "https://example.com/topic",
            "title": "Public discussion",
            "source_search_tool_call_id": 17,
        }
    ]


def legacy_public_trends_candidates_must_be_read_before_insight(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="public-trends-full-read")
    registry = ToolRegistry()
    url = "https://example.com/public-topic"
    opened: list[str] = []
    registry.register(
        ToolSpec(
            name="collect_public_trends",
            description="无Key公开热点",
            input_schema={"type": "object"},
            handler=lambda *_: {
                "query": "recent public topics",
                "candidates": [{"title": "Topic title", "url": url, "snippet": "short only"}],
            },
        )
    )

    def read_page(_, arguments):
        opened.append(arguments["url"])
        return {
            "url": url,
            "title": "Topic title",
            "text": "FULL_PUBLIC_TOPIC_BODY 普通人正在参与并讨论这个具体现象。" * 50,
        }

    registry.register(
        ToolSpec(
            name="read_web_page",
            description="阅读全文",
            input_schema={"type": "object"},
            handler=read_page,
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "public-trends",
                    "tool_name": "collect_public_trends",
                    "arguments": {"query": "recent public topics"},
                }
            )
        if len(prompts) == 2:
            assert "short only" in prompt
            assert "FULL_PUBLIC_TOPIC_BODY" not in prompt
            return json.dumps(
                {
                    "state_update": {
                        "current_strategy": "选择公共话题代表页面阅读全文",
                        "market_candidates": [
                            {
                                "candidate_ref": url,
                                "title": "Topic title",
                                "lens": "公共讨论",
                                "decision": "deep_read",
                                "reason": "摘要显示存在普通人参与信号，需正文核验",
                                "source_url": url,
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "发现一个值得阅读全文的公共讨论候选",
                    },
                    "consumed_tool_call_ids": [1],
                    "action": {
                        "type": "tool_call",
                        "step_id": "read-public-topic",
                        "tool_name": "read_web_page",
                        "arguments": {"url": url},
                        "purpose": "读取代表页面核验普通人参与",
                        "criterion_id": "article-understanding",
                        "expected_delta": "获得完整正文",
                        "stop_condition": "能保留或淘汰该候选",
                    },
                },
                ensure_ascii=False,
            )
        if len(prompts) == 3:
            assert "FULL_PUBLIC_TOPIC_BODY" in prompt
            assert "short only" not in prompt
            return json.dumps(
                {
                    "state_update": {
                        "market_insights": [
                            {
                                "title": "普通人参与的新现象",
                                "signal": "出现具体参与和讨论",
                                "phenomenon": "公共话题进入日常行为",
                                "why_interesting": "具备继续扩散的可能",
                                "source_refs": [url, "tool_call:2"],
                            }
                        ]
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "正文确认普通人参与和讨论",
                    },
                    "consumed_tool_call_ids": [2],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        return work_decision({"type": "complete", "message": "已阅读全文并完成提炼"})

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=6,
        default_tool_permissions={
            "web_search": "allow",
            "collect_public_trends": "allow",
            "read_web_page": "allow",
        },
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="寻找最近热点",
        objective="找到值得继续分析的公共讨论",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)
    assert result.outcome == "completed"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert opened == [url], json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)
    assert len(prompts) == 4


def test_market_insights_are_durable_beyond_checkpoint_limit(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="durable-insights")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="长期热点研究",
        objective="寻找普通人参与的文化趋势",
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "{}")
    insights = [
        MarketInsight(
            title=f"文化趋势 {index}",
            category="娱乐文化",
            signal=f"第 {index} 个具体现象",
            phenomenon="普通人正在形成新的参与行为",
            why_interesting="具有跨圈层传播潜力",
            source_refs=[f"https://example.com/{index}"],
        )
        for index in range(25)
    ]
    stored_ids = executor._persist_market_insights(thread, insights)
    assert len(stored_ids) == 25
    with database.sessions() as session:
        assert session.query(MarketInsightRecord).count() == 25
    related = executor._related_market_insights(thread, "文化趋势 普通人", limit=30)
    assert len(related) == 25


def test_market_discovery_batch_is_serialized_without_rejection(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="discovery-serialize")
    registry = ToolRegistry()
    calls: list[str] = []

    registry.register(
        ToolSpec(
            name="web_search",
            description="搜索公开网页",
            input_schema={"type": "object"},
            handler=lambda _, arguments: calls.append(str(arguments["query"]))
            or {"query": arguments["query"], "results": []},
        )
    )
    decision = json.dumps(
        {
            "actions": [
                {
                    "type": "tool_call",
                    "step_id": "first-search",
                    "tool_name": "web_search",
                    "arguments": {"query": "生活热点"},
                },
                {
                    "type": "tool_call",
                    "step_id": "stale-second-search",
                    "tool_name": "web_search",
                    "arguments": {"query": "娱乐热点"},
                },
            ]
        },
        ensure_ascii=False,
    )
    executor = WorkExecutor(
        database, runtime, registry, lambda *_: decision, max_steps_per_run=1
    )
    executor.set_tool_permission(agent_id, "web_search", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="串行搜索",
        objective="逐次判断生活与娱乐热点",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)

    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "yielded"
    assert calls == ["生活热点"]
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["serialized_discovery_batches"] == 1
    assert "rejected_unsafe_tool_batches" not in checkpoint


def test_unsafe_write_batch_is_rejected_before_execution(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="batch-safety")
    registry = ToolRegistry()
    calls: list[int] = []

    def write_tool(_, arguments):
        calls.append(int(arguments["value"]))
        return {"saved": True}

    registry.register(
        ToolSpec(
            name="write_tool",
            description="修改状态",
            input_schema={"type": "object"},
            handler=write_tool,
            risk_level="write",
        )
    )
    decisions = [
        json.dumps(
            {
                "actions": [
                    {"type": "tool_call", "step_id": "a", "tool_name": "write_tool", "arguments": {"value": 1}},
                    {"type": "tool_call", "step_id": "b", "tool_name": "write_tool", "arguments": {"value": 2}},
                ]
            }
        ),
        work_decision({"type": "complete", "message": "已改为顺序规划"}),
    ]
    executor = WorkExecutor(
        database, runtime, registry, lambda *_: decisions.pop(0), max_steps_per_run=3
    )
    executor.set_tool_permission(agent_id, "write_tool", "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="批量安全", objective="验证写入不能批量"
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)
    assert result.outcome == "completed"
    assert calls == []
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint["rejected_unsafe_tool_batches"] == 1


def test_normal_completion_does_not_create_quality_review(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="quality-no-noise")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="正常工作", objective="交付一份正常结果"
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    runtime.complete(claimed.id, claimed.execution_token, "已有完整成果")
    governance = WorkGovernanceService(database, runtime, {"work_governance": {}})
    result = governance.scan()
    assert result["detected"]["quality_review"] == 0
    with database.sessions() as session:
        assert session.query(WorkGovernanceIssue).count() == 0


def test_work_cleanup_api_clears_governance_records(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'cleanup.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n",
        encoding="utf-8",
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        controller_id = client.get("/api/agents").json()["controller"]["id"]
        with database.sessions() as session:
            thread = WorkThread(
                owner_agent_id=controller_id,
                title="历史任务",
                objective="用于治理清理测试",
                status="completed",
            )
            session.add(thread)
            session.flush()
            session.add(
                WorkGovernanceIssue(
                    issue_key="legacy-noise",
                    issue_type="quality_review",
                    status="open",
                    severity="info",
                    work_thread_id=thread.id,
                    summary="旧的自动复核噪音",
                    details_json="{}",
                    detected_by="test",
                )
            )
            session.commit()
        response = client.post(
            "/api/work/cleanup",
            json={"scope": "governance", "confirmation": "清理治理记录"},
        )
        assert response.status_code == 200
        assert response.json()["deleted"]["governance_issues"] == 1
        scan_response = client.post(
            f"/api/work/governance/scan?reviewer_agent_id={controller_id}"
        )
        assert scan_response.status_code == 200
        assert scan_response.json()["open"] == 0
        history_response = client.post(
            "/api/work/cleanup",
            json={"scope": "history", "confirmation": "清空工作历史"},
        )
        assert history_response.status_code == 200
        assert history_response.json()["deleted"]["threads"] == 1
        assert client.get("/api/work/threads").json() == []


def test_discovery_queue_skips_title_only_page_and_judges_next_full_body(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="durable-queue")
    registry = ToolRegistry()
    first = "https://example.com/title-only"
    second = "https://example.com/full-story"
    opened: list[str] = []
    registry.register(
        ToolSpec(
            "web_search",
            "搜索候选",
            {"type": "object"},
            lambda *_: {
                "results": [
                    {"title": "只有标题", "url": first, "snippet": "摘要不能判断"},
                    {"title": "真实活动", "url": second, "snippet": "仍然只是摘要"},
                ]
            },
        )
    )

    def read_page(_, arguments):
        url = arguments["url"]
        opened.append(url)
        if url == first:
            return {"url": url, "title": "只有标题", "text": "只有标题"}
        return {
            "url": url,
            "title": "真实活动",
            "text": "FULL_SECOND_BODY 普通人参加临时线下活动并分享现场体验。" * 40,
        }

    registry.register(
        ToolSpec("read_web_page", "阅读全文", {"type": "object"}, read_page)
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "discover",
                    "tool_name": "web_search",
                    "arguments": {"query": "普通人有趣活动"},
                }
            )
        if "FULL_SECOND_BODY" in prompt:
            with database.sessions() as session:
                material_id = session.scalar(
                    select(ToolCall.id).order_by(ToolCall.id.desc())
                )
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "正文显示普通人正在参与并传播临时线下活动",
                        "market_insights": [
                            {
                                "title": "临时线下活动成为真实社交入口",
                                "signal": "参与者主动到场并分享体验",
                                "phenomenon": "低包装活动正在吸引真实参与",
                                "why_interesting": "它体现了普通人对真实现场的偏好",
                                "source_refs": [second, f"tool_call:{material_id}"],
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "全文提供了具体参与行为",
                    },
                    "consumed_tool_call_ids": [material_id],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        return work_decision(
            {"type": "complete", "message": "已完成逐篇读取和提炼"}, progress=1
        )

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=8,
        default_tool_permissions={"web_search": "allow", "read_web_page": "allow"},
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="市场热点研究",
        objective="搜索普通人参与的有趣事件并阅读全文后提炼",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert opened == [first, second]
    assert all("摘要不能判断" not in prompt for prompt in prompts[1:])
    with database.sessions() as session:
        rows = session.scalars(
            select(ResearchCandidate)
            .where(ResearchCandidate.work_thread_id == thread.id)
            .order_by(ResearchCandidate.id)
        ).all()
        assert [row.status for row in rows] == ["content_insufficient", "insight_saved"]
        assert rows[0].content_chars == len("只有标题")
        assert rows[1].content_chars > 300


def test_public_social_post_body_is_used_when_html_page_is_only_a_title(
    tmp_path: Path,
):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="social-body")
    registry = ToolRegistry()
    url = "https://mastodon.social/@person/123"
    full_post = "SOCIAL_POST_BODY 我第一次参加社区交换晚餐，陌生人各自带一道家乡菜，现场又真实又混乱。" * 8
    registry.register(
        ToolSpec(
            "collect_public_trends",
            "公开社交热点",
            {"type": "object"},
            lambda *_: {
                "candidates": [
                    {
                        "platform": "mastodon",
                        "rank": 1,
                        "title": full_post[:300],
                        "snippet": full_post,
                        "url": url,
                    }
                ]
            },
        )
    )
    registry.register(
        ToolSpec(
            "read_web_page",
            "阅读全文",
            {"type": "object"},
            lambda *_: {"url": url, "title": "Mastodon", "text": "Mastodon"},
        )
    )
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "social-discovery",
                    "tool_name": "collect_public_trends",
                    "arguments": {
                        "sources": ["mastodon"],
                        "mastodon_kind": "statuses",
                    },
                }
            )
        if "SOCIAL_POST_BODY" in prompt and len(prompts) == 2:
            material_id = 2
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "原帖描述了陌生人通过家乡菜建立真实社交",
                        "market_insights": [
                            {
                                "title": "社区交换晚餐成为低门槛陌生人社交",
                                "signal": "参与者各自带一道菜并现场交流",
                                "phenomenon": "真实且不精心设计的聚会形成参与吸引力",
                                "why_interesting": "普通人可以低成本复制这种社交形式",
                                "source_refs": [url, f"tool_call:{material_id}"],
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "完整公开帖子提供了具体参与场景",
                    },
                    "consumed_tool_call_ids": [material_id],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        return work_decision({"type": "complete", "message": "已提炼原帖"}, progress=1)

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=6,
        default_tool_permissions={
            "web_search": "allow",
            "collect_public_trends": "allow",
            "read_web_page": "allow",
        },
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="社交热点",
        objective="寻找普通人参与的社交热点并提炼",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    with database.sessions() as session:
        row = session.scalar(
            select(ResearchCandidate).where(
                ResearchCandidate.work_thread_id == thread.id
            )
        )
        assert row.status == "insight_saved"
        assert row.content_kind == "public_social_post"
        assert row.content_chars == len(full_post)


def test_access_interstitials_trigger_same_domain_circuit_breaker(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="access-circuit")
    registry = ToolRegistry()
    blocked_urls = [
        "https://blocked.example/story-one",
        "https://blocked.example/story-two",
        "https://blocked.example/story-three",
    ]
    useful_url = "https://useful.example/community-event"
    opened: list[str] = []
    registry.register(
        ToolSpec(
            "web_search",
            "搜索候选",
            {"type": "object"},
            lambda *_: {
                "results": [
                    {"title": f"候选{index}", "url": url, "snippet": "仅用于召回"}
                    for index, url in enumerate([*blocked_urls, useful_url], start=1)
                ]
            },
        )
    )

    def read_page(_, arguments):
        url = arguments["url"]
        opened.append(url)
        if url in blocked_urls:
            return {
                "url": url,
                "title": "Just a moment...",
                "text": "Just a moment... Performing security verification. Verify you are human before proceeding.",
            }
        return {
            "url": url,
            "title": "社区晚餐",
            "text": "USEFUL_COMMUNITY_BODY 普通人带着家乡菜参加临时社区晚餐，并公开分享现场反应。" * 40,
        }

    registry.register(ToolSpec("read_web_page", "阅读全文", {"type": "object"}, read_page))
    prompts: list[str] = []

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "discover",
                    "tool_name": "web_search",
                    "arguments": {"query": "近期社区活动", "research_role": "live_event"},
                }
            )
        if "USEFUL_COMMUNITY_BODY" in prompt:
            with database.sessions() as session:
                material_id = session.scalar(
                    select(ToolCall.id).order_by(ToolCall.id.desc())
                )
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "正文确认普通人参加并分享社区晚餐",
                        "market_insights": [
                            {
                                "title": "社区晚餐成为低门槛真实社交",
                                "signal": "参与者带菜到场并分享体验",
                                "concrete_event": "普通人参加临时社区晚餐",
                                "phenomenon": "低包装活动吸引真实参与",
                                "why_interesting": "它把日常饮食变成陌生人社交入口",
                                "observed_discussion_signals": ["参与者公开分享现场反应"],
                                "inferred_questions": ["这种活动是否会进入更多城市"],
                                "source_refs": [useful_url, f"tool_call:{material_id}"],
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "取得具体参与行为",
                    },
                    "consumed_tool_call_ids": [material_id],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        return work_decision({"type": "complete", "message": "已完成"}, progress=1)

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=10,
        default_tool_permissions={"web_search": "allow", "read_web_page": "allow"},
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="市场热点研究",
        objective="寻找普通人参与的近期有趣事件",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert opened == [blocked_urls[0], useful_url, blocked_urls[1]]
    with database.sessions() as session:
        rows = session.scalars(
            select(ResearchCandidate)
            .where(ResearchCandidate.work_thread_id == thread.id)
            .order_by(ResearchCandidate.id)
        ).all()
        assert [row.status for row in rows] == [
                "content_insufficient",
                "insight_saved",
                "content_insufficient",
                "content_insufficient",
            ]
        assert rows[0].content_kind == "access_interstitial"
        assert rows[2].content_kind == "access_interstitial"
        assert rows[3].content_kind == "access_interstitial_skipped"


def test_discovery_queue_ignores_non_object_candidates(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="candidate-shape")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="候选结构兼容",
        objective="寻找公开事件",
    )
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "")

    result = executor._register_candidate_queue(
        thread,
        99,
        "web_search",
        {"query": "近期公开事件"},
        {
            "results": [
                "不可作为候选对象的字符串",
                {
                    "title": "可读取事件",
                    "url": "https://events.example/recent-event",
                    "snippet": "普通人参加近期公开活动。",
                },
            ]
        },
    )

    assert result["registered"] == 1
    assert result["queued"] == 1


def test_market_event_queue_filters_reference_pages_and_recent_cross_task_urls(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="market-source-quality")
    previous = runtime.create_thread(
        owner_agent_id=agent_id,
        title="上一轮市场热点",
        objective="寻找普通生活事件",
    )
    current = runtime.create_thread(
        owner_agent_id=agent_id,
        title="无人回归A",
        objective="验证全链路稳定性",
    )
    repeated_url = "https://local.example/events/repair-cafe"
    with database.sessions() as session:
        session.add(
            ResearchCandidate(
                work_thread_id=previous.id,
                owner_agent_id=agent_id,
                canonical_url=repeated_url,
                title="上轮已处理活动",
                status="insight_saved",
            )
        )
        # Match the real tool order: ResearchService has already persisted all
        # discovery hits in the current thread before WorkExecutor filters the
        # editorial read queue.
        session.add_all(
            [
                ResearchCandidate(
                    work_thread_id=current.id,
                    owner_agent_id=agent_id,
                    canonical_url=repeated_url,
                    title="预存的跨任务重复活动",
                    status="queued",
                ),
                ResearchCandidate(
                    work_thread_id=current.id,
                    owner_agent_id=agent_id,
                    canonical_url="https://en.wikipedia.org/wiki/Community",
                    title="预存百科页",
                    status="deferred",
                ),
            ]
        )
        session.commit()
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "")

    result = executor._register_candidate_queue(
        current,
        101,
        "web_search",
        {"query": "community event recap", "research_role": "general_discovery"},
        {
            "results": [
                {"title": "Community", "url": "https://en.wikipedia.org/wiki/Community"},
                {"title": "Meaning of community", "url": "https://dictionary.cambridge.org/dictionary/community"},
                {"title": "Repeated activity", "url": repeated_url},
                {"title": "Residents repair household items together", "url": "https://town.example/news/repair-day"},
            ]
        },
    )

    assert result["registered"] == 1
    with database.sessions() as session:
        rows = session.scalars(
            select(ResearchCandidate).where(ResearchCandidate.work_thread_id == current.id)
        ).all()
    queued = [row.canonical_url for row in rows if row.status == "queued"]
    assert queued == ["https://town.example/news/repair-day"]
    assert all(
        row.status == "rejected"
        for row in rows
        if row.canonical_url in {repeated_url, "https://en.wikipedia.org/wiki/Community"}
    )


def test_discovery_queue_marks_unselected_persisted_candidates_deferred(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, instance_id="candidate-deferred")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="代表候选选择",
        objective="每批只分析有限代表候选",
    )
    results = [
        {
            "title": f"同域候选{index}",
            "url": f"https://social.example/posts/{index}",
            "snippet": "公开讨论",
        }
        for index in range(8)
    ]
    with database.sessions() as session:
        for item in results:
            session.add(
                ResearchCandidate(
                    work_thread_id=thread.id,
                    owner_agent_id=agent_id,
                    canonical_url=item["url"],
                    title=item["title"],
                    status="queued",
                    sources_json='["public_trends"]',
                )
            )
        session.commit()
    executor = WorkExecutor(database, runtime, ToolRegistry(), lambda *_: "")

    result = executor._register_candidate_queue(
        thread,
        100,
        "collect_public_trends",
        {"query": "公开讨论"},
        {"results": results},
    )

    assert result["registered"] == 3
    with database.sessions() as session:
        rows = session.scalars(
            select(ResearchCandidate)
            .where(ResearchCandidate.work_thread_id == thread.id)
            .order_by(ResearchCandidate.id)
        ).all()
    assert [row.status for row in rows].count("queued") == 3
    assert [row.status for row in rows].count("deferred") == 5
    assert executor._runtime_criteria_facts(thread, {})["candidate_count"] == 8


def test_saved_market_lead_is_followed_up_before_old_broad_candidates(tmp_path: Path):
    database, agent_id = make_database(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="market-followup")
    registry = ToolRegistry()
    lead_url = "https://events.example/reset-to-real"
    broad_url = "https://reports.example/annual-trends"
    followup_url = "https://community.example/reset-to-real-reactions"
    opened: list[str] = []
    searches: list[dict[str, Any]] = []

    def search(_, arguments):
        searches.append(dict(arguments))
        if arguments.get("research_role") == "follow_up":
            return {
                "results": [
                    {
                        "title": "参与者公开讨论临时真实活动",
                        "url": followup_url,
                        "snippet": "候选摘要",
                    }
                ]
            }
        return {
            "results": [
                {"title": "Reset to Real活动", "url": lead_url, "snippet": "候选摘要"},
                {"title": "年度趋势报告", "url": broad_url, "snippet": "候选摘要"},
            ]
        }

    registry.register(ToolSpec("web_search", "搜索", {"type": "object"}, search))

    def read_page(_, arguments):
        url = arguments["url"]
        opened.append(url)
        if url == lead_url:
            body = "LEAD_BODY 年轻人参加临时、未经精心包装的线下活动，并把真实混乱视为体验的一部分。" * 40
        elif url == followup_url:
            body = "FOLLOWUP_BODY 参与者公开留言称他们更喜欢不可预测、可以共同完成的真实现场。" * 40
        else:
            body = "BROAD_BODY 这是一份宽泛年度营销趋势报告，没有具体参与事件。" * 40
        return {"url": url, "title": "正文", "text": body}

    registry.register(ToolSpec("read_web_page", "阅读全文", {"type": "object"}, read_page))
    prompts: list[str] = []

    def current_material_id() -> int:
        with database.sessions() as session:
            return int(session.scalar(select(ToolCall.id).order_by(ToolCall.id.desc())))

    def provider(_, __, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "discover-live-event",
                    "tool_name": "web_search",
                    "arguments": {"query": "recent unusual offline social events", "research_role": "live_event"},
                }
            )
        if "LEAD_BODY" in prompt:
            material_id = current_material_id()
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "发现真实、临时活动偏好",
                        "market_insights": [
                            {
                                "title": "真实混乱重新成为线下活动吸引力",
                                "signal": "年轻人参加未经精心包装的临时活动",
                                "concrete_event": "参与者把不可预测现场视为体验的一部分",
                                "phenomenon": "对完美展示的疲劳转化为真实现场偏好",
                                "why_interesting": "破碎和共同完成感本身正在成为体验价值",
                                "inferred_questions": ["参与者如何公开描述这种真实活动"],
                                "adjacent_searches": ["参与者如何公开描述这种真实活动"],
                                "source_refs": [lead_url, f"tool_call:{material_id}"],
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "形成一个值得继续追踪的现象",
                    },
                    "consumed_tool_call_ids": [material_id],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        if "深挖任务" in prompt and "参与者如何公开描述这种真实活动" in prompt:
            return work_decision(
                {
                    "type": "tool_call",
                    "step_id": "follow-up-public-response",
                    "tool_name": "web_search",
                    "arguments": {
                        "query": "参与者如何公开描述这种真实活动",
                        "research_role": "follow_up",
                    },
                    "purpose": "寻找参与者真实反应",
                    "criterion_id": "public-response",
                    "expected_delta": "获得公开讨论与参与表达",
                    "stop_condition": "找到可阅读全文的参与者反应",
                }
            )
        if "FOLLOWUP_BODY" in prompt:
            material_id = current_material_id()
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "深挖确认参与者偏好不可预测和共同完成",
                        "market_insights": [
                            {
                                "title": "真实活动偏好获得参与者表达支持",
                                "signal": "参与者公开表达对不可预测现场的偏好",
                                "concrete_event": "活动参与者公开留言描述真实体验",
                                "phenomenon": "真实感从审美主张进入参与行为",
                                "why_interesting": "它为前一现象补上了实际公众反应",
                                "observed_discussion_signals": ["更喜欢不可预测、可以共同完成的现场"],
                                "source_refs": [followup_url, f"tool_call:{material_id}"],
                            }
                        ],
                    },
                    "observation_assessment": {
                        "information_gain": "decisive",
                        "expectation_met": True,
                        "finding": "取得来源中明确的参与者表达",
                    },
                    "consumed_tool_call_ids": [material_id],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        if "BROAD_BODY" in prompt:
            material_id = current_material_id()
            return json.dumps(
                {
                    "state_update": {
                        "current_judgment": "宽泛趋势报告只作为背景，不形成热点",
                        "rejected_paths": [f"{broad_url}：缺少具体事件和参与反应"],
                    },
                    "observation_assessment": {
                        "information_gain": "low",
                        "expectation_met": False,
                        "finding": "仅获得背景信息",
                    },
                    "consumed_tool_call_ids": [material_id],
                    "action": {"type": "plan_update"},
                },
                ensure_ascii=False,
            )
        return work_decision({"type": "complete", "message": "已完成深挖"}, progress=1)

    executor = WorkExecutor(
        database,
        runtime,
        registry,
        provider,
        max_steps_per_run=14,
        default_tool_permissions={"web_search": "allow", "read_web_page": "allow"},
    )
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="市场热点深挖",
        objective="寻找具体有趣事件、公众参与和讨论，并形成中文洞察",
    )
    runtime.create_run(thread.id)
    claimed = runtime.claim_next(agent_id=agent_id)
    result = executor.execute(claimed.id, claimed.execution_token)

    assert result.outcome == "completed"
    assert opened == [lead_url, followup_url, broad_url]
    assert searches[1]["research_role"] == "follow_up"
    checkpoint = json.loads(result.run.checkpoint_json or "{}")
    assert checkpoint.get("market_followup_history")
    assert not checkpoint.get("market_followup_required")
