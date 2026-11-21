import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.business_service import BusinessError, BusinessService
from app.database import Database
from app.env_store import EnvStore
from app.main import create_app
from app.models import (
    AgentProfile,
    BusinessContentFact,
    BusinessContentSource,
    BusinessContentTheme,
    BusinessHotspot,
    BusinessObjectEvent,
    Evidence,
    ExpressionProposition,
    MarketInsightRecord,
    ProductCapability,
    ProductComponent,
    ProductDesignBrief,
    ProductFitAssessment,
    ProductOpportunity,
    WorkThread,
)


def make_business_service(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'business.db'}")
    database.create_tables()
    with database.sessions() as session:
        agent = AgentProfile(name="市场分析Agent")
        session.add(agent)
        session.flush()
        thread = WorkThread(
            owner_agent_id=agent.id,
            title="开放内容研究",
            objective="寻找现实内容并形成长期业务数据",
        )
        session.add(thread)
        session.flush()
        evidence = Evidence(
            work_thread_id=thread.id,
            owner_agent_id=agent.id,
            source_url="https://example.com/story",
            source_title="年轻人的低电量生活",
            published_at="2026-08-20",
            claim="公开讨论中出现把卧室视为恢复空间的表达。",
            excerpt="下班后只想回到不需要保持效率的房间。",
            confidence="high",
            source_type="independent_secondary",
            source_publisher="Example",
            content_hash="e" * 64,
        )
        session.add(evidence)
        session.add(
            Evidence(
                work_thread_id=thread.id,
                owner_agent_id=agent.id,
                source_url="https://another.example.org/story",
                source_title="卧室恢复空间观察",
                published_at="2026-08-21",
                claim="另一独立来源也记录了年轻人主动降低信息输入的生活实践。",
                excerpt="参与者把卧室作为退出效率要求和恢复精力的空间。",
                confidence="high",
                source_type="independent_secondary",
                source_publisher="Another Example",
                content_hash="f" * 64,
            )
        )
        session.flush()
        insight_data = {
            "title": "低电量生活与私人恢复空间",
            "category": "生活方式",
            "signal": "年轻人开始公开表达暂时退出效率竞争的需要。",
            "phenomenon": "卧室被重新理解为允许低效率和恢复能量的私人空间。",
            "why_interesting": "核心矛盾是持续效率要求与私人恢复需要之间的冲突。",
            "observed_discussion_signals": ["人们分享下班后降低社交和信息输入的做法。"],
            "participant_groups": ["城市年轻人"],
            "participation_behaviors": ["整理卧室", "减少信息输入"],
            "source_refs": [
                "https://example.com/story",
                "https://another.example.org/story",
            ],
            "confidence": "high",
        }
        insight = MarketInsightRecord(
            work_thread_id=thread.id,
            owner_agent_id=agent.id,
            title=insight_data["title"],
            category=insight_data["category"],
            source_refs_json=json.dumps(insight_data["source_refs"], ensure_ascii=False),
            insight_json=json.dumps(insight_data, ensure_ascii=False),
            search_text=json.dumps(insight_data, ensure_ascii=False).lower(),
            content_hash="i" * 64,
        )
        session.add(insight)
        session.commit()
        return database, BusinessService(database), agent.id, thread.id, insight.id


class NoopCodex:
    def close(self):
        return None


def make_api_client(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\nconversation:\n  model_retry_pause_seconds: 0\n",
        encoding="utf-8",
    )
    return TestClient(
        create_app(
            database=Database(f"sqlite:///{tmp_path / 'api.db'}"),
            env_store=EnvStore(tmp_path / ".env"),
            codex=NoopCodex(),
            settings_path=settings_path,
        )
    )


def test_market_insight_becomes_idempotent_business_data(tmp_path):
    database, service, agent_id, thread_id, insight_id = make_business_service(tmp_path)

    first = service.ingest_market_insight(insight_id)
    second = service.ingest_market_insight(insight_id)

    assert first["id"] == second["id"]
    assert first["owner_agent_id"] == agent_id
    assert first["active_work_thread_id"] == thread_id
    assert first["research_status"] == "verified"
    assert first["business_stage"] == "theme_formed"
    assert first["next_action"] == "基于已核验主题形成产品表达命题"
    assert len(first["sources"]) == 2
    assert len(first["facts"]) >= 4
    assert len(first["themes"]) == 1
    assert first["themes"][0]["people"] == ["城市年轻人"]

    with database.sessions() as session:
        assert session.scalar(select(func.count(BusinessHotspot.id))) == 1
        assert session.scalar(select(func.count(BusinessContentSource.id))) == 2
        assert session.scalar(select(func.count(BusinessContentTheme.id))) == 1
        fact_count = session.scalar(select(func.count(BusinessContentFact.id)))
        assert fact_count == len(first["facts"])


def test_internal_tool_trace_is_not_counted_as_external_source_and_evidence_is_created(tmp_path):
    database, service, _, thread_id, insight_id = make_business_service(tmp_path)
    with database.sessions() as session:
        insight = session.get(MarketInsightRecord, insight_id)
        insight.source_refs_json = json.dumps(
            ["https://example.com/story", "tool_call:700"], ensure_ascii=False
        )
        session.execute(delete(Evidence).where(Evidence.work_thread_id == thread_id))
        session.commit()

    hotspot = service.ingest_market_insight(insight_id)

    assert [item["source_url"] for item in hotspot["sources"]] == [
        "https://example.com/story"
    ]
    assert hotspot["sources"][0]["source_type"] == "web_page"
    assert hotspot["sources"][0]["publisher"] == "example.com"
    assert hotspot["research_status"] == "insufficient_evidence"
    assert hotspot["business_stage"] == "unprocessed"
    assert hotspot["themes"][0]["status"] == "insufficient_evidence"
    assert service.list_business_backlog(owner_agent_id=hotspot["owner_agent_id"])["stage"] == "research"
    with database.sessions() as session:
        evidence_count = session.scalar(
            select(func.count(Evidence.id)).where(Evidence.work_thread_id == thread_id)
        )
        assert evidence_count == 1
def test_agent_business_tools_can_scope_hotspots_to_current_work_thread(tmp_path):
    database, service, agent_id, first_thread_id, insight_id = make_business_service(
        tmp_path
    )
    first = service.ingest_market_insight(insight_id)
    with database.sessions() as session:
        second_thread = WorkThread(
            owner_agent_id=agent_id,
            title="全新隔离项目",
            objective="只处理本项目新发现的内容",
        )
        session.add(second_thread)
        session.flush()
        second_data = {
            "title": "社区花园晚间活动",
            "category": "社区生活",
            "signal": "居民在晚间共同维护社区花园。",
            "phenomenon": "公共空间成为低压力邻里连接入口。",
            "why_interesting": "陌生人之间出现轻量协作。",
            "source_refs": ["https://example.com/garden"],
        }
        second_insight = MarketInsightRecord(
            work_thread_id=second_thread.id,
            owner_agent_id=agent_id,
            title=second_data["title"],
            category=second_data["category"],
            source_refs_json=json.dumps(second_data["source_refs"], ensure_ascii=False),
            insight_json=json.dumps(second_data, ensure_ascii=False),
            search_text=json.dumps(second_data, ensure_ascii=False).lower(),
            content_hash="j" * 64,
        )
        session.add(second_insight)
        session.commit()
        second_thread_id, second_insight_id = second_thread.id, second_insight.id
    second = service.ingest_market_insight(second_insight_id)

    assert [item["id"] for item in service.list_hotspots(owner_agent_id=agent_id)] == [
        second["id"],
        first["id"],
    ]
    assert [
        item["id"]
        for item in service.list_hotspots(
            owner_agent_id=agent_id,
            work_thread_id=first_thread_id,
        )
    ] == [first["id"]]
    assert [
        item["id"]
        for item in service.list_hotspots(
            owner_agent_id=agent_id,
            work_thread_id=second_thread_id,
        )
    ] == [second["id"]]
    with pytest.raises(BusinessError, match="热点不属于当前项目"):
        service.get_hotspot(
            first["id"],
            owner_agent_id=agent_id,
            work_thread_id=second_thread_id,
        )


def test_open_associations_create_multiple_themes_and_business_backlog(tmp_path):
    database, service, agent_id, _, insight_id = make_business_service(tmp_path)
    with database.sessions() as session:
        insight = session.get(MarketInsightRecord, insight_id)
        data = json.loads(insight.insight_json)
        data["creative_theme_directions"] = [
            {
                "title": "允许孤独被看见",
                "inspiration_path": "从低电量生活的公开叙述联想到独处而不被催促的需要。",
                "human_meaning": "孤独不总是缺陷，也可能是恢复边界的方式。",
                "core_tension": "被社会连接与保护私人边界之间的张力",
                "emotions": ["孤独", "被理解"],
                "people": ["城市年轻人"],
                "scenes": ["独自回到卧室"],
                "symbols": ["安静房间", "微弱灯光"],
                "sensitivity_notes": ["不把抑郁或孤独浪漫化"],
            },
            {
                "title": "缓慢恢复",
                "inspiration_path": "从退出效率竞争联想到恢复并非瞬间完成。",
                "human_meaning": "人可以用自己的节奏逐渐恢复。",
                "emotions": ["松弛", "希望"],
                "scenes": ["休息日早晨"],
            },
        ]
        insight.insight_json = json.dumps(data, ensure_ascii=False)
        session.commit()

    hotspot = service.ingest_market_insight(insight_id)
    assert [item["title"] for item in hotspot["themes"]] == [
        "允许孤独被看见",
        "缓慢恢复",
    ]
    assert hotspot["themes"][0]["emotions"] == ["孤独", "被理解"]
    assert any("敏感性边界" in item for item in hotspot["themes"][0]["signals"])

    backlog = service.list_business_backlog(owner_agent_id=agent_id)
    assert backlog["stage"] == "expression_proposition"
    assert backlog["themes_waiting"] == 2
    assert backlog["themes"][0]["theme_title"] == "允许孤独被看见"


def test_business_backlog_advances_from_theme_to_product_fit(tmp_path):
    _, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)
    proposition = service.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": agent_id,
            "title": "私人恢复不需要证明效率",
            "core_expression": "表达休息本身具有正当性。",
        },
    )
    with service.database.sessions() as session:
        session.get(ExpressionProposition, proposition["id"]).status = "draft"
        # A sibling theme is still waiting, but the proposition already in
        # flight must finish product fit before the next theme is opened.
        session.add(
            BusinessContentTheme(
                hotspot_id=hotspot["id"],
                owner_agent_id=agent_id,
                title="同一热点的后续主题",
                narrative="等待下一轮独立分析。",
                status="active",
                fingerprint="sibling-theme-waiting",
            )
        )
        session.commit()
    product = service.create_product(
        {
            "capability_key": "sheet-backlog-test",
            "name": "床单",
            "product_type": "单品",
            "components": [{"name": "床单", "quantity": 1}],
        }
    )

    backlog = service.list_business_backlog(owner_agent_id=agent_id)
    assert backlog["stage"] == "product_fit"
    assert backlog["themes_waiting"] == 1
    assert backlog["propositions"][0]["proposition_id"] == proposition["id"]
    assert backlog["recent_expression_contexts"][0]["proposition_id"] == proposition["id"]
    assert backlog["recent_expression_contexts"][0]["title"] == proposition["title"]
    assert backlog["enabled_products"][0]["id"] == product["id"]


def test_product_capability_versions_preserve_old_components(tmp_path):
    _, service, *_ = make_business_service(tmp_path)

    first = service.create_product(
        {
            "capability_key": "bedding-set",
            "name": "床单式四件套",
            "product_type": "套件",
            "cost_min": 10000,
            "cost_max": 20000,
            "retail_min": 29900,
            "retail_max": 59900,
            "materials": ["全棉", "天丝"],
            "processes": ["数码直喷"],
            "components": [
                {"name": "被套", "quantity": 1, "supports_ab_sides": True},
                {"name": "床单", "quantity": 1},
                {"name": "枕套", "quantity": 2},
            ],
        }
    )
    second = service.create_product_version(
        first["id"],
        {
            "name": "床笠式四件套",
            "components": [
                {"name": "被套", "quantity": 1},
                {"name": "床笠", "quantity": 1},
                {"name": "枕套", "quantity": 2},
            ],
        },
    )

    assert first["version"] == 1
    assert second["version"] == 2
    assert first["capability_key"] == second["capability_key"]
    assert [item["name"] for item in service.get_product(first["id"])["components"]] == [
        "被套", "床单", "枕套"
    ]
    assert [item["name"] for item in second["components"]] == ["被套", "床笠", "枕套"]
    assert service.get_product(first["id"])["is_enabled"] is False
    assert second["is_enabled"] is True


def test_optional_intervention_is_audited_and_does_not_require_approval(tmp_path):
    database, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)

    updated = service.intervene_hotspot(
        hotspot["id"],
        {
            "actor_agent_id": agent_id,
            "reason": "暂时补充不同来源后再进入产品表达",
            "updates": {
                "research_status": "collecting",
                "automation_status": "queued",
                "next_action": "寻找另一类独立来源",
            },
        },
    )

    assert updated["research_status"] == "collecting"
    assert updated["automation_status"] == "queued"
    assert updated["next_action"] == "寻找另一类独立来源"
    intervention = next(
        event for event in updated["events"] if event["event_type"] == "manual_intervention"
    )
    assert intervention["reason"] == "暂时补充不同来源后再进入产品表达"
    assert intervention["before"]["research_status"] == "verified"
    assert intervention["after"]["research_status"] == "collecting"

    with database.sessions() as session:
        assert session.scalar(
            select(func.count(BusinessObjectEvent.id)).where(
                BusinessObjectEvent.event_type == "manual_intervention"
            )
        ) == 1


def test_expression_fit_creates_traceable_product_opportunity(tmp_path):
    _, service, agent_id, thread_id, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)
    theme_id = hotspot["themes"][0]["id"]
    proposition = service.create_expression_proposition(
        theme_id,
        {
            "owner_agent_id": agent_id,
            "title": "允许暂时失去效率的私人空间",
            "core_expression": "表达人在私人空间中卸下持续效率要求并逐渐恢复的过程。",
            "target_emotion": "被理解、放松，但不使用空洞治愈口号",
            "audience": ["城市年轻人"],
            "scenes": ["下班后回到卧室"],
            "purchase_reason": "消费者购买的是对私人恢复状态的认同和空间表达。",
            "exploration_angles": ["状态变化", "低强度幽默"],
            "avoid": ["直接复制躺平网络热词"],
            "risks": ["短期网络语言迅速过时"],
        },
    )
    product = service.create_product(
        {
            "capability_key": "pillowcase-pair",
            "name": "枕套两件套",
            "product_type": "两件套",
            "retail_min": 6900,
            "retail_max": 12900,
            "components": [{"name": "枕套", "quantity": 2}],
        }
    )

    fit = service.save_product_fit(
        {
            "proposition_id": proposition["id"],
            "product_capability_id": product["id"],
            "owner_agent_id": agent_id,
            "conclusion": "fit",
            "rationale": "枕套位于私人休息场景且适合低成本验证表达命题。",
            "target_audience": ["城市年轻人"],
            "scenes": ["卧室休息"],
            "risks": ["需要避免口号化"],
            "missing_information": [],
        }
    )

    assert fit["conclusion"] == "fit"
    assert fit["opportunity"]["status"] == "qualified"
    assert fit["opportunity"]["fit_assessment_id"] == fit["id"]
    assert fit["opportunity"]["next_action"] == "产品适配已通过，等待未来设计阶段启用"
    assert fit["opportunity"]["design_brief"] is None
    existing = service.get_existing_scoped_product_fit(
        proposition_id=proposition["id"],
        product_capability_id=product["id"],
        owner_agent_id=agent_id,
        work_thread_id=thread_id,
    )
    assert existing is not None
    assert existing["id"] == fit["id"]
    assert existing["opportunity"]["id"] == fit["opportunity"]["id"]
    updated_hotspot = service.get_hotspot(hotspot["id"])
    assert updated_hotspot["business_stage"] == "opportunity_created"
    assert updated_hotspot["next_action"] == "产品适配已通过，等待未来设计阶段启用"

    overridden = service.intervene_opportunity(
        fit["opportunity"]["id"],
        {
            "actor_agent_id": agent_id,
            "reason": "先观察主题持续性，不立即进入未来设计流程",
            "updates": {"status": "watching", "next_action": "七天后复核主题持续性"},
        },
    )
    assert overridden["status"] == "watching"
    assert overridden["next_action"] == "七天后复核主题持续性"


def test_product_fit_discards_future_design_relationship_gaps(tmp_path):
    _, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)
    proposition = service.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": agent_id,
            "title": "共同片段",
            "core_expression": "允许同一段经历留下不同的个人记忆。",
        },
    )
    product = service.create_product(
        {
            "capability_key": "design-gap-filter-set",
            "name": "关系过滤两件套",
            "product_type": "两件套",
            "components": [
                {"name": "被套", "quantity": 1},
                {"name": "枕套", "quantity": 1},
            ],
        }
    )

    fit = service.save_product_fit(
        {
            "proposition_id": proposition["id"],
            "product_capability_id": product["id"],
            "owner_agent_id": agent_id,
            "conclusion": "fit",
            "rationale": "通用产品参数足以完成当前产品适配。",
            "missing_information": [
                "后续确定被套与枕套的关系层次",
                "未来确定组件主次和画面排布",
            ],
        }
    )

    assert fit["missing_information"] == []
    assert fit["opportunity"]["status"] == "qualified"
    assert fit["opportunity"]["missing_information"] == []


def test_needs_validation_is_visible_as_a_validating_product_opportunity(tmp_path):
    _, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)
    proposition = service.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": agent_id,
            "title": "恢复空间待验证表达",
            "core_expression": "把休息空间理解为逐渐恢复的容器。",
        },
    )
    product = service.create_product(
        {
            "capability_key": "validation-sheet",
            "name": "验证床单",
            "product_type": "单品",
            "components": [{"name": "床单", "quantity": 1}],
        }
    )

    fit = service.save_product_fit(
        {
            "proposition_id": proposition["id"],
            "product_capability_id": product["id"],
            "owner_agent_id": agent_id,
            "conclusion": "needs_validation",
            "rationale": "主题有表达潜力，但购买动机仍需验证。",
            "missing_information": ["消费者是否愿意为该表达购买床品"],
        }
    )

    assert fit["opportunity"]["status"] == "candidate"
    assert fit["opportunity"]["missing_information"] == []
    assert fit["opportunity"]["next_action"] == "产品通用参数已补齐；市场验证暂缓，保留为候选机会"
    assert service.get_hotspot(hotspot["id"])["business_stage"] == "opportunity_created"


def test_empty_missing_information_never_keeps_validate_missing_next_action(tmp_path):
    _, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)
    proposition = service.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": agent_id,
            "title": "无需补缺的候选表达",
            "core_expression": "保留为候选机会等待未来业务阶段。",
        },
    )
    product = service.create_product(
        {
            "capability_key": "candidate-sheet",
            "name": "候选床单",
            "product_type": "单品",
            "components": [{"name": "床单", "quantity": 1}],
        }
    )

    fit = service.save_product_fit(
        {
            "proposition_id": proposition["id"],
            "product_capability_id": product["id"],
            "owner_agent_id": agent_id,
            "conclusion": "needs_validation",
            "rationale": "市场验证当前暂缓，但产品通用参数已经完整。",
            "missing_information": [],
        }
    )

    assert fit["opportunity"]["status"] == "candidate"
    assert fit["opportunity"]["missing_information"] == []
    assert fit["opportunity"]["next_action"] == "产品通用参数已补齐；市场验证暂缓，保留为候选机会"
    assert fit["opportunity"]["design_brief"] is None


def test_pending_product_fit_query_only_returns_unprocessed_propositions(tmp_path):
    _, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)
    proposition = service.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": agent_id,
            "title": "安静恢复空间",
            "core_expression": "给低电量生活留出无需解释的恢复时刻。",
        },
    )

    matched = service.list_pending_product_fits(
        owner_agent_id=agent_id,
        query="恢复",
    )
    missed = service.list_pending_product_fits(
        owner_agent_id=agent_id,
        query="海滩",
    )

    assert matched["total"] == 1
    assert matched["items"][0]["id"] == proposition["id"]
    assert missed == {"total": 0, "items": []}
    backlog = service.list_business_backlog(
        owner_agent_id=agent_id,
        limit=50,
    )
    assert [item["proposition_id"] for item in backlog["propositions"]] == [
        proposition["id"]
    ]

    product = service.create_product(
        {
            "capability_key": "pending-fit-sheet",
            "name": "通用床单",
            "product_type": "单品",
            "components": [{"name": "床单", "quantity": 1}],
        }
    )
    service.save_product_fit(
        {
            "proposition_id": proposition["id"],
            "product_capability_id": product["id"],
            "owner_agent_id": agent_id,
            "conclusion": "not_fit",
            "rationale": "记录明确不适配也代表该命题已经完成产品适配判断。",
        }
    )

    completed = service.list_pending_product_fits(owner_agent_id=agent_id)
    assert completed == {"total": 0, "items": []}


def test_expression_proposition_persists_normalized_creative_context(tmp_path):
    _, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)

    proposition = service.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": agent_id,
            "title": "雨停以后重新出门",
            "core_expression": "把恢复理解为重新接近日常，而不是立刻振作。",
            "theme_form": "雨后街角",
            "creative_seeds": [
                {
                    "type": " 景 ",
                    "content": " 路灯下逐渐变浅的水迹 ",
                    "source_relation": "从中断后的生活恢复联想到雨停后的街道。",
                    "creative_potential": "提供时间正在继续的具体线索。",
                    "ignored": "不应持久化",
                },
                {"type": "物", "content": "重新挂回门边的钥匙"},
                {"type": "物", "content": "   "},
            ],
        },
    )

    assert proposition["theme_form"] == "雨后街角"
    assert proposition["creative_seeds"] == [
        {
            "type": "景",
            "content": "路灯下逐渐变浅的水迹",
            "source_relation": "从中断后的生活恢复联想到雨停后的街道。",
            "creative_potential": "提供时间正在继续的具体线索。",
        },
        {
            "type": "物",
            "content": "重新挂回门边的钥匙",
            "source_relation": "",
            "creative_potential": "",
        },
    ]


def test_permanent_hotspot_delete_cascades_business_data_but_keeps_configuration(tmp_path):
    database, service, agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = service.ingest_market_insight(insight_id)
    proposition = service.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": agent_id,
            "title": "恢复空间表达",
            "core_expression": "让休息成为无需解释的日常状态。",
        },
    )
    product = service.create_product(
        {
            "capability_key": "delete-test-sheet",
            "name": "测试床单",
            "product_type": "单品",
            "components": [{"name": "床单", "quantity": 1}],
        }
    )
    service.save_product_fit(
        {
            "proposition_id": proposition["id"],
            "product_capability_id": product["id"],
            "owner_agent_id": agent_id,
            "conclusion": "fit",
            "rationale": "适合测试业务数据删除链路。",
        }
    )

    result = service.delete_hotspot(hotspot["id"])

    assert result["deleted"]["hotspots"] == 1
    assert result["deleted"]["opportunities"] == 1
    assert result["deleted"]["design_briefs"] == 0
    with database.sessions() as session:
        for model in (
            BusinessHotspot,
            BusinessContentSource,
            BusinessContentFact,
            BusinessContentTheme,
            ExpressionProposition,
            ProductFitAssessment,
            ProductOpportunity,
            ProductDesignBrief,
        ):
            assert session.scalar(select(func.count(model.id))) == 0
        assert session.scalar(
            select(func.count(BusinessObjectEvent.id)).where(
                BusinessObjectEvent.object_type != "product_capability"
            )
        ) == 0
        assert session.scalar(select(func.count(ProductCapability.id))) == 1
        assert session.scalar(select(func.count(MarketInsightRecord.id))) == 1


def test_agent_business_pagination_and_delete_all_api(tmp_path):
    with make_api_client(tmp_path) as client:
        agent_id = client.get("/api/chat/agents").json()[0]["id"]
        state = client.get(
            f"/api/business/agents/{agent_id}?hotspot_page=8&hotspot_page_size=2"
        )
        assert state.status_code == 200
        assert state.json()["hotspot_pagination"] == {
            "page": 1,
            "page_size": 2,
            "pages": 1,
            "total": 0,
        }
        rejected = client.request(
            "DELETE",
            f"/api/business/agents/{agent_id}/data",
            json={"confirmation": "删除"},
        )
        assert rejected.status_code == 400
        deleted = client.request(
            "DELETE",
            f"/api/business/agents/{agent_id}/data",
            json={"confirmation": "永久删除全部业务数据"},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"]["hotspots"] == 0


def test_business_api_exposes_products_and_agent_tools(tmp_path):
    with make_api_client(tmp_path) as client:
        product_response = client.post(
            "/api/business/products",
            json={
                "capability_key": "sheet",
                "name": "床单",
                "product_type": "单品",
                "retail_min": 9900,
                "retail_max": 19900,
                "materials": ["全棉"],
                "components": [{"name": "床单", "quantity": 1}],
            },
        )
        assert product_response.status_code == 201
        product = product_response.json()
        assert product["name"] == "床单"
        assert product["components"][0]["name"] == "床单"
        assert client.get("/api/business/products?enabled_only=true").json()[0]["id"] == product["id"]

        tools = {item["name"] for item in client.get("/api/tools").json()}
        assert {
            "list_business_hotspots",
            "get_business_hotspot",
            "get_expression_proposition",
            "save_expression_proposition",
            "list_product_capabilities",
            "save_product_fit_assessment",
            "list_product_opportunities",
        }.issubset(tools)

        page = client.get("/")
        assert page.status_code == 200
        assert 'id="businessPage"' in page.text
        assert 'id="businessConfigPage"' in page.text
        assert 'aria-label="基础配置"' in page.text
        assert 'aria-label="业务运营"' in page.text
        assert 'aria-label="执行协作"' in page.text
        assert 'aria-label="数据与系统"' in page.text
        assert 'class="nav-group-title" aria-expanded="true"' in page.text
        assert "setNavGroupCollapsed" in page.text
        assert "boss-agent-nav-" in page.text
        assert 'id="businessAgentSelect"' in page.text
        assert 'id="businessPendingFitQuery"' in page.text
        assert 'id="businessPendingFitRun"' in page.text
        assert 'id="businessDeleteAllButton"' in page.text
        assert 'id="businessHotspotPrev"' in page.text
        assert 'id="businessFinalScreen"' in page.text
        assert 'id="businessAbnormalScreen"' in page.text
        assert 'id="businessHotspotScreen"' in page.text
        assert 'id="businessFinalOutputList"' in page.text
        assert 'id="businessFinalPrev"' in page.text
        assert 'id="businessAbnormalNext"' in page.text
        assert 'id="businessPendingNext"' in page.text
        assert "scroll-snap-type: y mandatory" in page.text
        assert 'aria-label="业务屏幕导航"' in page.text
        assert 'data-business-screen-index="2"' in page.text
        assert "businessScreenWheelLocked" in page.text
        assert 'id="businessDetailDialog"' in page.text
        assert "const businessScreenPageSize = 15" in page.text
        assert "showBusinessDetail" in page.text
        assert 'id="businessProductForm"' in page.text
        assert 'id="businessProductComponents"' in page.text
        assert "loadBusinessAgents" in page.text
        assert "loadBusinessConfiguration" in page.text
        assert "saveBusinessProduct" in page.text
        assert "/api/business/agents/" in page.text
        assert "/api/business/hotspots/${hotspotId}/interventions" in page.text
        assert "/api/business/hotspots/${hotspotId}" in page.text
        assert "/api/business/agents/${selectedBusinessAgentId}/data" in page.text
        assert "/pending-product-fits" in page.text
        assert "/api/business/opportunities/${opportunityId}/interventions" in page.text
