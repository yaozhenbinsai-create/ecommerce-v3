import json

import pytest
from sqlalchemy import select

from app.design_service import DesignService, design_product_contract_text
from app.models import AgentProfile, DesignInputSnapshot, DesignJob
from tests.test_business_service import make_business_service


class DesignCodex:
    def __init__(self):
        self.calls = []

    def run_text_turn(self, prompt, model=None, session_key=None, timeout=None):
        self.calls.append(session_key)
        if ":route-review:" in session_key:
            return json.dumps({
                "selected_route_number": 2,
                "decision": "approved",
                "reason": "第二条最清楚地平衡陪伴和个人空间。",
                "comparisons": [
                    {"route_number": number, "strengths": ["结构清楚"],
                     "weaknesses": ["需母稿验证"], "scores": {"theme": 8}}
                    for number in (1, 2, 3)
                ],
                "master_requirements": ["保留大面积呼吸空间"],
            }, ensure_ascii=False)
        routes = []
        for number in (1, 2, 3):
            routes.append({
                "route_number": number, "name": f"路线{number}",
                "visual_story": f"视觉故事{number}", "visual_metaphor": f"隐喻{number}",
                "composition": f"构图{number}", "graphic_language": f"图形语言{number}",
                "color_strategy": f"色彩策略{number}",
                "component_relationship": f"组件关系{number}",
                "commercial_reason": f"商业理由{number}",
                "originality_boundary": "不使用受保护IP",
                "production_risks": ["需验证大尺寸细节"],
                "image_prompt_draft": f"母稿提示词{number}",
            })
        return json.dumps({
            "reference_principles": {"visual_goal": "低压力亲近感"},
            "routes": routes,
        }, ensure_ascii=False)


def make_design_opportunity(tmp_path):
    database, business, market_agent_id, _, insight_id = make_business_service(tmp_path)
    hotspot = business.ingest_market_insight(insight_id)
    proposition = business.create_expression_proposition(
        hotspot["themes"][0]["id"],
        {
            "owner_agent_id": market_agent_id,
            "title": "允许暂时失去效率的私人空间",
            "core_expression": "从持续效率要求转向私人恢复。",
            "target_emotion": "被理解和放松",
            "audience": ["城市年轻人"],
            "scenes": ["下班后的卧室"],
            "purchase_reason": "表达对私人恢复状态的认同。",
            "creative_seeds": [{"type": "概念", "content": "低电量时刻"}],
        },
    )
    product = business.create_product(
        {
            "capability_key": "design-test-set",
            "name": "可配置四件套",
            "product_type": "四件套",
            "retail_min": 39900,
            "retail_max": 69900,
            "materials": ["全棉"],
            "processes": ["数码直喷"],
            "components": [
                {"name": "枕套", "quantity": 2, "sizes": ["48×74cm"]},
                {"name": "床单", "quantity": 1, "sizes": ["230×250cm", "250×270cm"]},
                {"name": "被套", "quantity": 1, "sizes": ["200×230cm", "220×240cm"]},
                {"name": "床笠", "quantity": 1, "sizes": ["150×200cm", "180×200cm"]},
            ],
        }
    )
    fit = business.save_product_fit(
        {
            "proposition_id": proposition["id"],
            "product_capability_id": product["id"],
            "owner_agent_id": market_agent_id,
            "conclusion": "fit",
            "rationale": "表达与卧室使用场景一致。",
            "missing_information": [],
        }
    )
    with database.sessions() as session:
        designer = AgentProfile(
            name="设计Agent",
            role_responsibilities="产品视觉设计师",
            work_content="把产品机会转化为可生产的设计包",
        )
        session.add(designer)
        session.commit()
        designer_id = designer.id
    return database, fit["opportunity"]["id"], designer_id


def test_design_job_claim_is_idempotent_and_freezes_components(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    service = DesignService(database)

    first = service.claim_opportunity(opportunity_id, designer_id)
    second = service.claim_opportunity(opportunity_id, designer_id)

    assert first["created"] is True
    assert second["created"] is False
    assert second["id"] == first["id"]
    assert first["stage"] == "brief_ready"
    snapshot = first["input_snapshot"]
    assert snapshot["production"] == {
        "method": "digital_direct_print",
        "color_mode": "CMYK",
        "dpi": 300,
        "preferred_format": "TIFF",
        "fallback_formats": ["PDF", "PNG"],
        "traditional_separation_required": False,
    }
    assert [(item["name"], item["quantity"]) for item in snapshot["components"]] == [
        ("枕套", 2), ("床单", 1), ("被套", 1), ("床笠", 1)
    ]
    assert snapshot["components"][1]["sizes"] == ["230×250cm", "250×270cm"]
    with database.sessions() as session:
        assert session.scalar(select(DesignJob).where(DesignJob.id == first["id"])) is not None
        assert session.scalar(select(DesignInputSnapshot).where(
            DesignInputSnapshot.design_job_id == first["id"])) is not None


def test_design_product_contract_uses_frozen_component_quantities(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    snapshot = DesignService(database).claim_opportunity(
        opportunity_id, designer_id
    )["input_snapshot"]

    contract = json.loads(design_product_contract_text(snapshot))

    assert [(item["name"], item["quantity"]) for item in contract["components"]] == [
        ("枕套", 2), ("床单", 1), ("被套", 1), ("床笠", 1)
    ]
    assert contract["product"]["name"] == "可配置四件套"
    assert contract["production"]["dpi"] == 300


def test_design_job_transition_records_precise_stage(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    service = DesignService(database)
    job = service.claim_opportunity(opportunity_id, designer_id)

    updated = service.transition(
        job["id"],
        {
            "stage": "references_learning",
            "status": "in_progress",
            "automation_status": "running",
            "next_action": "提炼参考原则，不复制现成设计",
            "reason": "设计Agent开始第一节点",
            "actor_agent_id": designer_id,
        },
    )

    assert updated["stage"] == "references_learning"
    assert updated["status"] == "in_progress"
    assert updated["next_action"] == "提炼参考原则，不复制现成设计"


def test_design_inputs_exclude_claimed_opportunities(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    service = DesignService(database)

    before = service.list_inputs()
    assert before["total"] == 1
    assert before["items"][0]["id"] == opportunity_id
    assert before["items"][0]["product_name"] == "可配置四件套"

    service.claim_opportunity(opportunity_id, designer_id)
    after = service.list_inputs()
    assert after == {"items": [], "total": 0}


def test_text_automation_creates_three_routes_and_selects_one(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    service = DesignService(database)
    job = service.claim_opportunity(opportunity_id, designer_id)
    codex = DesignCodex()

    result = service.execute_text_automation(job["id"], codex)
    saved = service.get_job(job["id"])

    assert len(saved["routes"]) == 3
    assert [item["route_number"] for item in saved["routes"] if item["selected"]] == [2]
    assert len(saved["reviews"]) == 1
    assert saved["stage"] == "master_generating"
    assert saved["status"] == "deferred"
    assert saved["assets"] == []
    assert "安全暂缓" in result
    assert codex.calls == ["design:1:routes:v1", "design:1:route-review:v1"]


def test_master_automation_reworks_until_blind_review_approves(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    service = DesignService(database)
    job = service.claim_opportunity(opportunity_id, designer_id)
    codex = DesignCodex()
    generated_versions = []

    def generate(job_id, prompt, version):
        generated_versions.append((version, prompt))
        return {
            "uri": f"outputs/design/job-{job_id}/master-v{version}.png",
            "content_hash": f"{version:064x}", "model_id": 3,
            "model_name": "image-model", "protocol": "openai_images",
            "mime_type": "image/png", "width": 1024, "height": 1024,
        }

    def review(job_id, result, version):
        if version == 1:
            return {"artifact_type": "coordinated_design_board",
                    "is_complete_product_design": True,
                    "decision": "rework_required", "problems": ["层级弱"],
                    "required_changes": ["建立明确主次，不使用黑色轮廓线"]}
        return {"artifact_type": "coordinated_design_board",
                "is_complete_product_design": True,
                "decision": "approved", "problems": [], "required_changes": []}

    result = service.execute_text_automation(job["id"], codex, generate, review)
    saved = service.get_job(job["id"])

    assert [item[0] for item in generated_versions] == [1, 2]
    assert "建立明确主次" in generated_versions[1][1]
    assert [item["status"] for item in saved["assets"]] == ["rework_required", "approved"]
    assert [item["decision"] for item in saved["reviews"]][-2:] == [
        "rework_required", "approved"
    ]
    assert saved["stage"] == "master_approved"
    assert "可进入效果图阶段" in result


def test_master_automation_defers_after_three_failed_visual_reviews(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    service = DesignService(database)
    job = service.claim_opportunity(opportunity_id, designer_id)

    def generate(job_id, prompt, version):
        return {"uri": f"outputs/design/job-{job_id}/master-v{version}.png",
                "content_hash": f"{version:064x}"}

    def reject(job_id, result, version):
        return {"artifact_type": "coordinated_design_board",
                "is_complete_product_design": True,
                "decision": "rework_required", "required_changes": [f"重构版本{version}"]}

    result = service.execute_text_automation(job["id"], DesignCodex(), generate, reject)
    saved = service.get_job(job["id"])

    assert len(saved["assets"]) == 3
    assert all(item["status"] == "rework_required" for item in saved["assets"])
    assert saved["status"] == "deferred"
    assert saved["automation_status"] == "deferred"
    assert "三轮母稿均未通过" in result


def test_master_review_timeout_resumes_same_persisted_asset(tmp_path):
    database, opportunity_id, designer_id = make_design_opportunity(tmp_path)
    service = DesignService(database)
    job = service.claim_opportunity(opportunity_id, designer_id)
    generated = []

    def generate(job_id, prompt, version):
        generated.append(version)
        return {"uri": f"outputs/design/job-{job_id}/master-v{version}.png",
                "content_hash": f"{version:064x}", "mime_type": "image/png"}

    def timeout(job_id, result, version):
        raise RuntimeError("review timeout")

    with pytest.raises(RuntimeError, match="review timeout"):
        service.execute_text_automation(job["id"], DesignCodex(), generate, timeout)
    assert generated == [1]
    assert service.get_job(job["id"])["assets"][0]["status"] == "generated_pending_review"

    def approve(job_id, result, version):
        return {"artifact_type": "coordinated_design_board",
                "is_complete_product_design": True,
                "decision": "approved", "problems": [], "required_changes": []}

    result = service.execute_text_automation(job["id"], DesignCodex(), generate, approve)
    assert generated == [1]
    assert service.get_job(job["id"])["assets"][0]["status"] == "approved"
    assert "v1 已通过" in result
