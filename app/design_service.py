from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .business_service import BusinessError
from .database import Database
from .models import (
    AgentProfile,
    BusinessContentTheme,
    BusinessObjectEvent,
    DesignAsset,
    DesignComponentSpec,
    DesignInputSnapshot,
    DesignJob,
    DesignReview,
    DesignRoute,
    ExpressionProposition,
    ProductCapability,
    ProductComponent,
    ProductDesignBrief,
    ProductFitAssessment,
    ProductOpportunity,
    WorkThread,
    utc_now,
)


DESIGN_STAGES = (
    "intake", "brief_ready", "references_learning", "routes_generating",
    "routes_reviewing", "master_generating", "master_reviewing",
    "master_approved", "renders_generating", "renders_reviewing",
    "production_files_generating", "production_files_reviewing",
    "design_completed",
)
DESIGN_BRANCH_STAGES = {"rework_required", "waiting_retry", "deferred", "discarded", "failed"}


def design_product_contract_text(snapshot: dict[str, Any] | None) -> str:
    """Describe the frozen product without inventing a fixed bedding set."""
    if not isinstance(snapshot, dict):
        raise BusinessError("设计任务缺少冻结产品输入")
    product = snapshot.get("product")
    components = snapshot.get("components")
    if not isinstance(product, dict) or not isinstance(components, list) or not components:
        raise BusinessError("冻结产品输入缺少产品或组件")
    component_contract = []
    for component in components:
        if not isinstance(component, dict):
            raise BusinessError("冻结产品组件格式不正确")
        name = str(component.get("name") or "").strip()
        quantity = component.get("quantity")
        if not name or not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise BusinessError("冻结产品组件缺少有效名称或数量")
        component_contract.append({
            "name": name,
            "quantity": quantity,
            "sizes": component.get("sizes") or [],
            "materials": component.get("materials") or [],
            "constraints": component.get("constraints") or [],
        })
    return _dump({
        "product": {"name": product.get("name"), "type": product.get("type"),
                    "version": product.get("version")},
        "components": component_contract,
        "production": snapshot.get("production") or {},
    })


def _load(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def customer_design_knowledge() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "config" / "design_customer_knowledge.json"
    local = path.with_name("design_customer_knowledge.local.json")
    if local.is_file():
        path = local
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class DesignService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def claim_opportunity(self, opportunity_id: int, owner_agent_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            agent = session.get(AgentProfile, owner_agent_id)
            if agent is None:
                raise BusinessError("设计Agent不存在")
            opportunity = session.get(ProductOpportunity, opportunity_id)
            if opportunity is None:
                raise BusinessError("产品机会不存在")
            if opportunity.status not in {"qualified", "ready_for_design"}:
                raise BusinessError("产品机会尚未进入可设计状态")
            fit = session.get(ProductFitAssessment, opportunity.fit_assessment_id)
            product = session.get(ProductCapability, fit.product_capability_id) if fit else None
            proposition = session.get(ExpressionProposition, fit.proposition_id) if fit else None
            theme = session.get(BusinessContentTheme, proposition.theme_id) if proposition else None
            if not fit or not product or not proposition or not theme:
                raise BusinessError("产品机会上游数据不完整")
            brief = session.scalar(select(ProductDesignBrief).where(
                ProductDesignBrief.product_opportunity_id == opportunity.id
            ))
            components = session.scalars(select(ProductComponent).where(
                ProductComponent.product_capability_id == product.id
            ).order_by(ProductComponent.position, ProductComponent.id)).all()
            component_values = [{
                "id": item.id, "name": item.name, "quantity": item.quantity,
                "sizes": _load(item.sizes_json, []), "materials": _load(item.materials_json, []),
                "supported_pattern_modes": _load(item.supported_pattern_modes_json, []),
                "supports_ab_sides": item.supports_ab_sides,
                "constraints": _load(item.constraints_json, []), "position": item.position,
            } for item in components]
            brief_value = {
                "product": {"id": product.id, "capability_key": product.capability_key,
                    "version": product.version, "name": product.name, "type": product.product_type,
                    "materials": _load(product.materials_json, []), "processes": _load(product.processes_json, []),
                    "price": {"currency": product.currency, "retail_min": product.retail_min,
                        "retail_max": product.retail_max}},
                "components": component_values,
                "theme": {"id": theme.id, "title": theme.title, "narrative": theme.narrative,
                    "core_tension": theme.core_tension, "emotions": _load(theme.emotions_json, []),
                    "people": _load(theme.people_json, []), "scenes": _load(theme.scenes_json, [])},
                "proposition": {"id": proposition.id, "title": proposition.title,
                    "core_expression": proposition.core_expression, "target_emotion": proposition.target_emotion,
                    "audience": _load(proposition.audience_json, []), "scenes": _load(proposition.scenes_json, []),
                    "purchase_reason": proposition.purchase_reason, "theme_form": proposition.theme_form,
                    "creative_seeds": _load(proposition.creative_seeds_json, []),
                    "avoid": _load(proposition.avoid_json, []), "risks": _load(proposition.risks_json, [])},
                "production": {"method": "digital_direct_print", "color_mode": "CMYK",
                    "dpi": 300, "preferred_format": "TIFF", "fallback_formats": ["PDF", "PNG"],
                    "traditional_separation_required": False},
            }
            if brief is None:
                brief = ProductDesignBrief(product_opportunity_id=opportunity.id,
                    owner_agent_id=owner_agent_id, product_capability_id=product.id,
                    status="ready_for_design", brief_json=_dump(brief_value),
                    source_snapshot_json=_dump({"opportunity_id": opportunity.id,
                        "fit_assessment_id": fit.id, "proposition_id": proposition.id,
                        "theme_id": theme.id, "product_capability_id": product.id,
                        "product_version": product.version}))
                session.add(brief)
                session.flush()
            job = session.scalar(select(DesignJob).where(DesignJob.product_design_brief_id == brief.id))
            created = job is None
            if job is None:
                job = DesignJob(product_design_brief_id=brief.id,
                    product_opportunity_id=opportunity.id, owner_agent_id=owner_agent_id,
                    title=opportunity.title, stage="brief_ready", status="not_started",
                    automation_status="queued", next_action="学习参考原则并形成三条创意路线")
                session.add(job)
                session.flush()
                raw = _dump(brief_value)
                session.add(DesignInputSnapshot(design_job_id=job.id, version=1,
                    snapshot_json=raw, content_hash=sha256(raw.encode("utf-8")).hexdigest()))
                for component in components:
                    session.add(DesignComponentSpec(design_job_id=job.id,
                        component_id=component.id, version=1,
                        specification_json=_dump({"name": component.name,
                            "quantity": component.quantity, "sizes": _load(component.sizes_json, []),
                            "design_relation": "pending_master_design"})))
                session.add(BusinessObjectEvent(object_type="design_job", object_id=job.id,
                    event_type="claimed", actor_kind="agent", actor_agent_id=owner_agent_id,
                    reason="设计Agent领取产品机会", after_json=_dump({"stage": job.stage,
                        "owner_agent_id": owner_agent_id})))
                opportunity.status = "ready_for_design"
                opportunity.next_action = "由设计Agent按brief_ready阶段继续"
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                job = session.scalar(select(DesignJob).where(
                    DesignJob.product_design_brief_id == brief.id))
                created = False
            return {**self.get_job(job.id), "created": created}

    def start_automation(self, job_id: int, work_runtime: Any) -> dict[str, Any]:
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            if job is None:
                raise BusinessError("设计任务不存在")
            if job.stage in {"design_completed", "discarded", "failed"}:
                raise BusinessError("终态设计任务不能重新启动")
            if job.active_work_thread_id:
                thread = session.get(WorkThread, job.active_work_thread_id)
                if thread and thread.status in {"completed", "failed", "cancelled", "rolled_over"}:
                    run = work_runtime.reopen_thread(
                        thread.id,
                        trigger_kind="design_automation",
                        checkpoint={"design_job_id": job.id, "stage": job.stage},
                    )
                    job.status = "in_progress"
                    job.automation_status = "queued"
                    job.retry_count = 0
                    job.last_error = None
                    job.next_action = "从已持久化的当前设计阶段恢复"
                    session.commit()
                    return {"job": self.get_job(job.id),
                            "work_thread_id": thread.id, "work_run_id": run.id,
                            "created": True, "resumed": True}
                return {
                    "job": self.get_job(job.id),
                    "work_thread_id": job.active_work_thread_id,
                    "created": False,
                }
            owner_agent_id = job.owner_agent_id
            title = job.title

        thread = work_runtime.create_thread(
            owner_agent_id=owner_agent_id,
            title=f"自动设计：{title}",
            objective="根据冻结设计输入形成三条实质不同的创意路线，独立评审并选择唯一方向；未接通图片执行器前不得伪造母稿或生产文件。",
            success_criteria=[
                "形成三条实质不同且结构完整的创意路线",
                "使用隔离评审选择唯一方向并保存理由",
                "所有结果写入设计业务对象并可恢复、防重",
            ],
            constraints=["不得复制受保护IP", "不得伪造图片、效果图或生产文件"],
        )
        run = work_runtime.create_run(
            thread.id,
            trigger_kind="design_automation",
            checkpoint={"design_job_id": job_id, "stage": "brief_ready"},
        )
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            assert job is not None
            job.active_work_thread_id = thread.id
            job.status = "in_progress"
            job.automation_status = "queued"
            job.next_action = "自动生成三条创意路线并进行隔离评审"
            session.add(BusinessObjectEvent(
                object_type="design_job", object_id=job.id,
                event_type="automation_started", actor_kind="system",
                actor_agent_id=job.owner_agent_id,
                reason="创建专用设计自动化工作线程",
                after_json=_dump({"work_thread_id": thread.id, "work_run_id": run.id}),
                work_thread_id=thread.id,
            ))
            session.commit()
        return {"job": self.get_job(job_id), "work_thread_id": thread.id,
                "work_run_id": run.id, "created": True}

    @staticmethod
    def _model_json(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise BusinessError("设计模型没有返回JSON对象")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise BusinessError("设计模型返回的JSON无法解析") from exc
        if not isinstance(value, dict):
            raise BusinessError("设计模型返回格式无效")
        return value

    def execute_text_automation(
        self, job_id: int, codex: Any, image_generator: Any | None = None,
        image_reviewer: Any | None = None,
    ) -> str:
        job_data = self.get_job(job_id)
        snapshot = job_data.get("input_snapshot") or {}
        customer_knowledge = customer_design_knowledge()
        saved_routes = job_data.get("routes") or []
        if len(saved_routes) >= 3 and any(item.get("selected") for item in saved_routes):
            return self._generate_selected_master(job_id, image_generator, image_reviewer)

        routes = [item.get("route") or {} for item in saved_routes[:3]]
        if len(routes) < 3:
            with self.database.sessions() as session:
                job = session.get(DesignJob, job_id)
                assert job is not None
                job.stage = "routes_generating"
                job.status = "in_progress"
                job.automation_status = "running"
                job.started_at = job.started_at or utc_now()
                job.next_action = "生成三条实质不同的创意路线"
                session.commit()

            route_prompt = (
            "你是产品视觉设计师。只返回一个JSON对象，不要Markdown。根据冻结输入形成恰好三条实质不同的创意路线，"
            "不能只是换色。设计对象是完整产品而不是花型：每条路线都要统筹组件配色、整幅视觉作品、"
            "主题元素布局和组件关系。只能使用冻结产品能力中明确声明的工艺；当前产品以CMYK 300 DPI"
            "数码直喷输出被套与枕套的完整画面，不得称为印花，不得虚构刺绣、贴布、提花或绗缝工艺，"
            "禁止生成连续花型。JSON格式：{\"reference_principles\":{\"visual_goal\":\"\","
            "\"composition_discipline\":[],\"copyright_boundaries\":[],\"production_constraints\":[]},"
            "\"routes\":[{\"route_number\":1,\"name\":\"\",\"visual_story\":\"\","
            "\"visual_metaphor\":\"\",\"composition\":\"\",\"graphic_language\":\"\","
            "\"color_strategy\":\"\",\"component_relationship\":\"\",\"commercial_reason\":\"\","
            "\"originality_boundary\":\"\",\"production_risks\":[],\"image_prompt_draft\":\"\"}]}。"
            "必须适配真实组件且遵守avoid与risks。冻结输入："
            + json.dumps(snapshot, ensure_ascii=False)[:20000]
            + "。从真实商品参考中提炼的客户需求知识（只能学习规律，禁止复制具体元素）："
            + json.dumps(customer_knowledge, ensure_ascii=False)[:8000]
        )
            generated = self._model_json(codex.run_text_turn(
                route_prompt, session_key=f"design:{job_id}:routes:v1", timeout=300
            ))
            routes = generated.get("routes")
            if not isinstance(routes, list) or len(routes) != 3:
                raise BusinessError("设计模型必须返回恰好三条创意路线")
            required = {"name", "visual_story", "visual_metaphor", "composition",
                        "graphic_language", "color_strategy", "component_relationship",
                        "commercial_reason", "originality_boundary", "image_prompt_draft"}
            if any(not isinstance(item, dict) or not required.issubset(item) for item in routes):
                raise BusinessError("创意路线缺少必要字段")

            with self.database.sessions() as session:
                job = session.get(DesignJob, job_id)
                assert job is not None
                for index, route in enumerate(routes, 1):
                    existing = session.scalar(select(DesignRoute).where(
                        DesignRoute.design_job_id == job_id,
                        DesignRoute.route_number == index,
                        DesignRoute.version == 1,
                    ))
                    if existing is None:
                        route["route_number"] = index
                        route["reference_principles"] = generated.get("reference_principles") or {}
                        session.add(DesignRoute(design_job_id=job_id, route_number=index,
                            version=1, status="ready_for_review", route_json=_dump(route)))
                job.stage = "routes_reviewing"
                job.next_action = "隔离评审三条创意路线并选择唯一方向"
                session.commit()
        else:
            with self.database.sessions() as session:
                job = session.get(DesignJob, job_id)
                assert job is not None
                job.stage = "routes_reviewing"
                job.status = "in_progress"
                job.automation_status = "running"
                job.next_action = "复用已持久化路线，继续隔离评审"
                session.commit()

        review_prompt = (
            "你是独立设计评审员。你没有参与路线生成。只返回JSON对象，不要Markdown。"
            "从主题表达、原创性、美感、商业吸引力、家纺适配、生产可行性、版权风险七方面比较，"
            "必须选择唯一一条。格式：{\"selected_route_number\":1,\"decision\":\"approved\","
            "\"reason\":\"\",\"comparisons\":[{\"route_number\":1,\"strengths\":[],"
            "\"weaknesses\":[],\"scores\":{\"theme\":1,\"originality\":1,\"aesthetics\":1,"
            "\"commercial\":1,\"bedding_fit\":1,\"production\":1,\"copyright_safety\":1}}],"
            "\"master_requirements\":[]}。路线与冻结输入："
            + json.dumps({"input": snapshot, "routes": routes}, ensure_ascii=False)[:26000]
        )
        review = self._model_json(codex.run_text_turn(
            review_prompt, session_key=f"design:{job_id}:route-review:v1", timeout=300
        ))

        selected_number = int(review.get("selected_route_number") or 0)
        if selected_number not in {1, 2, 3} or review.get("decision") != "approved":
            raise BusinessError("路线评审没有批准唯一方向")

        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            assert job is not None
            rows = session.scalars(select(DesignRoute).where(
                DesignRoute.design_job_id == job_id, DesignRoute.version == 1
            )).all()
            selected_id = None
            for row in rows:
                row.selected = row.route_number == selected_number
                row.status = "approved" if row.selected else "not_selected"
                if row.selected:
                    selected_id = row.id
            if not session.scalar(select(DesignReview.id).where(
                DesignReview.design_job_id == job_id,
                DesignReview.stage == "routes_reviewing",
                DesignReview.target_type == "route_set",
            )):
                session.add(DesignReview(design_job_id=job_id,
                    stage="routes_reviewing", target_type="route_set",
                    target_id=selected_id, reviewer_agent_id=job.owner_agent_id,
                    decision="approved", review_json=_dump(review),
                    work_thread_id=job.active_work_thread_id))
            job.stage = "master_generating"
            job.status = "in_progress"
            job.automation_status = "running"
            job.next_action = "使用设计Agent图片方案生成唯一母稿"
            job.last_error = None
            session.commit()
        return self._generate_selected_master(job_id, image_generator, image_reviewer)

    def _generate_selected_master(
        self, job_id: int, image_generator: Any | None,
        image_reviewer: Any | None = None,
    ) -> str:
        job_data = self.get_job(job_id)
        candidates = [
            item for item in job_data.get("assets") or []
            if item.get("asset_key") == "master-candidate"
        ]
        approved = next((item for item in candidates
                         if item.get("status") == "approved"
                         and (item.get("specification") or {}).get("artifact_contract")
                         == "coordinated_design_board_v1"), None)
        if approved:
            return f"母稿 v{approved['version']} 已通过独立视觉评审。"
        legacy_approved_ids = [
            int(item["id"]) for item in candidates
            if item.get("status") == "approved"
        ]
        if legacy_approved_ids:
            with self.database.sessions() as session:
                job = session.get(DesignJob, job_id)
                assert job is not None
                for asset_id in legacy_approved_ids:
                    asset = session.get(DesignAsset, asset_id)
                    if asset is not None:
                        asset.status = "rework_required"
                        session.add(DesignReview(
                            design_job_id=job_id, stage="master_reviewing",
                            target_type="master_artwork", target_id=asset.id,
                            reviewer_agent_id=job.owner_agent_id,
                            decision="rework_required",
                            review_json=_dump({
                                "decision": "rework_required",
                                "gate": "coordinated_design_board_v1",
                                "problems": ["历史批准结果把完整产品设计错误降维成了连续花型"],
                                "required_changes": [
                                    "根据冻结产品输入重做全部真实组件的视觉总稿，并逐一核对数量、尺寸和工艺"
                                ],
                            }),
                            work_thread_id=job.active_work_thread_id,
                        ))
                session.commit()
            job_data = self.get_job(job_id)
            candidates = [item for item in job_data.get("assets") or []
                          if item.get("asset_key") == "master-candidate"]
        selected = next(
            (item for item in job_data.get("routes") or [] if item.get("selected")),
            None,
        )
        if selected is None:
            raise BusinessError("尚未选定唯一创意路线")
        if image_generator is None:
            with self.database.sessions() as session:
                job = session.get(DesignJob, job_id)
                assert job is not None
                job.stage = "master_generating"
                job.status = "deferred"
                job.automation_status = "deferred"
                job.next_action = "等待配置可执行的图片方案"
                job.last_error = "没有可调用的图片生成执行器"
                session.commit()
            return "已选定唯一创意路线；没有可执行图片通道，安全暂缓。"

        route = selected.get("route") or {}
        prompt = str(route.get("image_prompt_draft") or "").strip()
        if not prompt:
            raise BusinessError("选中路线缺少母稿图片提示词")
        if image_reviewer is None:
            self._defer_job(job_id, "等待配置独立视觉评审模型", "没有可调用的图片评审执行器")
            return "已选定唯一创意路线；没有独立视觉评审通道，安全暂缓。"

        pending = next(
            (item for item in reversed(candidates)
             if item.get("status") == "generated_pending_review"),
            None,
        )
        if pending:
            version = int(pending.get("version") or 0)
            specification = pending.get("specification") or {}
            result = {**specification, "uri": pending.get("uri")}
            with self.database.sessions() as session:
                job = session.get(DesignJob, job_id)
                assert job is not None
                job.stage = "master_reviewing"
                job.status = "in_progress"
                job.automation_status = "running"
                job.next_action = f"继续独立视觉盲评已持久化母稿 v{version}"
                session.commit()
            review = image_reviewer(job_id, result, version)
            decision = self._record_master_review(job_id, int(pending["id"]), review, "恢复评审")
            if decision == "approved":
                return f"母稿 v{version} 已通过独立视觉评审，可进入效果图阶段。"
            job_data = self.get_job(job_id)
            candidates = [item for item in job_data.get("assets") or []
                          if item.get("asset_key") == "master-candidate"]

        next_version = max((int(item.get("version") or 0) for item in candidates), default=0) + 1
        latest_review = next(
            (item for item in reversed(job_data.get("reviews") or [])
             if item.get("target_type") == "master_artwork"
             and item.get("decision") == "rework_required"),
            None,
        )
        if latest_review:
            required = (latest_review.get("review") or {}).get("required_changes") or []
            if required:
                prompt += "\n必须修正上一版问题：" + "；".join(map(str, required))

        for round_number in range(1, 4):
            version = next_version + round_number - 1
            with self.database.sessions() as session:
                job = session.get(DesignJob, job_id)
                assert job is not None
                job.stage = "master_generating"
                job.status = "in_progress"
                job.automation_status = "running"
                job.next_action = f"中转站正在生成母稿 v{version}（本轮 {round_number}/3）"
                session.commit()
            result = image_generator(job_id, prompt, version)
            asset_id = self._persist_master_candidate(job_id, selected, prompt, version, result)
            review = image_reviewer(job_id, result, version)
            decision = self._record_master_review(
                job_id, asset_id, review, f"本轮 {round_number}/3"
            )
            if decision == "approved":
                return f"母稿 v{version} 已通过独立视觉评审，可进入效果图阶段。"
            required = review.get("required_changes") or []
            prompt = str(route.get("image_prompt_draft") or "").strip()
            prompt += "\n上一版已被独立评审淘汰，不得仅做表面微调。必须完成：" + "；".join(map(str, required))

        self._defer_job(
            job_id, "三轮母稿均未通过；保留证据并等待下一轮自主重构",
            "独立视觉评审连续三轮判定返工，未生成效果图",
        )
        return "三轮母稿均未通过独立视觉门禁，已安全暂缓，未进入效果图阶段。"

    def _record_master_review(
        self, job_id: int, asset_id: int, review: dict[str, Any], round_label: str,
    ) -> str:
        decision = str(review.get("decision") or "").strip()
        if (review.get("artifact_type") != "coordinated_design_board"
                or review.get("is_complete_product_design") is not True):
            decision = "rework_required"
            review["decision"] = decision
            review.setdefault("problems", []).append(
                "资产类型门禁失败：输出不是包含完整产品设计的规范设计板"
            )
            review.setdefault("required_changes", []).append(
                "按冻结产品输入呈现全部真实组件、主效果、细节、配色和生产要求；禁止只交花型或只有排版没有产品"
            )
        if decision not in {"approved", "rework_required"}:
            raise BusinessError("独立视觉评审没有返回有效门禁结论")
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            asset = session.get(DesignAsset, asset_id)
            assert job is not None and asset is not None
            asset.status = decision
            session.add(DesignReview(
                design_job_id=job_id, stage="master_reviewing",
                target_type="master_artwork", target_id=asset.id,
                reviewer_agent_id=job.owner_agent_id, decision=decision,
                review_json=_dump(review), work_thread_id=job.active_work_thread_id,
            ))
            if decision == "approved":
                job.stage = "master_approved"
                job.status = "in_progress"
                job.automation_status = "queued"
                job.next_action = "基于批准母稿生成真实产品效果图"
                job.last_error = None
            else:
                job.stage = "master_generating"
                job.status = "in_progress"
                job.automation_status = "running"
                job.next_action = f"根据独立盲评自动返工母稿（{round_label}）"
            session.commit()
        return decision

    def _persist_master_candidate(
        self, job_id: int, selected: dict[str, Any], prompt: str,
        version: int, result: dict[str, Any],
    ) -> int:
        uri = str(result.get("uri") or "").strip()
        content_hash = str(result.get("content_hash") or "").strip()
        if not uri or not content_hash:
            raise BusinessError("图片执行器没有返回持久化文件与哈希")
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            assert job is not None
            asset = session.scalar(select(DesignAsset).where(
                DesignAsset.design_job_id == job_id,
                DesignAsset.asset_key == "master-candidate",
                DesignAsset.version == version,
            ))
            if asset is None:
                asset = DesignAsset(
                    design_job_id=job_id, asset_key="master-candidate",
                    asset_type="master_artwork", version=version,
                    status="generated_pending_review", uri=uri,
                    content_hash=content_hash,
                    specification_json=_dump({
                        "model_id": result.get("model_id"), "model_name": result.get("model_name"),
                        "protocol": result.get("protocol"), "prompt": prompt,
                        "selected_route_id": selected.get("id"), "width": result.get("width"),
                        "height": result.get("height"), "mime_type": result.get("mime_type"),
                        "production_ready": False,
                        "artifact_contract": "coordinated_design_board_v1",
                    }),
                )
                session.add(asset)
                session.flush()
            job.stage = "master_reviewing"
            job.status = "in_progress"
            job.automation_status = "running"
            job.next_action = f"独立视觉模型盲评母稿 v{version}"
            job.last_error = None
            session.commit()
            return asset.id

    def _defer_job(self, job_id: int, next_action: str, error: str) -> None:
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            assert job is not None
            job.status = "deferred"
            job.automation_status = "deferred"
            job.next_action = next_action
            job.last_error = error
            session.commit()

    def mark_automation_retry(self, job_id: int, error: str) -> bool:
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            if job is None:
                return False
            job.last_error = str(error)[:10000]
            job.retry_count += 1
            should_retry = job.retry_count < 3
            job.status = "waiting_retry" if should_retry else "deferred"
            job.automation_status = "waiting_retry" if should_retry else "deferred"
            job.next_action = (
                "官方模型暂时失败，将从当前设计阶段自动重试"
                if should_retry else
                "路线已安全保存；独立评审连续失败三次，等待模型恢复后从评审阶段继续"
            )
            session.commit()
            return should_retry

    def list_inputs(self, *, query: str | None = None, offset: int = 0,
                    limit: int = 15) -> dict[str, Any]:
        """List qualified market outputs which no design job has claimed yet."""
        with self.database.sessions() as session:
            claimed = select(DesignJob.product_opportunity_id)
            stmt = select(ProductOpportunity).where(
                ProductOpportunity.status.in_({"qualified", "ready_for_design"}),
                ~ProductOpportunity.id.in_(claimed),
            )
            if query:
                stmt = stmt.where(ProductOpportunity.title.contains(query.strip()))
            total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            opportunities = session.scalars(
                stmt.order_by(ProductOpportunity.updated_at.desc(), ProductOpportunity.id.desc())
                .offset(max(0, offset)).limit(min(100, max(1, limit)))
            ).all()
            items: list[dict[str, Any]] = []
            for opportunity in opportunities:
                fit = session.get(ProductFitAssessment, opportunity.fit_assessment_id)
                product = session.get(ProductCapability, fit.product_capability_id) if fit else None
                proposition = session.get(ExpressionProposition, fit.proposition_id) if fit else None
                theme = session.get(BusinessContentTheme, proposition.theme_id) if proposition else None
                items.append({
                    "id": opportunity.id,
                    "title": opportunity.title,
                    "status": opportunity.status,
                    "next_action": opportunity.next_action,
                    "core_expression": opportunity.core_expression,
                    "purchase_reason": opportunity.purchase_reason,
                    "risks": _load(opportunity.risks_json, []),
                    "theme_title": theme.title if theme else None,
                    "proposition_title": proposition.title if proposition else None,
                    "theme_form": proposition.theme_form if proposition else None,
                    "creative_seeds": _load(proposition.creative_seeds_json, []) if proposition else [],
                    "product_name": product.name if product else None,
                    "product_type": product.product_type if product else None,
                    "updated_at": _iso(opportunity.updated_at),
                })
            return {"items": items, "total": int(total)}

    def list_jobs(self, *, owner_agent_id: int | None = None, status: str | None = None,
                  stage: str | None = None, query: str | None = None,
                  offset: int = 0, limit: int = 15) -> dict[str, Any]:
        with self.database.sessions() as session:
            stmt = select(DesignJob)
            if owner_agent_id is not None: stmt = stmt.where(DesignJob.owner_agent_id == owner_agent_id)
            if status: stmt = stmt.where(DesignJob.status == status)
            if stage: stmt = stmt.where(DesignJob.stage == stage)
            if query: stmt = stmt.where(DesignJob.title.contains(query.strip()))
            total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            rows = session.scalars(stmt.order_by(DesignJob.priority.desc(), DesignJob.updated_at.desc())
                .offset(max(0, offset)).limit(min(100, max(1, limit)))).all()
            return {"items": [self._serialize_job(row) for row in rows], "total": int(total)}

    def get_job(self, job_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            if job is None: raise BusinessError("设计任务不存在")
            result = self._serialize_job(job)
            snapshot = session.scalar(select(DesignInputSnapshot).where(
                DesignInputSnapshot.design_job_id == job.id).order_by(DesignInputSnapshot.version.desc()))
            result["input_snapshot"] = _load(snapshot.snapshot_json, {}) if snapshot else None
            result["routes"] = [{"id": x.id, "route_number": x.route_number, "version": x.version,
                "status": x.status, "selected": x.selected, "route": _load(x.route_json, {})}
                for x in session.scalars(select(DesignRoute).where(DesignRoute.design_job_id == job.id)
                    .order_by(DesignRoute.route_number, DesignRoute.version)).all()]
            result["assets"] = [{"id": x.id, "asset_key": x.asset_key, "asset_type": x.asset_type,
                "version": x.version, "status": x.status, "source_asset_id": x.source_asset_id,
                "uri": x.uri, "specification": _load(x.specification_json, {})}
                for x in session.scalars(select(DesignAsset).where(DesignAsset.design_job_id == job.id)
                    .order_by(DesignAsset.id)).all()]
            result["reviews"] = [{"id": x.id, "stage": x.stage, "target_type": x.target_type,
                "target_id": x.target_id, "decision": x.decision, "review": _load(x.review_json, {}),
                "created_at": _iso(x.created_at)} for x in session.scalars(select(DesignReview).where(
                    DesignReview.design_job_id == job.id).order_by(DesignReview.id)).all()]
            result["components"] = [{"id": x.id, "component_id": x.component_id,
                "version": x.version, "specification": _load(x.specification_json, {})}
                for x in session.scalars(select(DesignComponentSpec).where(
                    DesignComponentSpec.design_job_id == job.id).order_by(DesignComponentSpec.id)).all()]
            return result

    def transition(self, job_id: int, values: dict[str, Any]) -> dict[str, Any]:
        stage = str(values.get("stage") or "").strip()
        if stage not in set(DESIGN_STAGES) | DESIGN_BRANCH_STAGES:
            raise BusinessError("设计阶段无效")
        reason = str(values.get("reason") or "").strip()
        if not reason: raise BusinessError("状态变更必须填写原因")
        with self.database.sessions() as session:
            job = session.get(DesignJob, job_id)
            if job is None: raise BusinessError("设计任务不存在")
            before = self._serialize_job(job)
            job.stage = stage
            job.status = str(values.get("status") or job.status)
            job.automation_status = str(values.get("automation_status") or job.automation_status)
            job.next_action = values.get("next_action")
            job.last_error = values.get("last_error")
            if stage == "design_completed":
                job.status, job.automation_status, job.finished_at = "completed", "completed", utc_now()
            session.add(BusinessObjectEvent(object_type="design_job", object_id=job.id,
                event_type="stage_changed", actor_kind=str(values.get("actor_kind") or "agent"),
                actor_agent_id=values.get("actor_agent_id"), reason=reason,
                before_json=_dump(before), after_json=_dump({"stage": job.stage, "status": job.status,
                    "automation_status": job.automation_status, "next_action": job.next_action})))
            session.commit()
            return self.get_job(job.id)

    @staticmethod
    def _serialize_job(job: DesignJob) -> dict[str, Any]:
        return {"id": job.id, "product_design_brief_id": job.product_design_brief_id,
            "product_opportunity_id": job.product_opportunity_id, "owner_agent_id": job.owner_agent_id,
            "title": job.title, "stage": job.stage, "status": job.status,
            "automation_status": job.automation_status, "active_work_thread_id": job.active_work_thread_id,
            "next_action": job.next_action, "last_error": job.last_error,
            "retry_count": job.retry_count, "priority": job.priority,
            "started_at": _iso(job.started_at), "finished_at": _iso(job.finished_at),
            "created_at": _iso(job.created_at), "updated_at": _iso(job.updated_at)}
