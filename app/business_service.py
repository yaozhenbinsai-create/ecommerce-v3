from __future__ import annotations

from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
import re
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from sqlalchemy import delete, func, select

from .database import Database
from .models import (
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
    ProductionRule,
    ResearchCandidate,
    WorkThread,
    utc_now,
)


class BusinessError(ValueError):
    pass


def _json_load(value: str | None, default: Any) -> Any:
    try:
        parsed = json.loads(value or "")
    except (TypeError, ValueError):
        return default
    return parsed


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _clean_text(value: Any, limit: int = 10000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _normalize_creative_seeds(values: Any) -> list[dict[str, str]]:
    seeds: list[dict[str, str]] = []
    for item in (values or [])[:50]:
        if not isinstance(item, dict):
            continue
        content = _clean_text(item.get("content"), 1000)
        if not content:
            continue
        seeds.append(
            {
                "type": _clean_text(item.get("type"), 100) or "其他",
                "content": content,
                "source_relation": _clean_text(item.get("source_relation"), 2000),
                "creative_potential": _clean_text(item.get("creative_potential"), 2000),
            }
        )
    return seeds


def _fingerprint(*values: Any) -> str:
    normalized = [re.sub(r"\s+", "", str(value or "")).lower() for value in values]
    return sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _as_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _external_source_refs(values: list[Any]) -> list[str]:
    """Return real external sources; execution-ledger refs are trace metadata."""
    return sorted(
        {
            str(value).strip()
            for value in values
            if str(value).strip().lower().startswith(("http://", "https://"))
        }
    )


def _source_publisher(source_url: str) -> str | None:
    hostname = (urlparse(source_url).hostname or "").lower()
    return hostname.removeprefix("www.")[:300] or None


def _published_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)
    except ValueError:
        match = re.search(r"\b(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", text)
        if not match:
            return None
        try:
            return datetime(*(int(item) for item in match.groups()), tzinfo=timezone.utc)
        except ValueError:
            return None


def _hotspot_evidence_quality(
    source_refs: list[str], evidence_by_url: dict[str, Evidence]
) -> tuple[bool, list[str]]:
    """Require corroboration and freshness before downstream business work."""
    evidence = [evidence_by_url[url] for url in source_refs if url in evidence_by_url]
    publishers = {
        str(item.source_publisher or _source_publisher(item.source_url) or "").lower()
        for item in evidence
        if str(item.source_publisher or _source_publisher(item.source_url) or "").strip()
    }
    cutoff = utc_now() - timedelta(days=180)
    recent = [item for item in evidence if (_published_datetime(item.published_at) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    gaps: list[str] = []
    if len(publishers) < 2:
        gaps.append("至少需要两个独立发布主体交叉支持")
    if not recent:
        gaps.append("至少需要一个发布日期明确且在180天内的来源")
    return not gaps, gaps


def _opportunity_state(conclusion: str, missing_information: list[Any]) -> tuple[str, str]:
    """Derive status and next action from one source of truth."""
    if missing_information:
        return "validating", "验证产品适配中的缺失信息"
    if conclusion == "fit":
        return "qualified", "产品适配已通过，等待未来设计阶段启用"
    return "candidate", "产品通用参数已补齐；市场验证暂缓，保留为候选机会"


def _actionable_product_fit_missing_information(values: list[Any]) -> list[str]:
    """Keep only gaps that belong to product fit in the current phase."""
    deferred_markers = (
        "消费者", "购买意愿", "购买证据", "市场验证", "市场接受", "搜索语言",
        "用户反馈", "使用反馈", "目标人群", "偏好", "母稿", "构图", "设计",
        "视觉", "可读性", "安全区", "组件关系", "关系层次", "层次关系",
        "主次", "构成关系", "画面关系", "元素关系", "排布", "布局", "比例",
        "图案尺度", "方向关系", "正反面关系", "被套与枕套", "床单与枕套",
    )
    result: list[str] = []
    for value in values:
        clean = _clean_text(value, 1000)
        if not clean or any(marker in clean for marker in deferred_markers):
            continue
        if clean not in result:
            result.append(clean)
    return result


HOTSPOT_LIFECYCLE = {"emerging", "active", "cooling", "expired", "evergreen"}
HOTSPOT_RESEARCH = {
    "discovered", "collecting", "verifying", "verified",
    "insufficient_evidence", "rejected",
}
HOTSPOT_STAGES = {
    "unprocessed", "theme_formed", "proposition_formed", "matching_products",
    "opportunity_created", "not_applicable", "ready_for_design", "designed",
    "production_ready",
}
AUTOMATION_STATUSES = {
    "queued", "running", "waiting_schedule", "retrying", "completed", "failed",
}
THEME_STATUSES = {
    "discovered", "researching", "active", "expression_ready", "watching",
    "insufficient_evidence", "expired", "archived",
}
PROPOSITION_STATUSES = {"draft", "reviewing", "qualified", "watching", "rejected"}
FIT_CONCLUSIONS = {"fit", "possible", "not_fit", "needs_validation"}
OPPORTUNITY_STATUSES = {
    "candidate", "validating", "qualified", "ready_for_design", "watching",
    "insufficient_evidence", "not_applicable", "rejected", "expired",
}


class BusinessService:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _event(
        session: Any,
        *,
        object_type: str,
        object_id: int,
        event_type: str,
        actor_kind: str = "system",
        actor_agent_id: int | None = None,
        reason: str | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        work_thread_id: int | None = None,
    ) -> None:
        session.add(
            BusinessObjectEvent(
                object_type=object_type,
                object_id=object_id,
                event_type=event_type,
                actor_kind=actor_kind,
                actor_agent_id=actor_agent_id,
                reason=_clean_text(reason, 10000) or None,
                before_json=_json_dump(before or {}),
                after_json=_json_dump(after or {}),
                work_thread_id=work_thread_id,
            )
        )

    @staticmethod
    def serialize_hotspot(row: BusinessHotspot) -> dict[str, Any]:
        return {
            "id": row.id,
            "owner_agent_id": row.owner_agent_id,
            "title": row.title,
            "summary": row.summary,
            "lifecycle_status": row.lifecycle_status,
            "research_status": row.research_status,
            "business_stage": row.business_stage,
            "automation_status": row.automation_status,
            "next_action": row.next_action,
            "missing_information": _json_load(row.missing_information_json, []),
            "tags": _json_load(row.tags_json, []),
            "active_work_thread_id": row.active_work_thread_id,
            "first_seen_at": _as_iso(row.first_seen_at),
            "last_activity_at": _as_iso(row.last_activity_at),
            "next_review_at": _as_iso(row.next_review_at),
            "archived_at": _as_iso(row.archived_at),
            "created_at": _as_iso(row.created_at),
            "updated_at": _as_iso(row.updated_at),
        }

    @staticmethod
    def serialize_product(row: ProductCapability, components: list[ProductComponent]) -> dict[str, Any]:
        return {
            "id": row.id,
            "capability_key": row.capability_key,
            "version": row.version,
            "name": row.name,
            "product_type": row.product_type,
            "is_enabled": row.is_enabled,
            "currency": row.currency,
            "cost_min": row.cost_min,
            "cost_max": row.cost_max,
            "retail_min": row.retail_min,
            "retail_max": row.retail_max,
            "minimum_order_quantity": row.minimum_order_quantity,
            "lead_time_days": row.lead_time_days,
            "materials": _json_load(row.materials_json, []),
            "processes": _json_load(row.processes_json, []),
            "packaging": _json_load(row.packaging_json, []),
            "constraints": _json_load(row.constraints_json, []),
            "quality_requirements": _json_load(row.quality_requirements_json, []),
            "effective_at": _as_iso(row.effective_at),
            "superseded_at": _as_iso(row.superseded_at),
            "components": [
                {
                    "id": item.id,
                    "name": item.name,
                    "quantity": item.quantity,
                    "sizes": _json_load(item.sizes_json, []),
                    "materials": _json_load(item.materials_json, []),
                    "supported_pattern_modes": _json_load(item.supported_pattern_modes_json, []),
                    "supports_ab_sides": item.supports_ab_sides,
                    "constraints": _json_load(item.constraints_json, []),
                    "position": item.position,
                }
                for item in sorted(components, key=lambda value: (value.position, value.id))
            ],
        }

    def ingest_market_insight(self, insight_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            insight = session.get(MarketInsightRecord, insight_id)
            if insight is None:
                raise BusinessError("市场洞察不存在")
            data = _json_load(insight.insight_json, {})
            raw_source_refs = [
                str(item).strip()
                for item in _json_load(insight.source_refs_json, [])
                if str(item).strip()
            ]
            source_refs = _external_source_refs(raw_source_refs)
            evidence_rows = session.scalars(
                select(Evidence).where(Evidence.work_thread_id == insight.work_thread_id)
            ).all()
            evidence_by_url = {row.source_url: row for row in evidence_rows}
            evidence_verified, evidence_gaps = _hotspot_evidence_quality(
                source_refs, evidence_by_url
            )
            hotspot_fp = _fingerprint(insight.title, insight.category or "", *source_refs)
            existing_link = session.scalar(
                select(BusinessContentSource).where(
                    BusinessContentSource.market_insight_id == insight.id
                ).order_by(BusinessContentSource.id)
            )
            hotspot = (
                session.get(BusinessHotspot, existing_link.hotspot_id)
                if existing_link is not None
                else session.scalar(
                    select(BusinessHotspot).where(
                        BusinessHotspot.fingerprint == hotspot_fp
                    )
                )
            )
            created = hotspot is None
            if hotspot is None:
                hotspot = BusinessHotspot(
                    owner_agent_id=insight.owner_agent_id,
                    title=insight.title,
                    summary=_clean_text(data.get("phenomenon") or data.get("signal") or data.get("why_interesting"), 10000),
                    fingerprint=hotspot_fp,
                    lifecycle_status="active",
                    research_status="verified" if evidence_verified else "insufficient_evidence",
                    business_stage="theme_formed" if evidence_verified else "unprocessed",
                    automation_status="completed" if evidence_verified else "queued",
                    next_action=(
                        "基于已核验主题形成产品表达命题"
                        if evidence_verified
                        else "；".join(evidence_gaps) or "补充可核验来源"
                    ),
                    missing_information_json=_json_dump(evidence_gaps),
                    tags_json=_json_dump([value for value in [insight.category] if value]),
                    active_work_thread_id=insight.work_thread_id,
                    first_seen_at=insight.created_at,
                    last_activity_at=utc_now(),
                )
                session.add(hotspot)
                session.flush()
            else:
                conflicting_hotspot = session.scalar(
                    select(BusinessHotspot.id).where(
                        BusinessHotspot.fingerprint == hotspot_fp,
                        BusinessHotspot.id != hotspot.id,
                    )
                )
                if conflicting_hotspot is None:
                    hotspot.fingerprint = hotspot_fp
                hotspot.last_activity_at = utc_now()
                hotspot.active_work_thread_id = insight.work_thread_id
                if evidence_verified:
                    hotspot.research_status = "verified"
                    hotspot.missing_information_json = "[]"
                    hotspot.automation_status = "completed"
                    hotspot.next_action = "基于已核验主题形成产品表达命题"
                elif hotspot.business_stage in {"unprocessed", "theme_formed"}:
                    hotspot.research_status = "insufficient_evidence"
                    hotspot.missing_information_json = _json_dump(evidence_gaps)
                    hotspot.automation_status = "queued"
                    hotspot.next_action = "；".join(evidence_gaps) or "补充可核验来源"
                if evidence_verified and hotspot.business_stage == "unprocessed":
                    hotspot.business_stage = "theme_formed"

            internal_sources = session.scalars(
                select(BusinessContentSource).where(
                    BusinessContentSource.hotspot_id == hotspot.id,
                    ~BusinessContentSource.source_url.startswith("http://"),
                    ~BusinessContentSource.source_url.startswith("https://"),
                )
            ).all()
            for internal_source in internal_sources:
                session.delete(internal_source)
            if internal_sources:
                session.flush()

            sources: list[BusinessContentSource] = []
            for source_url in source_refs:
                candidate = session.scalar(
                    select(ResearchCandidate).where(
                        ResearchCandidate.work_thread_id == insight.work_thread_id,
                        ResearchCandidate.canonical_url == source_url,
                    )
                )
                evidence = evidence_by_url.get(source_url)
                if evidence is None:
                    claim = _clean_text(
                        data.get("concrete_event")
                        or data.get("phenomenon")
                        or data.get("signal")
                        or insight.title,
                        10000,
                    )
                    excerpt = _clean_text(
                        (candidate.source_text if candidate else None)
                        or (candidate.snippet if candidate else None)
                        or data.get("concrete_event")
                        or data.get("phenomenon")
                        or data.get("signal"),
                        10000,
                    )
                    source_type = _clean_text(
                        candidate.content_kind if candidate else "web_page", 30
                    ) or "web_page"
                    evidence = Evidence(
                        work_thread_id=insight.work_thread_id,
                        owner_agent_id=insight.owner_agent_id,
                        source_url=source_url,
                        source_title=_clean_text(
                            candidate.title if candidate else insight.title, 1000
                        ),
                        claim=claim,
                        excerpt=excerpt,
                        confidence=str(data.get("confidence") or "medium"),
                        stance="supports",
                        source_type=source_type,
                        source_publisher=_source_publisher(source_url),
                        published_at=None,
                        verification_notes=(
                            "由已保存市场洞察关联的外部正文自动建立追溯记录；"
                            "尚未执行独立来源交叉复核。"
                        ),
                        review_status="unreviewed",
                        content_hash=_fingerprint(source_url, claim, excerpt),
                    )
                    session.add(evidence)
                    session.flush()
                    evidence_by_url[source_url] = evidence
                url_hash = _fingerprint(source_url)
                source = session.scalar(
                    select(BusinessContentSource).where(
                        BusinessContentSource.hotspot_id == hotspot.id,
                        BusinessContentSource.source_url_hash == url_hash,
                    )
                )
                if source is None:
                    source = BusinessContentSource(
                        hotspot_id=hotspot.id,
                        owner_agent_id=insight.owner_agent_id,
                        evidence_id=evidence.id if evidence else None,
                        market_insight_id=insight.id,
                        source_url=source_url,
                        source_url_hash=url_hash,
                        title=(evidence.source_title if evidence else insight.title)[:1000],
                        publisher=evidence.source_publisher if evidence else None,
                        source_type=evidence.source_type if evidence else "web_page",
                        published_at=evidence.published_at if evidence else None,
                        excerpt=evidence.excerpt if evidence else "",
                    )
                    session.add(source)
                    session.flush()
                else:
                    source.evidence_id = evidence.id if evidence else None
                    source.market_insight_id = insight.id
                    source.title = (
                        evidence.source_title if evidence else source.title or insight.title
                    )[:1000]
                    source.publisher = (
                        evidence.source_publisher if evidence else source.publisher
                    )
                    source.source_type = (
                        evidence.source_type if evidence else source.source_type or "web_page"
                    )
                    source.published_at = (
                        evidence.published_at if evidence else source.published_at
                    )
                    source.excerpt = (
                        evidence.excerpt if evidence else source.excerpt
                    )
                sources.append(source)

            facts: list[tuple[str, str, BusinessContentSource | None]] = []
            for key, fact_type in (
                ("concrete_event", "concrete_event"),
                ("signal", "market_signal"),
                ("phenomenon", "market_observation"),
                ("why_interesting", "interpretive_context"),
            ):
                value = _clean_text(data.get(key), 10000)
                if value:
                    facts.append((value, fact_type, sources[0] if sources else None))
            for value in list(data.get("observed_discussion_signals") or data.get("discussion_signals") or []):
                cleaned = _clean_text(value, 3000)
                if cleaned:
                    facts.append((cleaned, "public_discussion", sources[0] if sources else None))
            for evidence in evidence_rows:
                if evidence.source_url not in source_refs:
                    continue
                source = next((item for item in sources if item.source_url == evidence.source_url), None)
                facts.append((_clean_text(evidence.claim, 10000), "evidence_claim", source))
            for statement, fact_type, source in facts:
                fact_fp = _fingerprint(statement)
                existing_fact = session.scalar(
                    select(BusinessContentFact).where(
                        BusinessContentFact.hotspot_id == hotspot.id,
                        BusinessContentFact.fingerprint == fact_fp,
                    )
                )
                if existing_fact is None:
                    session.add(
                        BusinessContentFact(
                            hotspot_id=hotspot.id,
                            source_id=source.id if source else None,
                            owner_agent_id=insight.owner_agent_id,
                            statement=statement,
                            fact_type=fact_type,
                            confidence=str(data.get("confidence") or "medium"),
                            fingerprint=fact_fp,
                        )
                    )

            creative_directions = [
                item for item in list(data.get("creative_theme_directions") or [])
                if isinstance(item, dict) and _clean_text(item.get("title"), 500)
            ]
            if not creative_directions:
                creative_directions = [
                    {
                        "title": insight.title,
                        "inspiration_path": data.get("why_interesting") or data.get("phenomenon"),
                        "human_meaning": data.get("phenomenon") or data.get("signal"),
                        "core_tension": data.get("why_interesting"),
                        "people": data.get("participant_groups") or [],
                        "scenes": data.get("participation_behaviors") or [],
                    }
                ]
            for direction in creative_directions[:8]:
                theme_title = _clean_text(direction.get("title"), 500)
                theme_fp = _fingerprint(
                    hotspot.id,
                    theme_title,
                    direction.get("human_meaning"),
                    direction.get("inspiration_path"),
                )
                theme = session.scalar(
                    select(BusinessContentTheme).where(
                        BusinessContentTheme.hotspot_id == hotspot.id,
                        BusinessContentTheme.fingerprint == theme_fp,
                    )
                )
                if theme is None:
                    signals = list(
                        data.get("observed_discussion_signals")
                        or data.get("discussion_signals")
                        or []
                    )
                    symbols = [
                        _clean_text(item, 500)
                        for item in list(direction.get("symbols") or [])
                        if _clean_text(item, 500)
                    ]
                    sensitivity = [
                        _clean_text(item, 1000)
                        for item in list(direction.get("sensitivity_notes") or [])
                        if _clean_text(item, 1000)
                    ]
                    theme = BusinessContentTheme(
                        hotspot_id=hotspot.id,
                        owner_agent_id=insight.owner_agent_id,
                        title=theme_title,
                        narrative=_clean_text(
                            direction.get("human_meaning")
                            or data.get("phenomenon")
                            or data.get("signal"),
                            10000,
                        ) or theme_title,
                        core_tension=_clean_text(
                            direction.get("core_tension")
                            or direction.get("inspiration_path")
                            or data.get("why_interesting"),
                            5000,
                        ) or None,
                        emotions_json=_json_dump(direction.get("emotions") or []),
                        people_json=_json_dump(
                            direction.get("people")
                            or data.get("participant_groups")
                            or []
                        ),
                        scenes_json=_json_dump(
                            direction.get("scenes")
                            or data.get("participation_behaviors")
                            or []
                        ),
                        signals_json=_json_dump(
                            [
                                *signals,
                                *(["联想符号：" + "、".join(symbols)] if symbols else []),
                                *(["敏感性边界：" + "；".join(sensitivity)] if sensitivity else []),
                            ]
                        ),
                        status="active" if evidence_verified else "insufficient_evidence",
                        next_action=(
                            "形成不预设视觉答案的产品表达命题"
                            if evidence_verified
                            else "先补齐热点来源新鲜度与独立交叉证据"
                        ),
                        fingerprint=theme_fp,
                    )
                    session.add(theme)
                    session.flush()
            if created:
                self._event(
                    session,
                    object_type="hotspot",
                    object_id=hotspot.id,
                    event_type="created_from_market_insight",
                    actor_agent_id=insight.owner_agent_id,
                    after=self.serialize_hotspot(hotspot),
                    work_thread_id=insight.work_thread_id,
                )
            session.commit()
            return self.get_hotspot(hotspot.id)

    def ingest_market_insights(self, insight_ids: list[int]) -> list[int]:
        hotspot_ids: list[int] = []
        for insight_id in insight_ids:
            result = self.ingest_market_insight(int(insight_id))
            hotspot_ids.append(int(result["id"]))
        return hotspot_ids

    def list_hotspots(
        self,
        *,
        owner_agent_id: int | None = None,
        research_status: str | None = None,
        business_stage: str | None = None,
        work_thread_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self.database.sessions() as session:
            query = select(BusinessHotspot).where(BusinessHotspot.archived_at.is_(None))
            if owner_agent_id is not None:
                query = query.where(BusinessHotspot.owner_agent_id == owner_agent_id)
            if research_status:
                query = query.where(BusinessHotspot.research_status == research_status)
            if business_stage:
                query = query.where(BusinessHotspot.business_stage == business_stage)
            if work_thread_id is not None:
                query = query.where(
                    BusinessHotspot.active_work_thread_id.in_(
                        self._work_scope_thread_ids(session, work_thread_id)
                    )
                )
            rows = session.scalars(
                query.order_by(BusinessHotspot.last_activity_at.desc(), BusinessHotspot.id.desc())
                .offset(max(0, offset)).limit(min(200, max(1, limit)))
            ).all()
            return [self.serialize_hotspot(row) for row in rows]

    def count_hotspots(self, *, owner_agent_id: int | None = None) -> int:
        with self.database.sessions() as session:
            query = select(func.count(BusinessHotspot.id)).where(
                BusinessHotspot.archived_at.is_(None)
            )
            if owner_agent_id is not None:
                query = query.where(BusinessHotspot.owner_agent_id == owner_agent_id)
            return int(session.scalar(query) or 0)

    def list_business_backlog(
        self,
        *,
        owner_agent_id: int,
        work_thread_id: int | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return the next durable business work that research must not starve."""
        with self.database.sessions() as session:
            scoped_thread_ids: list[int] | None = None
            if work_thread_id is not None:
                current_thread = session.get(WorkThread, work_thread_id)
                if current_thread and current_thread.project_id is not None:
                    scoped_thread_ids = list(
                        session.scalars(
                            select(WorkThread.id).where(
                                WorkThread.project_id == current_thread.project_id
                            )
                        ).all()
                    )
                else:
                    scoped_thread_ids = [work_thread_id]
            theme_query = (
                select(BusinessContentTheme, BusinessHotspot)
                .join(BusinessHotspot, BusinessHotspot.id == BusinessContentTheme.hotspot_id)
                .outerjoin(
                    ExpressionProposition,
                    ExpressionProposition.theme_id == BusinessContentTheme.id,
                )
                .where(
                    BusinessContentTheme.owner_agent_id == owner_agent_id,
                    BusinessHotspot.research_status == "verified",
                    BusinessContentTheme.status.in_({"active", "researching"}),
                    ExpressionProposition.id.is_(None),
                )
            )
            proposition_query = (
                select(ExpressionProposition, BusinessContentTheme, BusinessHotspot)
                .join(BusinessContentTheme, BusinessContentTheme.id == ExpressionProposition.theme_id)
                .join(BusinessHotspot, BusinessHotspot.id == BusinessContentTheme.hotspot_id)
                .outerjoin(
                    ProductFitAssessment,
                    ProductFitAssessment.proposition_id == ExpressionProposition.id,
                )
                .where(
                    ExpressionProposition.owner_agent_id == owner_agent_id,
                    BusinessHotspot.research_status == "verified",
                    ExpressionProposition.status.in_(
                        {"draft", "qualified", "reviewing", "watching"}
                    ),
                    ProductFitAssessment.id.is_(None),
                )
            )
            if scoped_thread_ids is not None:
                theme_query = theme_query.where(
                    BusinessHotspot.active_work_thread_id.in_(scoped_thread_ids)
                )
                proposition_query = proposition_query.where(
                    BusinessHotspot.active_work_thread_id.in_(scoped_thread_ids)
                )
            themes_waiting = int(
                session.scalar(select(func.count()).select_from(theme_query.subquery()))
                or 0
            )
            propositions_waiting = int(
                session.scalar(
                    select(func.count()).select_from(proposition_query.subquery())
                )
                or 0
            )
            theme_rows = session.execute(
                theme_query
                .order_by(BusinessContentTheme.created_at, BusinessContentTheme.id)
                .limit(max(1, min(50, limit)))
            ).all()
            proposition_rows = session.execute(
                proposition_query
                .order_by(ExpressionProposition.created_at, ExpressionProposition.id)
                .limit(max(1, min(50, limit)))
            ).all()
            products = session.scalars(
                select(ProductCapability)
                .where(ProductCapability.is_enabled.is_(True))
                .order_by(ProductCapability.product_type, ProductCapability.id)
            ).all()
            product_components = session.scalars(
                select(ProductComponent).where(
                    ProductComponent.product_capability_id.in_(
                        [row.id for row in products]
                    )
                )
            ).all() if products else []
            components_by_product: dict[int, list[ProductComponent]] = {}
            for component in product_components:
                components_by_product.setdefault(
                    component.product_capability_id, []
                ).append(component)
            themes = [
                {
                    "hotspot_id": hotspot.id,
                    "hotspot_title": hotspot.title,
                    "theme_id": theme.id,
                    "theme_title": theme.title,
                    "narrative": theme.narrative,
                    "core_tension": theme.core_tension,
                    "emotions": _json_load(theme.emotions_json, []),
                    "people": _json_load(theme.people_json, []),
                    "scenes": _json_load(theme.scenes_json, []),
                    "signals": _json_load(theme.signals_json, []),
                    "next_action": theme.next_action,
                }
                for theme, hotspot in theme_rows
            ]
            propositions = [
                {
                    "hotspot_id": hotspot.id,
                    "theme_id": theme.id,
                    "theme_title": theme.title,
                    "proposition_id": proposition.id,
                    "title": proposition.title,
                    "core_expression": proposition.core_expression,
                    "target_emotion": proposition.target_emotion,
                    "audience": _json_load(proposition.audience_json, []),
                    "scenes": _json_load(proposition.scenes_json, []),
                    "purchase_reason": proposition.purchase_reason,
                    "risks": _json_load(proposition.risks_json, []),
                }
                for proposition, theme, hotspot in proposition_rows
            ]
            recent_query = (
                select(ExpressionProposition, BusinessContentTheme)
                .join(
                    BusinessContentTheme,
                    BusinessContentTheme.id == ExpressionProposition.theme_id,
                )
                .join(
                    BusinessHotspot,
                    BusinessHotspot.id == BusinessContentTheme.hotspot_id,
                )
                .where(ExpressionProposition.owner_agent_id == owner_agent_id)
            )
            if scoped_thread_ids is not None:
                recent_query = recent_query.where(
                    BusinessHotspot.active_work_thread_id.in_(scoped_thread_ids)
                )
            recent_rows = session.execute(
                recent_query.order_by(
                    ExpressionProposition.created_at.desc(),
                    ExpressionProposition.id.desc(),
                ).limit(8)
            ).all()
            return {
                "stage": (
                    # Finish the proposition already in flight before opening
                    # another theme.  This gives each object a complete
                    # expression -> product-fit transaction and prevents an
                    # unprocessed sibling theme from starving opportunities.
                    "product_fit"
                    if propositions
                    else "expression_proposition"
                    if themes
                    else "research"
                ),
                "themes_waiting": themes_waiting,
                "propositions_waiting": propositions_waiting,
                "themes": themes,
                "propositions": propositions,
                "recent_expression_contexts": [
                    {
                        "proposition_id": proposition.id,
                        "theme_title": theme.title,
                        "title": proposition.title,
                        "theme_form": proposition.theme_form,
                        "creative_hooks": [
                            {
                                "type": item.get("type"),
                                "content": item.get("content"),
                            }
                            for item in _json_load(
                                proposition.creative_seeds_json, []
                            )[:4]
                            if isinstance(item, dict)
                        ],
                    }
                    for proposition, theme in recent_rows
                ],
                "enabled_products": [
                    self.serialize_product(
                        row, components_by_product.get(row.id, [])
                    )
                    for row in products
                ],
            }

    def get_hotspot(
        self,
        hotspot_id: int,
        *,
        owner_agent_id: int | None = None,
        work_thread_id: int | None = None,
    ) -> dict[str, Any]:
        with self.database.sessions() as session:
            hotspot = session.get(BusinessHotspot, hotspot_id)
            if hotspot is None:
                raise BusinessError("热点不存在")
            if owner_agent_id is not None and hotspot.owner_agent_id != owner_agent_id:
                raise BusinessError("热点不属于当前Agent")
            if work_thread_id is not None and hotspot.active_work_thread_id not in set(
                self._work_scope_thread_ids(session, work_thread_id)
            ):
                raise BusinessError("热点不属于当前项目")
            sources = session.scalars(select(BusinessContentSource).where(BusinessContentSource.hotspot_id == hotspot_id).order_by(BusinessContentSource.id)).all()
            facts = session.scalars(select(BusinessContentFact).where(BusinessContentFact.hotspot_id == hotspot_id).order_by(BusinessContentFact.id)).all()
            themes = session.scalars(select(BusinessContentTheme).where(BusinessContentTheme.hotspot_id == hotspot_id).order_by(BusinessContentTheme.id)).all()
            events = session.scalars(select(BusinessObjectEvent).where(BusinessObjectEvent.object_type == "hotspot", BusinessObjectEvent.object_id == hotspot_id).order_by(BusinessObjectEvent.id.desc()).limit(100)).all()
            result = self.serialize_hotspot(hotspot)
            result.update(
                {
                    "sources": [
                        {"id": row.id, "evidence_id": row.evidence_id, "market_insight_id": row.market_insight_id, "source_url": row.source_url, "title": row.title, "publisher": row.publisher, "source_type": row.source_type, "published_at": row.published_at, "excerpt": row.excerpt, "access_status": row.access_status}
                        for row in sources
                    ],
                    "facts": [
                        {"id": row.id, "source_id": row.source_id, "statement": row.statement, "fact_type": row.fact_type, "stance": row.stance, "confidence": row.confidence, "uncertainty": row.uncertainty}
                        for row in facts
                    ],
                    "themes": [
                        {"id": row.id, "title": row.title, "narrative": row.narrative, "core_tension": row.core_tension, "emotions": _json_load(row.emotions_json, []), "people": _json_load(row.people_json, []), "scenes": _json_load(row.scenes_json, []), "signals": _json_load(row.signals_json, []), "status": row.status, "next_action": row.next_action}
                        for row in themes
                    ],
                    "events": [
                        {"id": row.id, "event_type": row.event_type, "actor_kind": row.actor_kind, "actor_agent_id": row.actor_agent_id, "reason": row.reason, "before": _json_load(row.before_json, {}), "after": _json_load(row.after_json, {}), "work_thread_id": row.work_thread_id, "created_at": _as_iso(row.created_at)}
                        for row in events
                    ],
                }
            )
            return result

    def create_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        components = list(payload.pop("components", []) or [])
        with self.database.sessions() as session:
            key = _clean_text(payload.get("capability_key"), 120) or f"product-{uuid4().hex[:12]}"
            if session.scalar(select(ProductCapability).where(ProductCapability.capability_key == key)):
                raise BusinessError("产品能力标识已经存在，请创建新版本")
            product = self._build_product(key, 1, payload)
            session.add(product)
            session.flush()
            self._replace_components(session, product.id, components)
            self._event(session, object_type="product_capability", object_id=product.id, event_type="created", after={"name": product.name, "version": product.version})
            session.commit()
            return self.get_product(product.id)

    @staticmethod
    def _build_product(key: str, version: int, payload: dict[str, Any]) -> ProductCapability:
        return ProductCapability(
            capability_key=key,
            version=version,
            name=_clean_text(payload.get("name"), 200),
            product_type=_clean_text(payload.get("product_type"), 120),
            is_enabled=bool(payload.get("is_enabled", True)),
            currency=_clean_text(payload.get("currency") or "CNY", 10),
            cost_min=payload.get("cost_min"), cost_max=payload.get("cost_max"),
            retail_min=payload.get("retail_min"), retail_max=payload.get("retail_max"),
            minimum_order_quantity=payload.get("minimum_order_quantity"),
            lead_time_days=payload.get("lead_time_days"),
            materials_json=_json_dump(payload.get("materials") or []),
            processes_json=_json_dump(payload.get("processes") or []),
            packaging_json=_json_dump(payload.get("packaging") or []),
            constraints_json=_json_dump(payload.get("constraints") or []),
            quality_requirements_json=_json_dump(payload.get("quality_requirements") or []),
        )

    @staticmethod
    def _replace_components(session: Any, product_id: int, components: list[dict[str, Any]]) -> None:
        for position, value in enumerate(components):
            session.add(
                ProductComponent(
                    product_capability_id=product_id,
                    name=_clean_text(value.get("name"), 200),
                    quantity=max(1, int(value.get("quantity") or 1)),
                    sizes_json=_json_dump(value.get("sizes") or []),
                    materials_json=_json_dump(value.get("materials") or []),
                    supported_pattern_modes_json=_json_dump(value.get("supported_pattern_modes") or []),
                    supports_ab_sides=bool(value.get("supports_ab_sides", False)),
                    constraints_json=_json_dump(value.get("constraints") or []),
                    position=int(value.get("position", position)),
                )
            )

    def create_product_version(self, product_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        requested_components = payload.pop("components", None)
        with self.database.sessions() as session:
            current = session.get(ProductCapability, product_id)
            if current is None:
                raise BusinessError("产品能力不存在")
            latest_version = session.scalar(select(func.max(ProductCapability.version)).where(ProductCapability.capability_key == current.capability_key)) or current.version
            old_components = session.scalars(select(ProductComponent).where(ProductComponent.product_capability_id == current.id).order_by(ProductComponent.position, ProductComponent.id)).all()
            base = self.serialize_product(current, list(old_components))
            merged = {**base, **payload}
            product = self._build_product(current.capability_key, int(latest_version) + 1, merged)
            session.add(product)
            session.flush()
            component_values = requested_components if requested_components is not None else base["components"]
            self._replace_components(session, product.id, list(component_values))
            current.is_enabled = False
            current.superseded_at = utc_now()
            self._event(session, object_type="product_capability", object_id=product.id, event_type="version_created", reason=f"supersedes:{current.id}", after={"name": product.name, "version": product.version})
            session.commit()
            return self.get_product(product.id)

    def list_products(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        with self.database.sessions() as session:
            query = select(ProductCapability)
            if enabled_only:
                query = query.where(ProductCapability.is_enabled.is_(True))
            rows = session.scalars(query.order_by(ProductCapability.capability_key, ProductCapability.version.desc())).all()
            ids = [row.id for row in rows]
            components = session.scalars(select(ProductComponent).where(ProductComponent.product_capability_id.in_(ids))).all() if ids else []
            grouped: dict[int, list[ProductComponent]] = {}
            for item in components:
                grouped.setdefault(item.product_capability_id, []).append(item)
            return [self.serialize_product(row, grouped.get(row.id, [])) for row in rows]

    def get_product(self, product_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            row = session.get(ProductCapability, product_id)
            if row is None:
                raise BusinessError("产品能力不存在")
            components = session.scalars(select(ProductComponent).where(ProductComponent.product_capability_id == product_id)).all()
            return self.serialize_product(row, list(components))

    def create_production_rule(self, payload: dict[str, Any]) -> dict[str, Any]:
        product_id = payload.get("product_capability_id")
        with self.database.sessions() as session:
            if product_id is not None and session.get(ProductCapability, int(product_id)) is None:
                raise BusinessError("产品能力不存在")
            row = ProductionRule(
                name=_clean_text(payload.get("name"), 200),
                scope=str(payload.get("scope") or ("product" if product_id else "global")),
                product_capability_id=int(product_id) if product_id else None,
                rule_type=_clean_text(payload.get("rule_type"), 80),
                rule_json=_json_dump(payload.get("rule") or {}),
                is_enabled=bool(payload.get("is_enabled", True)),
            )
            if not row.name or not row.rule_type:
                raise BusinessError("生产规则名称和类型不能为空")
            session.add(row)
            session.flush()
            self._event(session, object_type="production_rule", object_id=row.id, event_type="created", after={"name": row.name, "scope": row.scope, "rule_type": row.rule_type})
            session.commit()
            return self._serialize_production_rule(row)

    @staticmethod
    def _serialize_production_rule(row: ProductionRule) -> dict[str, Any]:
        return {
            "id": row.id,
            "name": row.name,
            "scope": row.scope,
            "product_capability_id": row.product_capability_id,
            "rule_type": row.rule_type,
            "rule": _json_load(row.rule_json, {}),
            "is_enabled": row.is_enabled,
            "created_at": _as_iso(row.created_at),
            "updated_at": _as_iso(row.updated_at),
        }

    def list_production_rules(
        self, *, product_capability_id: int | None = None, enabled_only: bool = True
    ) -> list[dict[str, Any]]:
        with self.database.sessions() as session:
            query = select(ProductionRule)
            if product_capability_id is not None:
                query = query.where(
                    (ProductionRule.product_capability_id.is_(None))
                    | (ProductionRule.product_capability_id == product_capability_id)
                )
            if enabled_only:
                query = query.where(ProductionRule.is_enabled.is_(True))
            rows = session.scalars(query.order_by(ProductionRule.scope, ProductionRule.id)).all()
            return [self._serialize_production_rule(row) for row in rows]

    def create_expression_proposition(
        self, theme_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        with self.database.sessions() as session:
            theme = session.get(BusinessContentTheme, theme_id)
            if theme is None:
                raise BusinessError("内容主题不存在")
            requested_owner = int(payload.get("owner_agent_id") or theme.owner_agent_id)
            if requested_owner != theme.owner_agent_id:
                raise BusinessError("只能为当前Agent负责的主题创建表达命题")
            title = _clean_text(payload.get("title"), 500)
            core_expression = _clean_text(payload.get("core_expression"), 10000)
            if not title or not core_expression:
                raise BusinessError("命题名称和核心表达不能为空")
            theme_form = _clean_text(payload.get("theme_form"), 100) or "概念"
            creative_seeds = _normalize_creative_seeds(payload.get("creative_seeds"))
            proposition_fp = _fingerprint(theme_id, title, core_expression)
            row = session.scalar(
                select(ExpressionProposition).where(
                    ExpressionProposition.theme_id == theme_id,
                    ExpressionProposition.fingerprint == proposition_fp,
                )
            )
            created = row is None
            if row is None:
                row = ExpressionProposition(
                    theme_id=theme_id,
                    owner_agent_id=requested_owner,
                    title=title,
                    core_expression=core_expression,
                    target_emotion=_clean_text(payload.get("target_emotion"), 5000) or None,
                    audience_json=_json_dump(payload.get("audience") or []),
                    scenes_json=_json_dump(payload.get("scenes") or []),
                    purchase_reason=_clean_text(payload.get("purchase_reason"), 5000) or None,
                    exploration_angles_json=_json_dump(payload.get("exploration_angles") or []),
                    theme_form=theme_form,
                    creative_seeds_json=_json_dump(creative_seeds),
                    avoid_json=_json_dump(payload.get("avoid") or []),
                    risks_json=_json_dump(payload.get("risks") or []),
                    status=str(payload.get("status") or "qualified"),
                    fingerprint=proposition_fp,
                )
                if row.status not in PROPOSITION_STATUSES:
                    raise BusinessError("非法表达命题状态")
                session.add(row)
                session.flush()
            theme.status = "expression_ready"
            hotspot = session.get(BusinessHotspot, theme.hotspot_id)
            if hotspot:
                hotspot.business_stage = "proposition_formed"
                hotspot.next_action = "读取已启用产品能力并判断产品适配"
                hotspot.last_activity_at = utc_now()
            if created:
                self._event(
                    session,
                    object_type="expression_proposition",
                    object_id=row.id,
                    event_type="created",
                    actor_kind=str(payload.get("actor_kind") or "agent"),
                    actor_agent_id=row.owner_agent_id,
                    after={"title": row.title, "status": row.status},
                    work_thread_id=hotspot.active_work_thread_id if hotspot else None,
                )
            session.commit()
            return self.get_expression_proposition(row.id)

    def get_expression_proposition(self, proposition_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            row = session.get(ExpressionProposition, proposition_id)
            if row is None:
                raise BusinessError("产品表达命题不存在")
            fits = session.scalars(
                select(ProductFitAssessment).where(
                    ProductFitAssessment.proposition_id == proposition_id
                ).order_by(ProductFitAssessment.id)
            ).all()
            return {
                "id": row.id,
                "theme_id": row.theme_id,
                "owner_agent_id": row.owner_agent_id,
                "title": row.title,
                "core_expression": row.core_expression,
                "target_emotion": row.target_emotion,
                "audience": _json_load(row.audience_json, []),
                "scenes": _json_load(row.scenes_json, []),
                "purchase_reason": row.purchase_reason,
                "exploration_angles": _json_load(row.exploration_angles_json, []),
                "theme_form": row.theme_form or "概念",
                "creative_seeds": _json_load(row.creative_seeds_json, []),
                "avoid": _json_load(row.avoid_json, []),
                "risks": _json_load(row.risks_json, []),
                "status": row.status,
                "fit_assessments": [self._serialize_fit(item) for item in fits],
                "created_at": _as_iso(row.created_at),
                "updated_at": _as_iso(row.updated_at),
            }

    def update_proposition_creative_context(
        self, proposition_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        theme_form = _clean_text(payload.get("theme_form"), 100)
        seeds = _normalize_creative_seeds(payload.get("creative_seeds"))
        if not theme_form or not seeds:
            raise BusinessError("主题形态和至少一个有效创意抓手不能为空")
        with self.database.sessions() as session:
            row = session.get(ExpressionProposition, proposition_id)
            if row is None:
                raise BusinessError("产品表达命题不存在")
            before = {
                "theme_form": row.theme_form,
                "creative_seeds": _json_load(row.creative_seeds_json, []),
            }
            row.theme_form = theme_form
            row.creative_seeds_json = _json_dump(seeds)
            self._event(
                session,
                object_type="expression_proposition",
                object_id=row.id,
                event_type="creative_context_updated",
                actor_kind=str(payload.get("actor_kind") or "agent"),
                actor_agent_id=row.owner_agent_id,
                reason="补充开放式创意抓手",
                before=before,
                after={"theme_form": theme_form, "creative_seeds": seeds},
            )
            session.commit()
        return self.get_expression_proposition(proposition_id)

    def list_pending_product_fits(
        self,
        *,
        owner_agent_id: int,
        query: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        with self.database.sessions() as session:
            statement = (
                select(
                    ExpressionProposition,
                    BusinessContentTheme,
                    BusinessHotspot,
                    WorkThread,
                )
                .join(
                    BusinessContentTheme,
                    BusinessContentTheme.id == ExpressionProposition.theme_id,
                )
                .join(
                    BusinessHotspot,
                    BusinessHotspot.id == BusinessContentTheme.hotspot_id,
                )
                .join(
                    WorkThread,
                    WorkThread.id == BusinessHotspot.active_work_thread_id,
                )
                .outerjoin(
                    ProductFitAssessment,
                    ProductFitAssessment.proposition_id == ExpressionProposition.id,
                )
                .where(
                    ExpressionProposition.owner_agent_id == owner_agent_id,
                    ProductFitAssessment.id.is_(None),
                    ExpressionProposition.status.in_(
                        {"draft", "reviewing", "qualified", "watching"}
                    ),
                )
            )
            clean_query = _clean_text(query, 200)
            if clean_query:
                pattern = f"%{clean_query}%"
                statement = statement.where(
                    ExpressionProposition.title.ilike(pattern)
                    | ExpressionProposition.core_expression.ilike(pattern)
                    | BusinessContentTheme.title.ilike(pattern)
                    | BusinessHotspot.title.ilike(pattern)
                )
            total = int(
                session.scalar(
                    select(func.count()).select_from(statement.subquery())
                )
                or 0
            )
            records = session.execute(
                statement.order_by(
                    ExpressionProposition.created_at,
                    ExpressionProposition.id,
                )
                .offset(max(0, offset))
                .limit(max(1, min(200, limit)))
            ).all()
            return {
                "total": total,
                "items": [
                    {
                        "id": proposition.id,
                        "title": proposition.title,
                        "core_expression": proposition.core_expression,
                        "status": proposition.status,
                        "theme_id": theme.id,
                        "theme_title": theme.title,
                        "hotspot_id": hotspot.id,
                        "hotspot_title": hotspot.title,
                        "work_thread_id": work_thread.id,
                        "project_id": work_thread.project_id,
                    }
                    for proposition, theme, hotspot, work_thread in records
                ],
            }

    @staticmethod
    def _serialize_fit(row: ProductFitAssessment) -> dict[str, Any]:
        return {
            "id": row.id,
            "proposition_id": row.proposition_id,
            "product_capability_id": row.product_capability_id,
            "owner_agent_id": row.owner_agent_id,
            "conclusion": row.conclusion,
            "rationale": row.rationale,
            "target_audience": _json_load(row.target_audience_json, []),
            "scenes": _json_load(row.scenes_json, []),
            "risks": _json_load(row.risks_json, []),
            "missing_information": _json_load(row.missing_information_json, []),
            "created_at": _as_iso(row.created_at),
            "updated_at": _as_iso(row.updated_at),
        }

    def get_existing_scoped_product_fit(
        self,
        *,
        proposition_id: int,
        product_capability_id: int,
        owner_agent_id: int,
        work_thread_id: int,
    ) -> dict[str, Any] | None:
        with self.database.sessions() as session:
            scoped_thread_ids = self._work_scope_thread_ids(session, work_thread_id)
            row = session.execute(
                select(ProductFitAssessment, ProductOpportunity)
                .join(
                    ExpressionProposition,
                    ExpressionProposition.id == ProductFitAssessment.proposition_id,
                )
                .join(
                    BusinessContentTheme,
                    BusinessContentTheme.id == ExpressionProposition.theme_id,
                )
                .join(
                    BusinessHotspot,
                    BusinessHotspot.id == BusinessContentTheme.hotspot_id,
                )
                .outerjoin(
                    ProductOpportunity,
                    ProductOpportunity.fit_assessment_id == ProductFitAssessment.id,
                )
                .where(
                    ProductFitAssessment.proposition_id == proposition_id,
                    ProductFitAssessment.product_capability_id == product_capability_id,
                    ProductFitAssessment.owner_agent_id == owner_agent_id,
                    BusinessHotspot.active_work_thread_id.in_(scoped_thread_ids),
                )
            ).first()
            if row is None:
                return None
            fit, opportunity = row
            result = self._serialize_fit(fit)
            result["opportunity"] = (
                self.get_opportunity(opportunity.id) if opportunity is not None else None
            )
            return result

    def save_product_fit(self, payload: dict[str, Any]) -> dict[str, Any]:
        conclusion = str(payload.get("conclusion") or "needs_validation")
        if conclusion not in FIT_CONCLUSIONS:
            raise BusinessError("非法产品适配结论")
        rationale = _clean_text(payload.get("rationale"), 10000)
        if not rationale:
            raise BusinessError("产品适配必须说明依据")
        with self.database.sessions() as session:
            proposition = session.get(ExpressionProposition, int(payload["proposition_id"]))
            product = session.get(ProductCapability, int(payload["product_capability_id"]))
            if proposition is None:
                raise BusinessError("产品表达命题不存在")
            if product is None:
                raise BusinessError("产品能力不存在")
            requested_owner = int(payload.get("owner_agent_id") or proposition.owner_agent_id)
            if requested_owner != proposition.owner_agent_id:
                raise BusinessError("只能评估当前Agent负责的产品表达命题")
            fit = session.scalar(
                select(ProductFitAssessment).where(
                    ProductFitAssessment.proposition_id == proposition.id,
                    ProductFitAssessment.product_capability_id == product.id,
                )
            )
            created = fit is None
            if fit is None:
                fit = ProductFitAssessment(
                    proposition_id=proposition.id,
                    product_capability_id=product.id,
                    owner_agent_id=requested_owner,
                    conclusion=conclusion,
                    rationale=rationale,
                )
                session.add(fit)
                session.flush()
            fit.conclusion = conclusion
            fit.rationale = rationale
            fit.target_audience_json = _json_dump(payload.get("target_audience") or [])
            fit.scenes_json = _json_dump(payload.get("scenes") or [])
            fit.risks_json = _json_dump(payload.get("risks") or [])
            fit.missing_information_json = _json_dump(
                _actionable_product_fit_missing_information(
                    payload.get("missing_information") or []
                )
            )
            missing_information = _json_load(fit.missing_information_json, [])
            theme = session.get(BusinessContentTheme, proposition.theme_id)
            hotspot = session.get(BusinessHotspot, theme.hotspot_id) if theme else None
            opportunity = session.scalar(
                select(ProductOpportunity).where(ProductOpportunity.fit_assessment_id == fit.id)
            )
            if conclusion in {"fit", "possible", "needs_validation"}:
                opportunity_status, opportunity_next_action = _opportunity_state(
                    conclusion, missing_information
                )
                if opportunity is None:
                    opportunity = ProductOpportunity(
                        fit_assessment_id=fit.id,
                        owner_agent_id=fit.owner_agent_id,
                        title=f"{proposition.title}｜{product.name}",
                        status=opportunity_status,
                        core_expression=proposition.core_expression,
                        purchase_reason=proposition.purchase_reason,
                        risks_json=fit.risks_json,
                        missing_information_json=fit.missing_information_json,
                        next_action=opportunity_next_action,
                    )
                    session.add(opportunity)
                    session.flush()
                else:
                    opportunity.status = opportunity_status
                    opportunity.risks_json = fit.risks_json
                    opportunity.missing_information_json = fit.missing_information_json
                    opportunity.next_action = opportunity_next_action
                if hotspot:
                    hotspot.business_stage = "opportunity_created"
                    hotspot.next_action = opportunity.next_action
                if (
                    conclusion == "fit"
                    and not missing_information
                    and payload.get("create_design_brief") is True
                ):
                    brief = session.scalar(
                        select(ProductDesignBrief).where(
                            ProductDesignBrief.product_opportunity_id == opportunity.id
                        )
                    )
                    product_components = session.scalars(
                        select(ProductComponent).where(
                            ProductComponent.product_capability_id == product.id
                        ).order_by(ProductComponent.position, ProductComponent.id)
                    ).all()
                    brief_payload = {
                        "product": self.serialize_product(product, list(product_components)),
                        "target_customer": _json_load(fit.target_audience_json, []),
                        "use_scenes": _json_load(fit.scenes_json, []),
                        "content_theme": theme.title if theme else None,
                        "core_tension": theme.core_tension if theme else None,
                        "target_emotion": proposition.target_emotion,
                        "core_expression": proposition.core_expression,
                        "purchase_reason": proposition.purchase_reason,
                        "commercial_positioning": opportunity.commercial_positioning,
                        "avoid": _json_load(proposition.avoid_json, []),
                        "risks": sorted(set(_json_load(proposition.risks_json, []) + _json_load(fit.risks_json, []))),
                    }
                    if brief is None:
                        brief = ProductDesignBrief(
                            product_opportunity_id=opportunity.id,
                            owner_agent_id=opportunity.owner_agent_id,
                            product_capability_id=product.id,
                            status="ready_for_design",
                            brief_json=_json_dump(brief_payload),
                            source_snapshot_json=_json_dump(
                                {
                                    "hotspot_id": hotspot.id if hotspot else None,
                                    "theme_id": theme.id if theme else None,
                                    "proposition_id": proposition.id,
                                    "fit_assessment_id": fit.id,
                                    "product_capability_id": product.id,
                                    "product_version": product.version,
                                }
                            ),
                        )
                        session.add(brief)
                    else:
                        brief.brief_json = _json_dump(brief_payload)
                    opportunity.status = "ready_for_design"
                    opportunity.next_action = "等待未来设计Agent领取设计任务书"
                    if hotspot:
                        hotspot.business_stage = "ready_for_design"
                        hotspot.next_action = opportunity.next_action
            else:
                if opportunity:
                    opportunity.status = "not_applicable" if conclusion == "not_fit" else "insufficient_evidence"
                    opportunity.next_action = "等待产品能力或新事实变化后复核"
                if hotspot:
                    hotspot.business_stage = "not_applicable" if conclusion == "not_fit" else "matching_products"
                    hotspot.next_action = "评估其他已启用产品能力" if conclusion == "not_fit" else "补充产品适配信息"
            if hotspot:
                hotspot.last_activity_at = utc_now()
            self._event(
                session,
                object_type="product_fit_assessment",
                object_id=fit.id,
                event_type="created" if created else "updated",
                actor_kind=str(payload.get("actor_kind") or "agent"),
                actor_agent_id=fit.owner_agent_id,
                after={"conclusion": fit.conclusion, "rationale": fit.rationale},
                work_thread_id=hotspot.active_work_thread_id if hotspot else None,
            )
            session.commit()
            result = self._serialize_fit(fit)
            result["opportunity"] = self.get_opportunity(opportunity.id) if opportunity else None
            return result

    def get_opportunity(self, opportunity_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            row = session.get(ProductOpportunity, opportunity_id)
            if row is None:
                raise BusinessError("产品机会不存在")
            fit = session.get(ProductFitAssessment, row.fit_assessment_id)
            proposition = session.get(ExpressionProposition, fit.proposition_id) if fit else None
            theme = session.get(BusinessContentTheme, proposition.theme_id) if proposition else None
            product = session.get(ProductCapability, fit.product_capability_id) if fit else None
            brief = session.scalar(
                select(ProductDesignBrief).where(
                    ProductDesignBrief.product_opportunity_id == row.id
                )
            )
            return {
                "id": row.id,
                "fit_assessment_id": row.fit_assessment_id,
                "owner_agent_id": row.owner_agent_id,
                "title": row.title,
                "proposition_id": proposition.id if proposition else None,
                "proposition_title": proposition.title if proposition else None,
                "theme_form": proposition.theme_form if proposition else None,
                "creative_seeds": (
                    _json_load(proposition.creative_seeds_json, [])
                    if proposition else []
                ),
                "theme_id": theme.id if theme else None,
                "theme_title": theme.title if theme else None,
                "product_capability_id": product.id if product else None,
                "product_name": product.name if product else None,
                "status": row.status,
                "core_expression": row.core_expression,
                "purchase_reason": row.purchase_reason,
                "commercial_positioning": row.commercial_positioning,
                "risks": _json_load(row.risks_json, []),
                "missing_information": _json_load(row.missing_information_json, []),
                "next_action": row.next_action,
                "fit_assessment": self._serialize_fit(fit) if fit else None,
                "design_brief": (
                    {
                        "id": brief.id,
                        "status": brief.status,
                        "product_capability_id": brief.product_capability_id,
                        "brief": _json_load(brief.brief_json, {}),
                        "source_snapshot": _json_load(brief.source_snapshot_json, {}),
                        "created_at": _as_iso(brief.created_at),
                        "updated_at": _as_iso(brief.updated_at),
                    }
                    if brief else None
                ),
                "created_at": _as_iso(row.created_at),
                "updated_at": _as_iso(row.updated_at),
            }

    def list_opportunities(
        self,
        *,
        owner_agent_id: int | None = None,
        status: str | None = None,
        work_thread_id: int | None = None,
        offset: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        with self.database.sessions() as session:
            query = select(ProductOpportunity)
            if owner_agent_id is not None:
                query = query.where(ProductOpportunity.owner_agent_id == owner_agent_id)
            if status:
                query = query.where(ProductOpportunity.status == status)
            if work_thread_id is not None:
                query = (
                    query.join(
                        ProductFitAssessment,
                        ProductFitAssessment.id == ProductOpportunity.fit_assessment_id,
                    )
                    .join(
                        ExpressionProposition,
                        ExpressionProposition.id == ProductFitAssessment.proposition_id,
                    )
                    .join(
                        BusinessContentTheme,
                        BusinessContentTheme.id == ExpressionProposition.theme_id,
                    )
                    .join(
                        BusinessHotspot,
                        BusinessHotspot.id == BusinessContentTheme.hotspot_id,
                    )
                    .where(
                        BusinessHotspot.active_work_thread_id.in_(
                            self._work_scope_thread_ids(session, work_thread_id)
                        )
                    )
                )
            rows = session.scalars(
                query.order_by(
                    ProductOpportunity.updated_at.desc(), ProductOpportunity.id.desc()
                )
                .offset(max(0, offset))
                .limit(min(500, max(1, limit)))
            ).all()
            ids = [row.id for row in rows]
        return [self.get_opportunity(row_id) for row_id in ids]

    def count_opportunities(self, *, owner_agent_id: int | None = None) -> int:
        with self.database.sessions() as session:
            query = select(func.count(ProductOpportunity.id))
            if owner_agent_id is not None:
                query = query.where(ProductOpportunity.owner_agent_id == owner_agent_id)
            return int(session.scalar(query) or 0)

    @staticmethod
    def _delete_hotspot_rows(session: Any, hotspot_ids: list[int]) -> dict[str, int]:
        hotspot_ids = sorted({int(item) for item in hotspot_ids if int(item) > 0})
        if not hotspot_ids:
            return {
                "hotspots": 0,
                "sources": 0,
                "facts": 0,
                "themes": 0,
                "propositions": 0,
                "fits": 0,
                "opportunities": 0,
                "design_briefs": 0,
                "events": 0,
            }
        theme_ids = list(
            session.scalars(
                select(BusinessContentTheme.id).where(
                    BusinessContentTheme.hotspot_id.in_(hotspot_ids)
                )
            ).all()
        )
        proposition_ids = (
            list(
                session.scalars(
                    select(ExpressionProposition.id).where(
                        ExpressionProposition.theme_id.in_(theme_ids)
                    )
                ).all()
            )
            if theme_ids
            else []
        )
        fit_ids = (
            list(
                session.scalars(
                    select(ProductFitAssessment.id).where(
                        ProductFitAssessment.proposition_id.in_(proposition_ids)
                    )
                ).all()
            )
            if proposition_ids
            else []
        )
        opportunity_ids = (
            list(
                session.scalars(
                    select(ProductOpportunity.id).where(
                        ProductOpportunity.fit_assessment_id.in_(fit_ids)
                    )
                ).all()
            )
            if fit_ids
            else []
        )
        counts = {
            "hotspots": len(hotspot_ids),
            "sources": int(
                session.scalar(
                    select(func.count(BusinessContentSource.id)).where(
                        BusinessContentSource.hotspot_id.in_(hotspot_ids)
                    )
                )
                or 0
            ),
            "facts": int(
                session.scalar(
                    select(func.count(BusinessContentFact.id)).where(
                        BusinessContentFact.hotspot_id.in_(hotspot_ids)
                    )
                )
                or 0
            ),
            "themes": len(theme_ids),
            "propositions": len(proposition_ids),
            "fits": len(fit_ids),
            "opportunities": len(opportunity_ids),
            "design_briefs": 0,
            "events": 0,
        }
        if opportunity_ids:
            counts["design_briefs"] = int(
                session.scalar(
                    select(func.count(ProductDesignBrief.id)).where(
                        ProductDesignBrief.product_opportunity_id.in_(opportunity_ids)
                    )
                )
                or 0
            )
            session.execute(
                delete(ProductDesignBrief).where(
                    ProductDesignBrief.product_opportunity_id.in_(opportunity_ids)
                )
            )
            session.execute(
                delete(ProductOpportunity).where(ProductOpportunity.id.in_(opportunity_ids))
            )
        if fit_ids:
            session.execute(
                delete(ProductFitAssessment).where(ProductFitAssessment.id.in_(fit_ids))
            )
        if proposition_ids:
            session.execute(
                delete(ExpressionProposition).where(
                    ExpressionProposition.id.in_(proposition_ids)
                )
            )
        if theme_ids:
            session.execute(
                delete(BusinessContentTheme).where(BusinessContentTheme.id.in_(theme_ids))
            )
        session.execute(
            delete(BusinessContentFact).where(
                BusinessContentFact.hotspot_id.in_(hotspot_ids)
            )
        )
        session.execute(
            delete(BusinessContentSource).where(
                BusinessContentSource.hotspot_id.in_(hotspot_ids)
            )
        )
        event_filters = [
            ("hotspot", hotspot_ids),
            ("expression_proposition", proposition_ids),
            ("product_fit_assessment", fit_ids),
            ("product_opportunity", opportunity_ids),
        ]
        for object_type, object_ids in event_filters:
            if not object_ids:
                continue
            result = session.execute(
                delete(BusinessObjectEvent).where(
                    BusinessObjectEvent.object_type == object_type,
                    BusinessObjectEvent.object_id.in_(object_ids),
                )
            )
            counts["events"] += int(result.rowcount or 0)
        session.execute(delete(BusinessHotspot).where(BusinessHotspot.id.in_(hotspot_ids)))
        return counts

    def delete_hotspot(self, hotspot_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            hotspot = session.get(BusinessHotspot, hotspot_id)
            if hotspot is None:
                raise BusinessError("热点不存在")
            owner_agent_id = hotspot.owner_agent_id
            counts = self._delete_hotspot_rows(session, [hotspot_id])
            session.commit()
            return {"deleted": counts, "owner_agent_id": owner_agent_id}

    def delete_agent_business(self, owner_agent_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            if session.get(AgentProfile, owner_agent_id) is None:
                raise BusinessError("Agent不存在")
            hotspot_ids = list(
                session.scalars(
                    select(BusinessHotspot.id).where(
                        BusinessHotspot.owner_agent_id == owner_agent_id
                    )
                ).all()
            )
            counts = self._delete_hotspot_rows(session, hotspot_ids)
            session.commit()
            return {"deleted": counts, "owner_agent_id": owner_agent_id}

    def intervene_opportunity(self, opportunity_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        reason = _clean_text(payload.get("reason"), 10000)
        if not reason:
            raise BusinessError("人工干预必须填写原因")
        with self.database.sessions() as session:
            row = session.get(ProductOpportunity, opportunity_id)
            if row is None:
                raise BusinessError("产品机会不存在")
            actor_agent_id = payload.get("actor_agent_id")
            if actor_agent_id is not None and session.get(AgentProfile, int(actor_agent_id)) is None:
                raise BusinessError("操作Agent不存在")
            before = {
                "status": row.status,
                "owner_agent_id": row.owner_agent_id,
                "next_action": row.next_action,
            }
            updates = payload.get("updates") or {}
            if "status" in updates:
                if updates["status"] not in OPPORTUNITY_STATUSES:
                    raise BusinessError("非法产品机会状态")
                row.status = updates["status"]
            if "owner_agent_id" in updates:
                owner_id = int(updates["owner_agent_id"])
                if session.get(AgentProfile, owner_id) is None:
                    raise BusinessError("目标Agent不存在")
                row.owner_agent_id = owner_id
            for field in ("title", "core_expression", "purchase_reason", "commercial_positioning", "next_action"):
                if field in updates:
                    setattr(row, field, _clean_text(updates[field], 10000) or None)
            after = {
                "status": row.status,
                "owner_agent_id": row.owner_agent_id,
                "next_action": row.next_action,
            }
            self._event(
                session,
                object_type="product_opportunity",
                object_id=row.id,
                event_type="manual_intervention",
                actor_kind="user",
                actor_agent_id=int(actor_agent_id) if actor_agent_id else None,
                reason=reason,
                before=before,
                after=after,
            )
            session.commit()
            return self.get_opportunity(row.id)

    def intervene_hotspot(self, hotspot_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        actor_agent_id = payload.get("actor_agent_id")
        reason = _clean_text(payload.get("reason"), 10000)
        if not reason:
            raise BusinessError("人工干预必须填写原因")
        with self.database.sessions() as session:
            hotspot = session.get(BusinessHotspot, hotspot_id)
            if hotspot is None:
                raise BusinessError("热点不存在")
            if actor_agent_id is not None and session.get(AgentProfile, int(actor_agent_id)) is None:
                raise BusinessError("操作Agent不存在")
            before = self.serialize_hotspot(hotspot)
            updates = payload.get("updates") or {}
            validators = {
                "lifecycle_status": HOTSPOT_LIFECYCLE,
                "research_status": HOTSPOT_RESEARCH,
                "business_stage": HOTSPOT_STAGES,
                "automation_status": AUTOMATION_STATUSES,
            }
            for field, allowed in validators.items():
                if field in updates:
                    if updates[field] not in allowed:
                        raise BusinessError(f"非法{field}")
                    setattr(hotspot, field, updates[field])
            if "owner_agent_id" in updates:
                new_owner = int(updates["owner_agent_id"])
                if session.get(AgentProfile, new_owner) is None:
                    raise BusinessError("目标Agent不存在")
                hotspot.owner_agent_id = new_owner
            for field in ("title", "summary", "next_action"):
                if field in updates:
                    setattr(hotspot, field, _clean_text(updates[field], 10000) or None)
            if "next_review_at" in updates:
                hotspot.next_review_at = updates["next_review_at"]
            hotspot.last_activity_at = utc_now()
            after = self.serialize_hotspot(hotspot)
            self._event(
                session,
                object_type="hotspot",
                object_id=hotspot.id,
                event_type="manual_intervention",
                actor_kind="user",
                actor_agent_id=int(actor_agent_id) if actor_agent_id else None,
                reason=reason,
                before=before,
                after=after,
                work_thread_id=hotspot.active_work_thread_id,
            )
            session.commit()
            return self.get_hotspot(hotspot.id)
    @staticmethod
    def _work_scope_thread_ids(session: Any, work_thread_id: int) -> list[int]:
        current = session.get(WorkThread, work_thread_id)
        if current and current.project_id is not None:
            return list(
                session.scalars(
                    select(WorkThread.id).where(
                        WorkThread.project_id == current.project_id
                    )
                ).all()
            )
        return [work_thread_id]
