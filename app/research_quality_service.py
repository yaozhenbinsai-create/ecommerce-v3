from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Callable
from urllib.parse import urlsplit

from sqlalchemy import select

from .database import Database
from .models import AgentProfile, Evidence, WorkEvent, WorkThread
from .tool_runtime import ToolContext
from .work_runtime import WorkConflict, WorkNotFound, json_text


SOURCE_TYPES = {
    "official_primary",
    "regulatory_primary",
    "independent_secondary",
    "aggregator",
    "community",
    "unknown",
}
PRIMARY_SOURCE_TYPES = {"official_primary", "regulatory_primary"}
REVIEW_STATUSES = {"unreviewed", "verified", "rejected", "needs_followup"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalized_publisher(evidence: Evidence) -> str:
    publisher = (evidence.source_publisher or "").strip().lower()
    if publisher:
        return re.sub(r"\s+", " ", publisher)
    return (urlsplit(evidence.source_url).hostname or "unknown").lower()


def evidence_quality_item(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "source_url": evidence.source_url,
        "source_title": evidence.source_title,
        "source_type": evidence.source_type,
        "source_publisher": evidence.source_publisher,
        "publisher_key": normalized_publisher(evidence),
        "claim": evidence.claim,
        "claim_key": evidence.claim_key,
        "confidence": evidence.confidence,
        "stance": evidence.stance,
        "verification_notes": evidence.verification_notes,
        "review_status": evidence.review_status,
        "reviewer_agent_id": evidence.reviewer_agent_id,
        "review_notes": evidence.review_notes,
        "reviewed_at": evidence.reviewed_at.isoformat() if evidence.reviewed_at else None,
    }


class ResearchQualityService:
    def __init__(
        self,
        database: Database,
        report_refresh: Callable[[int], Any] | None = None,
    ) -> None:
        self.database = database
        self.report_refresh = report_refresh

    @staticmethod
    def _controller(session: Any, agent_id: int) -> AgentProfile:
        agent = session.get(AgentProfile, agent_id)
        if not agent:
            raise WorkNotFound("Agent不存在")
        if not agent.is_controller:
            raise WorkConflict("只有总控Agent可以复核研究证据")
        return agent

    @staticmethod
    def summarize_rows(rows: list[Evidence]) -> dict[str, Any]:
        groups: dict[str, list[Evidence]] = {}
        for row in rows:
            key = (row.claim_key or "").strip() or f"evidence:{row.id}"
            groups.setdefault(key, []).append(row)
        claims: list[dict[str, Any]] = []
        for key, items in groups.items():
            publishers = {normalized_publisher(item) for item in items}
            supports = [item for item in items if item.stance == "supports"]
            refutes = [item for item in items if item.stance == "refutes"]
            supporting_publishers = {
                normalized_publisher(item) for item in supports
            }
            primary_support = any(
                item.source_type in PRIMARY_SOURCE_TYPES for item in supports
            )
            reviewed_support = any(
                item.review_status == "verified" for item in supports
            )
            contested = bool(supports and refutes)
            if contested:
                status = "contested"
            elif len(supporting_publishers) >= 2:
                status = "cross_checked"
            elif primary_support:
                status = "primary_supported"
            elif supports:
                status = "single_source"
            else:
                status = "context_only"
            claims.append(
                {
                    "claim_key": None if key.startswith("evidence:") else key,
                    "display_claim": items[0].claim,
                    "status": status,
                    "evidence_count": len(items),
                    "independent_publisher_count": len(publishers),
                    "support_count": len(supports),
                    "refute_count": len(refutes),
                    "primary_support": primary_support,
                    "reviewed_support": reviewed_support,
                    "evidence_ids": [item.id for item in items],
                }
            )
        status_counts: dict[str, int] = {}
        for claim in claims:
            status_counts[claim["status"]] = status_counts.get(claim["status"], 0) + 1
        source_type_counts: dict[str, int] = {}
        review_counts: dict[str, int] = {}
        for row in rows:
            source_type_counts[row.source_type] = source_type_counts.get(row.source_type, 0) + 1
            review_counts[row.review_status] = review_counts.get(row.review_status, 0) + 1
        return {
            "evidence_count": len(rows),
            "claim_count": len(claims),
            "claims": claims,
            "claim_status_counts": status_counts,
            "source_type_counts": source_type_counts,
            "review_status_counts": review_counts,
            "independent_publisher_count": len(
                {normalized_publisher(row) for row in rows}
            ),
            "primary_source_count": sum(
                1 for row in rows if row.source_type in PRIMARY_SOURCE_TYPES
            ),
            "contested_claim_count": status_counts.get("contested", 0),
            "cross_checked_claim_count": status_counts.get("cross_checked", 0),
            "guidance": [
                "单一高质量一手来源可以支持明确、低争议事实；不要为数量机械凑来源。",
                "重大、争议或高风险结论优先寻找不同发布主体的独立支持，并检查反驳证据。",
                "来源类型、置信度和复核状态是审计信息，不替代原文与指标口径判断。",
            ],
        }

    def summary(self, thread_id: int) -> dict[str, Any]:
        with self.database.sessions() as session:
            if not session.get(WorkThread, thread_id):
                raise WorkNotFound("工作线程不存在")
            rows = session.scalars(
                select(Evidence)
                .where(Evidence.work_thread_id == thread_id)
                .order_by(Evidence.id)
            ).all()
            result = self.summarize_rows(rows)
            result["work_thread_id"] = thread_id
            result["evidence"] = [evidence_quality_item(row) for row in rows]
            return result

    def review(
        self,
        evidence_id: int,
        *,
        reviewer_agent_id: int,
        status: str,
        notes: str,
    ) -> Evidence:
        if status not in REVIEW_STATUSES - {"unreviewed"}:
            raise WorkConflict("复核状态必须是 verified、rejected 或 needs_followup")
        notes = notes.strip()
        if not notes:
            raise WorkConflict("证据复核必须填写说明")
        with self.database.sessions() as session:
            self._controller(session, reviewer_agent_id)
            row = session.get(Evidence, evidence_id)
            if not row:
                raise WorkNotFound("证据不存在")
            row.review_status = status
            row.reviewer_agent_id = reviewer_agent_id
            row.review_notes = notes[:10000]
            row.reviewed_at = utc_now()
            session.add(
                WorkEvent(
                    work_thread_id=row.work_thread_id,
                    event_type="evidence_reviewed",
                    actor_kind="controller",
                    actor_id=str(reviewer_agent_id),
                    payload_json=json_text(
                        {
                            "evidence_id": row.id,
                            "review_status": status,
                            "notes": notes[:4000],
                        }
                    ),
                )
            )
            session.commit()
            session.refresh(row)
            thread_id = row.work_thread_id
        if self.report_refresh:
            try:
                self.report_refresh(thread_id)
            except Exception:
                # 复核是主账本动作；报告刷新属于可重建派生物，失败不能回滚复核。
                pass
        return row

    def assess_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if arguments:
            raise WorkConflict("assess_research_quality不接受参数")
        return self.summary(context.work_thread_id)

    def review_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        row = self.review(
            int(arguments["evidence_id"]),
            reviewer_agent_id=context.owner_agent_id,
            status=str(arguments["status"]),
            notes=str(arguments["notes"]),
        )
        return evidence_quality_item(row)
