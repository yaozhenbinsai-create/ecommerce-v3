from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .database import Database
from .models import (
    AgentProfile,
    Evidence,
    ResearchReport,
    ToolCall,
    WorkEvent,
    WorkThread,
)
from .research_runtime import serialize_evidence
from .research_quality_service import ResearchQualityService
from .work_runtime import WorkConflict, WorkNotFound


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _stored_json(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "-", value).strip(" .-")
    return (value[:80] or "research-report")


def _safe_source_url(value: str) -> str:
    parsed = urlsplit(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else "#"


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class ReportService:
    def __init__(self, database: Database, output_dir: Path | None = None) -> None:
        self.database = database
        self.output_dir = output_dir

    @staticmethod
    def _tree_threads(session: Any, root: WorkThread) -> list[WorkThread]:
        roots = [root]
        seen_roots = {root.id}
        cursor = root
        while cursor.continuation_of_thread_id:
            source = session.get(WorkThread, cursor.continuation_of_thread_id)
            if not source or source.id in seen_roots:
                break
            roots.append(source)
            seen_roots.add(source.id)
            cursor = source
        rows: list[WorkThread] = []
        seen: set[int] = set()
        for project_root in reversed(roots):
            if project_root.id not in seen:
                seen.add(project_root.id)
                rows.append(project_root)
            frontier = [project_root.id]
            while frontier:
                children = session.scalars(
                    select(WorkThread)
                    .where(WorkThread.parent_thread_id.in_(frontier))
                    .order_by(WorkThread.id)
                ).all()
                frontier = []
                for child in children:
                    if child.id in seen:
                        continue
                    seen.add(child.id)
                    rows.append(child)
                    frontier.append(child.id)
        return rows

    def artifact_paths(self, report: ResearchReport) -> dict[str, Path]:
        if self.output_dir is None:
            return {}
        directory = self.output_dir / f"work-{report.work_thread_id}"
        stem = (
            f"work-{report.work_thread_id}-v{report.version}-"
            f"{_safe_filename(report.title)}"
        )
        return {
            "markdown": directory / f"{stem}.md",
            "html": directory / f"{stem}.html",
        }

    def materialize(self, report: ResearchReport) -> dict[str, Path]:
        paths = self.artifact_paths(report)
        if not paths:
            return paths
        paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
        contents = {
            "markdown": report.markdown_content,
            "html": self.render_html(report),
        }
        for kind, path in paths.items():
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            temporary.write_text(contents[kind], encoding="utf-8")
            temporary.replace(path)
        return paths

    def generate(
        self, thread_id: int, *, allow_incomplete: bool = False
    ) -> tuple[ResearchReport, bool]:
        with self.database.sessions() as session:
            thread = session.get(WorkThread, thread_id)
            if not thread:
                raise WorkNotFound("工作线程不存在")
            if (
                not allow_incomplete
                and thread.status not in {"completed", "failed", "rolled_over", "cancelled"}
            ):
                raise WorkConflict("只有已结束或已换代的工作才能生成成果报告")
            tree_threads = self._tree_threads(session, thread)
            tree_ids = [item.id for item in tree_threads]
            agent_rows = session.scalars(
                select(AgentProfile).where(
                    AgentProfile.id.in_({item.owner_agent_id for item in tree_threads})
                )
            ).all()
            agents = {item.id: item.name for item in agent_rows}
            evidence_rows = session.scalars(
                select(Evidence)
                .where(Evidence.work_thread_id.in_(tree_ids))
                .order_by(Evidence.id)
            ).all()
            evidence = [serialize_evidence(item) for item in evidence_rows]
            quality = ResearchQualityService.summarize_rows(evidence_rows)
            completed_tool_calls = session.execute(
                select(ToolCall.id, ToolCall.work_thread_id, ToolCall.tool_name)
                .where(
                    ToolCall.work_thread_id.in_(tree_ids),
                    ToolCall.status == "completed",
                )
                .order_by(ToolCall.id)
            ).all()
            work_results = [
                {
                    "thread_id": item.id,
                    "parent_thread_id": item.parent_thread_id,
                    "continuation_of_thread_id": item.continuation_of_thread_id,
                    "title": item.title,
                    "owner_agent_id": item.owner_agent_id,
                    "owner_agent_name": agents.get(item.owner_agent_id, "未知Agent"),
                    "status": item.status,
                    "result_text": item.result_text or item.summary or "",
                }
                for item in tree_threads
            ]
            tool_counts: dict[str, int] = {}
            for _, _, tool_name in completed_tool_calls:
                tool_counts[str(tool_name)] = tool_counts.get(str(tool_name), 0) + 1
            tool_audit = {
                "completed_count": len(completed_tool_calls),
                "latest_tool_call_id": (
                    int(completed_tool_calls[-1][0]) if completed_tool_calls else None
                ),
                "counts_by_tool": tool_counts,
                "note": "完整参数和返回保存在工具审计账本，不进入用户成果正文。",
            }
            source = {
                "delivery_format": "concise-results-v1",
                "thread": {
                    "id": thread.id,
                    "continuation_of_thread_id": thread.continuation_of_thread_id,
                    "title": thread.title,
                    "objective": thread.objective,
                    "success_criteria": _stored_json(thread.success_criteria_json, []),
                    "constraints": _stored_json(thread.constraints_json, []),
                    "result_text": thread.result_text or "",
                    "finished_at": thread.finished_at.isoformat() if thread.finished_at else None,
                },
                "owner_agent_name": agents.get(thread.owner_agent_id, "未知Agent"),
                "work_results": work_results,
                "tool_audit": tool_audit,
                "evidence": evidence,
                "quality": quality,
            }
            source_hash = sha256(_json_text(source).encode("utf-8")).hexdigest()
            existing = session.scalar(
                select(ResearchReport).where(
                    ResearchReport.work_thread_id == thread_id,
                    ResearchReport.source_hash == source_hash,
                )
            )
            if existing:
                self.materialize(existing)
                return existing, True

            findings = []
            for index, item in enumerate(work_results, start=1):
                if not item["result_text"]:
                    continue
                findings.append(
                    {
                        "kind": "agent_result",
                        "evidence_ref": f"A{index}",
                        "claim": item["result_text"],
                        **item,
                    }
                )
            for index, item in enumerate(evidence, start=1):
                findings.append(
                    {
                        "kind": "evidence",
                        "evidence_ref": f"E{index}",
                        "claim": item["claim"],
                        "metric_name": item["metric_name"],
                        "metric_value": item["metric_value"],
                        "metric_unit": item["metric_unit"],
                        "confidence": item["confidence"],
                        "stance": item["stance"],
                    }
                )
            summary = thread.result_text or thread.summary or (
                "工作仍在进行，当前报告只保存已经形成的阶段成果。"
                if allow_incomplete
                else "工作已完成。"
            )
            recommendations: list[str] = []
            caveats = []
            if allow_incomplete and thread.status not in {
                "completed",
                "failed",
                "rolled_over",
                "cancelled",
            }:
                caveats.append(
                    "这是进行中的阶段成果，不代表任务完成；最终交付仍须通过全部成功标准。"
                )
            if thread.status == "failed":
                caveats.append("任务未满足全部成功标准；本报告保存结束时已有成果、工具依据和证据，不代表目标已经完整完成。")
            if not evidence:
                if completed_tool_calls:
                    caveats.append("任务树没有保存结构化网络证据；报告只保留Agent成果，完整工具过程可在审计账本核查。")
                else:
                    caveats.append("任务树没有保存结构化证据或成功工具结果；报告只能复述Agent最终结果，不能独立核验。")
            if any(item["confidence"] != "high" for item in evidence):
                caveats.append("部分证据可信度不是高；使用相关结论前应补充更可靠的一手来源。")
            if any(not item["published_at"] for item in evidence):
                caveats.append("部分来源没有明确发布日期，无法仅凭报告判断信息的新鲜度。")
            if quality["contested_claim_count"]:
                caveats.append("存在同时有支持和反驳证据的结论，使用前必须解决冲突或明确保留不确定性。")
            if quality["claim_status_counts"].get("single_source", 0):
                caveats.append("部分结论目前只有单一非一手来源；重大或争议决策前建议补充独立来源。")
            constraints = source["thread"]["constraints"]
            caveats.extend(f"任务限制：{item}" for item in constraints)
            if not caveats:
                caveats.append("报告是生成时点的证据快照，不代表来源在未来持续有效。")

            version = int(
                session.scalar(
                    select(func.max(ResearchReport.version)).where(
                        ResearchReport.work_thread_id == thread_id
                    )
                )
                or 0
            ) + 1
            report_kind = (
                "阶段成果"
                if allow_incomplete
                and thread.status not in {"completed", "failed", "rolled_over", "cancelled"}
                else "成果整理"
            )
            title = f"{thread.title}｜{report_kind}"
            markdown = self._markdown(
                title, summary, findings, recommendations, caveats, evidence, source, quality
            )
            report = ResearchReport(
                work_thread_id=thread.id,
                owner_agent_id=thread.owner_agent_id,
                version=version,
                title=title,
                source_hash=source_hash,
                executive_summary=summary,
                findings_json=_json_text(findings),
                recommendations_json=_json_text(recommendations),
                caveats_json=_json_text(caveats),
                evidence_snapshot_json=_json_text(evidence),
                markdown_content=markdown,
            )
            session.add(report)
            session.flush()
            session.add(
                WorkEvent(
                    work_thread_id=thread.id,
                    event_type="report_generated",
                    actor_kind="runtime",
                    actor_id="report-service",
                    payload_json=_json_text(
                        {"report_id": report.id, "version": version, "source_hash": source_hash}
                    ),
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.scalar(
                    select(ResearchReport).where(
                        ResearchReport.work_thread_id == thread_id,
                        ResearchReport.source_hash == source_hash,
                    )
                )
                if existing:
                    return existing, True
                raise
            session.refresh(report)
            self.materialize(report)
            return report, False

    def latest(self, thread_id: int) -> ResearchReport | None:
        with self.database.sessions() as session:
            if not session.get(WorkThread, thread_id):
                raise WorkNotFound("工作线程不存在")
            return session.scalar(
                select(ResearchReport)
                .where(ResearchReport.work_thread_id == thread_id)
                .order_by(ResearchReport.version.desc())
            )

    def get(self, thread_id: int, report_id: int) -> ResearchReport:
        with self.database.sessions() as session:
            report = session.get(ResearchReport, report_id)
            if not report or report.work_thread_id != thread_id:
                raise WorkNotFound("成果报告不存在")
            return report

    def render_html(self, report: ResearchReport) -> str:
        findings = _stored_json(report.findings_json, [])
        caveats = _stored_json(report.caveats_json, [])
        evidence = _stored_json(report.evidence_snapshot_json, [])
        collaboration_results = [
            item
            for item in findings
            if item.get("kind") == "agent_result"
            and item.get("thread_id") != report.work_thread_id
            and (item.get("result_text") or item.get("claim"))
        ]
        collaboration_items = "".join(
            '<li><strong>{} · {}</strong><div class="result">{}</div></li>'.format(
                escape(item.get("owner_agent_name") or "未知Agent"),
                escape(item.get("title") or "协作任务"),
                escape(item.get("result_text") or item.get("claim") or ""),
            )
            for item in collaboration_results
        )
        evidence_parts: list[str] = []
        for item in evidence:
            value = item.get("value_analysis") or {}
            details = value.get("specific_details") or []
            value_lines = []
            for label, key in (
                ("为什么现在值得看", "why_now"),
                ("反映的现象", "phenomenon"),
                ("内容角度", "content_angle"),
                ("小众价值", "niche_reason"),
            ):
                if value.get(key):
                    value_lines.append(
                        f"<div><b>{label}：</b>{escape(str(value[key]))}</div>"
                    )
            if details:
                value_lines.append(
                    "<div><b>具体细节：</b>{}</div>".format(
                        escape("；".join(str(detail) for detail in details))
                    )
                )
            value_block = (
                '<div class="value">{}</div>'.format("".join(value_lines))
                if value_lines
                else ""
            )
            evidence_parts.append(
                '<li><strong>{}</strong>{}<br><a href="{}">{}</a> · 可信度：{}</li>'.format(
                    escape(value.get("event_summary") or item.get("claim") or "已保存依据"),
                    value_block,
                    escape(_safe_source_url(item.get("source_url") or ""), quote=True),
                    escape(item.get("source_title") or item.get("source_url") or "查看来源"),
                    escape(item.get("confidence") or "unknown"),
                )
            )
        evidence_items = "".join(evidence_parts) or "<li>没有保存可直接展示的结构化来源。</li>"
        caveat_items = "".join(
            f"<li>{escape(str(value))}</li>" for value in caveats
        ) or "<li>无。</li>"
        generated_at = report.generated_at
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        generated = generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        collaboration_section = (
            f"<h2>Agent协作成果</h2><ol>{collaboration_items}</ol>"
            if collaboration_items
            else ""
        )
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(report.title)}</title><style>
        :root{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#1d2738;background:#eef3f9}}body{{margin:0}}main{{max-width:860px;margin:32px auto;padding:42px 54px;background:white;box-shadow:0 12px 40px #20395b1c}}h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:30px;padding-bottom:8px;border-bottom:1px solid #dfe6ef}}.meta{{color:#718096;font-size:13px}}.summary,.result{{font-size:16px;line-height:1.75;padding:14px 16px;background:#f2f6ff;border-left:4px solid #3d73e8;white-space:pre-wrap}}.value{{margin:8px 0;padding:10px 12px;background:#f7f9fc;border-radius:6px}}li{{margin:10px 0;line-height:1.65}}a{{color:#285fc7;overflow-wrap:anywhere}}@media(max-width:700px){{main{{margin:0;padding:24px 18px}}}}@media print{{body{{background:white}}main{{max-width:none;margin:0;padding:0;box-shadow:none}}a{{color:black}}}}
        </style></head><body><main><h1>{escape(report.title)}</h1><div class="meta">版本 {report.version} · 生成于 {generated} · 协作成果 {len(collaboration_results)} 项 · 关键依据 {len(evidence)} 条</div><h2>执行成果</h2><div class="summary">{escape(report.executive_summary)}</div>{collaboration_section}<h2>关键依据</h2><ol>{evidence_items}</ol><h2>未完成与限制</h2><ul>{caveat_items}</ul></main></body></html>'''

    def filename(self, report: ResearchReport, suffix: str) -> str:
        return f"{_safe_filename(report.title)}-v{report.version}.{suffix}"

    def _markdown(self, title: str, summary: str, findings: list[dict[str, Any]], recommendations: list[str], caveats: list[str], evidence: list[dict[str, Any]], source: dict[str, Any], quality: dict[str, Any]) -> str:
        collaboration_results = [
            item
            for item in findings
            if item.get("kind") == "agent_result"
            and item.get("thread_id") != source["thread"]["id"]
            and (item.get("result_text") or item.get("claim"))
        ]
        lines = [
            f"# {title}",
            "",
            "## 执行成果",
            "",
            summary,
        ]
        if collaboration_results:
            lines.extend(["", "## Agent协作成果", ""])
            for item in collaboration_results:
                lines.extend(
                    [
                        f"### {item.get('owner_agent_name') or '未知Agent'} · {item.get('title') or '协作任务'}",
                        "",
                        item.get("result_text") or item.get("claim") or "未提供",
                        "",
                    ]
                )
        lines.extend(["", "## 关键依据", ""])
        if evidence:
            for item in evidence:
                source_url = _safe_source_url(item.get("source_url") or "")
                source_title = item.get("source_title") or item.get("source_url") or "查看来源"
                confidence = item.get("confidence") or "unknown"
                lines.append(
                    f"- **{(item.get('value_analysis') or {}).get('event_summary') or item.get('claim') or '已保存依据'}** "
                    f"[{source_title}]({source_url}) · 可信度：{confidence}"
                )
                value = item.get("value_analysis") or {}
                for label, key in (
                    ("为什么现在值得看", "why_now"),
                    ("反映的现象", "phenomenon"),
                    ("具体细节", "specific_details"),
                    ("内容角度", "content_angle"),
                    ("小众价值", "niche_reason"),
                ):
                    field = value.get(key)
                    if field:
                        rendered = "；".join(str(part) for part in field) if isinstance(field, list) else str(field)
                        lines.append(f"  - {label}：{rendered}")
        else:
            lines.append("- 没有保存可直接展示的结构化来源。")
        lines.extend(["", "## 未完成与限制", ""])
        lines.extend([f"- {value}" for value in caveats] or ["- 无。"])
        lines.extend(
            [
                "",
                "---",
                "详细工具记录、原始返回和证据快照已保留在系统账本中，需要时可继续核查。",
                "",
            ]
        )
        return "\n".join(lines)


def serialize_report(report: ResearchReport, *, include_content: bool = True) -> dict[str, Any]:
    findings = _stored_json(report.findings_json, [])
    result = {
        "id": report.id, "work_thread_id": report.work_thread_id,
        "owner_agent_id": report.owner_agent_id, "version": report.version,
        "title": report.title, "source_hash": report.source_hash,
        "executive_summary": report.executive_summary,
        "findings": findings,
        "recommendations": _stored_json(report.recommendations_json, []),
        "caveats": _stored_json(report.caveats_json, []),
        "evidence_count": len(_stored_json(report.evidence_snapshot_json, [])),
        "agent_result_count": sum(
            1 for item in findings if item.get("kind") == "agent_result"
        ),
        "tool_result_count": sum(
            1 for item in findings if item.get("kind") == "tool_result"
        ),
        "preview_url": (
            f"/api/work/threads/{report.work_thread_id}/reports/{report.id}/preview"
        ),
        "download_urls": {
            "html": f"/api/work/threads/{report.work_thread_id}/reports/{report.id}/download?format=html",
            "markdown": f"/api/work/threads/{report.work_thread_id}/reports/{report.id}/download?format=md",
        },
        "generated_at": _iso_utc(report.generated_at),
    }
    if include_content:
        result["evidence_snapshot"] = _stored_json(report.evidence_snapshot_json, [])
        result["markdown_content"] = report.markdown_content
    return result
