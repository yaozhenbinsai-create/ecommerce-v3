from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.database import Database
from app.models import AgentProfile, Evidence, ResearchCandidate, ResearchProviderHealth, ToolCall, WorkEvent
from app.research_runtime import (
    ResearchError,
    ResearchService,
    filter_search_results,
    filter_relevant_results,
    normalize_search_query,
    relaxed_search_query,
    news_event_search_query,
    canonicalize_url,
    deduplicate_candidates,
    parse_duckduckgo_results,
    validate_public_url,
    assess_content_value,
    serialize_evidence,
)


def test_relaxed_search_query_keeps_event_evidence_anchors():
    value = relaxed_search_query(
        "2026 community social event participants local friendship gathering reported experience",
        [],
    )
    assert "community" in value
    assert "event recap attendees" in value

    cjk_value = relaxed_search_query(
        "2026年8月 社区 兴趣活动 活动回顾 参与者 线下 聚会 工作坊",
        [],
    )
    assert "社区" in cjk_value
    assert "活动回顾 主办方 参与者" in cjk_value

    news_value = news_event_search_query(
        "Repair Cafe community event recap local news participants 2026"
    )
    assert news_value == "Repair Cafe volunteers residents"
from app.research_quality_service import ResearchQualityService
from app.tool_runtime import ToolContext, ToolRegistry, ToolSpec
from app.work_executor import WorkExecutor
from app.work_runtime import WorkRuntime


PUBLIC_RESOLVER = lambda _host, _port: ["93.184.216.34"]


class FakeStreamResponse:
    def __init__(self, status_code=200, headers=None, content=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)

    def iter_bytes(self):
        yield self._content


class FakeHttpClient:
    responses = []
    requested = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url, headers=None):
        self.requested.append(url)
        return self.responses.pop(0)


def make_service(tmp_path: Path) -> tuple[Database, ResearchService, int]:
    database = Database(f"sqlite:///{tmp_path / 'research.db'}")
    database.create_tables()
    with database.sessions() as session:
        agent = AgentProfile(name="市场Agent")
        session.add(agent)
        session.commit()
        agent_id = agent.id
    service = ResearchService(database, resolver=PUBLIC_RESOLVER)
    return database, service, agent_id


def test_public_url_guard_rejects_local_and_private_networks():
    with pytest.raises(ResearchError):
        validate_public_url("http://127.0.0.1/admin")
    with pytest.raises(ResearchError):
        validate_public_url("http://metadata.internal/", lambda *_: ["169.254.169.254"])
    with pytest.raises(ResearchError):
        validate_public_url("file:///etc/passwd")
    assert (
        validate_public_url("https://example.com/page", PUBLIC_RESOLVER)
        == "https://example.com/page"
    )


def test_public_url_guard_tolerates_teredo_dns_answer_with_public_ipv4():
    resolver = lambda *_: ["185.45.5.35", "2001::1"]

    assert validate_public_url("https://www.google.com/search?q=test", resolver) == (
        "https://www.google.com/search?q=test"
    )


def test_public_url_guard_still_rejects_mixed_public_and_private_networks():
    with pytest.raises(ResearchError):
        validate_public_url(
            "https://mixed.example/", lambda *_: ["93.184.216.34", "127.0.0.1"]
        )


def test_search_and_page_reader_parse_bounded_public_content(tmp_path: Path, monkeypatch):
    _, service, _ = make_service(tmp_path)
    brave_results = b'''<html><body><div class="snippet" data-type="web"><a href="https://example.com/report"><div class="search-snippet-title">Example App</div></a><div class="desktop-default-regular">More than 50 million registered users</div></div></body></html>'''
    rss = b"""<?xml version='1.0'?><rss><channel><item><title>Fallback App</title>
    <link>https://example.com/report</link><description>More than 50 million users</description>
    </item></channel></rss>"""
    html = b'<html><head><title>Official report</title><script>ignore</script></head><body><h1>Growth</h1><p>We passed 50 million registered users.</p><a href="/more">More evidence</a><a href="https://outside.test/">Outside</a></body></html>'
    FakeHttpClient.responses = [
        FakeStreamResponse(headers={"content-type": "text/html"}, content=brave_results),
        FakeStreamResponse(headers={"content-type": "text/html; charset=utf-8"}, content=html),
    ]
    FakeHttpClient.requested = []
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)

    searched = service.search("registered users app", 5)
    page = service.read_page(searched["results"][0]["url"])

    assert searched["results"][0]["title"] == "Example App"
    assert searched["provider"] == "brave_html"
    assert searched["offset"] == 0
    assert page["title"] == "Official report"
    assert "50 million registered users" in page["text"]
    assert "ignore" not in page["text"]
    assert page["links"] == [
        {"url": "https://example.com/more", "text": "More evidence"}
    ]


def test_content_value_rejects_headline_and_accepts_concrete_story():
    headline = assess_content_value({"event_summary": "某活动登上热榜"})
    assert headline["ready"] is False
    assert "why_now" in headline["missing"]

    rich = assess_content_value(
        {
            "category": "小众",
            "event_summary": "世界七叶树果锦标赛公布今年比赛日期并开放社区参与",
            "why_now": "比赛日期刚刚确认，参赛名额和社区筹款正在形成新一轮讨论",
            "phenomenon": "传统地方游戏正借年度赛事、慈善筹款和社群传播重新获得关注",
            "specific_details": ["比赛定于10月11日举行", "赛事自1965年以来为慈善筹款超过48.5万英镑"],
            "affected_groups": ["民间体育爱好者", "地方社区"],
            "discussion_evidence": "参赛名额已满，社区围绕规则和卫冕者持续讨论",
            "niche_reason": "它是规则完整、全球参赛但大众媒体较少覆盖的民间运动",
            "content_angle": "可整理成传统民间运动如何靠社区赛事持续六十年的故事",
        }
    )
    assert rich["ready"] is True
    assert rich["score"] >= 9
    assert rich["missing"] == []


def test_no_key_hacker_news_layer_standardizes_and_persists_candidates(
    tmp_path: Path, monkeypatch
):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, instance_id="public-trends")
    thread = runtime.create_thread(owner_agent_id=agent_id, title="热点", objective="找热点")
    run = runtime.create_run(thread.id)
    context = ToolContext(thread.id, run.id, agent_id, 1)
    FakeHttpClient.responses = [
        FakeStreamResponse(headers={"content-type": "application/json"}, content=b"[42]"),
        FakeStreamResponse(
            headers={"content-type": "application/json"},
            content=b'{"id":42,"title":"Useful open source launch","url":"https://example.com/launch","by":"alice","score":321,"descendants":87,"time":1786800000}',
        ),
    ]
    FakeHttpClient.requested = []
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)

    result = service.collect_public_trends(
        context, sources=["hacker_news"], limit=1
    )

    assert result["candidate_only"] is True
    assert result["candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["platform"] == "hacker_news"
    assert candidate["engagement"] == {"score": 321, "comments": 87}
    assert candidate["needs_deep_read"] is True
    assert service.list_candidates(thread.id)[0]["sources"] == [
        "public_trends_hacker_news"
    ]


def test_public_trends_broad_default_does_not_start_from_hacker_news(
    tmp_path: Path, monkeypatch
):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, instance_id="public-trends-default")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="大众生活热点",
        objective="寻找普通人参与的热点",
    )
    run = runtime.create_run(thread.id)
    context = ToolContext(thread.id, run.id, agent_id, 1)
    called: list[str] = []
    monkeypatch.setattr(
        service,
        "_collect_mastodon",
        lambda limit, *, instance, kind: called.append("mastodon") or [],
    )
    monkeypatch.setattr(
        service,
        "_collect_hacker_news",
        lambda limit: called.append("hacker_news") or [],
    )

    result = service.collect_public_trends(context, limit=5)

    assert result["sources"] == ["mastodon"]
    assert called == ["mastodon"]


def test_keyworded_mastodon_trends_do_not_persist_unrelated_global_posts(
    tmp_path: Path, monkeypatch
):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, instance_id="public-trends-query-filter")
    thread = runtime.create_thread(owner_agent_id=agent_id, title="文化活动", objective="找活动")
    run = runtime.create_run(thread.id)
    context = ToolContext(thread.id, run.id, agent_id, 1)
    monkeypatch.setattr(
        service,
        "_collect_mastodon",
        lambda limit, *, instance, kind: [
            {"title": "Unrelated political post", "snippet": "election argument", "url": "https://social.example/politics"},
            {"title": "Community art festival workshop", "snippet": "local hobby gathering", "url": "https://social.example/art"},
        ],
    )

    result = service.collect_public_trends(
        context,
        sources=["mastodon"],
        query="community art festival workshop local culture hobby gathering",
        limit=5,
        mastodon_kind="statuses",
    )

    assert [item["url"] for item in result["candidates"]] == ["https://social.example/art"]


def test_evidence_persists_content_value_for_reports(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, instance_id="value-evidence")
    thread = runtime.create_thread(owner_agent_id=agent_id, title="热点", objective="找热点")
    run = runtime.create_run(thread.id)
    context = ToolContext(thread.id, run.id, agent_id, 1)
    content_value = {
        "category": "生活",
        "event_summary": "城市社区启动夜间公共厨房计划并公布首批运营数据",
        "why_now": "首批三个月运营数据本周刚刚公开",
        "phenomenon": "居民正在用共享空间降低独居者晚餐成本并增加邻里互动",
        "specific_details": ["覆盖六个社区", "每周服务约八百人次"],
        "affected_groups": ["独居青年"],
        "discussion_evidence": "公开反馈中主要讨论价格和晚间安全",
        "niche_reason": "",
        "content_angle": "适合整理为城市共享厨房是否能改善独居生活的案例",
    }
    saved = service.save_evidence(
        context,
        {
            "source_url": "https://example.com/community-kitchen",
            "source_title": "Community kitchen report",
            "claim": "公共厨房公布首批运营数据",
            "excerpt": "Six neighborhoods now serve roughly 800 visits per week.",
            "source_type": "official_primary",
            "source_publisher": "Example City",
            "claim_key": "community-kitchen-launch",
            "verification_notes": "已读取原文",
            "content_value": content_value,
        },
    )
    assert saved["content_value"]["ready"] is True
    row = service.list_evidence(thread.id)[0]
    serialized = serialize_evidence(row)
    assert serialized["value_ready"] is True
    assert serialized["value_analysis"]["category"] == "生活"


def test_discovers_relevant_pages_from_site_owned_sitemap(tmp_path: Path, monkeypatch):
    _, service, _ = make_service(tmp_path)
    robots = b"User-agent: *\nSitemap: https://example.com/sitemap-index.xml\n"
    index = b'''<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.com/articles.xml</loc><lastmod>2026-08-01</lastmod></sitemap>
    </sitemapindex>'''
    urls = b'''<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/about</loc><lastmod>2025-01-01</lastmod></url>
      <url><loc>https://example.com/reports/gemini-400-million-users</loc><lastmod>2026-08-01</lastmod></url>
      <url><loc>https://example.com/products/search</loc><lastmod>2026-07-01</lastmod></url>
    </urlset>'''
    FakeHttpClient.responses = [
        FakeStreamResponse(headers={"content-type": "text/plain"}, content=robots),
        FakeStreamResponse(headers={"content-type": "application/xml"}, content=index),
        FakeStreamResponse(headers={"content-type": "application/xml"}, content=urls),
    ]
    FakeHttpClient.requested = []
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)

    found = service.discover_site_pages(
        "https://example.com/", "Example Gemini 400 million users", max_results=2
    )

    assert found["source"] == "site_robots_and_sitemaps"
    assert found["discovered_url_count"] == 3
    assert found["matching_url_count"] == 1
    assert found["results"][0]["url"].endswith("gemini-400-million-users")
    assert found["results"][0]["matched_terms"] == ["gemini", "400"]
    assert FakeHttpClient.requested == [
        "https://example.com/robots.txt",
        "https://example.com/sitemap-index.xml",
        "https://example.com/articles.xml",
    ]


def test_discovers_and_ranks_official_atom_feed(tmp_path: Path, monkeypatch):
    _, service, _ = make_service(tmp_path)
    homepage = b'''<html><head><link rel="alternate" type="application/atom+xml" href="/updates.atom"></head></html>'''
    atom = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>General update</title><link href="https://example.com/general"/><updated>2026-08-01</updated><summary>Other news</summary></entry>
      <entry><title>Gemini reaches 400 million users</title><link href="https://example.com/gemini?utm_source=feed"/><updated>2026-08-12</updated><summary>Official monthly active users update</summary></entry>
    </feed>'''
    FakeHttpClient.responses = [
        FakeStreamResponse(headers={"content-type": "text/html"}, content=homepage),
        FakeStreamResponse(headers={"content-type": "application/atom+xml"}, content=atom),
    ]
    FakeHttpClient.requested = []
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)

    found = service.discover_site_feeds(
        "https://example.com/", "Gemini 400 million users", max_results=5, max_feeds=1
    )

    assert found["source"] == "site_rss_atom"
    assert found["feeds_checked"] == [
        {"url": "https://example.com/updates.atom", "kind": "atom"}
    ]
    assert found["results"][0]["url"] == "https://example.com/gemini"
    assert "gemini" in found["results"][0]["matched_terms"]


def test_bounded_directory_exploration_stays_on_site_and_deduplicates(tmp_path: Path, monkeypatch):
    _, service, _ = make_service(tmp_path)
    start = b'''<html><head><title>News directory</title></head><body>
      <a href="/reports/gemini?utm_source=nav">Gemini users report</a>
      <a href="/reports/gemini">Duplicate</a>
      <a href="https://outside.test/report">Outside</a>
    </body></html>'''
    report = b'''<html><head><title>Gemini official report</title></head><body><p>Gemini has 400 million monthly active users.</p></body></html>'''
    FakeHttpClient.responses = [
        FakeStreamResponse(headers={"content-type": "text/html"}, content=start),
        FakeStreamResponse(headers={"content-type": "text/html"}, content=report),
    ]
    FakeHttpClient.requested = []
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)

    found = service.explore_site_directory(
        "https://example.com/news", "Gemini 400 million users", max_pages=2, max_depth=1
    )

    assert found["visited_count"] == 2
    assert found["pages_visited"] == [
        "https://example.com/news",
        "https://example.com/reports/gemini",
    ]
    assert all("outside.test" not in item for item in found["pages_visited"])
    assert found["results"][0]["url"] == "https://example.com/reports/gemini"


def test_candidate_pool_deduplicates_cross_channel_tracking_urls(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, instance_id="candidate-pool")
    thread = runtime.create_thread(owner_agent_id=agent_id, title="候选池", objective="去重")
    run = runtime.create_run(thread.id)
    context = ToolContext(thread.id, run.id, agent_id, 1)
    first = service._persist_candidates(
        context,
        "web_search",
        [{"url": "https://EXAMPLE.com/report/?utm_source=search", "title": "Report"}],
    )
    second = service._persist_candidates(
        context,
        "site_feed",
        [{"url": "https://example.com/report#section", "title": "Official report"}],
    )
    assert canonicalize_url(first[0]["url"]) == "https://example.com/report"
    assert second[0]["candidate_sources"] == ["web_search", "site_feed"]
    listed = service.list_candidates(thread.id)
    assert listed[0]["sources"] == ["web_search", "site_feed"]
    with database.sessions() as session:
        rows = session.scalars(select(ResearchCandidate)).all()
        assert len(rows) == 1


def test_provider_health_persists_failures_and_recovery(tmp_path: Path):
    database, service, _ = make_service(tmp_path)
    service._record_provider_health("brave_html", outcome="failure", latency_ms=100, error="captcha")
    service._record_provider_health("brave_html", outcome="failure", latency_ms=110, error="captcha")
    service._record_provider_health("brave_html", outcome="failure", latency_ms=120, error="captcha")
    failed = service.provider_health()[0]
    assert failed["status"] == "unavailable"
    assert failed["consecutive_failures"] == 3
    service._record_provider_health("brave_html", outcome="success", latency_ms=80)
    recovered = service.provider_health()[0]
    assert recovered["status"] == "healthy"
    assert recovered["consecutive_failures"] == 0
    with database.sessions() as session:
        assert session.query(ResearchProviderHealth).count() == 1


def test_new_research_tools_and_health_api_are_registered(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'research-api.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    from app.main import create_app
    from app.env_store import EnvStore
    from fastapi.testclient import TestClient

    class NoopCodex:
        def close(self):
            return None

    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        tools = {item["name"] for item in client.get("/api/tools").json()}
        assert {
            "collect_public_trends",
            "assess_content_value",
            "discover_site_feeds",
            "explore_site_directory",
            "get_research_provider_health",
            "list_research_candidates",
            "assess_research_quality",
            "review_research_evidence",
        } <= tools
        assert client.get("/api/research/provider-health").status_code == 200
        assert client.get("/api/research/provider-health").json() == []


def test_sitemap_partial_parse_keeps_complete_url_records():
    partial = b'''<?xml version="1.0"?><urlset>
      <url><loc>https://google.example/about</loc></url>
      <url><loc>https://google.example/products/lens-update</loc><lastmod>2026-08-01</lastmod></url>
      <url><loc>https://google.example/incomplete'''
    kind, rows = ResearchService._sitemap_rows(partial, allow_partial=True)
    assert kind == "urlset"
    assert rows == [
        {"url": "https://google.example/about", "lastmod": ""},
        {
            "url": "https://google.example/products/lens-update",
            "lastmod": "2026-08-01",
        },
    ]


def test_discovery_permission_inherits_web_search_but_can_be_overridden(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="permissions")
    registry = ToolRegistry()
    registry.register(ToolSpec("web_search", "搜索", {}, service.search_tool))
    registry.register(
        ToolSpec("collect_public_trends", "公开热点", {}, service.collect_public_trends_tool)
    )
    registry.register(
        ToolSpec("assess_content_value", "内容价值", {}, service.assess_content_value_tool)
    )
    registry.register(
        ToolSpec("discover_site_pages", "官网发现", {}, service.discover_site_pages_tool)
    )
    registry.register(
        ToolSpec("discover_site_feeds", "订阅发现", {}, service.discover_site_feeds_tool)
    )
    registry.register(
        ToolSpec("explore_site_directory", "目录探索", {}, service.explore_site_directory_tool)
    )
    registry.register(
        ToolSpec("assess_research_quality", "质量", {}, lambda *_: {})
    )
    executor = WorkExecutor(database, runtime, registry, lambda *_: "{}")

    executor.set_tool_permission(agent_id, "web_search", "allow")
    assert executor.list_tool_permissions(agent_id)["discover_site_pages"] == "allow"
    assert executor.list_tool_permissions(agent_id)["collect_public_trends"] == "allow"
    assert executor.list_tool_permissions(agent_id)["assess_content_value"] == "allow"
    assert executor.list_tool_permissions(agent_id)["discover_site_feeds"] == "allow"
    assert executor.list_tool_permissions(agent_id)["explore_site_directory"] == "allow"
    assert executor.list_tool_permissions(agent_id)["assess_research_quality"] == "allow"

    executor.set_tool_permission(agent_id, "discover_site_pages", "deny")
    assert executor.list_tool_permissions(agent_id)["discover_site_pages"] == "deny"


def test_research_rhythm_requires_inspection_after_repeated_discovery(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="rhythm")
    registry = ToolRegistry()
    registry.register(ToolSpec("web_search", "搜索", {}, service.search_tool))
    registry.register(ToolSpec("read_web_page", "阅读", {}, service.read_page_tool))
    executor = WorkExecutor(database, runtime, registry, lambda *_: "{}")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="节奏测试", objective="测试研究节奏"
    )
    run = runtime.create_run(thread.id)
    with database.sessions() as session:
        for index in range(8):
            session.add(
                ToolCall(
                    work_thread_id=thread.id,
                    work_run_id=run.id,
                    owner_agent_id=agent_id,
                    step_id=f"search-{index}",
                    action_key=f"rhythm-{index}",
                    tool_name="web_search",
                    arguments_json="{}",
                    status="completed",
                    permission="allow",
                    retry_safety="safe",
                    execution_token=1,
                )
            )
        session.commit()
        rhythm = executor._research_rhythm(session, thread.id)
        assert rhythm["guard_active"] is True
        assert rhythm["discovery_calls_without_page_inspection"] == 8
        session.add(
            ToolCall(
                work_thread_id=thread.id,
                work_run_id=run.id,
                owner_agent_id=agent_id,
                step_id="read",
                action_key="rhythm-read",
                tool_name="read_web_page",
                arguments_json="{}",
                status="completed",
                permission="allow",
                retry_safety="safe",
                execution_token=1,
            )
        )
        session.commit()
        assert executor._research_rhythm(session, thread.id)["guard_active"] is False


def test_search_domain_filter_rejects_provider_scope_leakage():
    results = [
        {"title": "Allowed", "url": "https://news.example.com/report", "snippet": "ok"},
        {"title": "Wrong", "url": "https://irrelevant.test/report", "snippet": "no"},
        {"title": "Lookalike", "url": "https://example.com.attacker.test/", "snippet": "no"},
    ]
    assert filter_search_results(results, ["example.com"]) == [results[0]]
    assert filter_search_results(results, []) == results


def test_search_query_normalization_and_relevance_filter():
    assert normalize_search_query(
        'site:blog.google Gemini "400 million" users', ["blog.google"]
    ) == 'Gemini "400 million" users site:blog.google'
    results = [
        {"title": "Gemini reaches 400 million", "url": "https://example.com/1", "snippet": "monthly active users"},
        {"title": "Example homepage", "url": "https://example.com/2", "snippet": "general news"},
    ]
    assert filter_relevant_results(
        results, 'Gemini "400 million" monthly active users'
    ) == [results[0]]


def test_search_relevance_does_not_accept_unrelated_result_by_year_only():
    unrelated = [
        {
            "title": "2026 FIFA World Cup Schedule",
            "url": "https://example.com/world-cup",
            "snippet": "Dates, stadiums and all 104 matches in 2026.",
        }
    ]
    assert filter_relevant_results(
        unrelated,
        "2026 recent community repair cafe clothing swap mending event participants",
    ) == []
    assert filter_relevant_results(
        unrelated,
        "2026 recent community beach cleanup volunteer activity discussion",
    ) == []


def test_search_relevance_uses_chinese_subject_terms_not_only_year():
    results = [
        {
            "title": "2026 年日历",
            "url": "https://example.com/calendar",
            "snippet": "全年日期和周数。",
        },
        {
            "title": "社区食物互助行动",
            "url": "https://example.com/community-fridge",
            "snippet": "志愿者补充共享冰箱并记录邻里响应。",
        },
    ]
    assert filter_relevant_results(
        results, "2026 近期 社区食物互助 志愿者 邻里响应 现场报道"
    ) == [results[1]]


def test_duckduckgo_parser_decodes_redirect_and_snippet():
    document = '''<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fblog.example.com%2Freport">Official &amp; report</a><a class="result__snippet">More than <b>400 million</b> monthly users.</a></div>'''
    results = parse_duckduckgo_results(document)
    assert results == [
        {
            "title": "Official & report",
            "url": "https://blog.example.com/report",
            "snippet": "More than 400 million monthly users.",
        }
    ]


def test_reader_revalidates_redirect_and_rejects_private_target(tmp_path: Path, monkeypatch):
    _, service, _ = make_service(tmp_path)
    FakeHttpClient.responses = [
        FakeStreamResponse(status_code=302, headers={"location": "http://127.0.0.1/secret"})
    ]
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)
    with pytest.raises(ResearchError, match="禁止访问"):
        service.read_page("https://example.com/start")


def test_reader_truncates_when_response_exceeds_limit(tmp_path: Path, monkeypatch):
    _, service, _ = make_service(tmp_path)
    FakeHttpClient.responses = [
        FakeStreamResponse(
            headers={"content-type": "text/plain", "content-length": "1000001"},
            content=b"ignored",
        )
    ]
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)
    page = service.read_page("https://example.com/large")
    assert page["truncated"] is True
    assert page["text"] == "ignored"


def test_reader_fallback_keeps_original_source_url_after_direct_403(tmp_path: Path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'reader-fallback.db'}")
    database.create_tables()
    service = ResearchService(
        database,
        settings={
            "research": {
                "reader_fallback_endpoint": "https://reader.example.invalid/{url}",
                "request_attempts": 1,
            }
        },
        resolver=PUBLIC_RESOLVER,
    )
    FakeHttpClient.responses = [
        FakeStreamResponse(status_code=403),
        FakeStreamResponse(
            headers={"content-type": "text/plain; charset=utf-8"},
            content=b"Title: Official report\nThe product has 400 million monthly users.",
        ),
    ]
    FakeHttpClient.requested = []
    monkeypatch.setattr("app.research_runtime.httpx.Client", FakeHttpClient)

    page = service.read_page("https://example.com/official")

    assert page["url"] == "https://example.com/official"
    assert page["resolved_url"] == "https://example.com/official"
    assert page["fetched_via"] == "reader_fallback"
    assert page["reader_url"] == "https://reader.example.invalid/https://example.com/official"
    assert "400 million monthly users" in page["text"]
    assert FakeHttpClient.requested == [
        "https://example.com/official",
        "https://reader.example.invalid/https://example.com/official",
    ]


def test_evidence_is_scoped_and_deduplicated(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="research")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="App规模研究", objective="寻找超过5000万注册用户的App"
    )
    context = ToolContext(thread.id, 1, agent_id, 1)
    values = {
        "source_url": "https://example.com/report",
        "source_title": "Official report",
        "claim": "Example App超过5000万注册用户",
        "excerpt": "We passed 50 million registered users.",
        "metric_name": "registered users",
        "metric_value": "50000000",
        "metric_unit": "users",
        "confidence": "high",
        "stance": "supports",
    }

    first = service.save_evidence(context, values)
    second = service.save_evidence(context, values)

    assert first["reused"] is False
    assert second == {"evidence_id": first["evidence_id"], "reused": True}
    with database.sessions() as session:
        rows = session.scalars(select(Evidence)).all()
        assert len(rows) == 1
        assert rows[0].metric_name == "registered users"


def test_research_quality_groups_independent_sources_and_conflicts(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="quality")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="交叉验证", objective="核对同一用户规模结论"
    )
    context = ToolContext(thread.id, 1, agent_id, 1)
    common = {
        "claim_key": "example-users-2026",
        "metric_name": "monthly users",
        "metric_value": "100000000",
        "metric_unit": "users",
        "confidence": "high",
    }
    service.save_evidence(context, {
        **common,
        "source_url": "https://company.example/report",
        "source_title": "Company report",
        "source_type": "official_primary",
        "source_publisher": "Example Company",
        "claim": "产品月度用户超过1亿",
        "excerpt": "More than 100 million monthly users.",
        "stance": "supports",
        "verification_notes": "正文明确给出月度用户口径",
    })
    service.save_evidence(context, {
        **common,
        "source_url": "https://regulator.example/filing",
        "source_title": "Regulatory filing",
        "source_type": "regulatory_primary",
        "source_publisher": "Market Regulator",
        "claim": "产品月度用户超过1亿",
        "excerpt": "The filing confirms more than 100 million monthly users.",
        "stance": "supports",
    })
    service.save_evidence(context, {
        **common,
        "source_url": "https://analyst.example/correction",
        "source_title": "Metric correction",
        "source_type": "independent_secondary",
        "source_publisher": "Independent Analyst",
        "claim": "披露可能是累计账户而非月度用户",
        "excerpt": "The number refers to registered accounts, not monthly users.",
        "stance": "refutes",
    })
    quality = ResearchQualityService(database).summary(thread.id)
    assert quality["independent_publisher_count"] == 3
    assert quality["primary_source_count"] == 2
    assert quality["contested_claim_count"] == 1
    claim = quality["claims"][0]
    assert claim["status"] == "contested"
    assert claim["support_count"] == 2
    assert claim["refute_count"] == 1


def test_controller_evidence_review_is_audited(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'evidence-review.db'}")
    database.create_tables()
    with database.sessions() as session:
        controller = AgentProfile(name="总控", is_controller=True)
        worker = AgentProfile(name="市场", is_controller=False)
        session.add_all([controller, worker])
        session.commit()
        controller_id, worker_id = controller.id, worker.id
    runtime = WorkRuntime(database)
    thread = runtime.create_thread(owner_agent_id=worker_id, title="证据复核", objective="复核")
    with database.sessions() as session:
        row = Evidence(
            work_thread_id=thread.id,
            owner_agent_id=worker_id,
            source_url="https://example.com/official",
            source_title="Official",
            claim="已披露用户数据",
            excerpt="Official excerpt",
            source_type="official_primary",
            source_publisher="Example",
            claim_key="example-users",
            confidence="high",
            stance="supports",
            content_hash="f" * 64,
        )
        session.add(row)
        session.commit()
        evidence_id = row.id
    quality = ResearchQualityService(database)
    reviewed = quality.review(
        evidence_id,
        reviewer_agent_id=controller_id,
        status="verified",
        notes="已核对原文、指标和时间",
    )
    assert reviewed.review_status == "verified"
    assert reviewed.reviewer_agent_id == controller_id
    with database.sessions() as session:
        event = session.scalar(
            select(WorkEvent).where(WorkEvent.event_type == "evidence_reviewed")
        )
        assert event is not None
    with pytest.raises(Exception, match="只有总控"):
        quality.review(
            evidence_id,
            reviewer_agent_id=worker_id,
            status="rejected",
            notes="员工不能复核",
        )


def test_candidate_tool_uses_small_bounded_pages(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="candidate-pages")
    thread = runtime.create_thread(
        owner_agent_id=agent_id,
        title="候选分页",
        objective="候选结果不能撑爆工作上下文",
    )
    run = runtime.create_run(thread.id)
    with database.sessions() as session:
        for index in range(25):
            session.add(
                ResearchCandidate(
                    work_thread_id=thread.id,
                    owner_agent_id=agent_id,
                    canonical_url=f"https://example.com/{index}",
                    title="标题" * 400,
                    snippet="摘要" * 2000,
                    sources_json='["test"]',
                )
            )
        session.commit()

    first = service.list_candidates_tool(
        ToolContext(thread.id, run.id, agent_id, 1), {}
    )

    assert first["limit"] == 10
    assert len(first["candidates"]) == 10
    assert first["has_more"] is True
    assert first["next_offset"] == 10
    assert max(len(item["title"]) for item in first["candidates"]) == 500
    assert max(len(item["snippet"]) for item in first["candidates"]) == 800

    second = service.list_candidates_tool(
        ToolContext(thread.id, run.id, agent_id, 1),
        {"offset": first["next_offset"], "limit": 100},
    )
    assert second["limit"] == 20
    assert len(second["candidates"]) == 15
    assert second["has_more"] is False


def test_agent_autonomously_searches_reads_and_saves_evidence(tmp_path: Path):
    database, service, agent_id = make_service(tmp_path)
    runtime = WorkRuntime(database, lease_seconds=60, instance_id="research")
    registry = ToolRegistry()
    registry.register(ToolSpec("web_search", "搜索", {}, service.search_tool))
    registry.register(ToolSpec("read_web_page", "读网页", {}, service.read_page_tool))
    registry.register(
        ToolSpec("save_evidence", "存证据", {}, service.save_evidence_tool, risk_level="write")
    )
    registry.register(ToolSpec("list_evidence", "读证据", {}, service.list_evidence_tool))
    service.search = lambda *_: {
        "query": "app registered users",
        "results": [{"title": "Official", "url": "https://example.com/report", "snippet": "50 million"}],
    }
    service.read_page = lambda *_: {
        "url": "https://example.com/report",
        "title": "Official",
        "text": "We passed 50 million registered users.",
        "truncated": False,
    }
    actions = [
        {"type": "tool_call", "step_id": "search", "tool_name": "web_search", "arguments": {"query": "app registered users"}, "purpose": "寻找候选来源", "criterion_id": "registered-users", "expected_delta": "得到官方候选URL", "stop_condition": "找到相关候选"},
        {"type": "plan_update"},
        {"type": "tool_call", "step_id": "evidence", "tool_name": "save_evidence", "arguments": {
            "source_url": "https://example.com/report", "source_title": "Official",
            "claim": "Example App超过5000万注册用户", "excerpt": "We passed 50 million registered users.",
            "metric_name": "registered users", "metric_value": "50000000", "metric_unit": "users",
            "confidence": "high", "stance": "supports"}, "purpose": "保存已经核验的正式证据", "criterion_id": "registered-users", "expected_delta": "成功标准获得证据引用", "stop_condition": "证据持久化成功"},
        {"type": "complete", "message": "找到1个符合条件的App并保存证据"},
    ]

    def provider(_, __, prompt):
        action = actions.pop(0)
        if action["type"] == "complete":
            assert "evidence_id" in prompt
        payload = {
            "state_summary": "研究中",
            "progress": 0.5,
            "plan": [],
            "action": action,
        }
        if action.get("step_id") == "read":
            payload.update(
                {
                    "state_update": {
                        "current_strategy": "先阅读搜索召回的官方代表页面",
                        "open_questions": ["正文是否给出注册用户口径"],
                    },
                    "observation_assessment": {
                        "information_gain": "useful",
                        "expectation_met": True,
                        "finding": "搜索找到一个官方候选页面",
                    },
                    "consumed_tool_call_ids": [1],
                }
            )
        elif action["type"] == "plan_update":
            payload.update(
                {
                    "state_update": {
                        "current_judgment": "正文明确给出5000万注册用户",
                        "verified_facts": [
                            "Official正文写明50 million registered users，来源https://example.com/report，tool_call:2"
                        ],
                        "open_questions": [],
                    },
                    "observation_assessment": {
                        "information_gain": "decisive",
                        "expectation_met": True,
                        "finding": "全文确认了指标数值与注册用户口径",
                    },
                    "consumed_tool_call_ids": [2],
                }
            )
        return __import__("json").dumps(payload, ensure_ascii=False)

    executor = WorkExecutor(database, runtime, registry, provider, max_steps_per_run=6)
    for name in ("web_search", "read_web_page", "save_evidence", "list_evidence"):
        executor.set_tool_permission(agent_id, name, "allow")
    thread = runtime.create_thread(
        owner_agent_id=agent_id, title="真实研究循环", objective="寻找注册用户超过5000万的App"
    )
    runtime.create_run(thread.id)
    run = runtime.claim_next(agent_id=agent_id)
    assert run is not None

    result = executor.execute(run.id, run.execution_token)

    assert result.outcome == "completed"
    assert result.steps_executed == 5
    evidence = service.list_evidence(thread.id)
    assert len(evidence) == 1
    assert evidence[0].confidence == "high"
