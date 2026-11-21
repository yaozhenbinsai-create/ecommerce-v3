from __future__ import annotations

from hashlib import sha256
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import base64
import gzip
import io
import ipaddress
import json
import math
import re
import socket
import time
from typing import Any, Callable
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlencode, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .database import Database
from .models import Evidence, ResearchCandidate, ResearchProviderHealth
from .research_quality_service import SOURCE_TYPES
from .tool_runtime import ToolContext, ToolRegistryError


class ResearchError(ToolRegistryError):
    pass


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored = 0
        self._in_title = False
        self.links: list[dict[str, str]] = []
        self._link_href = ""
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if tag == "title":
            self._in_title = True
        if tag == "a" and not self._ignored:
            self._link_href = dict(attrs).get("href") or ""
            self._link_text = []
        if tag in {"p", "div", "article", "section", "br", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._link_href:
            anchor = re.sub(r"\s+", " ", " ".join(self._link_text)).strip()
            self.links.append({"url": self._link_href, "text": anchor[:500]})
            self._link_href = ""
            self._link_text = []

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._link_href:
            self._link_text.append(data)
        self.parts.append(data)

    def result(self) -> tuple[str, str]:
        title = re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.parts))
        text = re.sub(r"\n\s*\n+", "\n", text).strip()
        return title, text


class _FeedLinkHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.feeds: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "link":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        rel = {item.lower() for item in values.get("rel", "").split()}
        content_type = values.get("type", "").lower()
        href = values.get("href", "").strip()
        if href and "alternate" in rel and any(
            marker in content_type for marker in ("rss", "atom", "xml")
        ):
            self.feeds.append(href)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonicalize_url(value: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not hostname:
        return str(value or "").strip().split("#", 1)[0]
    port = parsed.port
    netloc = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    tracking = {"fbclid", "gclid", "mc_cid", "mc_eid"}
    query = [
        (key, item)
        for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
        if not key.lower().startswith("utm_") and key.lower() not in tracking
        for item in values
    ]
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, urlencode(query, doseq=True), ""))


def deduplicate_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in results:
        canonical = canonicalize_url(str(item.get("url") or ""))
        if not canonical:
            continue
        current = merged.get(canonical)
        enriched = {**item, "url": canonical}
        if current is None:
            merged[canonical] = enriched
            continue
        if len(str(enriched.get("title") or "")) > len(str(current.get("title") or "")):
            current["title"] = enriched.get("title")
        if len(str(enriched.get("snippet") or "")) > len(str(current.get("snippet") or "")):
            current["snippet"] = enriched.get("snippet")
        current["score"] = max(float(current.get("score") or 0), float(enriched.get("score") or 0))
    return list(merged.values())


def _clean_value_text(value: Any, limit: int = 2000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _clean_value_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _clean_value_text(item, 500)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def assess_content_value(value: Any) -> dict[str, Any]:
    """Score whether a found item is useful content, rather than merely a headline."""
    raw = value if isinstance(value, dict) else {}
    analysis = {
        "category": _clean_value_text(raw.get("category"), 100),
        "event_summary": _clean_value_text(raw.get("event_summary")),
        "why_now": _clean_value_text(raw.get("why_now")),
        "phenomenon": _clean_value_text(raw.get("phenomenon")),
        "specific_details": _clean_value_list(raw.get("specific_details")),
        "affected_groups": _clean_value_list(raw.get("affected_groups")),
        "discussion_evidence": _clean_value_text(raw.get("discussion_evidence")),
        "niche_reason": _clean_value_text(raw.get("niche_reason")),
        "content_angle": _clean_value_text(raw.get("content_angle")),
    }
    score = 0
    score += 2 if len(analysis["event_summary"]) >= 12 else 0
    score += 2 if len(analysis["why_now"]) >= 8 else 0
    score += 2 if len(analysis["phenomenon"]) >= 10 else 0
    score += min(3, len(analysis["specific_details"]))
    score += 1 if analysis["affected_groups"] else 0
    score += 1 if len(analysis["discussion_evidence"]) >= 8 else 0
    score += 1 if len(analysis["niche_reason"]) >= 8 else 0
    score += 2 if len(analysis["content_angle"]) >= 10 else 0
    required = {
        "event_summary": len(analysis["event_summary"]) >= 12,
        "why_now": len(analysis["why_now"]) >= 8,
        "specific_details": len(analysis["specific_details"]) >= 2,
        "content_angle": len(analysis["content_angle"]) >= 10,
    }
    missing = [name for name, present in required.items() if not present]
    if not analysis["phenomenon"] and not analysis["discussion_evidence"]:
        missing.append("phenomenon_or_discussion_evidence")
    ready = score >= 9 and not missing
    return {
        "score": score,
        "max_score": 14,
        "ready": ready,
        "missing": missing,
        "analysis": analysis,
        "rule": "标题、榜单排名和一句摘要只能作为候选；达到9分且通过必填项后才可作为热点成果。",
    }


def _bing_target(href: str) -> str:
    parsed = urlsplit(href)
    if parsed.hostname and parsed.hostname.endswith("bing.com") and parsed.path.startswith("/ck/a"):
        encoded = (parse_qs(parsed.query).get("u") or [""])[0]
        if encoded.startswith("a1"):
            payload = encoded[2:]
            payload += "=" * (-len(payload) % 4)
            try:
                decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
                if decoded.startswith(("http://", "https://")):
                    return decoded
            except (ValueError, UnicodeDecodeError):
                pass
    return href


class _BingResultsHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_result = 0
        self._in_title = 0
        self._in_snippet = 0
        self._href = ""
        self._title: list[str] = []
        self._snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "li" and "b_algo" in classes:
            self._in_result = 1
            self._href, self._title, self._snippet = "", [], []
            return
        if not self._in_result:
            return
        if tag == "h2":
            self._in_title += 1
        elif tag == "a" and self._in_title and not self._href:
            self._href = values.get("href") or ""
        elif tag == "p" or "b_caption" in classes:
            self._in_snippet += 1

    def handle_endtag(self, tag: str) -> None:
        if not self._in_result:
            return
        if tag == "h2" and self._in_title:
            self._in_title -= 1
        elif tag == "p" and self._in_snippet:
            self._in_snippet -= 1
        elif tag == "li":
            title = re.sub(r"\s+", " ", " ".join(self._title)).strip()
            snippet = re.sub(r"\s+", " ", " ".join(self._snippet)).strip()
            href = _bing_target(self._href)
            if title and href.startswith(("http://", "https://")):
                self.results.append({"title": title, "url": href, "snippet": snippet[:1000]})
            self._in_result = 0

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)


class _BraveResultsHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_result = 0
        self._in_title = 0
        self._in_snippet = 0
        self._href = ""
        self._title: list[str] = []
        self._snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "div" and "snippet" in classes and values.get("data-type") == "web":
            self._in_result = 1
            self._href, self._title, self._snippet = "", [], []
            return
        if not self._in_result:
            return
        if tag == "a" and not self._href:
            href = values.get("href") or ""
            if href.startswith(("http://", "https://")):
                self._href = href
        if "search-snippet-title" in classes:
            self._in_title += 1
        elif "desktop-default-regular" in classes or "snippet-description" in classes:
            self._in_snippet += 1

    def handle_endtag(self, tag: str) -> None:
        if not self._in_result:
            return
        if tag == "div" and self._in_title:
            self._in_title -= 1
            return
        if tag == "div" and self._in_snippet:
            self._in_snippet -= 1
            return
        if tag == "div" and not self._in_title and not self._in_snippet:
            title = re.sub(r"\s+", " ", " ".join(self._title)).strip()
            snippet = re.sub(r"\s+", " ", " ".join(self._snippet)).strip()
            if title and self._href:
                self.results.append(
                    {"title": title, "url": self._href, "snippet": snippet[:1000]}
                )
            self._in_result = 0

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)


def parse_brave_results(document: str) -> list[dict[str, str]]:
    starts = [
        match.start()
        for match in re.finditer(
            r'<div\s+class="[^"]*\bsnippet\b[^"]*"[^>]*data-type="web"[^>]*>',
            document,
            flags=re.I,
        )
    ]
    results: list[dict[str, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(document)
        block = document[start:end]
        link_match = re.search(r'<a\s+[^>]*href="(https?://[^"]+)"', block, flags=re.I)
        title_match = re.search(
            r'<div\s+[^>]*class="[^"]*search-snippet-title[^"]*"[^>]*>(.*?)</div>',
            block,
            flags=re.I | re.S,
        )
        if not link_match or not title_match:
            continue
        title = html.unescape(re.sub(r"<[^>]+>", " ", title_match.group(1)))
        title = re.sub(r"\s+", " ", title).strip()
        snippet_match = re.search(
            r'<div\s+[^>]*class="[^"]*desktop-default-regular[^"]*"[^>]*>(.*?)</div>',
            block,
            flags=re.I | re.S,
        )
        snippet = ""
        if snippet_match:
            snippet = html.unescape(re.sub(r"<[^>]+>", " ", snippet_match.group(1)))
            snippet = re.sub(r"\s+", " ", snippet).strip()
        if title:
            results.append(
                {
                    "title": title,
                    "url": html.unescape(link_match.group(1)),
                    "snippet": snippet[:1000],
                }
            )
    return results


def parse_duckduckgo_results(document: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.I | re.S,
    )
    matches = list(pattern.finditer(document))
    for index, match in enumerate(matches):
        raw_url = html.unescape(match.group(1))
        parsed = urlsplit(raw_url)
        if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
            target = (parse_qs(parsed.query).get("uddg") or [""])[0]
            raw_url = target or raw_url
        title = html.unescape(re.sub(r"<[^>]+>", " ", match.group(2)))
        title = re.sub(r"\s+", " ", title).strip()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        block = document[match.end():block_end]
        snippet_match = re.search(
            r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
            block,
            flags=re.I | re.S,
        )
        snippet = ""
        if snippet_match:
            snippet = html.unescape(re.sub(r"<[^>]+>", " ", snippet_match.group(1)))
            snippet = re.sub(r"\s+", " ", snippet).strip()
        if title and raw_url.startswith(("http://", "https://")):
            results.append({"title": title, "url": raw_url, "snippet": snippet[:1000]})
    return results


Resolver = Callable[[str, int], list[str]]


# Some Windows DNS/proxy stacks advertise the Teredo transition prefix together
# with a usable public IPv4 address. Python classifies that prefix as non-global,
# even though it is neither a loopback nor an internal network destination.
# Ignore only that transition answer when a public IPv4 answer is also present;
# real private, loopback and link-local answers remain a hard failure.
_TEREDO_TRANSITION_NETWORK = ipaddress.ip_network("2001::/32")


def _resolve_host(hostname: str, port: int) -> list[str]:
    return list(
        {
            item[4][0]
            for item in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        }
    )


def validate_public_url(url: str, resolver: Resolver = _resolve_host) -> str:
    if not isinstance(url, str) or len(url) > 4000:
        raise ResearchError("网址无效或过长")
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ResearchError("只允许访问公开的HTTP或HTTPS网址")
    if parsed.username or parsed.password:
        raise ResearchError("网址不能包含用户名或密码")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise ResearchError("禁止访问本机或内部网络")
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not literal_ip.is_global:
        raise ResearchError("禁止访问本机、内网、保留或非公网地址")
    try:
        addresses = resolver(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except (OSError, socket.gaierror) as exc:
        raise ResearchError(f"网址解析失败：{exc}") from exc
    if not addresses:
        raise ResearchError("网址没有可访问的公网地址")
    resolved_addresses = [ipaddress.ip_address(address) for address in addresses]
    has_public_ipv4 = any(
        isinstance(address, ipaddress.IPv4Address) and address.is_global
        for address in resolved_addresses
    )
    for ip in resolved_addresses:
        if ip.is_global:
            continue
        if (
            has_public_ipv4
            and isinstance(ip, ipaddress.IPv6Address)
            and ip in _TEREDO_TRANSITION_NETWORK
        ):
            continue
        if not ip.is_global:
            raise ResearchError("禁止访问本机、内网、保留或非公网地址")
    return url.strip()


def filter_search_results(
    results: list[dict[str, str]], domains: list[str]
) -> list[dict[str, str]]:
    """Enforce domain scope when a provider ignores site: constraints."""
    if not domains:
        return results
    filtered: list[dict[str, str]] = []
    for item in results:
        hostname = (urlsplit(str(item.get("url") or "")).hostname or "").lower()
        if hostname and any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in domains
        ):
            filtered.append(item)
    return filtered


_SEARCH_STOPWORDS = {
    "about", "active", "app", "application", "company", "earnings", "for",
    "from", "how", "million", "official", "report", "search", "the", "this",
    "users", "with", "year", "recent", "recap", "event", "participants",
}

_SEARCH_CJK_STOPWORDS = {
    "近期", "最近", "已发生", "已结束", "活动", "事件", "普通人", "公众",
    "参与", "参与者", "公开", "公开报道", "现场", "现场报道", "报道", "复盘",
    "讨论", "反应", "2026年",
}


def search_relevance_tokens(query: str) -> tuple[list[str], list[str]]:
    latin = [
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._+-]*", query)
        if len(token) >= 3 and token.lower() not in _SEARCH_STOPWORDS
    ]
    cjk = [
        token
        for token in re.findall(r"[\u3400-\u9fff]{2,}", query)
        if token not in _SEARCH_CJK_STOPWORDS
    ]
    tokens = list(dict.fromkeys([*latin, *cjk]))[:24]
    semantic = [token for token in tokens if not any(char.isdigit() for char in token)]
    numeric = [token for token in tokens if any(char.isdigit() for char in token)]
    return semantic, numeric


def normalize_search_query(query: str, domains: list[str]) -> str:
    base_query = re.sub(r"\bsite:[^\s()]+", " ", query, flags=re.I)
    base_query = re.sub(r"\b(?:AND|OR)\b(?=\s*(?:AND|OR|\)|$))", " ", base_query, flags=re.I)
    base_query = re.sub(r"^\s*(?:AND|OR)\b|\b(?:AND|OR)\s*$", " ", base_query, flags=re.I)
    base_query = re.sub(r"\s+", " ", base_query).strip()
    if not domains:
        return base_query
    site_filter = " OR ".join(f"site:{domain}" for domain in domains[:5])
    if len(domains) == 1:
        return f"{base_query} {site_filter}".strip()
    return f"{base_query} ({site_filter})".strip()


def relaxed_search_query(query: str, domains: list[str]) -> str:
    """Shorten an over-specified empty query without changing its subject."""
    base_query = re.sub(r"\bsite:[^\s()]+", " ", str(query or ""), flags=re.I)
    parts = re.findall(r"[\u3400-\u9fff]{2,}|[a-zA-Z0-9][a-zA-Z0-9._+-]*", base_query)
    generic = {
        "近期", "最近", "具体", "公开", "讨论", "参与者", "活动回顾",
        "recent", "specific", "public", "discussion", "participants",
    }
    kept: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.lower()
        if part in generic or key in generic or key in seen or key.isdigit():
            continue
        kept.append(part)
        seen.add(key)
        if len(kept) >= 6:
            break
    if len(kept) < 2:
        return normalize_search_query(query, domains)
    has_cjk = bool(re.search(r"[\u3400-\u9fff]", query))
    anchors = ["活动回顾", "主办方", "参与者"] if has_cjk else ["event", "recap", "attendees"]
    subject_terms = kept[:4]
    return normalize_search_query(" ".join([*subject_terms, *anchors]), domains)


def news_event_search_query(query: str) -> str:
    """Build a compact news query around the concrete subject and participants."""
    parts = re.findall(r"[\u3400-\u9fff]{2,}|[a-zA-Z0-9][a-zA-Z0-9._+-]*", str(query or ""))
    generic = {
        "recent", "community", "local", "news", "newspaper", "event", "recap",
        "attendees", "participants", "reported", "experience", "2026",
        "近期", "最近", "社区", "活动", "事件", "新闻", "报道", "回顾", "活动回顾",
        "参与", "参与者", "普通人", "真实体验", "2026年",
    }
    kept: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.lower()
        if key in generic or part in generic or key.isdigit() or key in seen:
            continue
        kept.append(part)
        seen.add(key)
        if len(kept) >= 4:
            break
    if not kept:
        kept = parts[:3]
    anchors = ["主办方", "志愿者", "居民"] if re.search(r"[\u3400-\u9fff]", query) else ["volunteers", "residents"]
    return " ".join([*kept, *anchors]).strip()


def filter_relevant_results(
    results: list[dict[str, str]], query: str
) -> list[dict[str, str]]:
    semantic, numeric = search_relevance_tokens(query)
    if not semantic and not numeric:
        return results
    filtered: list[dict[str, str]] = []
    for item in results:
        haystack = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        semantic_matches = {token for token in semantic if token.lower() in haystack}
        numeric_matches = {token for token in numeric if token.lower() in haystack}
        # A year is a time constraint, not evidence that the result is about
        # the requested subject.  Numeric-only matching is allowed only when
        # the query itself contains no semantic term.
        if semantic:
            required = 1 if len(semantic) == 1 else 2
            relevant = len(semantic_matches) >= required
        else:
            relevant = bool(numeric_matches)
        if relevant:
            filtered.append(item)
    return filtered


class ResearchService:
    def __init__(
        self,
        database: Database,
        settings: dict[str, Any] | None = None,
        *,
        resolver: Resolver = _resolve_host,
    ) -> None:
        config = (settings or {}).get("research") or {}
        self.database = database
        self.resolver = resolver
        self.search_endpoint = str(
            config.get("search_endpoint") or "https://www.bing.com/search?format=rss&q={query}"
        )
        self.search_html_endpoint = str(
            config.get("search_html_endpoint")
            or "https://www.bing.com/search?setlang=en-US&cc=US&q={query}"
        )
        self.search_brave_endpoint = str(
            config.get("search_brave_endpoint")
            or "https://search.brave.com/search?source=web&q={query}&offset={offset}"
        )
        self.reader_fallback_endpoint = str(
            config.get("reader_fallback_endpoint") or ""
        ).strip()
        self.search_duckduckgo_endpoint = str(
            config.get("search_duckduckgo_endpoint") or ""
        ).strip()
        self.timeout_seconds = min(60.0, max(1.0, float(config.get("timeout_seconds", 20))))
        self.max_response_bytes = min(
            5_000_000, max(10_000, int(config.get("max_response_bytes", 1_000_000)))
        )
        self.max_page_chars = min(
            100_000, max(2_000, int(config.get("max_page_chars", 30_000)))
        )
        self.max_feed_bytes = min(
            5_000_000, max(100_000, int(config.get("max_feed_bytes", 2_000_000)))
        )
        self.max_directory_pages = min(
            30, max(1, int(config.get("max_directory_pages", 12)))
        )
        self.request_attempts = min(5, max(1, int(config.get("request_attempts", 3))))
        self.user_agent = str(
            config.get("user_agent")
            or "BossAgentResearch/1.0 (local read-only research application)"
        )[:500]

    def _record_provider_health(
        self,
        provider: str,
        *,
        outcome: str,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        with self.database.sessions() as session:
            row = session.get(ResearchProviderHealth, provider)
            if not row:
                row = ResearchProviderHealth(
                    provider=provider,
                    status="unknown",
                    success_count=0,
                    empty_count=0,
                    failure_count=0,
                    consecutive_failures=0,
                )
                session.add(row)
            row.last_checked_at = utc_now()
            row.last_latency_ms = max(0, latency_ms)
            if outcome == "success":
                row.status = "healthy"
                row.success_count += 1
                row.consecutive_failures = 0
                row.last_error = None
                row.last_success_at = row.last_checked_at
            elif outcome == "empty":
                row.status = "degraded"
                row.empty_count += 1
                row.consecutive_failures = 0
                row.last_error = error or "Provider返回空结果"
            else:
                row.failure_count += 1
                row.consecutive_failures += 1
                row.status = "unavailable" if row.consecutive_failures >= 3 else "degraded"
                row.last_error = (error or "Provider调用失败")[:2000]
            session.commit()

    def provider_health(self) -> list[dict[str, Any]]:
        with self.database.sessions() as session:
            rows = session.scalars(
                select(ResearchProviderHealth).order_by(ResearchProviderHealth.provider)
            ).all()
        return [
            {
                "provider": row.provider,
                "status": row.status,
                "success_count": row.success_count,
                "empty_count": row.empty_count,
                "failure_count": row.failure_count,
                "consecutive_failures": row.consecutive_failures,
                "last_latency_ms": row.last_latency_ms,
                "last_error": row.last_error,
                "last_checked_at": row.last_checked_at.isoformat(),
                "last_success_at": row.last_success_at.isoformat() if row.last_success_at else None,
            }
            for row in rows
        ]

    def list_candidates(
        self, thread_id: int, offset: int = 0, limit: int = 50
    ) -> list[dict[str, Any]]:
        offset = min(10_000, max(0, int(offset)))
        limit = min(100, max(1, int(limit)))
        with self.database.sessions() as session:
            rows = session.scalars(
                select(ResearchCandidate)
                .where(ResearchCandidate.work_thread_id == thread_id)
                .order_by(ResearchCandidate.last_seen_at.desc(), ResearchCandidate.id.desc())
                .offset(offset)
                .limit(limit)
            ).all()
        result = []
        for row in rows:
            source_metadata = json.loads(row.sources_json or "[]")
            source_metadata = source_metadata if isinstance(source_metadata, list) else []
            sources = [
                str(entry.get("provider") or entry.get("tool_name") or "").strip()
                if isinstance(entry, dict)
                else str(entry).strip()
                for entry in source_metadata
            ]
            result.append({
                "id": row.id,
                "work_thread_id": row.work_thread_id,
                "url": row.canonical_url,
                "title": row.title,
                "snippet": row.snippet,
                "sources": [source for source in sources if source],
                "source_metadata": source_metadata,
                "status": row.status,
                "discovery_tool_call_id": row.discovery_tool_call_id,
                "read_tool_call_id": row.read_tool_call_id,
                "source_rank": row.source_rank,
                "content_kind": row.content_kind,
                "content_chars": row.content_chars,
                "disposition_reason": row.disposition_reason,
                "first_seen_at": row.first_seen_at.isoformat(),
                "last_seen_at": row.last_seen_at.isoformat(),
            })
        return result

    def _persist_candidates(
        self, context: ToolContext | None, source: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        unique = deduplicate_candidates(results)
        if context is None:
            return unique
        with self.database.sessions() as session:
            for item in unique:
                canonical = canonicalize_url(str(item.get("url") or ""))
                row = session.scalar(
                    select(ResearchCandidate).where(
                        ResearchCandidate.work_thread_id == context.work_thread_id,
                        ResearchCandidate.canonical_url == canonical,
                    )
                )
                if not row:
                    row = ResearchCandidate(
                        work_thread_id=context.work_thread_id,
                        owner_agent_id=context.owner_agent_id,
                        canonical_url=canonical,
                    )
                    session.add(row)
                raw_sources = json.loads(row.sources_json or "[]")
                sources = []
                for entry in raw_sources if isinstance(raw_sources, list) else []:
                    if isinstance(entry, dict):
                        sources.append(entry)
                    elif str(entry).strip():
                        sources.append({"provider": str(entry).strip()})
                if not any(entry.get("provider") == source for entry in sources):
                    sources.append({"provider": source})
                row.sources_json = json.dumps(sources, ensure_ascii=False)
                row.title = str(item.get("title") or row.title)[:1000]
                row.snippet = str(item.get("snippet") or row.snippet)[:4000]
                row.last_seen_at = utc_now()
                item["candidate_sources"] = [
                    str(entry.get("provider") or entry.get("tool_name") or "").strip()
                    for entry in sources
                    if isinstance(entry, dict)
                    and str(entry.get("provider") or entry.get("tool_name") or "").strip()
                ]
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
        return unique

    def _get(
        self,
        url: str,
        *,
        allow_truncate: bool = False,
        max_bytes: int | None = None,
    ) -> tuple[str, str, bytes, bool]:
        current = validate_public_url(url, self.resolver)
        response_limit = min(
            5_000_000,
            max(10_000, int(max_bytes if max_bytes is not None else self.max_response_bytes)),
        )
        headers = {"User-Agent": self.user_agent, "Accept": "application/json,text/html,application/rss+xml,application/xml,text/plain;q=0.9,*/*;q=0.1"}
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=False) as client:
            for _ in range(4):
                validate_public_url(current, self.resolver)
                redirect_target: str | None = None
                last_request_error: httpx.RequestError | None = None
                for attempt in range(self.request_attempts):
                    try:
                        with client.stream("GET", current, headers=headers) as response:
                            if response.status_code in {301, 302, 303, 307, 308}:
                                location = response.headers.get("location")
                                if not location:
                                    raise ResearchError("网页重定向缺少目标地址")
                                redirect_target = validate_public_url(
                                    urljoin(current, location), self.resolver
                                )
                                break
                            if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < self.request_attempts:
                                time.sleep(0.2 * (attempt + 1))
                                continue
                            response.raise_for_status()
                            declared = response.headers.get("content-length")
                            declared_too_large = bool(
                                declared and int(declared) > response_limit
                            )
                            if declared_too_large and not allow_truncate:
                                raise ResearchError("网页内容超过读取上限")
                            chunks: list[bytes] = []
                            total = 0
                            truncated = declared_too_large
                            for chunk in response.iter_bytes():
                                remaining = response_limit - total
                                if remaining <= 0:
                                    truncated = True
                                    break
                                if len(chunk) > remaining:
                                    chunks.append(chunk[:remaining])
                                    total += remaining
                                    truncated = True
                                    break
                                chunks.append(chunk)
                                total += len(chunk)
                            return (
                                current,
                                response.headers.get("content-type", ""),
                                b"".join(chunks),
                                truncated,
                            )
                    except httpx.HTTPStatusError as exc:
                        raise ResearchError(f"网页返回HTTP {exc.response.status_code}") from exc
                    except httpx.RequestError as exc:
                        last_request_error = exc
                        if attempt + 1 < self.request_attempts:
                            time.sleep(0.2 * (attempt + 1))
                            continue
                    except ValueError as exc:
                        raise ResearchError("网页Content-Length无效") from exc
                if redirect_target:
                    current = redirect_target
                    continue
                if last_request_error:
                    raise ResearchError(f"网页读取失败：{last_request_error}") from last_request_error
        raise ResearchError("网页重定向次数过多")

    def _get_json(self, url: str, *, max_bytes: int = 1_000_000) -> Any:
        _, _, content, truncated = self._get(
            url, allow_truncate=False, max_bytes=max_bytes
        )
        if truncated:
            raise ResearchError("公开热点接口返回内容超过读取上限")
        try:
            return json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResearchError("公开热点接口没有返回有效JSON") from exc

    @staticmethod
    def _iso_from_unix(value: Any) -> str | None:
        try:
            return datetime.fromtimestamp(int(value), timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            return None

    def _collect_hacker_news(self, limit: int) -> list[dict[str, Any]]:
        ids = self._get_json(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            max_bytes=500_000,
        )
        if not isinstance(ids, list):
            raise ResearchError("Hacker News热门列表格式无效")
        results: list[dict[str, Any]] = []
        for rank, story_id in enumerate(ids[:limit], start=1):
            item = self._get_json(
                f"https://hacker-news.firebaseio.com/v0/item/{int(story_id)}.json",
                max_bytes=200_000,
            )
            if not isinstance(item, dict) or item.get("deleted") or item.get("dead"):
                continue
            title = _clean_value_text(item.get("title"), 1000)
            if not title:
                continue
            discussion_url = f"https://news.ycombinator.com/item?id={int(story_id)}"
            source_url = str(item.get("url") or discussion_url)
            score = int(item.get("score") or 0)
            comments = int(item.get("descendants") or 0)
            results.append(
                {
                    "platform": "hacker_news",
                    "rank": rank,
                    "external_id": str(story_id),
                    "title": title,
                    "snippet": f"{score} points · {comments} comments",
                    "summary": "公开榜单候选，必须继续读取原文与讨论后才能作为证据。",
                    "url": source_url,
                    "discussion_url": discussion_url,
                    "author": _clean_value_text(item.get("by"), 200),
                    "published_at": self._iso_from_unix(item.get("time")),
                    "engagement": {"score": score, "comments": comments},
                    "hot_reason": f"Hacker News Top Stories 当前第{rank}名，{score}分、{comments}条评论",
                    "needs_deep_read": True,
                }
            )
        return results

    def _collect_mastodon(
        self, limit: int, *, instance: str, kind: str
    ) -> list[dict[str, Any]]:
        base = validate_public_url(instance.rstrip("/"), self.resolver).rstrip("/")
        payload = self._get_json(
            f"{base}/api/v1/trends/{kind}?limit={limit}", max_bytes=1_500_000
        )
        if not isinstance(payload, list):
            raise ResearchError("Mastodon趋势接口格式无效")
        results: list[dict[str, Any]] = []
        for rank, item in enumerate(payload[:limit], start=1):
            if not isinstance(item, dict):
                continue
            history = item.get("history") if isinstance(item.get("history"), list) else []
            latest = history[0] if history and isinstance(history[0], dict) else {}
            if kind == "links":
                title = _clean_value_text(item.get("title"), 1000)
                url = str(item.get("url") or "")
                snippet = _clean_value_text(item.get("description"), 1500)
                author = _clean_value_text(item.get("author_name"), 200)
            elif kind == "tags":
                name = _clean_value_text(item.get("name"), 300)
                title = f"#{name}" if name else ""
                url = str(item.get("url") or f"{base}/tags/{quote(name)}")
                snippet = "Mastodon公开趋势标签"
                author = ""
            else:
                account = item.get("account") if isinstance(item.get("account"), dict) else {}
                content = re.sub(r"<[^>]+>", " ", str(item.get("content") or ""))
                title = _clean_value_text(content, 1000)
                url = str(item.get("url") or item.get("uri") or "")
                snippet = title
                author = _clean_value_text(account.get("acct"), 200)
            if not title or not url:
                continue
            uses = int(latest.get("uses") or 0)
            accounts = int(latest.get("accounts") or 0)
            engagement = {
                "uses": uses,
                "accounts": accounts,
                "favourites": int(item.get("favourites_count") or 0),
                "reblogs": int(item.get("reblogs_count") or 0),
                "replies": int(item.get("replies_count") or 0),
            }
            results.append(
                {
                    "platform": "mastodon",
                    "rank": rank,
                    "external_id": str(item.get("id") or item.get("name") or url),
                    "title": title,
                    "snippet": snippet,
                    "summary": "单个公开Mastodon实例的趋势候选，不代表全网热度。",
                    "url": url,
                    "author": author,
                    "published_at": item.get("created_at"),
                    "engagement": engagement,
                    "hot_reason": f"{base}公开趋势{kind}当前第{rank}名",
                    "scope_note": f"仅代表实例 {base} 的公开趋势",
                    "needs_deep_read": True,
                }
            )
        return results

    def _collect_bluesky(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        endpoint = (
            "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?"
            + urlencode({"q": query.strip(), "sort": "top", "limit": limit})
        )
        payload = self._get_json(endpoint, max_bytes=2_000_000)
        posts = payload.get("posts") if isinstance(payload, dict) else None
        if not isinstance(posts, list):
            raise ResearchError("Bluesky公开搜索接口格式无效")
        results: list[dict[str, Any]] = []
        for rank, item in enumerate(posts[:limit], start=1):
            if not isinstance(item, dict):
                continue
            author = item.get("author") if isinstance(item.get("author"), dict) else {}
            record = item.get("record") if isinstance(item.get("record"), dict) else {}
            text = _clean_value_text(record.get("text"), 1500)
            uri = str(item.get("uri") or "")
            handle = _clean_value_text(author.get("handle"), 300)
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            url = f"https://bsky.app/profile/{quote(handle)}/post/{quote(rkey)}" if handle and rkey else str(item.get("uri") or "")
            if not text or not url:
                continue
            engagement = {
                "likes": int(item.get("likeCount") or 0),
                "reposts": int(item.get("repostCount") or 0),
                "replies": int(item.get("replyCount") or 0),
                "quotes": int(item.get("quoteCount") or 0),
            }
            results.append(
                {
                    "platform": "bluesky",
                    "rank": rank,
                    "external_id": uri,
                    "title": text[:300],
                    "snippet": text,
                    "summary": f"Bluesky公开搜索中与“{query.strip()}”相关的高互动候选，不是平台全局热榜。",
                    "url": url,
                    "author": handle,
                    "published_at": record.get("createdAt") or item.get("indexedAt"),
                    "engagement": engagement,
                    "hot_reason": f"Bluesky公开搜索 top 排序第{rank}名",
                    "scope_note": "仅代表指定关键词的公开搜索结果",
                    "needs_deep_read": True,
                }
            )
        return results

    def collect_public_trends(
        self,
        context: ToolContext,
        *,
        sources: list[str] | None = None,
        query: str = "",
        limit: int = 10,
        mastodon_instance: str = "https://mastodon.social",
        mastodon_kind: str = "links",
    ) -> dict[str, Any]:
        allowed = {"hacker_news", "mastodon", "bluesky"}
        # Hacker News is valuable but strongly technology-biased. Broad calls
        # default to a general public-social source; Agents can request HN
        # explicitly when the research lens is technology.
        chosen = list(dict.fromkeys(sources or ["mastodon"]))
        if any(source not in allowed for source in chosen):
            raise ResearchError("sources只支持hacker_news、mastodon、bluesky")
        limit = min(20, max(1, int(limit)))
        if mastodon_kind not in {"links", "statuses", "tags"}:
            raise ResearchError("mastodon_kind只支持links、statuses或tags")
        batches: dict[str, list[dict[str, Any]]] = {}
        errors: list[dict[str, str]] = []
        for source in chosen:
            started = time.perf_counter()
            provider = f"public_trends_{source}"
            try:
                if source == "hacker_news":
                    items = self._collect_hacker_news(limit)
                elif source == "mastodon":
                    items = self._collect_mastodon(
                        limit, instance=mastodon_instance, kind=mastodon_kind
                    )
                else:
                    items = self._collect_bluesky(query, limit)
                    if not query.strip():
                        raise ResearchError("使用Bluesky公开搜索时必须提供query")
                if query.strip() and source in {"mastodon", "hacker_news"}:
                    # These endpoints expose a global trend feed and ignore the
                    # caller's query. Never label unrelated global posts as
                    # query-matched candidates; filter locally or return empty.
                    items = filter_relevant_results(items, query)
                outcome = "success" if items else "empty"
                self._record_provider_health(
                    provider,
                    outcome=outcome,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                batches[source] = self._persist_candidates(
                    context, provider, items
                )
            except ResearchError as exc:
                self._record_provider_health(
                    provider,
                    outcome="failure",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error=str(exc),
                )
                errors.append({"source": source, "error": str(exc)})
                batches[source] = []
        items = [item for source in chosen for item in batches[source]]
        return {
            "retrieved_at": utc_now().isoformat(),
            "sources": chosen,
            "query": query.strip() or None,
            "candidate_count": len(items),
            "candidates": items,
            "errors": errors,
            "candidate_only": True,
            "next_step": "由Agent按意外性、反差、跨圈层传播、现实关联和谈资自主挑选真正有趣的候选，再读取原文与讨论核验事实；仅标题或榜单排名不得保存为最终成果。",
        }

    def search(
        self,
        query: str,
        max_results: int = 8,
        offset: int = 0,
        domains: list[str] | None = None,
        research_role: str | None = None,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query or len(query) > 500:
            raise ResearchError("搜索词不能为空且不能超过500字")
        max_results = min(20, max(1, int(max_results)))
        offset = min(100, max(0, int(offset)))
        clean_domains: list[str] = []
        for domain in domains or []:
            value = str(domain).strip().lower()
            if not re.fullmatch(r"[a-z0-9.-]+", value) or ".." in value:
                raise ResearchError("domains只能包含有效域名")
            clean_domains.append(value)
        effective_query = normalize_search_query(query, clean_domains)
        first = offset + 1
        event_role = str(research_role or "") in {
            "live_event",
            "public_discussion",
            "general_discovery",
        }
        if event_role:
            news_query = news_event_search_query(query)
            news_endpoint = (
                "https://www.bing.com/news/search?q={query}&format=rss"
            ).format(query=quote_plus(news_query))
            started = time.perf_counter()
            try:
                news_url, _, news_content, _ = self._get(news_endpoint)
                news_root = ET.fromstring(news_content)
                news_results: list[dict[str, str]] = []
                for item in news_root.findall(".//item")[:max_results]:
                    title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    if "bing.com/news/apiclick.aspx" in link.lower():
                        link = (parse_qs(urlsplit(link).query).get("url") or [link])[0]
                    description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
                    description = re.sub(r"\s+", " ", description).strip()
                    if title and link:
                        news_results.append(
                            {
                                "title": title,
                                "url": link,
                                "snippet": description[:1000],
                                "published_at": (item.findtext("pubDate") or "").strip(),
                            }
                        )
                news_results = deduplicate_candidates(news_results)
                if news_results:
                    self._record_provider_health(
                        "bing_news_rss",
                        outcome="success",
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                    return {
                        "query": query,
                        "effective_query": effective_query,
                        "relaxed_query": news_query,
                        "result_count": len(news_results),
                        "empty_result": False,
                        "offset": offset,
                    "provider": "bing_news_rss",
                        "provider_url": news_url,
                        "domain_scope_relaxed": bool(clean_domains),
                        "results": news_results,
                    }
                self._record_provider_health(
                    "bing_news_rss",
                    outcome="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            except (ResearchError, ET.ParseError) as exc:
                self._record_provider_health(
                    "bing_news_rss",
                    outcome="failure",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error=str(exc),
                )
        brave_endpoint = self.search_brave_endpoint.format(
            query=quote_plus(effective_query), offset=offset
        )
        started = time.perf_counter()
        try:
            brave_url, _, brave_content, _ = self._get(
                brave_endpoint, allow_truncate=True
            )
            brave_results = parse_brave_results(
                brave_content.decode("utf-8", errors="replace")
            )
            brave_results = filter_search_results(brave_results, clean_domains)
            brave_results = filter_relevant_results(brave_results, query)
            if brave_results:
                self._record_provider_health(
                    "brave_html", outcome="success", latency_ms=int((time.perf_counter() - started) * 1000)
                )
                return {
                    "query": query,
                    "effective_query": effective_query,
                    "offset": offset,
                    "provider": "brave_html",
                    "provider_url": brave_url,
                    "results": deduplicate_candidates(brave_results)[:max_results],
                }
            self._record_provider_health(
                "brave_html", outcome="empty", latency_ms=int((time.perf_counter() - started) * 1000)
            )
        except ResearchError as exc:
            self._record_provider_health(
                "brave_html", outcome="failure", latency_ms=int((time.perf_counter() - started) * 1000), error=str(exc)
            )
        if self.search_duckduckgo_endpoint:
            duckduckgo_endpoint = self.search_duckduckgo_endpoint.format(
                query=quote_plus(effective_query), offset=offset
            )
            started = time.perf_counter()
            try:
                duckduckgo_url, _, duckduckgo_content, _ = self._get(
                    duckduckgo_endpoint, allow_truncate=True
                )
                duckduckgo_results = parse_duckduckgo_results(
                    duckduckgo_content.decode("utf-8", errors="replace")
                )
                duckduckgo_results = filter_search_results(
                    duckduckgo_results, clean_domains
                )
                duckduckgo_results = filter_relevant_results(
                    duckduckgo_results, query
                )
                if duckduckgo_results:
                    self._record_provider_health(
                        "duckduckgo_html", outcome="success", latency_ms=int((time.perf_counter() - started) * 1000)
                    )
                    return {
                        "query": query,
                        "effective_query": effective_query,
                        "offset": offset,
                        "provider": "duckduckgo_html",
                        "provider_url": duckduckgo_url,
                        "results": deduplicate_candidates(duckduckgo_results)[:max_results],
                    }
                self._record_provider_health(
                    "duckduckgo_html", outcome="empty", latency_ms=int((time.perf_counter() - started) * 1000)
                )
            except ResearchError as exc:
                self._record_provider_health(
                    "duckduckgo_html", outcome="failure", latency_ms=int((time.perf_counter() - started) * 1000), error=str(exc)
                )
        html_endpoint = self.search_html_endpoint.format(
            query=quote_plus(effective_query), first=first
        )
        started = time.perf_counter()
        try:
            html_url, _, html_content, _ = self._get(html_endpoint)
            parser = _BingResultsHTML()
            parser.feed(html_content.decode("utf-8", errors="replace"))
            html_results = filter_search_results(parser.results, clean_domains)
            html_results = filter_relevant_results(html_results, query)
            if html_results:
                self._record_provider_health(
                    "bing_html", outcome="success", latency_ms=int((time.perf_counter() - started) * 1000)
                )
                return {
                    "query": query,
                    "effective_query": effective_query,
                    "offset": offset,
                    "provider": "bing_html",
                    "provider_url": html_url,
                    "results": deduplicate_candidates(html_results)[:max_results],
                }
            self._record_provider_health(
                "bing_html", outcome="empty", latency_ms=int((time.perf_counter() - started) * 1000)
            )
        except ResearchError as exc:
            self._record_provider_health(
                "bing_html", outcome="failure", latency_ms=int((time.perf_counter() - started) * 1000), error=str(exc)
            )
        endpoint = self.search_endpoint.format(
            query=quote_plus(effective_query), first=first
        )
        started = time.perf_counter()
        try:
            final_url, _, content, _ = self._get(endpoint)
        except ResearchError as exc:
            self._record_provider_health(
                "bing_rss", outcome="failure", latency_ms=int((time.perf_counter() - started) * 1000), error=str(exc)
            )
            raise
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            self._record_provider_health(
                "bing_rss", outcome="failure", latency_ms=int((time.perf_counter() - started) * 1000), error="搜索服务没有返回有效RSS"
            )
            raise ResearchError("搜索服务没有返回有效RSS") from exc
        results: list[dict[str, str]] = []
        for item in root.findall(".//item")[:max_results]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
            description = re.sub(r"\s+", " ", description).strip()
            if link:
                results.append({"title": title, "url": link, "snippet": description[:1000]})
        results = filter_search_results(results, clean_domains)
        results = filter_relevant_results(results, query)
        results = deduplicate_candidates(results)
        relaxed_query = None
        if not results:
            candidate_query = relaxed_search_query(query, clean_domains)
            if candidate_query and candidate_query != effective_query:
                relaxed_query = candidate_query
                relaxed_endpoint = self.search_endpoint.format(
                    query=quote_plus(candidate_query), first=first
                )
                try:
                    relaxed_url, _, relaxed_content, _ = self._get(relaxed_endpoint)
                    relaxed_root = ET.fromstring(relaxed_content)
                    relaxed_results: list[dict[str, str]] = []
                    for item in relaxed_root.findall(".//item")[:max_results]:
                        title = (item.findtext("title") or "").strip()
                        link = (item.findtext("link") or "").strip()
                        description = re.sub(
                            r"<[^>]+>", " ", item.findtext("description") or ""
                        )
                        description = re.sub(r"\s+", " ", description).strip()
                        if link:
                            relaxed_results.append(
                                {"title": title, "url": link, "snippet": description[:1000]}
                            )
                    results = deduplicate_candidates(
                        filter_relevant_results(
                            filter_search_results(relaxed_results, clean_domains),
                            candidate_query,
                        )
                    )
                    if results:
                        final_url = relaxed_url
                except (ResearchError, ET.ParseError):
                    results = []
        self._record_provider_health(
            "bing_rss",
            outcome="success" if results else "empty",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return {
            "query": query,
            "effective_query": effective_query,
            "relaxed_query": relaxed_query,
            "result_count": len(results),
            "empty_result": not bool(results),
            "offset": offset,
            "provider": "bing_rss",
            "provider_url": final_url,
            "results": results,
        }

    def read_page(self, url: str) -> dict[str, Any]:
        original_url = validate_public_url(url, self.resolver)
        fetched_via = "direct"
        reader_url: str | None = None
        try:
            final_url, content_type, content, response_truncated = self._get(
                original_url, allow_truncate=True
            )
        except ResearchError as exc:
            if not self.reader_fallback_endpoint or not any(
                marker in str(exc)
                for marker in ("HTTP 401", "HTTP 403", "HTTP 406", "HTTP 429")
            ):
                raise
            reader_url = self.reader_fallback_endpoint.format(
                url=quote(original_url, safe=":/?&=%#")
            )
            final_url, content_type, content, response_truncated = self._get(
                reader_url, allow_truncate=True
            )
            fetched_via = "reader_fallback"
        lowered = content_type.lower()
        if lowered and not any(kind in lowered for kind in ("html", "text/plain", "xml")):
            raise ResearchError(f"不支持的网页内容类型：{content_type}")
        encoding_match = re.search(r"charset=([\w.-]+)", content_type, flags=re.I)
        encoding = encoding_match.group(1) if encoding_match else "utf-8"
        try:
            decoded = content.decode(encoding, errors="replace")
        except LookupError:
            decoded = content.decode("utf-8", errors="replace")
        if "html" in lowered or "<html" in decoded[:1000].lower():
            parser = _ReadableHTML()
            parser.feed(decoded)
            title, text = parser.result()
            links = self._normalize_page_links(
                parser.links,
                final_url if fetched_via == "direct" else original_url,
            )
        else:
            title, text = "", re.sub(r"\s+", " ", decoded).strip()
            links = []
        truncated = response_truncated or len(text) > self.max_page_chars
        return {
            "url": original_url,
            "resolved_url": final_url if fetched_via == "direct" else original_url,
            "fetched_via": fetched_via,
            "reader_url": reader_url,
            "title": title[:1000],
            "content_type": content_type,
            "text": text[: self.max_page_chars],
            "truncated": truncated,
            "links": links,
        }

    @staticmethod
    def _normalize_page_links(
        links: list[dict[str, str]], base_url: str, max_links: int = 80
    ) -> list[dict[str, str]]:
        base_host = (urlsplit(base_url).hostname or "").lower()
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in links:
            target = urljoin(base_url, str(item.get("url") or "")).split("#", 1)[0]
            parsed = urlsplit(target)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.hostname.lower() != base_host
                or target in seen
            ):
                continue
            seen.add(target)
            normalized.append({"url": target, "text": str(item.get("text") or "")[:500]})
            if len(normalized) >= max_links:
                break
        return normalized

    @staticmethod
    def _decode_sitemap(content: bytes) -> bytes:
        if not content.startswith(b"\x1f\x8b"):
            return content
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as archive:
                decoded = archive.read(5_000_001)
        except OSError as exc:
            raise ResearchError("站点地图gzip内容无效") from exc
        if len(decoded) > 5_000_000:
            raise ResearchError("站点地图解压后超过上限")
        return decoded

    @staticmethod
    def _sitemap_rows(
        content: bytes, *, allow_partial: bool = False
    ) -> tuple[str, list[dict[str, str]]]:
        decoded = ResearchService._decode_sitemap(content)
        try:
            root = ET.fromstring(decoded)
        except ET.ParseError as exc:
            if not allow_partial:
                raise ResearchError("站点地图不是有效XML") from exc
            document = decoded.decode("utf-8", errors="replace")
            lowered = document[:2000].lower()
            if "sitemapindex" in lowered:
                kind, row_tag = "sitemapindex", "sitemap"
            elif "urlset" in lowered:
                kind, row_tag = "urlset", "url"
            else:
                raise ResearchError("截断内容无法识别为站点地图") from exc
            rows: list[dict[str, str]] = []
            pattern = rf"<(?:[\w.-]+:)?{row_tag}\b[^>]*>(.*?)</(?:[\w.-]+:)?{row_tag}>"
            for match in re.finditer(pattern, document, flags=re.I | re.S):
                block = match.group(1)
                location_match = re.search(
                    r"<(?:[\w.-]+:)?loc\b[^>]*>(.*?)</(?:[\w.-]+:)?loc>",
                    block,
                    flags=re.I | re.S,
                )
                if not location_match:
                    continue
                lastmod_match = re.search(
                    r"<(?:[\w.-]+:)?lastmod\b[^>]*>(.*?)</(?:[\w.-]+:)?lastmod>",
                    block,
                    flags=re.I | re.S,
                )
                rows.append(
                    {
                        "url": html.unescape(location_match.group(1).strip()),
                        "lastmod": html.unescape(lastmod_match.group(1).strip())[:80]
                        if lastmod_match
                        else "",
                    }
                )
                if len(rows) >= 10_000:
                    break
            if not rows:
                raise ResearchError("截断的站点地图没有完整网址记录") from exc
            return kind, rows
        kind = root.tag.rsplit("}", 1)[-1].lower()
        if kind not in {"sitemapindex", "urlset"}:
            raise ResearchError("XML不是sitemapindex或urlset")
        rows: list[dict[str, str]] = []
        row_tag = "sitemap" if kind == "sitemapindex" else "url"
        for element in root:
            if element.tag.rsplit("}", 1)[-1].lower() != row_tag:
                continue
            values = {
                child.tag.rsplit("}", 1)[-1].lower(): (child.text or "").strip()
                for child in element
            }
            location = values.get("loc", "")
            if location:
                rows.append({"url": html.unescape(location), "lastmod": values.get("lastmod", "")[:80]})
            if len(rows) >= 10_000:
                break
        return kind, rows

    def discover_site_pages(
        self,
        base_url: str,
        query: str,
        max_results: int = 12,
        max_sitemaps: int = 12,
    ) -> dict[str, Any]:
        """Discover candidate pages from a public site's own robots/sitemaps."""
        base_url = validate_public_url(str(base_url or "").strip(), self.resolver)
        query = str(query or "").strip()
        if not query or len(query) > 500:
            raise ResearchError("发现目标不能为空且不能超过500字")
        max_results = min(30, max(1, int(max_results)))
        max_sitemaps = min(30, max(1, int(max_sitemaps)))
        parsed_base = urlsplit(base_url)
        origin = f"{parsed_base.scheme}://{parsed_base.netloc}"
        robots_url = urljoin(origin, "/robots.txt")
        sitemap_queue: list[str] = []
        warnings: list[str] = []
        try:
            _, _, robots_content, _ = self._get(robots_url, allow_truncate=True)
            robots_text = robots_content.decode("utf-8", errors="replace")
            for match in re.finditer(r"^\s*Sitemap\s*:\s*(\S+)", robots_text, flags=re.I | re.M):
                sitemap_queue.append(urljoin(origin, match.group(1).strip()))
        except ResearchError as exc:
            warnings.append(f"robots.txt读取失败：{exc}")
        if not sitemap_queue:
            sitemap_queue.extend(
                [urljoin(origin, path) for path in ("/sitemap.xml", "/sitemap_index.xml", "/news-sitemap.xml")]
            )

        visited: set[str] = set()
        visited_order: list[str] = []
        pages: dict[str, dict[str, str]] = {}
        while sitemap_queue and len(visited) < max_sitemaps:
            sitemap_url = sitemap_queue.pop(0).split("#", 1)[0]
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            visited_order.append(sitemap_url)
            try:
                final_url, _, content, truncated = self._get(
                    sitemap_url, allow_truncate=True, max_bytes=5_000_000
                )
                kind, rows = self._sitemap_rows(content, allow_partial=truncated)
            except ResearchError as exc:
                if len(warnings) < 10:
                    warnings.append(f"{sitemap_url}：{exc}")
                continue
            if kind == "sitemapindex":
                for row in rows:
                    child = urljoin(final_url, row["url"]).split("#", 1)[0]
                    if child not in visited and child not in sitemap_queue:
                        sitemap_queue.append(child)
            else:
                for row in rows:
                    target = urljoin(final_url, row["url"]).split("#", 1)[0]
                    target_parts = urlsplit(target)
                    if target_parts.scheme in {"http", "https"} and target_parts.hostname:
                        pages.setdefault(target, {"url": target, "lastmod": row["lastmod"]})
                    if len(pages) >= 20_000:
                        break
                if len(pages) >= 20_000:
                    break

        site_tokens = {
            token
            for token in re.split(r"[^a-zA-Z0-9]+", parsed_base.hostname or "")
            if len(token) >= 2
        } | {"www", "blog", "com", "org", "net"}
        tokens = list(
            dict.fromkeys(
                token.lower()
                for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._+-]*|[\u4e00-\u9fff]{2,}", query)
                if len(token) >= 2
                and token.lower() not in _SEARCH_STOPWORDS
                and token.lower() not in site_tokens
            )
        )[:30]
        page_matches: list[tuple[dict[str, str], list[str]]] = []
        document_frequency = {token: 0 for token in tokens}
        for page in pages.values():
            page_parts = urlsplit(page["url"])
            haystack = unquote(f"{page_parts.path} {page_parts.query}").lower()
            matched = [token for token in tokens if token in haystack]
            page_matches.append((page, matched))
            for token in matched:
                document_frequency[token] += 1
        ranked: list[dict[str, Any]] = []
        document_count = max(1, len(pages))
        for page, matched in page_matches:
            score = round(
                sum(
                    (math.log((document_count + 1) / (document_frequency[token] + 1)) + 1)
                    * (1 + math.log1p(len(token)))
                    for token in matched
                ),
                4,
            )
            ranked.append({**page, "score": score, "matched_terms": matched})
        ranked.sort(key=lambda item: (item["score"], item["lastmod"], item["url"]), reverse=True)
        matching = [item for item in ranked if item["score"] > 0]
        selected = (matching or ranked)[:max_results]
        return {
            "base_url": base_url,
            "query": query,
            "source": "site_robots_and_sitemaps",
            "robots_url": robots_url,
            "sitemaps_checked": visited_order,
            "discovered_url_count": len(pages),
            "matching_url_count": len(matching),
            "results": selected,
            "warnings": warnings,
        }

    @staticmethod
    def _feed_entries(content: bytes, base_url: str) -> tuple[str, list[dict[str, str]]]:
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ResearchError("订阅源不是有效RSS或Atom XML") from exc
        root_name = root.tag.rsplit("}", 1)[-1].lower()
        entries: list[dict[str, str]] = []
        if root_name == "rss" or root.find(".//channel") is not None:
            kind = "rss"
            for item in root.findall(".//item"):
                link = (item.findtext("link") or "").strip()
                title = (item.findtext("title") or "").strip()
                description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
                published = (item.findtext("pubDate") or item.findtext("date") or "").strip()
                if link:
                    entries.append(
                        {
                            "url": canonicalize_url(urljoin(base_url, link)),
                            "title": title[:1000],
                            "snippet": re.sub(r"\s+", " ", description).strip()[:2000],
                            "published_at": published[:100],
                        }
                    )
        elif root_name == "feed":
            kind = "atom"
            namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
            prefix = f"{{{namespace}}}" if namespace else ""
            for entry in root.findall(f".//{prefix}entry"):
                link = ""
                for link_node in entry.findall(f"{prefix}link"):
                    rel = (link_node.attrib.get("rel") or "alternate").lower()
                    href = (link_node.attrib.get("href") or "").strip()
                    if href and rel in {"alternate", ""}:
                        link = href
                        break
                title = (entry.findtext(f"{prefix}title") or "").strip()
                summary = (
                    entry.findtext(f"{prefix}summary")
                    or entry.findtext(f"{prefix}content")
                    or ""
                )
                published = (
                    entry.findtext(f"{prefix}published")
                    or entry.findtext(f"{prefix}updated")
                    or ""
                ).strip()
                if link:
                    entries.append(
                        {
                            "url": canonicalize_url(urljoin(base_url, link)),
                            "title": re.sub(r"<[^>]+>", " ", title)[:1000],
                            "snippet": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", summary)).strip()[:2000],
                            "published_at": published[:100],
                        }
                    )
        else:
            raise ResearchError("XML不是RSS或Atom订阅源")
        return kind, deduplicate_candidates(entries)

    def discover_site_feeds(
        self,
        base_url: str,
        query: str,
        max_results: int = 20,
        max_feeds: int = 5,
    ) -> dict[str, Any]:
        base_url = validate_public_url(str(base_url or "").strip(), self.resolver)
        query = str(query or "").strip()
        if len(query) > 500:
            raise ResearchError("发现目标不能超过500字")
        max_results = min(50, max(1, int(max_results)))
        max_feeds = min(10, max(1, int(max_feeds)))
        parsed = urlsplit(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        candidates: list[str] = []
        warnings: list[str] = []
        try:
            final_url, content_type, content, _ = self._get(base_url, allow_truncate=True)
            decoded = content.decode("utf-8", errors="replace")
            if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
                parser = _FeedLinkHTML()
                parser.feed(decoded)
                candidates.extend(urljoin(final_url, item) for item in parser.feeds)
        except ResearchError as exc:
            warnings.append(f"首页订阅发现失败：{exc}")
        candidates.extend(
            urljoin(origin, path)
            for path in ("/feed", "/feed.xml", "/rss", "/rss.xml", "/atom.xml", "/index.xml")
        )
        feed_urls = list(dict.fromkeys(canonicalize_url(item) for item in candidates))[:max_feeds]
        entries: list[dict[str, Any]] = []
        checked: list[dict[str, str]] = []
        query_tokens = [
            token.lower() for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._+-]*|[\u4e00-\u9fff]{2,}", query)
            if len(token) >= 2 and token.lower() not in _SEARCH_STOPWORDS
        ][:30]
        for feed_url in feed_urls:
            try:
                resolved, _, content, truncated = self._get(
                    feed_url, allow_truncate=True, max_bytes=self.max_feed_bytes
                )
                if truncated:
                    warnings.append(f"{feed_url}：订阅源超过上限，已截断且未解析")
                    continue
                kind, rows = self._feed_entries(content, resolved)
                checked.append({"url": resolved, "kind": kind})
                for row in rows:
                    haystack = f"{row.get('title', '')} {row.get('snippet', '')} {unquote(urlsplit(row['url']).path)}".lower()
                    matched = [token for token in query_tokens if token in haystack]
                    row["matched_terms"] = matched
                    row["score"] = len(matched)
                    row["feed_url"] = resolved
                    entries.append(row)
            except ResearchError as exc:
                if len(warnings) < 12:
                    warnings.append(f"{feed_url}：{exc}")
        unique = deduplicate_candidates(entries)
        unique.sort(key=lambda item: (float(item.get("score") or 0), item.get("published_at") or ""), reverse=True)
        matching = [item for item in unique if item.get("score", 0) > 0]
        return {
            "base_url": base_url,
            "query": query,
            "source": "site_rss_atom",
            "feeds_checked": checked,
            "discovered_entry_count": len(unique),
            "matching_entry_count": len(matching),
            "results": (matching or unique)[:max_results],
            "warnings": warnings,
        }

    def explore_site_directory(
        self,
        start_url: str,
        query: str,
        max_pages: int = 12,
        max_depth: int = 2,
        max_results: int = 20,
    ) -> dict[str, Any]:
        start_url = validate_public_url(str(start_url or "").strip(), self.resolver)
        query = str(query or "").strip()
        if not query or len(query) > 500:
            raise ResearchError("探索目标不能为空且不能超过500字")
        max_pages = min(self.max_directory_pages, max(1, int(max_pages)))
        max_depth = min(3, max(0, int(max_depth)))
        max_results = min(50, max(1, int(max_results)))
        host = (urlsplit(start_url).hostname or "").lower()
        tokens = [
            token.lower() for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._+-]*|[\u4e00-\u9fff]{2,}", query)
            if len(token) >= 2 and token.lower() not in _SEARCH_STOPWORDS
        ][:30]
        queue: list[tuple[str, int]] = [(canonicalize_url(start_url), 0)]
        queued = {canonicalize_url(start_url)}
        visited: list[str] = []
        found: list[dict[str, Any]] = []
        warnings: list[str] = []
        while queue and len(visited) < max_pages:
            current, depth = queue.pop(0)
            try:
                page = self.read_page(current)
            except ResearchError as exc:
                if len(warnings) < 12:
                    warnings.append(f"{current}：{exc}")
                continue
            visited.append(current)
            haystack = f"{page.get('title', '')} {page.get('text', '')[:5000]} {unquote(urlsplit(current).path)}".lower()
            matched = [token for token in tokens if token in haystack]
            if matched:
                found.append(
                    {
                        "url": current,
                        "title": page.get("title") or "",
                        "snippet": str(page.get("text") or "")[:1000],
                        "depth": depth,
                        "matched_terms": matched,
                        "score": len(matched),
                    }
                )
            if depth >= max_depth:
                continue
            links = page.get("links") or []
            ranked_links = sorted(
                links,
                key=lambda item: sum(
                    1 for token in tokens
                    if token in f"{item.get('text', '')} {unquote(urlsplit(item.get('url', '')).path)}".lower()
                ),
                reverse=True,
            )
            for item in ranked_links:
                target = canonicalize_url(str(item.get("url") or ""))
                if (
                    target in queued
                    or (urlsplit(target).hostname or "").lower() != host
                    or len(queued) >= max_pages * 8
                ):
                    continue
                queued.add(target)
                queue.append((target, depth + 1))
        unique = deduplicate_candidates(found)
        unique.sort(key=lambda item: (float(item.get("score") or 0), -int(item.get("depth") or 0)), reverse=True)
        return {
            "start_url": start_url,
            "query": query,
            "source": "bounded_site_directory",
            "pages_visited": visited,
            "visited_count": len(visited),
            "queued_count": len(queued),
            "results": unique[:max_results],
            "warnings": warnings,
        }

    def save_evidence(self, context: ToolContext, values: dict[str, Any]) -> dict[str, Any]:
        required = {"source_url", "source_title", "claim", "excerpt"}
        missing = [name for name in required if not str(values.get(name) or "").strip()]
        if missing:
            raise ResearchError(f"证据缺少字段：{', '.join(sorted(missing))}")
        source_url = validate_public_url(str(values["source_url"]), self.resolver)
        confidence = str(values.get("confidence") or "medium")
        stance = str(values.get("stance") or "supports")
        if confidence not in {"low", "medium", "high"}:
            raise ResearchError("confidence必须是low、medium或high")
        if stance not in {"supports", "refutes", "context"}:
            raise ResearchError("stance必须是supports、refutes或context")
        source_type = str(values.get("source_type") or "unknown")
        if source_type not in SOURCE_TYPES:
            raise ResearchError("source_type不是支持的来源类型")
        content_value_supplied = isinstance(values.get("content_value"), dict)
        value_result = assess_content_value(values.get("content_value"))
        normalized = json.dumps(
            {"source_url": source_url, "claim": values["claim"], "excerpt": values["excerpt"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = sha256(normalized.encode("utf-8")).hexdigest()
        with self.database.sessions() as session:
            existing = session.scalar(
                select(Evidence).where(
                    Evidence.work_thread_id == context.work_thread_id,
                    Evidence.content_hash == content_hash,
                )
            )
            if existing:
                changed = False
                if existing.source_type == "unknown" and source_type != "unknown":
                    existing.source_type = source_type
                    changed = True
                for field, limit in (
                    ("source_publisher", 300),
                    ("claim_key", 200),
                    ("verification_notes", 10000),
                ):
                    value = values.get(field)
                    if value and not getattr(existing, field):
                        setattr(existing, field, str(value)[:limit])
                        changed = True
                if value_result["score"] > existing.value_score:
                    existing.value_analysis_json = json.dumps(
                        value_result["analysis"], ensure_ascii=False
                    )
                    existing.value_score = value_result["score"]
                    changed = True
                if changed:
                    session.commit()
                result = {"evidence_id": existing.id, "reused": True}
                if content_value_supplied:
                    result["content_value"] = value_result
                if changed:
                    result["metadata_updated"] = True
                return result
            row = Evidence(
                work_thread_id=context.work_thread_id,
                owner_agent_id=context.owner_agent_id,
                source_url=source_url,
                source_title=str(values["source_title"])[:1000],
                published_at=(str(values.get("published_at"))[:80] if values.get("published_at") else None),
                claim=str(values["claim"])[:10000],
                excerpt=str(values["excerpt"])[:20000],
                metric_name=(str(values.get("metric_name"))[:200] if values.get("metric_name") else None),
                metric_value=(str(values.get("metric_value"))[:200] if values.get("metric_value") else None),
                metric_unit=(str(values.get("metric_unit"))[:100] if values.get("metric_unit") else None),
                confidence=confidence,
                stance=stance,
                source_type=source_type,
                source_publisher=(str(values.get("source_publisher"))[:300] if values.get("source_publisher") else None),
                claim_key=(str(values.get("claim_key"))[:200] if values.get("claim_key") else None),
                verification_notes=(str(values.get("verification_notes"))[:10000] if values.get("verification_notes") else None),
                value_analysis_json=json.dumps(
                    value_result["analysis"], ensure_ascii=False
                ),
                value_score=value_result["score"],
                content_hash=content_hash,
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.scalar(
                    select(Evidence).where(
                        Evidence.work_thread_id == context.work_thread_id,
                        Evidence.content_hash == content_hash,
                    )
                )
                if not existing:
                    raise
                result = {"evidence_id": existing.id, "reused": True}
                if content_value_supplied:
                    result["content_value"] = value_result
                return result
            session.refresh(row)
            result = {"evidence_id": row.id, "reused": False}
            if content_value_supplied:
                result["content_value"] = value_result
            return result

    def list_evidence(self, thread_id: int) -> list[Evidence]:
        with self.database.sessions() as session:
            return session.scalars(
                select(Evidence).where(Evidence.work_thread_id == thread_id).order_by(Evidence.id)
            ).all()

    def search_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        value = self.search(
            arguments.get("query"),
            arguments.get("max_results", 8),
            arguments.get("offset", 0),
            arguments.get("domains"),
            arguments.get("research_role"),
        )
        value["results"] = self._persist_candidates(
            context, str(value.get("provider") or "web_search"), value.get("results") or []
        )
        return value

    def read_page_tool(self, _: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.read_page(str(arguments.get("url") or ""))

    def discover_site_pages_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        value = self.discover_site_pages(
            str(arguments.get("base_url") or ""),
            str(arguments.get("query") or ""),
            arguments.get("max_results", 12),
            arguments.get("max_sitemaps", 12),
        )
        value["results"] = self._persist_candidates(
            context, "site_sitemap", value.get("results") or []
        )
        return value

    def discover_site_feeds_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        value = self.discover_site_feeds(
            str(arguments.get("base_url") or ""),
            str(arguments.get("query") or ""),
            arguments.get("max_results", 20),
            arguments.get("max_feeds", 5),
        )
        value["results"] = self._persist_candidates(
            context, "site_feed", value.get("results") or []
        )
        return value

    def explore_site_directory_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        value = self.explore_site_directory(
            str(arguments.get("start_url") or ""),
            str(arguments.get("query") or ""),
            arguments.get("max_pages", self.max_directory_pages),
            arguments.get("max_depth", 2),
            arguments.get("max_results", 20),
        )
        value["results"] = self._persist_candidates(
            context, "site_directory", value.get("results") or []
        )
        return value

    def provider_health_tool(
        self, _: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if arguments:
            raise ResearchError("get_research_provider_health不接受参数")
        return {"providers": self.provider_health()}

    def list_candidates_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        offset = min(10_000, max(0, int(arguments.get("offset") or 0)))
        limit = min(20, max(1, int(arguments.get("limit") or 10)))
        rows = self.list_candidates(
            context.work_thread_id,
            offset,
            limit + 1,
        )
        candidates = []
        for item in rows[:limit]:
            value = dict(item)
            value["title"] = str(value.get("title") or "")[:500]
            value["snippet"] = str(value.get("snippet") or "")[:800]
            candidates.append(value)
        has_more = len(rows) > limit
        return {
            "offset": offset,
            "limit": limit,
            "candidates": candidates,
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
            "instruction": (
                "如需更多候选，保持limit不变并使用next_offset翻页；"
                "不要扩大单页来绕过上下文预算。"
            ),
        }

    def collect_public_trends_tool(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.collect_public_trends(
            context,
            sources=arguments.get("sources"),
            query=str(arguments.get("query") or ""),
            limit=arguments.get("limit", 10),
            mastodon_instance=str(
                arguments.get("mastodon_instance") or "https://mastodon.social"
            ),
            mastodon_kind=str(arguments.get("mastodon_kind") or "links"),
        )

    def assess_content_value_tool(
        self, _: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return assess_content_value(arguments)

    def save_evidence_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.save_evidence(context, arguments)

    def list_evidence_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            raise ResearchError("list_evidence不接受参数")
        return {"evidence": [serialize_evidence(item) for item in self.list_evidence(context.work_thread_id)]}


def serialize_evidence(item: Evidence) -> dict[str, Any]:
    try:
        value_analysis = json.loads(item.value_analysis_json or "{}")
    except json.JSONDecodeError:
        value_analysis = {}
    return {
        "id": item.id,
        "work_thread_id": item.work_thread_id,
        "owner_agent_id": item.owner_agent_id,
        "source_url": item.source_url,
        "source_title": item.source_title,
        "published_at": item.published_at,
        "retrieved_at": item.retrieved_at.isoformat(),
        "claim": item.claim,
        "excerpt": item.excerpt,
        "metric_name": item.metric_name,
        "metric_value": item.metric_value,
        "metric_unit": item.metric_unit,
        "confidence": item.confidence,
        "stance": item.stance,
        "source_type": item.source_type,
        "source_publisher": item.source_publisher,
        "claim_key": item.claim_key,
        "verification_notes": item.verification_notes,
        "value_analysis": value_analysis,
        "value_score": item.value_score,
        "value_ready": item.value_score >= 9,
        "review_status": item.review_status,
        "reviewer_agent_id": item.reviewer_agent_id,
        "review_notes": item.review_notes,
        "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
        "created_at": item.created_at.isoformat(),
    }
