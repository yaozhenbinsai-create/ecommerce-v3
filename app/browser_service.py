from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, Route, sync_playwright

from .research_runtime import Resolver, _resolve_host, validate_public_url
from .tool_runtime import ToolContext, ToolRegistryError


class BrowserToolError(ToolRegistryError):
    pass


@dataclass
class BrowserSession:
    context: BrowserContext
    page: Page
    updated_at: float


class BrowserService:
    BLOCKED_ACTION_WORDS = (
        "submit", "send", "publish", "post", "buy", "purchase", "checkout",
        "delete", "remove", "upload", "login", "log in", "logout", "sign in",
        "sign out", "register", "subscribe", "unsubscribe", "提交", "发送",
        "发布", "购买", "结账", "删除", "移除", "上传", "登录", "退出", "注册", "订阅",
    )
    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        resolver: Resolver = _resolve_host,
        executable_path: str | None = None,
        public_url_validator: Callable[[str], str] | None = None,
    ) -> None:
        self.resolver = resolver
        self.executable_path = executable_path or self._find_browser()
        self.public_url_validator = public_url_validator
        self.executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._sessions: dict[int, BrowserSession] = {}
        self.configure(settings or {})

    @staticmethod
    def _find_browser() -> str:
        candidates = (
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise BrowserToolError("未找到可用的Chrome或Edge浏览器")

    def configure(self, settings: dict[str, Any]) -> None:
        config = settings.get("browser") or {}
        self.navigation_timeout_ms = min(120_000, max(3_000, int(config.get("navigation_timeout_ms", 30_000))))
        self.max_page_chars = min(100_000, max(2_000, int(config.get("max_page_chars", 30_000))))
        self.max_elements = min(300, max(10, int(config.get("max_elements", 120))))
        self.max_response_bytes = min(20_000_000, max(100_000, int(config.get("max_response_bytes", 5_000_000))))
        self.max_sessions = min(16, max(1, int(config.get("max_sessions", 4))))
        self.idle_timeout_seconds = min(3600, max(30, int(config.get("idle_timeout_seconds", 600))))

    def _run(self, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="boss-agent-browser"
                )
            return self.executor.submit(function).result(timeout=max(30, self.navigation_timeout_ms / 1000 + 15))
        except BrowserToolError:
            raise
        except Exception as exc:
            raise BrowserToolError(f"动态浏览器操作失败：{exc}") from exc

    def _ensure_runtime(self) -> None:
        if self._browser and self._browser.is_connected():
            return
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            executable_path=self.executable_path,
            args=["--disable-extensions", "--disable-sync", "--no-first-run"],
        )

    def _validate_url(self, value: str) -> str:
        if self.public_url_validator:
            return self.public_url_validator(value)
        return validate_public_url(value, self.resolver)

    def _route(self, route: Route) -> None:
        request = route.request
        if request.method.upper() not in {"GET", "HEAD"}:
            route.abort("blockedbyclient")
            return
        if request.resource_type in {"image", "media", "font"}:
            route.abort("blockedbyclient")
            return
        try:
            self._validate_url(request.url)
        except Exception:
            route.abort("blockedbyclient")
            return
        try:
            response = route.fetch(max_redirects=0, timeout=self.navigation_timeout_ms)
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self.max_response_bytes:
                route.abort("blockedbyclient")
                return
            body = response.body()
            if len(body) > self.max_response_bytes:
                route.abort("blockedbyclient")
                return
            route.fulfill(response=response, body=body)
        except Exception:
            route.abort("failed")

    def _create_session(self, thread_id: int) -> BrowserSession:
        self._ensure_runtime()
        assert self._browser is not None
        context = self._browser.new_context(
            accept_downloads=False,
            java_script_enabled=True,
            service_workers="block",
            viewport={"width": 1365, "height": 900},
        )
        context.set_default_timeout(self.navigation_timeout_ms)
        context.set_default_navigation_timeout(self.navigation_timeout_ms)
        context.route("**/*", self._route)
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.dismiss())
        page.on("popup", lambda popup: popup.close())
        page.on("download", lambda download: download.cancel())
        session = BrowserSession(context=context, page=page, updated_at=time.monotonic())
        self._sessions[thread_id] = session
        return session

    def _cleanup_idle(self) -> None:
        now = time.monotonic()
        for key, session in list(self._sessions.items()):
            if now - session.updated_at > self.idle_timeout_seconds:
                session.context.close()
                self._sessions.pop(key, None)

    def _session(self, thread_id: int) -> BrowserSession:
        self._cleanup_idle()
        now = time.monotonic()
        session = self._sessions.get(thread_id)
        if not session:
            while len(self._sessions) >= self.max_sessions:
                oldest_id = min(
                    self._sessions,
                    key=lambda item: self._sessions[item].updated_at,
                )
                self._sessions.pop(oldest_id).context.close()
            session = self._create_session(thread_id)
        session.updated_at = now
        return session

    @staticmethod
    def _normalize_text(value: str, limit: int) -> str:
        return re.sub(r"\s+", " ", value or "").strip()[:limit]

    def _observe(self, session: BrowserSession) -> dict[str, Any]:
        page = session.page
        if page.url != "about:blank":
            self._validate_url(page.url)
        text = page.locator("body").inner_text(timeout=self.navigation_timeout_ms)
        text = text[: self.max_page_chars]
        elements = page.locator(
            "a[href], button, input:not([type=hidden]), textarea, select, [role=button]"
        )
        count = min(elements.count(), self.max_elements)
        items: list[dict[str, Any]] = []
        for index in range(count):
            element = elements.nth(index)
            try:
                if not element.is_visible():
                    continue
                tag = element.evaluate("el => el.tagName.toLowerCase()")
                input_type = (element.get_attribute("type") or "").lower()
                item = {
                    "id": index + 1,
                    "tag": tag,
                    "type": input_type or None,
                    "text": self._normalize_text(
                        element.inner_text() if tag not in {"input", "textarea", "select"} else "",
                        300,
                    ),
                    "aria_label": self._normalize_text(element.get_attribute("aria-label") or "", 200),
                    "placeholder": self._normalize_text(element.get_attribute("placeholder") or "", 200),
                    "href": element.get_attribute("href") if tag == "a" else None,
                    "name": self._normalize_text(element.get_attribute("name") or "", 200),
                }
                items.append(item)
            except Exception:
                continue
        return {
            "url": page.url,
            "title": page.title(),
            "text": text,
            "text_truncated": len(text) >= self.max_page_chars,
            "elements": items,
            "element_count": len(items),
            "safety": {
                "mode": "read_only",
                "untrusted_page_content": True,
                "blocked_methods": ["POST", "PUT", "PATCH", "DELETE"],
                "downloads": "blocked",
                "uploads": "blocked",
                "authentication_state": "ephemeral_guest",
            },
        }

    def open(self, thread_id: int, url: str) -> dict[str, Any]:
        url = self._validate_url(url)

        def operation() -> dict[str, Any]:
            session = self._session(thread_id)
            session.page.goto(url, wait_until="domcontentloaded")
            session.page.wait_for_timeout(500)
            session.updated_at = time.monotonic()
            return self._observe(session)

        return self._run(operation)

    def observe(self, thread_id: int) -> dict[str, Any]:
        return self._run(lambda: self._observe(self._session(thread_id)))

    def _element(self, session: BrowserSession, element_id: int):
        locator = session.page.locator(
            "a[href], button, input:not([type=hidden]), textarea, select, [role=button]"
        ).nth(max(0, int(element_id) - 1))
        if locator.count() != 1 or not locator.is_visible():
            raise BrowserToolError("元素不存在或已不可见，请重新观察页面")
        return locator

    def click(self, thread_id: int, element_id: int) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            session = self._session(thread_id)
            element = self._element(session, element_id)
            tag = element.evaluate("el => el.tagName.toLowerCase()")
            input_type = (element.get_attribute("type") or "").lower()
            href = element.get_attribute("href") or ""
            safe_label = " ".join(
                filter(
                    None,
                    [
                        element.inner_text(),
                        element.get_attribute("aria-label") or "",
                        element.get_attribute("title") or "",
                        element.get_attribute("value") or "",
                        urlsplit(urljoin(session.page.url, href)).path if href else "",
                    ],
                )
            ).lower()
            form_method = element.evaluate(
                "el => (el.form?.method || '').toLowerCase()"
            )
            get_search = form_method == "get" and any(
                word in safe_label for word in ("search", "find", "filter", "搜索", "查询", "筛选")
            )
            if any(word in safe_label for word in self.BLOCKED_ACTION_WORDS) and not get_search:
                raise BrowserToolError("该元素可能触发外部写入或身份操作，只读浏览器已拒绝")
            if input_type in {"submit", "file", "image"}:
                if input_type in {"file", "image"}:
                    raise BrowserToolError("只读浏览器禁止提交或上传")
                form_method = element.evaluate(
                    "el => (el.form?.method || 'get').toLowerCase()"
                )
                if form_method != "get":
                    raise BrowserToolError("只读浏览器只允许GET搜索表单")
            if tag == "a":
                target = self._validate_url(urljoin(session.page.url, href))
                session.page.goto(target, wait_until="domcontentloaded")
            else:
                safe_interaction = (
                    input_type in {"checkbox", "radio"}
                    or get_search
                    or element.get_attribute("aria-expanded") is not None
                    or (element.get_attribute("role") or "").lower() == "tab"
                    or any(
                        word in safe_label
                        for word in (
                            "next", "previous", "more", "expand", "show", "hide",
                            "filter", "search", "find", "sort", "tab", "下一", "上一",
                            "更多", "展开", "显示", "隐藏", "筛选", "搜索", "查询", "排序", "标签",
                        )
                    )
                )
                if not safe_interaction:
                    raise BrowserToolError("按钮用途不明确，只读浏览器默认拒绝点击")
                element.click()
                session.page.wait_for_timeout(300)
                if session.page.url != "about:blank":
                    self._validate_url(session.page.url)
            session.updated_at = time.monotonic()
            return self._observe(session)

        return self._run(operation)

    def fill(self, thread_id: int, element_id: int, value: str) -> dict[str, Any]:
        value = str(value or "")[:2000]

        def operation() -> dict[str, Any]:
            session = self._session(thread_id)
            element = self._element(session, element_id)
            tag = element.evaluate("el => el.tagName.toLowerCase()")
            input_type = (element.get_attribute("type") or "text").lower()
            if tag not in {"input", "textarea"} or input_type in {"file", "password", "email", "tel", "hidden", "submit"}:
                raise BrowserToolError("只允许填写普通搜索或筛选文本框")
            form_method = element.evaluate(
                "el => (el.form?.method || 'get').toLowerCase()"
            )
            if form_method != "get":
                raise BrowserToolError("只允许填写普通搜索或筛选文本框")
            element.fill(value)
            session.updated_at = time.monotonic()
            return self._observe(session)

        return self._run(operation)

    def select(self, thread_id: int, element_id: int, value: str) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            session = self._session(thread_id)
            element = self._element(session, element_id)
            if element.evaluate("el => el.tagName.toLowerCase()") != "select":
                raise BrowserToolError("目标元素不是下拉选择框")
            element.select_option(str(value)[:500])
            session.page.wait_for_timeout(300)
            session.updated_at = time.monotonic()
            return self._observe(session)

        return self._run(operation)

    def close_thread(self, thread_id: int) -> dict[str, Any]:
        if self.executor is None:
            return {"closed": False, "work_thread_id": thread_id}
        def operation() -> dict[str, Any]:
            session = self._sessions.pop(thread_id, None)
            if session:
                session.context.close()
            return {"closed": bool(session), "work_thread_id": thread_id}

        return self._run(operation)

    def status(self) -> dict[str, Any]:
        def snapshot() -> dict[str, Any]:
            self._cleanup_idle()
            return {
                "available": bool(self.executable_path),
                "browser": Path(self.executable_path).stem,
                "active_sessions": len(self._sessions),
                "mode": "read_only",
                "limits": {
                    "navigation_timeout_ms": self.navigation_timeout_ms,
                    "max_page_chars": self.max_page_chars,
                    "max_elements": self.max_elements,
                    "max_response_bytes": self.max_response_bytes,
                    "max_sessions": self.max_sessions,
                    "idle_timeout_seconds": self.idle_timeout_seconds,
                },
            }
        if self.executor is None:
            return {
                "available": bool(self.executable_path),
                "browser": Path(self.executable_path).stem,
                "active_sessions": 0,
                "mode": "read_only",
                "limits": {
                    "navigation_timeout_ms": self.navigation_timeout_ms,
                    "max_page_chars": self.max_page_chars,
                    "max_elements": self.max_elements,
                    "max_response_bytes": self.max_response_bytes,
                    "max_sessions": self.max_sessions,
                    "idle_timeout_seconds": self.idle_timeout_seconds,
                },
            }
        return self._run(snapshot)

    def close(self) -> None:
        if self.executor is None:
            return
        def operation() -> dict[str, Any]:
            for session in list(self._sessions.values()):
                session.context.close()
            self._sessions.clear()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
            self._browser = None
            self._playwright = None
            return {"closed": True}

        try:
            self._run(operation)
        finally:
            executor = self.executor
            self.executor = None
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)

    def open_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.open(context.work_thread_id, str(arguments.get("url") or ""))

    def observe_tool(self, context: ToolContext, _: dict[str, Any]) -> dict[str, Any]:
        return self.observe(context.work_thread_id)

    def click_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.click(context.work_thread_id, int(arguments.get("element_id") or 0))

    def fill_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.fill(context.work_thread_id, int(arguments.get("element_id") or 0), str(arguments.get("value") or ""))

    def select_tool(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.select(context.work_thread_id, int(arguments.get("element_id") or 0), str(arguments.get("value") or ""))

    def close_tool(self, context: ToolContext, _: dict[str, Any]) -> dict[str, Any]:
        return self.close_thread(context.work_thread_id)
