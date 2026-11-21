from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

from app.browser_service import BrowserService, BrowserToolError
from app.database import Database
from app.env_store import EnvStore
from app.main import create_app
from app.models import AgentProfile


HTML = b"""<!doctype html><html><head><title>Dynamic Market</title></head><body>
<h1 id='title'>Loading</h1>
<form method='get' action='/search'><input name='q' placeholder='Search market'><button type='submit'>Search</button></form>
<button id='expand' onclick="document.querySelector('#detail').hidden=false">Expand details</button>
<p id='detail' hidden>Dynamic market detail: growth 42 percent.</p>
<select id='region' onchange="document.querySelector('#selected').textContent='Region '+this.value"><option value='global'>Global</option><option value='asia'>Asia</option></select>
<p id='selected'></p>
<form method='post' action='/publish'><input name='title'><button type='submit'>Publish report</button></form>
<a href='/next'>Next page</a><a href='/delete/item'>Delete item</a><a href='/login'>Login</a>
<script>document.querySelector('#title').textContent='Rendered market dashboard'</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    post_count = 0

    def do_GET(self):
        body = HTML
        if self.path.startswith("/next"):
            body = b"<html><title>Next</title><body><h1>Page two</h1><a href='/'>Back</a></body></html>"
        elif self.path.startswith("/search"):
            body = f"<html><title>Search</title><body>Search result {self.path}</body></html>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        type(self).post_count += 1
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_):
        return


@pytest.fixture()
def browser_site():
    Handler.post_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=2)


def local_validator(value: str) -> str:
    if not value.startswith("http://127.0.0.1:"):
        raise BrowserToolError("测试只允许本机站点")
    return value


def find_element(result, *, text=None, placeholder=None, tag=None):
    for item in result["elements"]:
        if text and text not in item["text"]:
            continue
        if placeholder and placeholder != item["placeholder"]:
            continue
        if tag and tag != item["tag"]:
            continue
        return item["id"]
    raise AssertionError("element not found")


def test_dynamic_browser_observe_click_fill_select_and_close(browser_site):
    service = BrowserService(
        {"browser": {"navigation_timeout_ms": 10_000}},
        public_url_validator=local_validator,
    )
    try:
        opened = service.open(10, browser_site)
        assert opened["title"] == "Dynamic Market"
        assert "Rendered market dashboard" in opened["text"]
        assert opened["safety"]["mode"] == "read_only"
        input_id = find_element(opened, placeholder="Search market")
        filled = service.fill(10, input_id, "ai agents")
        assert filled["url"] == browser_site + "/"
        expand_id = find_element(filled, text="Expand details")
        expanded = service.click(10, expand_id)
        assert "growth 42 percent" in expanded["text"]
        select_id = find_element(expanded, tag="select")
        selected = service.select(10, select_id, "asia")
        assert "Region asia" in selected["text"]
        next_id = find_element(selected, text="Next page")
        page_two = service.click(10, next_id)
        assert page_two["title"] == "Next"
        assert page_two["url"].endswith("/next")
        assert service.close_thread(10)["closed"] is True
        assert service.status()["active_sessions"] == 0
    finally:
        service.close()


def test_dynamic_browser_blocks_post_login_and_delete(browser_site):
    service = BrowserService(public_url_validator=local_validator)
    try:
        opened = service.open(11, browser_site)
        publish_id = find_element(opened, text="Publish report")
        with pytest.raises(BrowserToolError, match="外部写入|提交"):
            service.click(11, publish_id)
        assert Handler.post_count == 0
        delete_id = find_element(service.observe(11), text="Delete item")
        with pytest.raises(BrowserToolError, match="外部写入|身份"):
            service.click(11, delete_id)
        login_id = find_element(service.observe(11), text="Login")
        with pytest.raises(BrowserToolError, match="外部写入|身份"):
            service.click(11, login_id)
        with pytest.raises(BrowserToolError, match="普通搜索"):
            text_input = service.observe(11)["elements"]
            publish_input = next(item["id"] for item in text_input if item["name"] == "title")
            service.fill(11, publish_input, "should not publish")
    finally:
        service.close()


class NoopCodex:
    def close(self):
        return None


def test_browser_api_registration_settings_and_network_permission(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'browser.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
        workspace_path=tmp_path / "workspace",
    )
    with TestClient(app) as client:
        tools = {item["name"] for item in client.get("/api/tools").json()}
        browser_tools = {
            "browser_open",
            "browser_observe",
            "browser_click",
            "browser_fill",
            "browser_select",
            "browser_close",
        }
        assert browser_tools <= tools
        status = client.get("/api/browser/status")
        assert status.status_code == 200
        assert status.json()["mode"] == "read_only"
        defaults = client.get("/api/settings/browser").json()
        defaults["max_elements"] = 75
        saved = client.put("/api/settings/browser", json=defaults)
        assert saved.status_code == 200
        assert app.state.browser_service.max_elements == 75
        with database.sessions() as session:
            worker = AgentProfile(name="联网员工", is_controller=False)
            blocked = AgentProfile(name="离线员工", is_controller=False)
            session.add_all([worker, blocked])
            session.commit()
            worker_id, blocked_id = worker.id, blocked.id
        app.state.work_executor.set_tool_permission(worker_id, "web_search", "allow")
        for name in browser_tools:
            assert app.state.work_executor.list_tool_permissions(worker_id)[name] == "allow"
            assert app.state.work_executor.list_tool_permissions(blocked_id)[name] == "deny"
