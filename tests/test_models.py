from datetime import datetime, timedelta, timezone
from pathlib import Path
from secrets import token_urlsafe
import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from app.database import Database
from app.env_store import EnvStore
from app.main import create_app
from app.chat_runtime import ChatRuntime, ModelCallResult, ModelRunError, PlanRunError
from app.codex_client import CodexProtocolError
from app.memory_service import MemoryService
from app.models import (
    AgentProfile,
    AgentRuntimeState,
    ChatMessage,
    ConversationSummary,
    ModelEntry,
    PendingChatTask,
    RoutingPlan,
    RoutingPlanModel,
    WorkEvent,
    WorkRun,
    WorkThread,
)
from app.work_runtime import WorkRuntime


# Credentials exist only in this process and back local test transports.
TEST_MODEL_CLAUDE_KEY = token_urlsafe(24)
TEST_MODEL_CURRENT_KEY = token_urlsafe(24)
TEST_MODEL_NEW_KEY = token_urlsafe(24)
TEST_MODEL_PRIMARY_KEY = token_urlsafe(24)
TEST_MODEL_REQUIRED_KEY = token_urlsafe(24)
TEST_MODEL_SAVED_KEY = token_urlsafe(24)
TEST_MODEL_VALUE_KEY = token_urlsafe(24)


class FakeCodex:
    def __init__(self, connected: bool = True) -> None:
        self.connected = connected
        self.session_keys: list[str | None] = []
        self.reset_prefixes: list[str] = []

    def reset_sessions(self, prefix: str) -> None:
        self.reset_prefixes.append(prefix)

    def inspect_account(self):
        if not self.connected:
            return {"account": None, "providerCapabilities": {}, "models": []}
        return {
            "account": {"type": "chatgpt", "planType": "pro"},
            "providerCapabilities": {
                "imageGeneration": True,
                "namespaceTools": True,
                "webSearch": True,
            },
            "models": [],
        }

    def start_chatgpt_login(self):
        return {
            "type": "chatgpt",
            "authUrl": "https://example.test/oauth",
            "loginId": "login-test",
        }

    def run_text_turn(
        self,
        prompt: str,
        model: str | None = None,
        session_key: str | None = None,
    ):
        self.session_keys.append(session_key)
        if session_key and session_key.startswith("health:model:"):
            return "OK"
        if "请压缩下面的对话上下文" in prompt:
            return "目标：继续当前对话。\n已确认：保留历史关键内容。\n下一步：回答用户。"
        if "每日长期记忆整理" in prompt:
            return '{"actions":[]}'
        if "整理最终回复" in prompt:
            return "总控已汇总子Agent结果"
        if "系统唯一总控Agent" in prompt:
            match = re.search(r'"id":\s*(\d+)', prompt)
            agent_id = int(match.group(1)) if match else 0
            return json.dumps(
                {
                    "agent_ids": [agent_id],
                    "instructions": {str(agent_id): "请完成专业任务"},
                    "direct_response": "",
                },
                ensure_ascii=False,
            )
        if "只完成总控交给你的任务" in prompt:
            return "子Agent已完成任务"
        if "严格按照岗位资料工作" in prompt:
            return "子Agent直接回复"
        assert "不得新增职责" in prompt
        return json.dumps(
            {
                "role_responsibilities": "负责产品需求梳理。",
                "work_content": "整理需求并输出说明。",
                "personality": "严谨、直接。",
                "preferences": "喜欢简洁方案。",
            },
            ensure_ascii=False,
        )

    def close(self):
        return None


def test_chat_runtime_reports_each_model_attempt_and_fallback(tmp_path: Path, monkeypatch):
    runtime = ChatRuntime(EnvStore(tmp_path / ".env"), FakeCodex(), {}, retry_pause_seconds=0)
    primary = ModelEntry(
        id=1,
        connection_key="primary",
        name="主模型",
        access_method="chatgpt_pro",
        protocol="openai",
        model_type="chat",
        model_identifier="primary",
        is_enabled=True,
    )
    backup = ModelEntry(
        id=2,
        connection_key="backup",
        name="备用模型",
        access_method="chatgpt_pro",
        protocol="openai",
        model_type="chat",
        model_identifier="backup",
        is_enabled=True,
    )
    attempts = []

    def fake_call(model, prompt, session_key=None):
        if model.id == 1:
            raise ModelRunError("主模型暂时不可用", "transient")
        return ModelCallResult("备用成功", 11, 4, 15, "provider")

    monkeypatch.setattr(runtime, "call_model_result", fake_call)
    result = runtime.run_models(
        [primary, backup],
        "工作提示",
        attempts_per_model=1,
        on_attempt=attempts.append,
    )
    assert result.model_id == 2
    assert result.total_tokens == 15
    assert [item["status"] for item in attempts] == ["failed", "completed"]
    assert attempts[0]["error_kind"] == "transient"
    assert attempts[1]["model"].name == "备用模型"


def test_chat_runtime_keeps_three_network_retries_before_fallback(
    tmp_path: Path, monkeypatch
):
    primary = ModelEntry(
        id=1,
        connection_key="primary-retry",
        name="主模型",
        model_type="chat",
        access_method="api",
        protocol="openai",
        model_identifier="primary",
        is_enabled=True,
    )
    fallback = ModelEntry(
        id=2,
        connection_key="fallback-retry",
        name="备用模型",
        model_type="chat",
        access_method="api",
        protocol="openai",
        model_identifier="fallback",
        is_enabled=True,
    )
    runtime = ChatRuntime(
        EnvStore(tmp_path / ".env"),
        FakeCodex(),
        {"retry": {"attempts_per_model": 3}},
        retry_pause_seconds=0,
    )
    calls: list[str] = []

    def fake_call(model, *_args, **_kwargs):
        calls.append(model.name)
        if model.id == primary.id:
            raise ModelRunError("短暂网络波动", "transient")
        return ModelCallResult("备用成功", 5, 2, 7, "provider")

    monkeypatch.setattr(runtime, "call_model_result", fake_call)
    result = runtime.run_models([primary, fallback], "执行任务")

    assert result.text == "备用成功"
    assert calls == ["主模型", "主模型", "主模型", "备用模型"]


def test_chat_runtime_rechecks_hard_deadline_before_network_retry(
    tmp_path: Path, monkeypatch
):
    model = ModelEntry(
        id=1,
        connection_key="deadline-retry",
        name="主模型",
        model_type="chat",
        access_method="api",
        protocol="openai",
        model_identifier="primary",
        is_enabled=True,
    )
    runtime = ChatRuntime(
        EnvStore(tmp_path / ".env"), FakeCodex(), {}, retry_pause_seconds=0
    )
    calls = 0
    checks = 0

    def fake_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ModelRunError("短暂网络波动", "transient")

    def before_attempt():
        nonlocal checks
        checks += 1
        return checks == 1

    monkeypatch.setattr(runtime, "call_model_result", fake_call)
    with pytest.raises(PlanRunError) as exc_info:
        runtime.run_models(
            [model],
            "执行任务",
            attempts_per_model=3,
            before_attempt=before_attempt,
        )

    assert exc_info.value.kind == "deadline"
    assert calls == 1


def test_chat_runtime_uses_single_attempt_for_half_open_model(
    tmp_path: Path, monkeypatch
):
    primary = ModelEntry(
        id=1,
        connection_key="half-open",
        name="探测模型",
        model_type="chat",
        access_method="api",
        protocol="openai",
        model_identifier="probe",
        is_enabled=True,
    )
    fallback = ModelEntry(
        id=2,
        connection_key="healthy-fallback",
        name="健康备用",
        model_type="chat",
        access_method="api",
        protocol="openai",
        model_identifier="fallback",
        is_enabled=True,
    )
    runtime = ChatRuntime(
        EnvStore(tmp_path / ".env"),
        FakeCodex(),
        {"retry": {"attempts_per_model": 3}},
        retry_pause_seconds=0,
    )
    calls: list[str] = []

    def fake_call(model, *_args, **_kwargs):
        calls.append(model.name)
        if model.id == primary.id:
            raise ModelRunError("探测仍失败", "transient")
        return ModelCallResult("备用成功", 5, 2, 7, "provider")

    monkeypatch.setattr(runtime, "call_model_result", fake_call)
    result = runtime.run_models(
        [primary, fallback],
        "执行任务",
        attempts_by_model={primary.id: 1, fallback.id: 3},
    )

    assert result.text == "备用成功"
    assert calls == ["探测模型", "健康备用"]


def make_client(tmp_path: Path, codex: FakeCodex | None = None) -> TestClient:
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    env_store = EnvStore(tmp_path / ".env")
    (tmp_path / "settings.yaml").write_text(
        "conversation:\n  model_retry_pause_seconds: 0\n", encoding="utf-8"
    )
    return TestClient(
        create_app(
            database=database,
            env_store=env_store,
            codex=codex or FakeCodex(),
            settings_path=tmp_path / "settings.yaml",
        )
    )


def test_work_dashboard_keeps_detail_state_across_polling(tmp_path: Path):
    with make_client(tmp_path) as client:
        html = client.get("/").text

    assert "openWorkDetailKeys" in html
    assert 'details[data-work-details-key][open]' in html
    assert 'data-work-details-key="audit-${project.id}"' in html
    assert 'data-work-details-key="result-${item.id}"' in html
    assert "等待Agent成果" in html
    assert "waiting_dependency" in html
    assert "projectDisplayExecution" in html
    assert "'waiting_retry', 'running'" in html
    assert 'id="clearChatButton"' in html
    assert "清空当前对话和对话摘要" in html
    assert "method: 'DELETE'" in html


def test_clear_current_agent_chat_resets_context_but_preserves_internal_records(
    tmp_path: Path,
):
    codex = FakeCodex()
    with make_client(tmp_path, codex) as client:
        agent_id = client.get("/api/chat/agents").json()[0]["id"]
        database = Database(f"sqlite:///{tmp_path / 'test.db'}")
        with database.sessions() as session:
            user_message = ChatMessage(
                conversation_agent_id=agent_id,
                sender_kind="user",
                recipient_agent_id=agent_id,
                content="需要清空的用户消息",
                status="completed",
            )
            agent_message = ChatMessage(
                conversation_agent_id=agent_id,
                sender_kind="agent",
                sender_agent_id=agent_id,
                content="需要清空的Agent回复",
                status="completed",
            )
            internal_message = ChatMessage(
                conversation_agent_id=agent_id,
                sender_kind="agent",
                sender_agent_id=agent_id,
                recipient_agent_id=agent_id,
                content="保留的内部协作审计",
                is_internal=True,
                status="completed",
            )
            session.add_all([user_message, agent_message, internal_message])
            session.flush()
            session.add(
                PendingChatTask(
                    conversation_agent_id=agent_id,
                    user_message_id=user_message.id,
                    request_text=user_message.content,
                    task_kind="chat",
                    status="completed",
                    next_retry_at=datetime.now(timezone.utc),
                    result_message_id=agent_message.id,
                )
            )
            session.add(
                ConversationSummary(
                    conversation_agent_id=agent_id,
                    summary="旧对话摘要",
                    through_message_id=agent_message.id,
                )
            )
            session.commit()

        cleared = client.delete(f"/api/chat/agents/{agent_id}/messages")
        assert cleared.status_code == 200
        assert cleared.json() == {
            "status": "cleared",
            "agent_id": agent_id,
            "messages_deleted": 2,
            "chat_tasks_deleted": 1,
            "summary_cleared": True,
            "long_term_memory_preserved": True,
            "work_history_preserved": True,
        }
        assert client.get(f"/api/chat/agents/{agent_id}/messages").json() == []
        internal = client.get("/api/chat/internal").json()
        assert [item["content"] for item in internal] == ["保留的内部协作审计"]
        assert codex.reset_prefixes == [f"agent:{agent_id}:main"]

        with database.sessions() as session:
            assert session.query(ConversationSummary).count() == 0
            assert session.query(PendingChatTask).count() == 0
            remaining = session.query(ChatMessage).all()
            assert len(remaining) == 1 and remaining[0].is_internal is True

            blocked_message = ChatMessage(
                conversation_agent_id=agent_id,
                sender_kind="user",
                recipient_agent_id=agent_id,
                content="仍在重试的聊天",
                status="waiting_retry",
            )
            session.add(blocked_message)
            session.flush()
            session.add(
                PendingChatTask(
                    conversation_agent_id=agent_id,
                    user_message_id=blocked_message.id,
                    request_text=blocked_message.content,
                    task_kind="chat",
                    status="waiting_retry",
                    next_retry_at=datetime.now(timezone.utc) + timedelta(hours=1),
                )
            )
            session.commit()

        blocked = client.delete(f"/api/chat/agents/{agent_id}/messages")
        assert blocked.status_code == 409
        assert "聊天任务执行中" in blocked.json()["detail"]
        assert len(client.get(f"/api/chat/agents/{agent_id}/messages").json()) == 1


def test_custom_model_crud_and_secret_storage(tmp_path: Path):
    with make_client(tmp_path) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/models").json() == []

        created = client.post(
            "/api/models/custom",
            json={
                "name": "图片中转站",
                "protocol": "openai_images",
                "model_type": "image",
                "model_identifier": "image-model",
                "base_url": "https://relay.example/v1/",
                "context_window": 128000,
                "max_output_tokens": 8192,
                "api_key": TEST_MODEL_VALUE_KEY,
                "image_generation": True,
                "image_edit": True,
                "is_enabled": True,
            },
        )
        assert created.status_code == 201
        model = created.json()
        assert model["base_url"] == "https://relay.example/v1"
        assert model["context_window"] == 128000
        assert model["max_output_tokens"] == 8192
        assert model["has_credential"] is True
        assert TEST_MODEL_VALUE_KEY not in str(model)
        assert TEST_MODEL_VALUE_KEY in (tmp_path / ".env").read_text(encoding="utf-8")

        updated = client.put(
            f"/api/models/{model['id']}",
            json={"name": "图片中转站 2", "api_key": TEST_MODEL_NEW_KEY},
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "图片中转站 2"
        env_text = (tmp_path / ".env").read_text(encoding="utf-8")
        assert TEST_MODEL_NEW_KEY in env_text
        assert TEST_MODEL_VALUE_KEY not in env_text

        deleted = client.delete(f"/api/models/{model['id']}")
        assert deleted.status_code == 204
        assert client.get("/api/models").json() == []
        assert TEST_MODEL_NEW_KEY not in (tmp_path / ".env").read_text(encoding="utf-8")


def test_model_discovery_and_connection_test_use_current_or_saved_key(
    tmp_path: Path, monkeypatch
):
    requests: list[dict] = []

    class FakeResponse:
        def __init__(self, body, status_code=200):
            self.body = body
            self.status_code = status_code
            self.is_success = status_code < 400
            self.text = ""

        def json(self):
            return self.body

    catalog_body = {
                "data": [
                    {"id": "gpt-beta", "owned_by": "relay"},
                    {"id": "gpt-alpha", "owned_by": "relay"},
                ]
            }

    class FakeHttpClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, headers=None, params=None):
            requests.append({"method": "GET", "url": url, "headers": headers or {}, "params": params})
            return FakeResponse(catalog_body)

        def post(self, url, headers=None, json=None):
            requests.append({"method": "POST", "url": url, "headers": headers or {}, "json": json})
            if json["model"] == "not-found":
                return FakeResponse({"error": {"message": "model not found"}}, 404)
            return FakeResponse({"choices": [{"message": {"content": "OK"}}]})

    with make_client(tmp_path) as client:
        monkeypatch.setattr("app.main.httpx.Client", FakeHttpClient)
        created = client.post(
            "/api/models/custom",
            json={
                "name": "测试中转站",
                "protocol": "openai",
                "model_type": "chat",
                "model_identifier": "gpt-alpha",
                "base_url": "https://relay.example/v1",
                "api_key": TEST_MODEL_SAVED_KEY,
            },
        ).json()

        discovered = client.post(
            "/api/models/discover",
            json={
                "protocol": "openai",
                "base_url": "https://new-relay.example/v1/",
                "api_key": TEST_MODEL_CURRENT_KEY,
            },
        )
        assert discovered.status_code == 200
        assert [item["id"] for item in discovered.json()["models"]] == [
            "gpt-alpha",
            "gpt-beta",
        ]
        assert requests[-1]["url"] == "https://new-relay.example/v1/models"
        assert requests[-1]["headers"]["Authorization"] == ('Bearer ' + TEST_MODEL_CURRENT_KEY)
        assert TEST_MODEL_CURRENT_KEY not in discovered.text

        tested = client.post(
            "/api/models/test",
            json={
                "model_id": created["id"],
                "protocol": "openai",
                "base_url": "https://relay.example/v1",
                "model_identifier": "gpt-alpha",
            },
        )
        assert tested.status_code == 200
        assert tested.json()["request_kind"] == "Chat Completions"
        assert requests[-1]["url"] == "https://relay.example/v1/chat/completions"
        assert requests[-1]["headers"]["Authorization"] == ('Bearer ' + TEST_MODEL_SAVED_KEY)
        saved = next(
            item for item in client.get("/api/models").json() if item["id"] == created["id"]
        )
        assert saved["health_status"] == "healthy"
        assert "gpt-alpha" in saved["health_message"]
        assert saved["health_checked_at"] is not None
        assert saved["circuit_reset_at"] is not None

        unchanged_save = client.put(
            f"/api/models/{created['id']}",
            json={
                "name": "测试中转站",
                "protocol": "openai",
                "model_type": "chat",
                "model_identifier": "gpt-alpha",
                "base_url": "https://relay.example/v1/",
                "context_window": None,
                "max_output_tokens": None,
                "image_generation": False,
                "image_edit": False,
                "is_enabled": True,
            },
        )
        assert unchanged_save.status_code == 200
        assert unchanged_save.json()["health_status"] == "healthy"
        assert "gpt-alpha" in unchanged_save.json()["health_message"]

        missing = client.post(
            "/api/models/test",
            json={
                "model_id": created["id"],
                "protocol": "openai",
                "base_url": "https://relay.example/v1",
                "model_identifier": "not-found",
            },
        )
        assert missing.status_code == 404
        saved = next(
            item for item in client.get("/api/models").json() if item["id"] == created["id"]
        )
        assert saved["health_status"] == "error"
        assert saved["health_checked_at"] is not None
        assert saved["circuit_reset_at"] is None


def test_successful_model_probe_wakes_waiting_work_and_resets_circuit(
    tmp_path: Path, monkeypatch
):
    database = Database(f"sqlite:///{tmp_path / 'probe-wake.db'}")
    env_store = EnvStore(tmp_path / ".env")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "conversation:\n  model_retry_pause_seconds: 0\n"
        "work_runtime:\n  auto_execute_enabled: false\n",
        encoding="utf-8",
    )

    class FakeResponse:
        is_success = True
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    app = create_app(
        database=database,
        env_store=env_store,
        codex=FakeCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        monkeypatch.setattr("app.main.httpx.Client", FakeHttpClient)
        model = client.post(
            "/api/models/custom",
            json={
                "name": "恢复模型",
                "protocol": "openai",
                "model_type": "chat",
                "model_identifier": "gpt-recovered",
                "base_url": "https://relay.example/v1",
                "api_key": TEST_MODEL_SAVED_KEY,
            },
        ).json()
        with database.sessions() as session:
            plan = RoutingPlan(name="恢复方案", plan_type="chat")
            session.add(plan)
            session.flush()
            session.add(
                RoutingPlanModel(
                    plan_id=plan.id,
                    model_id=model["id"],
                    position=1,
                )
            )
            agent = AgentProfile(name="恢复员工", chat_plan_id=plan.id)
            session.add(agent)
            session.commit()
            agent_id = agent.id

        runtime = WorkRuntime(database, lease_seconds=60, instance_id="probe-wake")
        thread = runtime.create_thread(
            owner_agent_id=agent_id,
            title="等待模型恢复",
            objective="恢复后继续原任务",
        )
        run = runtime.create_run(thread.id)
        with database.sessions() as session:
            thread_row = session.get(WorkThread, thread.id)
            run_row = session.get(WorkRun, run.id)
            assert thread_row is not None and run_row is not None
            thread_row.status = "waiting_retry"
            run_row.status = "waiting_retry"
            run_row.next_wake_at = datetime.now(timezone.utc) + timedelta(minutes=30)
            run_row.last_error = "上游模型暂时不可用"
            session.commit()

        tested = client.post(
            "/api/models/test",
            json={
                "model_id": model["id"],
                "protocol": "openai",
                "model_type": "chat",
                "base_url": "https://relay.example/v1",
                "model_identifier": "gpt-recovered",
            },
        )

        assert tested.status_code == 200
        assert tested.json()["resumed_work_runs"] == 1
        with database.sessions() as session:
            recovered_model = session.get(ModelEntry, model["id"])
            recovered_thread = session.get(WorkThread, thread.id)
            recovered_run = session.get(WorkRun, run.id)
            event = session.query(WorkEvent).filter_by(
                work_run_id=run.id,
                event_type="model_health_recovered",
            ).one()
            assert recovered_model is not None
            assert recovered_model.circuit_reset_at is not None
            assert recovered_thread is not None
            assert recovered_thread.status == "queued"
            assert recovered_run is not None
            assert recovered_run.status == "queued"
            assert recovered_run.next_wake_at is None
            assert event.actor_id == f"model:{model['id']}"


def test_claude_discovery_adds_v1_and_required_headers(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class FakeResponse:
        is_success = True
        status_code = 200
        text = ""

        def json(self):
            return {
                "data": [
                    {
                        "id": "claude-example",
                        "display_name": "Claude Example",
                        "max_input_tokens": 200000,
                        "max_tokens": 32000,
                    }
                ]
            }

    class FakeHttpClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, headers=None, params=None):
            captured.update(url=url, headers=headers, params=params)
            return FakeResponse()

    with make_client(tmp_path) as client:
        monkeypatch.setattr("app.main.httpx.Client", FakeHttpClient)
        response = client.post(
            "/api/models/discover",
            json={
                "protocol": "claude",
                "base_url": "https://api.anthropic.example",
                "api_key": TEST_MODEL_CLAUDE_KEY,
            },
        )
        assert response.status_code == 200
        assert captured["url"] == "https://api.anthropic.example/v1/models"
        assert captured["headers"] == {
            "anthropic-version": "2023-06-01",
            "x-api-key": TEST_MODEL_CLAUDE_KEY,
        }
        assert captured["params"] == {"limit": 1000}
        assert response.json()["models"][0]["context_window"] == 200000
        assert response.json()["models"][0]["max_output_tokens"] == 32000


def test_custom_model_requires_api_key(tmp_path: Path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/models/custom",
            json={
                "name": "缺少密钥",
                "protocol": "openai",
                "model_type": "chat",
                "model_identifier": "test-model",
                "base_url": "https://relay.example/v1",
            },
        )
        assert response.status_code == 422
        assert client.get("/api/models").json() == []


def test_connection_test_makes_real_protocol_specific_calls(tmp_path: Path, monkeypatch):
    requests: list[dict] = []

    class FakeResponse:
        is_success = True
        status_code = 200
        text = ""

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    class FakeHttpClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            if url.endswith("/images/generations"):
                return FakeResponse({"data": [{"b64_json": "dGVzdA=="}]})
            if url.endswith("/responses") and json and json.get("tools"):
                return FakeResponse(
                    {
                        "output": [
                            {
                                "type": "image_generation_call",
                                "result": "dGVzdA==",
                            }
                        ]
                    }
                )
            if url.endswith("/responses"):
                return FakeResponse(
                    {"output": [{"content": [{"type": "output_text", "text": "OK"}]}]}
                )
            if url.endswith("/messages"):
                return FakeResponse({"content": [{"type": "text", "text": "OK"}]})
            return FakeResponse({"choices": [{"message": {"content": "OK"}}]})

    cases = [
        (
            "openai_responses",
            "chat",
            "https://openai.example/v1/responses",
            "OpenAI Responses",
        ),
        (
            "claude",
            "chat",
            "https://claude.example/v1/messages",
            "Claude Messages",
        ),
        (
            "claude_openai",
            "chat",
            "https://claude.example/v1/chat/completions",
            "Chat Completions",
        ),
        (
            "openai_images",
            "image",
            "https://image.example/v1/images/generations",
            "OpenAI Images API",
        ),
        (
            "openai_responses",
            "image",
            "https://image.example/v1/responses",
            "Responses 图片工具",
        ),
    ]

    with make_client(tmp_path) as client:
        monkeypatch.setattr("app.main.httpx.Client", FakeHttpClient)
        for protocol, model_type, endpoint, request_kind in cases:
            base_url = endpoint.rsplit(
                    "/images/generations" if protocol == "openai_images" else (
                        "/responses" if protocol == "openai_responses" else (
                        "/chat/completions" if protocol in {"openai", "claude_openai"} else (
                            "/responses" if protocol == "openai_responses" else "/messages"
                        )
                        )
                    ),
                1,
            )[0]
            if protocol in {"claude", "claude_openai"}:
                base_url = "https://claude.example"
            response = client.post(
                "/api/models/test",
                json={
                    "protocol": protocol,
                    "model_type": model_type,
                    "model_identifier": "test-model",
                    "base_url": base_url,
                    "api_key": TEST_MODEL_REQUIRED_KEY,
                },
            )
            assert response.status_code == 200
            assert response.json()["endpoint"] == endpoint
            assert response.json()["request_kind"] == request_kind
            assert requests[-1]["url"] == endpoint


def test_custom_image_endpoint_and_upstream_error_expose_actual_url(
    tmp_path: Path, monkeypatch
):
    requests: list[str] = []

    class FakeResponse:
        is_success = False
        status_code = 503
        text = ""

        def json(self):
            return {"error": {"message": "image route unavailable"}}

    class FakeHttpClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, headers=None, json=None):
            requests.append(url)
            return FakeResponse()

    with make_client(tmp_path) as client:
        monkeypatch.setattr("app.main.httpx.Client", FakeHttpClient)
        response = client.post(
            "/api/models/test",
            json={
                "protocol": "custom_image",
                "model_type": "image",
                "model_identifier": "relay-image-model",
                "base_url": "https://relay.example/v1",
                "api_key": TEST_MODEL_REQUIRED_KEY,
                "image_generation_endpoint": "/custom/image/create",
            },
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["endpoint"] == "https://relay.example/v1/custom/image/create"
        assert "image route unavailable" in detail["message"]
        assert requests == ["https://relay.example/v1/custom/image/create"]


def test_image_model_rejects_chat_and_claude_protocols(tmp_path: Path):
    with make_client(tmp_path) as client:
        for protocol in ("openai", "claude", "claude_openai"):
            response = client.post(
                "/api/models/custom",
                json={
                    "name": f"invalid-{protocol}",
                    "protocol": protocol,
                    "model_type": "image",
                    "model_identifier": "image-model",
                    "base_url": "https://relay.example/v1",
                    "api_key": TEST_MODEL_REQUIRED_KEY,
                    "image_generation": True,
                },
            )
            assert response.status_code == 422


def test_new_native_and_compatibility_protocols_call_expected_endpoints(
    tmp_path: Path, monkeypatch
):
    requests: list[dict] = []

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class FakeHttpClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            if url.endswith("/responses"):
                return FakeResponse(
                    {
                        "output": [
                            {
                                "content": [
                                    {"type": "output_text", "text": "responses-ok"}
                                ]
                            }
                        ]
                    }
                )
            return FakeResponse(
                {"choices": [{"message": {"content": "compatibility-ok"}}]}
            )

    env_store = EnvStore(tmp_path / ".env")
    credential_env = env_store.create_secret(TEST_MODEL_REQUIRED_KEY)
    runtime = ChatRuntime(env_store, FakeCodex(), {})
    model = ModelEntry(
        id=1,
        connection_key="custom-test",
        name="协议测试",
        access_method="custom",
        protocol="openai_responses",
        model_type="chat",
        model_identifier="test-model",
        base_url="https://relay.example/v1",
        credential_env=credential_env,
        is_enabled=True,
    )
    monkeypatch.setattr("app.chat_runtime.httpx.Client", FakeHttpClient)

    assert runtime.call_model(model, "hello") == "responses-ok"
    assert requests[-1]["url"] == "https://relay.example/v1/responses"
    assert requests[-1]["json"]["input"] == "hello"

    model.protocol = "claude_openai"
    model.base_url = "https://api.anthropic.example"
    assert runtime.call_model(model, "hello") == "compatibility-ok"
    assert requests[-1]["url"] == "https://api.anthropic.example/v1/chat/completions"
    assert requests[-1]["headers"]["Authorization"] == ('Bearer ' + TEST_MODEL_REQUIRED_KEY)


def test_conversation_settings_can_be_saved_from_api(tmp_path: Path):
    with make_client(tmp_path) as client:
        defaults = client.get("/api/settings/conversation")
        assert defaults.status_code == 200
        assert defaults.json()["unknown_context_window"] == 32768

        payload = {
            "unknown_context_window": 64000,
            "default_max_output_tokens": 8000,
            "reserve_ratio": 0.12,
            "reserve_tokens_floor": 5000,
            "reserve_tokens_ceiling": 20000,
            "auto_compaction_enabled": True,
            "compaction_trigger_ratio": 0.75,
            "keep_recent_tokens": 12000,
            "summary_max_tokens": 3000,
        }
        saved = client.put("/api/settings/conversation", json=payload)
        assert saved.status_code == 200
        assert client.get("/api/settings/conversation").json() == payload
        settings_text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
        assert "unknown_context_window: 64000" in settings_text

        invalid = {**payload, "reserve_tokens_floor": 30000}
        assert client.put("/api/settings/conversation", json=invalid).status_code == 422

        memory = client.get("/api/settings/memory")
        assert memory.status_code == 200
        assert memory.json()["enabled"] is True
        retry = client.put(
            "/api/settings/retry",
            json={
                "request_timeout_seconds": 300,
                "attempts_per_model": 2,
                "retry_delay_minutes": 15,
            },
        )
        assert retry.status_code == 200
        assert retry.json()["request_timeout_seconds"] == 300
        assert retry.json()["retry_delay_minutes"] == 15
        runtime = client.put(
            "/api/settings/agent-runtime",
            json={
                "heartbeat_enabled": True,
                "heartbeat_interval_minutes": 20,
                "queue_scan_seconds": 3,
            },
        )
        assert runtime.status_code == 200
        assert runtime.json()["queue_scan_seconds"] == 3


def test_chatgpt_pro_creates_exactly_two_models(tmp_path: Path):
    with make_client(tmp_path) as client:
        connected = client.post("/api/pro/connect", json={})
        assert connected.status_code == 200
        assert connected.json()["status"] == "connected"
        assert {item["model_type"] for item in connected.json()["models"]} == {
            "chat",
            "image",
        }

        second_connect = client.post("/api/pro/connect", json={})
        assert second_connect.status_code == 200
        models = client.get("/api/models").json()
        assert len(models) == 2
        image = next(item for item in models if item["model_type"] == "image")
        assert image["image_generation"] is True
        assert image["image_edit"] is True
        assert image["has_credential"] is False


def test_chatgpt_pro_saved_models_can_be_health_checked(tmp_path: Path):
    fake_codex = FakeCodex()
    with make_client(tmp_path, fake_codex) as client:
        assert client.post("/api/pro/connect", json={}).status_code == 200
        models = client.get("/api/models").json()
        chat = next(item for item in models if item["model_type"] == "chat")
        image = next(item for item in models if item["model_type"] == "image")

        chat_probe = client.post(f"/api/models/{chat['id']}/test", json={})
        assert chat_probe.status_code == 200
        assert chat_probe.json()["request_kind"] == "chat_completion"
        saved_chat = next(
            item for item in client.get("/api/models").json()
            if item["id"] == chat["id"]
        )
        assert saved_chat["health_status"] == "healthy"
        assert "health:model:" in str(fake_codex.session_keys)

        image_probe = client.post(f"/api/models/{image['id']}/test", json={})
        assert image_probe.status_code == 200
        assert image_probe.json()["request_kind"] == "image_capability"
        saved_image = next(
            item for item in client.get("/api/models").json()
            if item["id"] == image["id"]
        )
        assert saved_image["health_status"] == "healthy"
        assert "未实际生成图片" in saved_image["health_message"]

        status = client.get("/api/pro/status")
        assert status.status_code == 200
        assert {
            item["id"]: item["health_status"] for item in status.json()["models"]
        } == {chat["id"]: "healthy", image["id"]: "healthy"}


def test_chatgpt_login_returns_official_authorization_url(tmp_path: Path):
    with make_client(tmp_path, FakeCodex(connected=False)) as client:
        response = client.post("/api/pro/connect", json={})
        assert response.status_code == 200
        assert response.json() == {
            "status": "authorization_required",
            "auth_url": "https://example.test/oauth",
            "login_id": "login-test",
        }


def test_routing_plan_order_and_model_delete_protection(tmp_path: Path):
    with make_client(tmp_path) as client:
        def create_model(name: str, model_type: str) -> int:
            response = client.post(
                "/api/models/custom",
                json={
                    "name": name,
                        "protocol": "openai_images" if model_type == "image" else "openai",
                    "model_type": model_type,
                        "model_identifier": name.lower(),
                        "base_url": "https://relay.example/v1",
                        "api_key": TEST_MODEL_PRIMARY_KEY,
                        "image_generation": model_type == "image",
                },
            )
            assert response.status_code == 201
            return response.json()["id"]

        chat_a = create_model("Chat A", "chat")
        chat_b = create_model("Chat B", "chat")
        chat_c = create_model("Chat C", "chat")
        image_a = create_model("Image A", "image")

        created = client.post(
            "/api/plans",
            json={
                "name": "默认对话",
                "plan_type": "chat",
                "model_ids": [chat_b, chat_a],
            },
        )
        assert created.status_code == 201
        plan = created.json()
        assert [item["model"]["id"] for item in plan["items"]] == [chat_b, chat_a]
        assert plan["model_count"] == 2

        wrong_type = client.post(
            "/api/plans",
            json={
                "name": "错误方案",
                "plan_type": "chat",
                "model_ids": [image_a],
            },
        )
        assert wrong_type.status_code == 400

        protected = client.delete(f"/api/models/{chat_b}")
        assert protected.status_code == 409
        detail = protected.json()["detail"]
        assert detail["references"][0]["name"] == "默认对话"

        updated = client.put(
            f"/api/plans/{plan['id']}",
            json={
                "name": "默认对话",
                "plan_type": "chat",
                "model_ids": [chat_a, chat_c],
            },
        )
        assert updated.status_code == 200
        assert [item["model"]["id"] for item in updated.json()["items"]] == [
            chat_a,
            chat_c,
        ]

        assert client.delete(f"/api/models/{chat_b}").status_code == 204
        assert client.delete(f"/api/plans/{plan['id']}").status_code == 204


def test_direct_chat_and_controller_delegation(tmp_path: Path):
    codex = FakeCodex()
    with make_client(tmp_path, codex) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        chat_model = next(
            model for model in connected["models"] if model["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "对话执行方案",
                "plan_type": "chat",
                "model_ids": [chat_model["id"]],
            },
        ).json()
        agents = client.get("/api/agents").json()
        controller = agents["controller"]
        controller_payload = {
            "name": "总控Agent",
            "role_responsibilities": "识别任务并安排子Agent",
            "work_content": "分配任务并汇总结果",
            "personality": "严谨",
            "preferences": "结果清晰",
            "chat_plan_id": plan["id"],
            "image_plan_id": None,
            "ai_organized": False,
        }
        assert client.put(
            f"/api/agents/{controller['id']}", json=controller_payload
        ).status_code == 200
        child_payload = {
            "name": "需求Agent",
            "role_responsibilities": "负责需求分析",
            "work_content": "整理需求",
            "personality": "认真",
            "preferences": "简洁",
            "chat_plan_id": plan["id"],
            "image_plan_id": None,
            "ai_organized": False,
        }
        child = client.post("/api/agents", json=child_payload).json()

        direct = client.post(
            f"/api/chat/agents/{child['id']}/messages",
            json={"content": "请直接回答"},
        )
        assert direct.status_code == 200
        assert direct.json()["message"]["content"] == "子Agent直接回复"
        assert f"agent:{child['id']}:main:model:{chat_model['id']}" in codex.session_keys

        delegated = client.post(
            f"/api/chat/agents/{controller['id']}/messages",
            json={"content": "请让合适的子Agent完成需求分析"},
        )
        assert delegated.status_code == 200
        assert delegated.json()["message"]["content"] == "总控已汇总子Agent结果"
        internal = client.get("/api/chat/internal").json()
        assert len(internal) == 2
        assert internal[0]["sender"]["name"] == "总控Agent"
        assert internal[0]["recipient"]["name"] == "需求Agent"
        assert internal[1]["sender"]["name"] == "需求Agent"
        assert internal[1]["recipient"]["name"] == "总控Agent"
        runtime_agents = client.get("/api/runtime/agents").json()
        assert {item["status"] for item in runtime_agents} == {"waiting"}
        assert all(item["main_session_key"].endswith(":main") for item in runtime_agents)
        tasks = client.get("/api/runtime/tasks").json()
        assert any(item["task_kind"] == "chat" for item in tasks)
        assert any(
            item["task_kind"] == "delegated" and item["parent_task_id"]
            for item in tasks
        )
        assert codex.session_keys.count(
            f"agent:{child['id']}:main:model:{chat_model['id']}"
        ) >= 2
        assert client.delete(f"/api/agents/{child['id']}").status_code == 409


def test_controller_chat_launches_persistent_work_and_reports_terminal_result(
    tmp_path: Path,
):
    database = Database(f"sqlite:///{tmp_path / 'controller-work.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n"
        "  auto_execute_enabled: false\n"
        "  queue_scan_seconds: 60\n",
        encoding="utf-8",
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=FakeCodex(),
        settings_path=settings_path,
    )
    with TestClient(app) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        chat_model = next(
            model for model in connected["models"] if model["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "总控工作方案",
                "plan_type": "chat",
                "model_ids": [chat_model["id"]],
            },
        ).json()
        controller = client.get("/api/agents").json()["controller"]
        assert client.put(
            f"/api/agents/{controller['id']}",
            json={
                "name": "总控Agent",
                "role_responsibilities": "管理公司员工Agent",
                "work_content": "创建、委派和验收正式工作",
                "personality": "严谨",
                "preferences": "先执行后汇报",
                "chat_plan_id": plan["id"],
                "image_plan_id": None,
                "ai_organized": False,
            },
        ).status_code == 200

        launched = client.post(
            f"/api/chat/agents/{controller['id']}/work",
            json={"content": "安排员工核验今天的公开市场信息并提交证据"},
        )
        assert launched.status_code == 201
        payload = launched.json()
        assert payload["status"] == "queued"
        assert payload["work_thread"]["owner_agent_id"] == controller["id"]
        assert payload["work_thread"]["status"] == "queued"
        assert payload["work_run"]["trigger_kind"] == "controller_chat"
        assert payload["scheduler_enabled"] is True
        assert client.get("/api/settings/work-scheduler").json() == {
            "auto_execute_enabled": True,
            "queue_scan_seconds": 60,
            "max_concurrent_agents": 4,
        }
        assert payload["message"]["work_thread"]["id"] == payload["work_thread"]["id"]

        messages = client.get(
            f"/api/chat/agents/{controller['id']}/messages"
        ).json()
        assert len(messages) == 2
        work_message = messages[-1]
        assert work_message["status"] == "work_queued"
        assert work_message["work_thread"]["status"] == "queued"

        claimed = client.post(
            "/api/work/runs/claim", params={"agent_id": controller["id"]}
        ).json()["run"]
        completed = client.post(
            f"/api/work/runs/{claimed['id']}/complete",
            json={
                "execution_token": claimed["execution_token"],
                "result_text": "员工已经完成核验并提交可追溯结果",
            },
        )
        assert completed.status_code == 200

        messages = client.get(
            f"/api/chat/agents/{controller['id']}/messages"
        ).json()
        work_message = messages[-1]
        assert work_message["status"] == "completed"
        assert work_message["work_thread"]["status"] == "completed"
        assert work_message["work_thread"]["result_text"] == (
            "员工已经完成核验并提交可追溯结果"
        )
        assert work_message["work_thread"]["report"]["id"] == completed.json()[
            "report"
        ]["id"]
        assert work_message["work_thread"]["report"]["preview_url"].endswith(
            "/preview"
        )
        assert work_message["work_thread"]["report"]["download_urls"][
            "markdown"
        ].endswith("format=md")
        assert "员工已经完成核验" in work_message["content"]

        continued = client.post(
            f"/api/chat/agents/{controller['id']}/work",
            json={
                "content": "继续完成刚才项目的后续工作",
                "continuation_of_thread_id": payload["work_thread"]["id"],
            },
        )
        assert continued.status_code == 201
        continuation = continued.json()
        assert continuation["work_thread"]["continuation_of_thread_id"] == (
            payload["work_thread"]["id"]
        )
        assert continuation["work_thread"]["objective"] == (
            payload["work_thread"]["objective"]
        )
        assert continuation["work_run"]["checkpoint"][
            "continuation_snapshot"
        ]["source_thread_id"] == payload["work_thread"]["id"]
        original = client.get(
            f"/api/work/threads/{payload['work_thread']['id']}"
        ).json()
        assert original["status"] == "completed"
        assert original["continued_by_thread_id"] == continuation["work_thread"]["id"]

        continued_run = client.post(
            "/api/work/runs/claim", params={"agent_id": controller["id"]}
        ).json()["run"]
        assert continued_run["work_thread_id"] == continuation["work_thread"]["id"]
        assert client.post(
            f"/api/work/runs/{continued_run['id']}/fail",
            json={
                "execution_token": continued_run["execution_token"],
                "error": "达到本轮预算，需要由新根线程接力",
                "result_text": "已保留阶段成果",
            },
        ).status_code == 200
        automatic = client.post(
            f"/api/chat/agents/{controller['id']}/work",
            json={
                "content": "重新开始完成之前未完成工作",
                "continuation_of_thread_id": continuation["work_thread"]["id"],
            },
        )
        assert automatic.status_code == 201
        assert automatic.json()["work_thread"]["continuation_of_thread_id"] == (
            continuation["work_thread"]["id"]
        )
        duplicate = client.post(
            f"/api/chat/agents/{controller['id']}/work",
            json={
                "content": "继续之前未完成的任务",
                "continuation_of_thread_id": automatic.json()["work_thread"]["id"],
            },
        )
        assert duplicate.status_code == 409
        assert "尚未结束" in duplicate.json()["detail"]
        automatic_run = client.post(
            "/api/work/runs/claim", params={"agent_id": controller["id"]}
        ).json()["run"]
        finished_continuation = client.post(
            f"/api/work/runs/{automatic_run['id']}/complete",
            json={
                "execution_token": automatic_run["execution_token"],
                "result_text": "续作已经完成剩余工作",
            },
        )
        assert finished_continuation.status_code == 200
        report = finished_continuation.json()["report"]
        markdown = client.get(report["download_urls"]["markdown"]).text
        assert "员工已经完成核验并提交可追溯结果" in markdown
        assert "续作已经完成剩余工作" in markdown

        fresh = client.post(
            f"/api/chat/agents/{controller['id']}/work",
            json={"content": "持续24小时开展全新的市场分析，不继承旧任务"},
        )
        assert fresh.status_code == 201
        assert fresh.json()["work_thread"]["continuation_of_thread_id"] is None
        assert fresh.json()["work_thread"]["objective"] == (
            "持续24小时开展全新的市场分析，不继承旧任务"
        )


def test_controller_runs_multi_round_shared_agent_discussion(tmp_path: Path):
    class DiscussionCodex(FakeCodex):
        def __init__(self):
            super().__init__()
            self.participant_prompts: list[str] = []
            self.moderation_rounds = 0

        def run_text_turn(
            self,
            prompt: str,
            model: str | None = None,
            session_key: str | None = None,
        ):
            self.session_keys.append(session_key)
            if "系统唯一总控Agent" in prompt:
                ids = [int(value) for value in re.findall(r'"id":\s*(\d+)', prompt)]
                child_ids = list(dict.fromkeys(ids))[-2:]
                return json.dumps(
                    {
                        "agent_ids": child_ids,
                        "instructions": {
                            str(child_ids[0]): "先提出方案",
                            str(child_ids[1]): "评审并补充方案",
                        },
                        "direct_response": "",
                    },
                    ensure_ascii=False,
                )
            if "参与多Agent长对话" in prompt:
                self.participant_prompts.append(prompt)
                is_research = "名称：研究Agent" in prompt
                round_two = "当前讨论轮次：第2轮" in prompt
                if is_research and not round_two:
                    return json.dumps(
                        {
                            "message": "研究Agent提出初步方案A。",
                            "decision": "continue",
                            "reason": "需要评审",
                        },
                        ensure_ascii=False,
                    )
                if not is_research and not round_two:
                    assert "研究Agent提出初步方案A" in prompt
                    return json.dumps(
                        {
                            "message": "评审Agent回应方案A并提出修正B。",
                            "decision": "continue",
                            "reason": "需要验证修正",
                        },
                        ensure_ascii=False,
                    )
                if is_research:
                    assert "评审Agent回应方案A并提出修正B" in prompt
                    return json.dumps(
                        {
                            "message": "研究Agent接受修正B并补齐证据。",
                            "decision": "finish",
                            "reason": "观点已充分",
                        },
                        ensure_ascii=False,
                    )
                assert "研究Agent接受修正B并补齐证据" in prompt
                return json.dumps(
                    {
                        "message": "评审Agent确认结论可以结束。",
                        "decision": "finish",
                        "reason": "已形成共识",
                    },
                    ensure_ascii=False,
                )
            if "正在主持多Agent长对话" in prompt:
                self.moderation_rounds += 1
                ids = [int(value) for value in re.findall(r'"id":\s*(\d+)', prompt)]
                child_ids = list(dict.fromkeys(ids))[-2:]
                if self.moderation_rounds == 1:
                    return json.dumps(
                        {
                            "action": "continue",
                            "agent_ids": child_ids,
                            "instructions": {
                                str(child_ids[0]): "结合评审意见补齐证据",
                                str(child_ids[1]): "复核修订后的结论",
                            },
                            "final_response": "",
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "action": "finish",
                        "agent_ids": child_ids,
                        "instructions": {},
                        "final_response": "两轮讨论完成，双方形成最终共识。",
                    },
                    ensure_ascii=False,
                )
            return super().run_text_turn(prompt, model, session_key)

    codex = DiscussionCodex()
    with make_client(tmp_path, codex) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        chat_model = next(
            model for model in connected["models"] if model["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "多轮讨论方案",
                "plan_type": "chat",
                "model_ids": [chat_model["id"]],
            },
        ).json()
        controller = client.get("/api/agents").json()["controller"]
        base = {
            "role_responsibilities": "参与团队讨论",
            "work_content": "提出并评审方案",
            "personality": "严谨",
            "preferences": "依据共享记录发言",
            "chat_plan_id": plan["id"],
            "image_plan_id": None,
            "ai_organized": True,
        }
        client.put(
            f"/api/agents/{controller['id']}",
            json={**base, "name": "总控Agent"},
        )
        client.post("/api/agents", json={**base, "name": "研究Agent"})
        client.post("/api/agents", json={**base, "name": "评审Agent"})

        response = client.post(
            f"/api/chat/agents/{controller['id']}/messages",
            json={"content": "请组织团队共同讨论方案"},
        )
        assert response.status_code == 200
        assert response.json()["message"]["content"] == "两轮讨论完成，双方形成最终共识。"
        assert codex.moderation_rounds == 2
        assert len(codex.participant_prompts) == 4
        internal = client.get("/api/chat/internal").json()
        assert len(internal) == 8
        assert sum("第1轮" in item["content"] for item in internal) == 4
        assert sum("第2轮" in item["content"] for item in internal) == 4
        assert any("建议继续" in item["content"] for item in internal)
        assert any("建议结束" in item["content"] for item in internal)


def test_all_models_fail_creates_persistent_retry(tmp_path: Path):
    class FailingCodex(FakeCodex):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def run_text_turn(
            self,
            prompt: str,
            model: str | None = None,
            session_key: str | None = None,
        ):
            self.calls += 1
            raise CodexProtocolError("temporary failure")

    failing = FailingCodex()
    with make_client(tmp_path, failing) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        chat_model = next(
            model for model in connected["models"] if model["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "失败测试方案",
                "plan_type": "chat",
                "model_ids": [chat_model["id"]],
            },
        ).json()
        controller = client.get("/api/agents").json()["controller"]
        payload = {
            "name": "总控Agent",
            "role_responsibilities": "总控",
            "work_content": "协调",
            "personality": "严谨",
            "preferences": "稳定",
            "chat_plan_id": plan["id"],
            "image_plan_id": None,
            "ai_organized": False,
        }
        client.put(f"/api/agents/{controller['id']}", json=payload)
        response = client.post(
            f"/api/chat/agents/{controller['id']}/messages",
            json={"content": "测试失败恢复"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "waiting_retry"
        assert failing.calls == 3
        messages = client.get(
            f"/api/chat/agents/{controller['id']}/messages"
        ).json()
        assert messages[-1]["status"] == "waiting_retry"


def test_context_is_compacted_and_original_messages_are_preserved(tmp_path: Path):
    with make_client(tmp_path) as client:
        model = client.post(
            "/api/models/custom",
            json={
                "name": "小上下文模型",
                "protocol": "openai",
                "model_type": "chat",
                "model_identifier": "small-context",
                "base_url": "https://relay.example/v1",
                "context_window": 6000,
                "max_output_tokens": 1000,
            },
        ).json()
        # Use the Pro adapter so the fake model can execute without an API key.
        connected = client.post("/api/pro/connect", json={}).json()
        pro_model = next(
            item for item in connected["models"] if item["model_type"] == "chat"
        )
        client.put(
            f"/api/models/{pro_model['id']}",
            json={"context_window": 6000, "max_output_tokens": 1000},
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "压缩测试方案",
                "plan_type": "chat",
                "model_ids": [pro_model["id"]],
            },
        ).json()
        controller = client.get("/api/agents").json()["controller"]
        child = client.post(
            "/api/agents",
            json={
                "name": "压缩Agent",
                "role_responsibilities": "持续对话",
                "work_content": "回答问题",
                "personality": "严谨",
                "preferences": "保留上下文",
                "chat_plan_id": plan["id"],
                "image_plan_id": None,
                "ai_organized": True,
            },
        ).json()
        long_text = "重要背景" * 900
        assert client.post(
            f"/api/chat/agents/{child['id']}/messages", json={"content": long_text}
        ).status_code == 200
        assert client.post(
            f"/api/chat/agents/{child['id']}/messages", json={"content": "继续"}
        ).status_code == 200
        messages = client.get(
            f"/api/chat/agents/{child['id']}/messages"
        ).json()
        assert any(item["content"] == long_text for item in messages)
        database = Database(f"sqlite:///{tmp_path / 'test.db'}")
        with database.sessions() as session:
            summary = session.query(ConversationSummary).filter_by(
                conversation_agent_id=child["id"]
            ).one()
            assert "继续当前对话" in summary.summary


def test_sqlite_vector_memory_crud_and_search(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'memory.db'}")
    database.create_tables()
    service = MemoryService(
        database,
        {"memory": {"enabled": True, "vector_dimensions": 128, "minimum_score": 0.05}},
    )
    entry = service.add(None, "用户喜欢简洁的开发方案")
    results = service.search("简洁开发方案", include_all_agents=True)
    assert results and results[0][0].id == entry.id
    updated = service.update(entry.id, "用户喜欢简洁且逐步实施的开发方案")
    assert "逐步实施" in updated.content
    assert service.soft_delete(entry.id).status == "deleted"
    assert service.search("逐步实施", include_all_agents=True) == []
    assert service.restore(entry.id).status == "active"


def test_controller_retry_resumes_after_completed_child(tmp_path: Path):
    class FailingSynthesisCodex(FakeCodex):
        def __init__(self):
            super().__init__()
            self.child_calls = 0
            self.synthesis_calls = 0

        def run_text_turn(
            self,
            prompt: str,
            model: str | None = None,
            session_key: str | None = None,
        ):
            if "整理最终回复" in prompt:
                self.synthesis_calls += 1
                if self.synthesis_calls <= 3:
                    raise CodexProtocolError("temporary synthesis failure")
                return "断点恢复后的最终结果"
            if "只完成总控交给你的任务" in prompt:
                self.child_calls += 1
            return super().run_text_turn(prompt, model, session_key)

    codex = FailingSynthesisCodex()
    with make_client(tmp_path, codex) as client:
        connected = client.post("/api/pro/connect", json={}).json()
        chat_model = next(
            model for model in connected["models"] if model["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "断点方案",
                "plan_type": "chat",
                "model_ids": [chat_model["id"]],
            },
        ).json()
        controller = client.get("/api/agents").json()["controller"]
        base = {
            "role_responsibilities": "协调",
            "work_content": "执行任务",
            "personality": "严谨",
            "preferences": "稳定",
            "chat_plan_id": plan["id"],
            "image_plan_id": None,
            "ai_organized": True,
        }
        client.put(
            f"/api/agents/{controller['id']}",
            json={**base, "name": "总控Agent"},
        )
        client.post("/api/agents", json={**base, "name": "执行Agent"})
        response = client.post(
            f"/api/chat/agents/{controller['id']}/messages",
            json={"content": "执行需要子Agent的任务"},
        )
        assert response.json()["status"] == "waiting_retry"
        assert codex.child_calls == 1

    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    with database.sessions() as session:
        task = session.query(PendingChatTask).filter_by(task_kind="chat").one()
        assert json.loads(task.checkpoint_json)["child_results"]
        task.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    with make_client(tmp_path, codex):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with database.sessions() as session:
                if (
                    session.query(PendingChatTask)
                    .filter_by(task_kind="chat")
                    .one()
                    .status
                    == "completed"
                ):
                    break
            time.sleep(0.05)
        else:
            raise AssertionError("重试任务没有完成")
    assert codex.child_calls == 1


def test_runtime_recovers_interrupted_agent_after_restart(tmp_path: Path):
    with make_client(tmp_path):
        pass
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    with database.sessions() as session:
        controller = session.query(AgentProfile).filter_by(is_controller=True).one()
        message = ChatMessage(
            conversation_agent_id=controller.id,
            sender_kind="user",
            recipient_agent_id=controller.id,
            content="重启恢复测试",
            status="completed",
        )
        session.add(message)
        session.flush()
        task = PendingChatTask(
            conversation_agent_id=controller.id,
            user_message_id=message.id,
            request_text=message.content,
            task_kind="chat",
            status="running",
            next_retry_at=datetime.now(timezone.utc),
        )
        session.add(task)
        session.flush()
        runtime = session.query(AgentRuntimeState).filter_by(
            agent_id=controller.id
        ).one()
        runtime.status = "running"
        runtime.active_task_id = task.id
        session.commit()

        with make_client(tmp_path) as client:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                states = client.get("/api/runtime/agents").json()
                tasks = client.get("/api/runtime/tasks").json()
                if (
                    states[0]["status"] != "running"
                    and tasks[0]["status"] != "running"
                ):
                    break
                time.sleep(0.05)
            assert states[0]["status"] in {"waiting", "waiting_retry"}
            assert tasks[0]["status"] != "running"


def test_agent_crud_pagination_organization_and_plan_protection(tmp_path: Path):
    with make_client(tmp_path) as client:
        initial = client.get("/api/agents").json()
        controller = initial["controller"]
        assert controller["name"] == "总控Agent"
        assert controller["needs_ai_organization"] is True
        assert client.delete(f"/api/agents/{controller['id']}").status_code == 409

        raw_payload = {
            "name": "需求Agent",
            "role_responsibilities": "管需求",
            "work_content": "写需求",
            "personality": "严谨",
            "preferences": "简洁",
            "chat_plan_id": None,
            "image_plan_id": None,
            "ai_organized": False,
        }
        created = client.post("/api/agents", json=raw_payload)
        assert created.status_code == 201
        raw_agent = created.json()
        assert raw_agent["parent"]["name"] == "总控Agent"
        assert raw_agent["chat_plan"] is None
        assert raw_agent["needs_ai_organization"] is True

        for index in range(10):
            payload = {**raw_payload, "name": f"分页Agent {index}"}
            assert client.post("/api/agents", json=payload).status_code == 201
        assert len(client.get("/api/agents?page=1").json()["items"]) == 10
        second_page = client.get("/api/agents?page=2").json()
        assert len(second_page["items"]) == 1
        assert second_page["pagination"]["total"] == 11

        connected = client.post("/api/pro/connect", json={}).json()
        chat_model = next(
            model for model in connected["models"] if model["model_type"] == "chat"
        )
        plan = client.post(
            "/api/plans",
            json={
                "name": "Agent整理方案",
                "plan_type": "chat",
                "model_ids": [chat_model["id"]],
            },
        ).json()
        organized = client.post(
            "/api/agents/organize",
            json={
                "chat_plan_id": plan["id"],
                "role_responsibilities": "管需求",
                "work_content": "写需求",
                "personality": "严谨",
                "preferences": "简洁",
            },
        )
        assert organized.status_code == 200
        content = organized.json()["content"]
        updated = client.put(
            f"/api/agents/{raw_agent['id']}",
            json={
                "name": raw_agent["name"],
                **content,
                "chat_plan_id": plan["id"],
                "image_plan_id": None,
                "ai_organized": True,
            },
        )
        assert updated.status_code == 200
        assert updated.json()["needs_ai_organization"] is False

        protected = client.delete(f"/api/plans/{plan['id']}")
        assert protected.status_code == 409
        assert protected.json()["detail"]["references"][0]["name"] == "需求Agent"
        assert client.delete(f"/api/agents/{raw_agent['id']}").status_code == 204
        assert client.delete(f"/api/plans/{plan['id']}").status_code == 204
