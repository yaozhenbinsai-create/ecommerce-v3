from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .codex_client import CodexProtocolError
from .env_store import EnvStore
from .models import ModelEntry


class ModelRunError(RuntimeError):
    def __init__(self, message: str, kind: str = "transient") -> None:
        self.kind = kind
        super().__init__(message)


class PlanRunError(RuntimeError):
    def __init__(
        self,
        errors: list[str],
        *,
        kind: str = "transient",
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.errors = errors
        self.kind = kind
        self.checkpoint = checkpoint
        super().__init__("；".join(errors))


@dataclass
class PlanRunResult:
    text: str
    model_id: int
    model_name: str
    attempts: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_source: str = "estimated"


@dataclass
class ModelCallResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_source: str


class ChatRuntime:
    def __init__(
        self,
        env_store: EnvStore,
        codex: Any,
        settings: dict[str, Any],
        retry_pause_seconds: float = 1.0,
    ) -> None:
        self.env_store = env_store
        self.codex = codex
        self.settings = settings
        self.retry_pause_seconds = retry_pause_seconds

    def run_models(
        self,
        models: list[ModelEntry],
        prompt: str,
        attempts_per_model: int | None = None,
        session_key: str | None = None,
        on_attempt: Callable[[dict[str, Any]], None] | None = None,
        attempts_by_model: dict[int, int] | None = None,
        timeout_seconds: float | None = None,
        before_attempt: Callable[[], bool] | None = None,
    ) -> PlanRunResult:
        attempts_per_model = attempts_per_model or int(
            self.settings.get("retry", {}).get("attempts_per_model", 3)
        )
        errors: list[str] = []
        total_attempts = 0
        enabled = [model for model in models if model.is_enabled]
        if not enabled:
            raise PlanRunError(["方案中没有已启用模型"])
        for model in enabled:
            model_attempts = max(
                1,
                int(
                    (attempts_by_model or {}).get(model.id, attempts_per_model)
                ),
            )
            for attempt in range(1, model_attempts + 1):
                if before_attempt is not None and not before_attempt():
                    raise PlanRunError(
                        ["工作硬截止时间已到，禁止开始新的模型尝试"],
                        kind="deadline",
                    )
                total_attempts += 1
                started = time.perf_counter()
                try:
                    if timeout_seconds is None:
                        call = self.call_model_result(
                            model, prompt, session_key=session_key
                        )
                    else:
                        call = self.call_model_result(
                            model,
                            prompt,
                            session_key=session_key,
                            timeout_seconds=timeout_seconds,
                        )
                    text = call.text.strip()
                    if not text:
                        raise ModelRunError("模型返回空内容")
                    if on_attempt:
                        on_attempt(
                            {
                                "attempt_number": total_attempts,
                                "model": model,
                                "status": "completed",
                                "output": text,
                                "input_tokens": call.input_tokens,
                                "output_tokens": call.output_tokens,
                                "total_tokens": call.total_tokens,
                                "usage_source": call.usage_source,
                                "duration_ms": int((time.perf_counter() - started) * 1000),
                            }
                        )
                    return PlanRunResult(
                        text=text,
                        model_id=model.id,
                        model_name=model.name,
                        attempts=total_attempts,
                        input_tokens=call.input_tokens,
                        output_tokens=call.output_tokens,
                        total_tokens=call.total_tokens,
                        usage_source=call.usage_source,
                    )
                except CodexProtocolError as exc:
                    error = ModelRunError(str(exc), self.classify_error(str(exc)))
                except ModelRunError as exc:
                    error = exc
                else:
                    continue
                exc = error
                if on_attempt:
                    on_attempt(
                        {
                            "attempt_number": total_attempts,
                            "model": model,
                            "status": "failed",
                            "error_kind": error.kind,
                            "error": str(error),
                            "output": "",
                            "input_tokens": self.estimate_tokens(prompt),
                            "output_tokens": 0,
                            "total_tokens": self.estimate_tokens(prompt),
                            "usage_source": "estimated",
                            "duration_ms": int((time.perf_counter() - started) * 1000),
                        }
                    )
                if error.kind == "context_overflow":
                    errors.append(f"{model.name} 第{attempt}次：{exc}")
                    raise PlanRunError(errors, kind="context_overflow") from exc
                errors.append(f"{model.name} 第{attempt}次：{exc}")
                if error.kind in {"auth", "configuration", "invalid_request"}:
                    break
                if model.access_method == "chatgpt_pro":
                    close = getattr(self.codex, "close", None)
                    if close:
                        close()
                if attempt < model_attempts and self.retry_pause_seconds:
                    time.sleep(self.retry_pause_seconds)
        raise PlanRunError(errors)

    @staticmethod
    def classify_error(message: str, status_code: int | None = None) -> str:
        lowered = message.lower()
        if status_code in {401, 403}:
            return "auth"
        if status_code == 413 or any(
            marker in lowered
            for marker in (
                "context_length",
                "context window",
                "maximum context",
                "too many tokens",
                "prompt is too long",
                "上下文",
            )
        ):
            return "context_overflow"
        if status_code in {400, 404, 405, 422}:
            return "invalid_request"
        if status_code in {408, 409, 425, 429} or (
            status_code is not None and status_code >= 500
        ):
            return "transient"
        if any(marker in lowered for marker in ("api key", "未配置", "标识不完整")):
            return "configuration"
        return "transient"

    def call_model(
        self, model: ModelEntry, prompt: str, session_key: str | None = None
    ) -> str:
        return self.call_model_result(model, prompt, session_key=session_key).text

    @staticmethod
    def estimate_tokens(value: str) -> int:
        ascii_count = sum(1 for char in value if ord(char) < 128)
        other_count = len(value) - ascii_count
        return max(1, int(ascii_count / 4 + other_count / 1.5 + 0.999))

    def call_model_result(
        self,
        model: ModelEntry,
        prompt: str,
        session_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ModelCallResult:
        if model.access_method == "chatgpt_pro":
            model_session = f"{session_key}:model:{model.id}" if session_key else None
            text = self.codex.run_text_turn(
                prompt, model.model_identifier, session_key=model_session
            )
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(text)
            return ModelCallResult(
                text, input_tokens, output_tokens, input_tokens + output_tokens, "estimated"
            )
        api_key = self.env_store.get(model.credential_env)
        if not api_key:
            raise ModelRunError("未配置API Key", "configuration")
        base_url = (model.base_url or "").rstrip("/")
        if not base_url or not model.model_identifier:
            raise ModelRunError("模型地址或标识不完整", "configuration")
        timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(
                self.settings.get("retry", {}).get(
                    "request_timeout_seconds", 300
                )
            )
        )
        max_tokens = model.max_output_tokens or self.settings.get(
            "conversation", {}
        ).get("default_max_output_tokens", 4096)
        try:
            with httpx.Client(trust_env=False, timeout=timeout) as client:
                if model.protocol == "claude":
                    endpoint = (
                        f"{base_url}/messages"
                        if base_url.endswith("/v1")
                        else f"{base_url}/v1/messages"
                    )
                    response = client.post(
                        endpoint,
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        json={
                            "model": model.model_identifier,
                            "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    text = "".join(
                        part.get("text", "")
                        for part in (body.get("content") or [])
                        if part.get("type") == "text"
                    )
                    usage = body.get("usage") or {}
                    input_tokens = int(usage.get("input_tokens") or self.estimate_tokens(prompt))
                    output_tokens = int(usage.get("output_tokens") or self.estimate_tokens(text))
                    return ModelCallResult(
                        text, input_tokens, output_tokens, input_tokens + output_tokens,
                        "provider" if usage else "estimated"
                    )
                if model.protocol == "openai_responses":
                    response = client.post(
                        f"{base_url}/responses",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model.model_identifier,
                            "input": prompt,
                            "max_output_tokens": max_tokens,
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    text = str(body["output_text"]) if body.get("output_text") else "".join(
                        str(part.get("text") or "") for item in (body.get("output") or [])
                        for part in (item.get("content") or []) if part.get("type") in {"output_text", "text"}
                    )
                    usage = body.get("usage") or {}
                    input_tokens = int(usage.get("input_tokens") or self.estimate_tokens(prompt))
                    output_tokens = int(usage.get("output_tokens") or self.estimate_tokens(text))
                    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
                    return ModelCallResult(
                        text, input_tokens, output_tokens, total_tokens,
                        "provider" if usage else "estimated"
                    )
                chat_base_url = base_url
                if (
                    model.protocol == "claude_openai"
                    and not chat_base_url.endswith("/v1")
                ):
                    chat_base_url = f"{chat_base_url}/v1"
                response = client.post(
                    f"{chat_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model.model_identifier,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                    },
                )
                response.raise_for_status()
                body = response.json()
                text = body["choices"][0]["message"]["content"]
                usage = body.get("usage") or {}
                input_tokens = int(usage.get("prompt_tokens") or self.estimate_tokens(prompt))
                output_tokens = int(usage.get("completion_tokens") or self.estimate_tokens(text))
                total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
                return ModelCallResult(
                    text, input_tokens, output_tokens, total_tokens,
                    "provider" if usage else "estimated"
                )
        except httpx.HTTPStatusError as exc:
            details = exc.response.text[:500]
            status = exc.response.status_code
            raise ModelRunError(
                f"HTTP {status}：{details}", self.classify_error(details, status)
            ) from exc
        except httpx.RequestError as exc:
            raise ModelRunError(f"连接失败：{exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelRunError("响应格式不正确") from exc
