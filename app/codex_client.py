from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


class CodexProtocolError(RuntimeError):
    pass


class CodexAppServer:
    """Small persistent JSON-RPC client for the official Codex app-server."""

    def __init__(self, cwd: Path, timeout_seconds: int = 30) -> None:
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.RLock()
        self._next_id = 0
        self._notifications: deque[dict[str, Any]] = deque(maxlen=100)
        self._stderr: deque[str] = deque(maxlen=20)
        self._active_turns = 0
        self._threads: dict[str, str] = {}

    @property
    def process_pid(self) -> int | None:
        process = self._process
        return process.pid if process and process.poll() is None else None

    def restart_if_idle(self) -> bool:
        if self._active_turns:
            return False
        self.close()
        return True

    def reset_sessions(self, prefix: str) -> None:
        with self._lock:
            keys = [key for key in self._threads if key.startswith(prefix)]
            for key in keys:
                self._threads.pop(key, None)

    def inspect_account(self) -> dict[str, Any]:
        account_result = self.request("account/read", {"refreshToken": False})
        account = account_result.get("account")
        result: dict[str, Any] = {
            "account": account,
            "requiresOpenaiAuth": account_result.get("requiresOpenaiAuth"),
            "providerCapabilities": {},
            "models": [],
        }
        if not account:
            return result
        result["providerCapabilities"] = self.request(
            "modelProvider/capabilities/read", {}
        )
        model_result = self.request("model/list", {})
        result["models"] = model_result.get("data") or model_result.get("models") or []
        return result

    def start_chatgpt_login(self) -> dict[str, Any]:
        return self.request("account/login/start", {"type": "chatgpt"}, timeout=60)

    def run_text_turn(
        self,
        prompt: str,
        model: str | None = None,
        timeout: int = 120,
        session_key: str | None = None,
    ) -> str:
        with self._lock:
            self._active_turns += 1
            try:
                self._ensure_started()
                self._notifications.clear()
                thread_id = self._threads.get(session_key) if session_key else None
                if not thread_id:
                    thread = self._request_locked(
                        "thread/start",
                        {"cwd": str(self.cwd), **({"model": model} if model else {})},
                        self.timeout_seconds,
                    )
                    thread_id = thread["thread"]["id"]
                    if session_key:
                        self._threads[session_key] = thread_id
                turn = self._request_locked(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": prompt}],
                    },
                    self.timeout_seconds,
                )
                turn_id = turn["turn"]["id"]
                pending = list(self._notifications)
                self._notifications.clear()
                text_parts: list[str] = []
                deadline = time.monotonic() + timeout

                while time.monotonic() < deadline:
                    if pending:
                        message = pending.pop(0)
                    else:
                        try:
                            message = self._messages.get(
                                timeout=max(0.1, deadline - time.monotonic())
                            )
                        except queue.Empty as exc:
                            raise CodexProtocolError("Codex text turn timed out") from exc
                    method = message.get("method")
                    params = message.get("params") or {}
                    if method == "item/agentMessage/delta":
                        delta = params.get("delta")
                        if isinstance(delta, str):
                            text_parts.append(delta)
                    elif method == "turn/completed":
                        completed_turn = params.get("turn") or {}
                        if completed_turn.get("id") != turn_id:
                            continue
                        if completed_turn.get("status") != "completed":
                            raise CodexProtocolError(
                                f"Codex turn ended with {completed_turn.get('status')}"
                            )
                        result = "".join(text_parts).strip()
                        if not result:
                            raise CodexProtocolError("Codex returned empty text")
                        return result
                    elif method == "error":
                        raise CodexProtocolError(json.dumps(params, ensure_ascii=False))
                raise CodexProtocolError("Codex text turn timed out")
            finally:
                self._active_turns -= 1

    def request(
        self, method: str, params: dict[str, Any], timeout: int | None = None
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_started()
            return self._request_locked(method, params, timeout or self.timeout_seconds)

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._threads.clear()
            if not process or process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _codex_home(self) -> Path:
        return (self.cwd / 'data' / 'official-client').resolve()

    def _process_environment(self):
        # Each deployment owns its official login. The CLI only needs the OS runtime environment.
        keep = {'PATH', 'PATHEXT', 'WINDIR', 'SYSTEMROOT', 'SYSTEMDRIVE', 'COMSPEC', 'PROGRAMDATA',
                'PROGRAMFILES', 'PROGRAMFILES(X86)', 'USERPROFILE', 'USERNAME', 'USER', 'HOME',
                'HOMEDRIVE', 'HOMEPATH', 'APPDATA', 'LOCALAPPDATA', 'TEMP', 'TMP', 'TMPDIR', 'LANG', 'LANGUAGE', 'TERM'}
        env = {name: value for name, value in os.environ.items() if name.upper() in keep or name.upper().startswith('LC_')}
        home = self._codex_home()
        home.mkdir(parents=True, mode=0o700, exist_ok=True)
        env['CODEX_HOME'] = str(home)
        return env

    def _ensure_started(self) -> None:
        if self._process and self._process.poll() is None:
            return
        self._messages = queue.Queue()
        command = (
            ["cmd.exe", "/d", "/c", "codex", "app-server", "--listen", "stdio://"]
            if os.name == "nt"
            else ["codex", "app-server", "--listen", "stdio://"]
        )
        self._process = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=self._process_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        threading.Thread(
            target=self._read_stdout, args=(self._process.stdout,), daemon=True
        ).start()
        threading.Thread(
            target=self._read_stderr, args=(self._process.stderr,), daemon=True
        ).start()
        self._request_locked(
            "initialize",
            {
                "clientInfo": {
                    "name": "boss_agent",
                    "title": "Boss Agent",
                    "version": "0.1.0",
                }
            },
            self.timeout_seconds,
        )
        self._send({"method": "initialized", "params": {}})

    def _request_locked(
        self, method: str, params: dict[str, Any], timeout: int
    ) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"method": method, "id": request_id, "params": params})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                details = " | ".join(self._stderr)
                raise CodexProtocolError(
                    f"Codex request timed out: {method}. {details}".strip()
                ) from exc
            if message.get("id") != request_id:
                self._notifications.append(message)
                continue
            if "error" in message:
                raise CodexProtocolError(
                    json.dumps(message["error"], ensure_ascii=False)
                )
            return message.get("result") or {}
        raise CodexProtocolError(f"Codex request timed out: {method}")

    def _send(self, message: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise CodexProtocolError("Codex app-server is not running")
        self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def _read_stdout(self, stream: Any) -> None:
        for raw_line in iter(stream.readline, ""):
            try:
                self._messages.put(json.loads(raw_line))
            except json.JSONDecodeError:
                continue

    def _read_stderr(self, stream: Any) -> None:
        for raw_line in iter(stream.readline, ""):
            line = raw_line.strip()
            if line:
                self._stderr.append(line)
