from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

try:
    import psutil
except ImportError:  # Allows the settings page to load before dependencies are installed.
    psutil = None  # type: ignore[assignment]


@dataclass
class ProcessSnapshot:
    root_pid: int | None
    process_count: int
    rss_mb: float
    uptime_hours: float
    warning: bool
    restarted: bool = False


class OwnedProcessWatchdog:
    """Monitors only the process tree started by this application."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self._breaches = 0
        self.last_snapshot = ProcessSnapshot(None, 0, 0.0, 0.0, False)

    def inspect(self, config: dict[str, Any]) -> ProcessSnapshot:
        if psutil is None:
            self.last_snapshot = ProcessSnapshot(None, 0, 0.0, 0.0, False)
            return self.last_snapshot
        root_pid = getattr(self.owner, "process_pid", None)
        if not root_pid:
            self._breaches = 0
            self.last_snapshot = ProcessSnapshot(None, 0, 0.0, 0.0, False)
            return self.last_snapshot
        try:
            root = psutil.Process(root_pid)
            processes = [root, *root.children(recursive=True)]
            rss = sum(process.memory_info().rss for process in processes if process.is_running())
            uptime = max(0.0, (time.time() - root.create_time()) / 3600)
        except (psutil.Error, OSError):
            self._breaches = 0
            self.last_snapshot = ProcessSnapshot(None, 0, 0.0, 0.0, False)
            return self.last_snapshot
        rss_mb = rss / 1024 / 1024
        warning = rss_mb >= int(config["warning_mb"])
        should_recycle = (
            rss_mb >= int(config["restart_mb"])
            or uptime >= int(config["max_uptime_hours"])
        )
        self._breaches = self._breaches + 1 if should_recycle else 0
        restarted = False
        if self._breaches >= int(config["consecutive_breaches"]):
            restart = getattr(self.owner, "restart_if_idle", None)
            if restart:
                restarted = bool(restart())
            if restarted:
                self._breaches = 0
        self.last_snapshot = ProcessSnapshot(
            root_pid=root_pid,
            process_count=len(processes),
            rss_mb=round(rss_mb, 2),
            uptime_hours=round(uptime, 2),
            warning=warning,
            restarted=restarted,
        )
        return self.last_snapshot
