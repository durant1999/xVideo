from __future__ import annotations

import json
import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any

from .settings import get_settings


logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


def store_path() -> Path:
    settings = get_settings()
    job_root = Path(settings.job_root)
    root = job_root if job_root.is_absolute() else Path(settings.repo_root) / job_root
    return root / "devices.json"


def _load() -> list[dict[str, Any]]:
    path = store_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001 - corrupted registry should not break BFF startup.
        logger.warning("Could not read device registry %s: %s", path, exc)
        return []
    if not isinstance(data, list):
        return []
    return [
        item
        for item in data
        if isinstance(item, dict) and isinstance(item.get("push_token"), str)
    ]


def _save(items: list[dict[str, Any]]) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def add_device(push_token: str, platform: str) -> None:
    push_token = push_token.strip()
    platform = (platform or "unknown").strip() or "unknown"
    if not push_token:
        raise ValueError("push_token is required")

    with _LOCK:
        items = _load()
        for item in items:
            if item.get("push_token") == push_token:
                item["platform"] = platform
                _save(items)
                return
        items.append({"push_token": push_token, "platform": platform})
        _save(items)


def _send_expo(messages: list[dict[str, Any]]) -> None:
    if not messages:
        return
    for start in range(0, len(messages), 100):
        chunk = messages[start : start + 100]
        req = urllib.request.Request(
            EXPO_PUSH_URL,
            data=json.dumps(chunk).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=15).read()
        except Exception as exc:  # pragma: no cover - depends on external Expo service.
            logger.warning("[push] send failed: %s", exc)


def notify_job_done(job: dict[str, Any]) -> None:
    """Blocking; call from asyncio.to_thread."""
    tokens = [item["push_token"] for item in _load()]
    if not tokens:
        return

    status = str(job.get("status") or "unknown")
    title = "分析完成" if status == "succeeded" else f"任务{status}"
    body = str(job.get("source") or job.get("job_id") or "")[:48]
    messages = [
        {
            "to": token,
            "title": title,
            "body": body,
            "data": {"jobId": job.get("job_id")},
        }
        for token in tokens
    ]
    _send_expo(messages)
