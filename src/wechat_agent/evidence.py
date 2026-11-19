"""Private 0700/0600 evidence directory management and payload/log redaction."""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {
    "transaction_id",
    "reservation",
    "api_key",
    "clipboard_key",
    "token",
    "secret",
    "password",
    "image_base64",
}

_PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def redact_text(value: str) -> str:
    value = _ID_CARD_PATTERN.sub("[REDACTED_ID]", value)
    value = _PHONE_PATTERN.sub("[REDACTED_PHONE]", value)
    return value


def redact_dict(data: Any) -> Any:
    """Recursively mask sensitive keys, base64 blobs, and phone/ID patterns for safe log export."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            key_lower = str(k).lower()
            if key_lower in SENSITIVE_KEYS:
                if key_lower == "image_base64" and isinstance(v, str):
                    cleaned[k] = f"<redacted_base64 len={len(v)}>"
                else:
                    cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_dict(v)
        return cleaned
    if isinstance(data, list):
        return [redact_dict(item) for item in data]
    if isinstance(data, str):
        return redact_text(data)
    return data


def _chmod_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)
    return path


class RunEvidence:
    """Per-run private artifact store under .local-data/runs/<run_id>/ with 0700/0600 permissions."""

    def __init__(self, root: Path, run_id: str | None = None) -> None:
        self.root = Path(root)
        _chmod_private_dir(self.root)
        if run_id is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = f"{stamp}-{secrets.token_hex(3)}"
        self.run_id = run_id
        self.dir = self.root / run_id
        _chmod_private_dir(self.dir)
        self.paste_evidence_dir = self.dir / "paste_stages"
        _chmod_private_dir(self.paste_evidence_dir)
        self._events_path = self.dir / "events.jsonl"

    def write_payload(self, text: str) -> Path:
        return _write_private_file(self.dir / "payload.txt", text)

    def image_output_path(self, step_name: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", step_name).strip("_") or "frame"
        return self.dir / f"{safe}.jpg"

    def secure_existing_file(self, path: str | Path | None) -> None:
        if not path:
            return
        p = Path(path)
        if p.is_file():
            os.chmod(p, 0o600)

    def write_json(self, name: str, payload: dict[str, Any], *, redact: bool = False) -> Path:
        data = redact_dict(payload) if redact else payload
        return _write_private_file(
            self.dir / name,
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        )

    def log_event(self, step: str, details: dict[str, Any] | None = None) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "step": step,
            "details": redact_dict(details or {}),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        fd = os.open(str(self._events_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line)
        os.chmod(self._events_path, 0o600)
