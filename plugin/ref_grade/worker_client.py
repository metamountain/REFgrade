"""HTTP-Client für den REF-Grade-Worker – ohne krita, ohne Qt."""

from __future__ import annotations

import json
from urllib.request import urlopen, Request


class WorkerError(Exception):
    """HTTP-Fehler oder Fehlerantwort vom Worker."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"[{status}] {message}")


def call(port: int, path: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    """GET (ohne payload) oder POST JSON; dict zurückgeben, WorkerError bei Fehler."""
    url = f"http://127.0.0.1:{port}{path}"
    if payload is None:
        req = Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )

    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        code = getattr(e, "code", 500)
        try:
            msg_data = json.loads(e.read().decode("utf-8"))
            msg = msg_data.get("error", str(e))
        except Exception:
            msg = str(e)
        raise WorkerError(code, msg) from None

    if "error" in data:
        raise WorkerError(200, data["error"])
    return data
