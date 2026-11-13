"""Thread-local keep-alive JSON HTTP client.

The embedding and rerank gateways are hit once per query. Opening a new TCP
connection each time exhausts ephemeral ports under benchmark concurrency
(observed as ``Cannot assign requested address``) and adds latency. Each
worker thread keeps one HTTP/1.1 connection per base URL.
"""

from __future__ import annotations

import http.client
import json
import threading
from urllib.parse import urlsplit


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class JsonHttp:
    def __init__(self, base: str, timeout: float):
        parts = urlsplit(base.rstrip("/"))
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"invalid base URL: {base}")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.prefix = parts.path.rstrip("/")
        self.timeout = timeout
        self._local = threading.local()

    def _connection(self) -> http.client.HTTPConnection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            connection = cls(self.host, self.port, timeout=self.timeout)
            self._local.connection = connection
        return connection

    def _drop(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            try:
                connection.close()
            finally:
                self._local.connection = None

    def post(self, path: str, body: dict, headers: dict[str, str] | None = None) -> dict:
        target = self.prefix + "/" + path.lstrip("/")
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
        request_headers.update(headers or {})
        last: Exception | None = None
        for _attempt in range(2):
            connection = self._connection()
            try:
                connection.request("POST", target, body=payload, headers=request_headers)
                response = connection.getresponse()
                raw = response.read()
                if response.status >= 400:
                    raise HttpError(response.status, raw.decode("utf-8", "replace"))
                value = json.loads(raw.decode("utf-8", "replace"))
                if not isinstance(value, dict):
                    raise ValueError("response is not a JSON object")
                return value
            except HttpError:
                raise
            except (http.client.HTTPException, OSError, ValueError) as exc:
                last = exc
                self._drop()
        raise RuntimeError(f"request failed: {last}") from last
