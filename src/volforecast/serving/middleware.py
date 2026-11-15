"""Pure-ASGI middleware: request id + structured access log, and HTTP metrics.

Order (outermost first): CORS, metrics, request context. Metrics sit outside the request context
so even unhandled errors are counted, and the ``route`` label is the matched route *template*
(``/v1/forecast``), or ``unmatched`` when no route matched, never the raw path.
"""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from volforecast.logging import get_logger
from volforecast.serving.metrics import UNMATCHED_ROUTE, observe_request

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
_QUIET_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else UNMATCHED_ROUTE


class RequestContextMiddleware:
    """Accepts/echoes ``X-Request-ID``, binds it to the log context, logs one line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = ""
        for k, v in scope["headers"]:
            if k == b"x-request-id":
                incoming = v.decode("latin-1")
                break
        rid = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        structlog.contextvars.bind_contextvars(request_id=rid)
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = rid
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope["path"]
            emit = log.debug if path in _QUIET_PATHS and status < 400 else log.info
            emit(
                "request",
                method=scope["method"],
                path=path,
                route=route_template(scope),
                status=status,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                client=(scope.get("client") or ("-", 0))[0],
            )
            structlog.contextvars.unbind_contextvars("request_id")


class MetricsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            observe_request(
                route_template(scope), scope["method"], status, time.perf_counter() - start
            )
