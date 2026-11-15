"""One error envelope for every non-2xx response (``{"error": {code, message, request_id}}``)."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from volforecast.logging import get_logger

log = get_logger(__name__)

_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "unavailable",
}


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers or {}
        self.details = details


def request_id_of(request: Request) -> str | None:
    rid = getattr(request.state, "request_id", None)
    return None if rid is None else str(rid)


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    rid = request_id_of(request)
    hdrs = dict(headers or {})
    if rid:
        hdrs.setdefault("X-Request-ID", rid)
    body: dict[str, Any] = {"error": {"code": code, "message": message, "request_id": rid}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status_code, headers=hdrs)


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(request, exc.status_code, exc.code, exc.message, exc.headers, exc.details)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _CODES.get(exc.status_code, f"http_{exc.status_code}")
    try:
        default = HTTPStatus(exc.status_code).phrase
    except ValueError:
        default = "error"
    message = str(exc.detail) if exc.detail else default
    return error_response(request, exc.status_code, code, message, dict(exc.headers or {}))


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        {
            "loc": [str(p) for p in e.get("loc", ())],
            "msg": str(e.get("msg", "")),
            "type": str(e.get("type", "")),
        }
        for e in exc.errors()
    ]
    first = details[0] if details else {"loc": [], "msg": "invalid request"}
    message = f"invalid {'.'.join(first['loc'])}: {first['msg']}"
    return error_response(request, 422, "validation_error", message, details=details)


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled_exception", err=repr(exc), path=request.url.path)
    return error_response(request, 500, "internal_error", "internal server error")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)
