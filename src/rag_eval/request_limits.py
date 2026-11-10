"""Bound total HTTP request bodies before JSON parsing or indexing.

ASGI streaming guard; works even when Content-Length is omitted. Not a replacement
for reverse-proxy limits, timeouts or connection throttling.
"""
from __future__ import annotations

from starlette.responses import JSONResponse

MAX_REQUEST_BYTES = 512 * 1024


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app, max_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    declared = int(value)
                except (ValueError, UnicodeError):
                    await JSONResponse(
                        {"detail": "Invalid Content-Length"}, status_code=400
                    )(scope, receive, send)
                    return
                if declared < 0:
                    await JSONResponse(
                        {"detail": "Invalid Content-Length"}, status_code=400
                    )(scope, receive, send)
                    return
                if declared > self.max_bytes:
                    await JSONResponse(
                        {"detail": "Request body exceeds 512 KiB"}, status_code=413
                    )(scope, receive, send)
                    return

        body_size = 0
        response_started = False

        async def checked_receive():
            nonlocal body_size
            message = await receive()
            if message["type"] == "http.request":
                body_size += len(message.get("body", b""))
                if body_size > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def checked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, checked_receive, checked_send)
        except _BodyTooLarge:
            if response_started:
                raise
            await JSONResponse(
                {"detail": "Request body exceeds 512 KiB"}, status_code=413
            )(scope, receive, send)
