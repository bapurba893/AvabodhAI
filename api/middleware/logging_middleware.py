"""
api/middleware/logging_middleware.py
------------------------------------
Logs every request and response — method, path, status, time taken.
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000, 2)
        # Logged whenever present — not enforced here (that's
        # api.dependencies.get_tenant_id's job), just surfaced so a
        # tenant-specific incident can be traced straight from the logs.
        tenant_id = request.headers.get("x-tenant-id", "-")
        logger.info(
            "%s %s -> %d | %.2fms | tenant=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            tenant_id,
        )
        return response