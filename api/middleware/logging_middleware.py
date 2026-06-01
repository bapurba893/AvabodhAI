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
        logger.info(
            "%s %s -> %d | %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response