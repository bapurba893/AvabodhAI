"""
api/middleware/tenant_guard_middleware.py
------------------------------------------
Defense-in-depth backstop for multi-tenancy.

Real tenant scoping happens per-route, via api.dependencies.get_tenant_id
as a FastAPI dependency — that's what actually threads tenant_id into the
DB queries. This middleware does NOT replace that. What it does is close
a specific future failure mode: someone adds a new endpoint later, forgets
to declare `tenant_id: str = Depends(get_tenant_id)` on it, and that route
silently serves unscoped (all-tenant) data with no error and no warning.

This middleware runs before routing and rejects any request to a
non-exempt path that doesn't carry an X-Tenant-ID header — so a route
missing its per-route dependency fails loudly with a 400 instead of
quietly leaking. It intentionally does NOT validate the value beyond
presence (that's still get_tenant_id's job, so error messages and any
future validation logic stay in one place).
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from utils.logger import get_logger

logger = get_logger(__name__)

# Paths that are legitimately tenant-agnostic — ops/monitoring endpoints
# and the API's own docs. Everything else must carry X-Tenant-ID.
_EXEMPT_PATHS = {
    "/", "/health", "/health/", "/health/db",
    "/docs", "/redoc", "/openapi.json",
}


class TenantGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # CORS preflight requests never carry custom headers like
        # X-Tenant-ID — that's the actual request that follows, once the
        # browser has the preflight's OK. Rejecting OPTIONS here would
        # break CORS entirely for any browser-based frontend, since the
        # preflight would never reach CORSMiddleware to get answered.
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path not in _EXEMPT_PATHS and not path.startswith("/health"):
            if not request.headers.get("x-tenant-id", "").strip():
                logger.warning("Rejected request with no X-Tenant-ID: %s %s", request.method, path)
                return JSONResponse(
                    status_code=400,
                    content={"detail": "X-Tenant-ID header is required"},
                )
        return await call_next(request)