"""
api/middleware/tenant_guard_middleware.py
------------------------------------------
Defense-in-depth backstop for multi-tenancy AND department isolation.

Real scoping happens per-route, via api.dependencies.get_tenant_id /
get_org_unit_id as FastAPI dependencies — that's what actually threads
both into the DB queries. This middleware does NOT replace that. What it
does is close a specific future failure mode: someone adds a new
endpoint later, forgets to declare
`tenant_id: str = Depends(get_tenant_id)` (or the org_unit_id
equivalent) on it, and that route silently serves unscoped data with no
error and no warning.

This middleware runs before routing and rejects any request to a
non-exempt path that doesn't carry BOTH X-Tenant-ID and X-Org-Unit-ID —
so a route missing either per-route dependency fails loudly with a 400
instead of quietly leaking. It intentionally does NOT validate the
values beyond presence (that's still get_tenant_id/get_org_unit_id's
job, so error messages and any future validation logic stay in one
place).
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from utils.logger import get_logger

logger = get_logger(__name__)

# Paths that are legitimately tenant-agnostic — ops/monitoring endpoints
# and the API's own docs. Everything else must carry both headers.
_EXEMPT_PATHS = {
    "/", "/health", "/health/", "/health/db",
    "/docs", "/redoc", "/openapi.json",
}


def _is_signed_file_link(path: str) -> bool:
    """
    GET /documents/{id}/file is deliberately authorized via a signed
    token in the URL, NOT via headers — it's meant to work as a plain
    clickable link/<iframe src>, and browsers don't attach custom
    headers to those. Blanket-requiring X-Org-Unit-ID here would break
    that endpoint outright, so it's exempted from the header check —
    its own real authorization (utils.signed_link.verify_file_token)
    happens inside the route handler itself, not weakened by this.
    """
    return path.startswith("/documents/") and path.endswith("/file")


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
        if path not in _EXEMPT_PATHS and not path.startswith("/health") and not _is_signed_file_link(path):
            missing = []
            if not request.headers.get("x-tenant-id", "").strip():
                missing.append("X-Tenant-ID")
            if not request.headers.get("x-org-unit-id", "").strip():
                missing.append("X-Org-Unit-ID")
            if missing:
                logger.warning("Rejected request missing %s: %s %s", missing, request.method, path)
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"{' and '.join(missing)} header is required"},
                )
        return await call_next(request)