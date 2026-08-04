"""
api/dependencies.py
--------------------
Shared FastAPI dependencies.

get_tenant_id() is the ONE place tenant identity enters the KB container.
Every route that reads or writes document_summaries / document_chunks /
chat_threads / chat_messages depends on this and threads the value all the
way down to the raw SQL WHERE clauses in pipeline/retriever.py and
api/routes/search.py — there is no other mechanism enforcing isolation
between tenants.

Trust boundary: this header is trusted as coming from the Security Gateway
(or an equivalent internal caller) — NOT from the end user directly. That
only holds if the KB container is unreachable except through that trusted
caller (internal network / service mesh / mTLS). If the KB container is
ever exposed directly to end users, this header must instead be derived
from a verified token (JWT claim, session lookup, etc.), not trusted as
plain input.
"""

from fastapi import Header, HTTPException

_MAX_TENANT_ID_LENGTH = 128


def get_tenant_id(x_tenant_id: str = Header(..., alias="X-Tenant-ID")) -> str:
    """
    FastAPI dependency — every tenant-scoped route takes this as a parameter:

        tenant_id: str = Depends(get_tenant_id)

    Missing, blank, or oversized header -> 400, rejected before it ever
    reaches the database layer.
    """
    tenant_id = (x_tenant_id or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header is required")
    if len(tenant_id) > _MAX_TENANT_ID_LENGTH:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header is too long")
    return tenant_id