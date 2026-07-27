from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from uuid import UUID


class KBUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kbDocumentId: UUID
    status: str


class KBStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kbDocumentId: UUID
    status: str
    error_message: Optional[str] = None


class KBRetrieveRequest(BaseModel):
    owner_id: str
    query: str
    filters: Optional[Dict[str, Any]] = None


class KBChatRequest(BaseModel):
    owner_id: str
    message: str
    conversation_id: str


class KBChatResponse(BaseModel):
    response: str
