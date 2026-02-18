from pydantic import BaseModel, Field
from typing import Optional


class DocumentCreateResponse(BaseModel):
    """Схема для ответа на запрос создания документа."""

    id: int = Field(..., json_schema_extra={"example": 1})
    filename: str = Field(..., json_schema_extra={"example": "example.pdf"})
    chunks_count: int = Field(..., ge=1, json_schema_extra={"example": 5})


class DocumentUploadResponse(BaseModel):
    """Схема для ответа на запрос загрузки документа."""

    id: int = Field(..., json_schema_extra={"example": 1})
    filename: str = Field(..., json_schema_extra={"example": "example.pdf"})
    chunks_count: int = Field(..., ge=1, json_schema_extra={"example": 5})
