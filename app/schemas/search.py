from pydantic import BaseModel, Field
from typing import List, Optional


class SearchResult(BaseModel):
    """Схема для результата поиска."""

    id: int = Field(..., json_schema_extra={"example": 1})
    text: str = Field(
        ..., json_schema_extra={"example": "Пример текста из документа..."}
    )
    filename: str = Field(..., json_schema_extra={"example": "example.pdf"})
    similarity: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.95})


class LexicalSearchResponse(BaseModel):
    """Схема для ответа на лексический поиск."""

    results: List[SearchResult] = Field(
        ...,
        json_schema_extra={
            "example": [
                {
                    "id": 1,
                    "text": "Пример текста из документа...",
                    "filename": "example.pdf",
                    "similarity": 0.95,
                }
            ]
        },
    )


class SemanticSearchResponse(BaseModel):
    """Схема для ответа на семантический поиск."""

    results: List[SearchResult] = Field(
        ...,
        json_schema_extra={
            "example": [
                {
                    "id": 1,
                    "text": "Пример текста из документа...",
                    "filename": "example.pdf",
                    "similarity": 0.95,
                }
            ]
        },
    )


class HybridSearchResponse(BaseModel):
    """Схема для ответа на гибридный поиск."""

    results: List[SearchResult] = Field(
        ...,
        json_schema_extra={
            "example": [
                {
                    "id": 1,
                    "text": "Пример текста из документа...",
                    "filename": "example.pdf",
                    "similarity": 0.95,
                }
            ]
        },
    )
