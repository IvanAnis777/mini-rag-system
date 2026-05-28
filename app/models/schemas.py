"""Pydantic схемы для API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Запрос пользователя."""
    question: str = Field(..., description="Вопрос пользователя", min_length=1, max_length=1000)
    language: Optional[str] = Field(default="ru", description="Язык ответа (ru/en)")
    max_chunks: Optional[int] = Field(default=5, description="Максимальное количество контекстных чунков")


class DocumentChunk(BaseModel):
    """Чунк документа из векторного хранилища."""
    content: str = Field(..., description="Содержимое чунка")
    similarity_score: float = Field(..., description="Оценка схожести", ge=0.0, le=1.0)
    document_id: Optional[str] = Field(None, description="ID исходного документа")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Метаданные чунка")


class QueryResponse(BaseModel):
    """Ответ системы."""
    answer: str = Field(..., description="Ответ в формате Markdown")
    sources: List[str] = Field(default_factory=list, description="Список источников с цитированием")
    query_rewritten: str = Field(..., description="Переформулированный запрос")
    context_chunks: List[DocumentChunk] = Field(default_factory=list, description="Использованные чунки")
    confidence: float = Field(..., description="Уверенность в ответе", ge=0.0, le=1.0)
    processing_time: float = Field(..., description="Время обработки в секундах")


class AgenticQueryResponse(BaseModel):
    """Ответ агентного RAG (LangGraph) с трейсом самокоррекции."""
    answer: str = Field(..., description="Итоговый ответ")
    sources: List[str] = Field(default_factory=list, description="Источники с цитированием")
    documents_used: int = Field(..., description="Сколько релевантных чунков попало в контекст")
    transforms: int = Field(..., description="Число переформулировок запроса")
    generations: int = Field(..., description="Число попыток генерации")
    grounded: bool = Field(..., description="Ответ обоснован контекстом (не галлюцинация)")
    answers_question: bool = Field(..., description="Ответ отвечает по существу")
    trace: List[str] = Field(default_factory=list, description="Лог шагов графа")
    processing_time: float = Field(..., description="Время обработки в секундах")


class DocumentUploadRequest(BaseModel):
    """Запрос на загрузку документа."""
    content: str = Field(..., description="Содержимое документа")
    title: str = Field(..., description="Название документа")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Метаданные документа")


class DocumentUploadResponse(BaseModel):
    """Ответ на загрузку документа."""
    document_id: str = Field(..., description="ID загруженного документа")
    chunks_created: int = Field(..., description="Количество созданных чунков")
    message: str = Field(..., description="Сообщение о статусе")


class HealthResponse(BaseModel):
    """Ответ о состоянии системы."""
    status: str = Field(..., description="Статус системы")
    services: Dict[str, str] = Field(..., description="Статус сервисов")
    timestamp: str = Field(..., description="Время проверки")


class SearchRequest(BaseModel):
    """Запрос поиска по векторному хранилищу."""
    query: str = Field(..., description="Поисковый запрос")
    limit: Optional[int] = Field(default=5, description="Количество результатов")
    threshold: Optional[float] = Field(default=0.7, description="Порог схожести")


class SearchResponse(BaseModel):
    """Ответ поиска."""
    results: List[DocumentChunk] = Field(..., description="Найденные чунки")
    total_found: int = Field(..., description="Общее количество найденных результатов")
    query_embedding_time: float = Field(..., description="Время векторизации запроса")
    search_time: float = Field(..., description="Время поиска")


class ErrorResponse(BaseModel):
    """Ответ об ошибке."""
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали") 