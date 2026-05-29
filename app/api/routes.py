"""API роуты для Mini RAG системы."""

import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.orm import Session

from app.models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadRequest, DocumentUploadResponse,
    HealthResponse, SearchRequest, SearchResponse, ErrorResponse, AgenticQueryResponse
)
from app.services.rag_service import get_rag_service
from app.services.vector_store import get_vector_store_service
from app.services.llama_client import get_llama_client
from app.services.embedding_service import get_embedding_service
from app.core.database import get_db

# Создаем роутер
router = APIRouter()

# Инициализируем сервисы
rag_service = get_rag_service()
vector_store = get_vector_store_service()
llama_client = get_llama_client()
embedding_service = get_embedding_service()


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, http_request: Request):
    """Обработка пользовательского запроса."""
    try:
        # Получаем информацию о клиенте
        user_ip = http_request.client.host
        user_agent = http_request.headers.get("user-agent")
        
        # Обрабатываем запрос
        response = await rag_service.process_query(
            request=request,
            user_ip=user_ip,
            user_agent=user_agent
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )


@router.post("/query/agentic", response_model=AgenticQueryResponse)
async def process_query_agentic(request: QueryRequest):
    """Агентный RAG на LangGraph: retrieve → grade → (transform↺) → generate → self-check.

    В отличие от линейного /query умеет отсеивать нерелевантные документы,
    переформулировать запрос при слабом поиске и перегенерировать ответ при галлюцинации.
    Возвращает трейс шагов графа.
    """
    from app.services.rag_graph import get_agentic_rag_graph

    start_time = time.time()
    try:
        graph = get_agentic_rag_graph()
        result = await graph.ainvoke(
            {"question": request.question, "query": request.question}
        )
        docs = result.get("documents", [])
        sources = [
            f"▲{i} " + " ".join(d.get("content", "").split()[:12])
            for i, d in enumerate(docs, 1)
        ]
        return AgenticQueryResponse(
            answer=result.get("generation", ""),
            sources=sources,
            documents_used=len(docs),
            transforms=result.get("transforms", 0),
            generations=result.get("generations", 0),
            grounded=bool(result.get("grounded", False)),
            answers_question=bool(result.get("answers_question", False)),
            trace=result.get("trace", []),
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка агентной обработки: {str(e)}")


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(request: DocumentUploadRequest):
    """Загрузка нового документа в векторное хранилище."""
    try:
        document_id, chunks_created = await vector_store.add_document(
            title=request.title,
            content=request.content,
            metadata=request.metadata
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            chunks_created=chunks_created,
            message=f"Документ успешно загружен. Создано чанков: {chunks_created}"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка загрузки документа: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Удаление документа из векторного хранилища."""
    try:
        success = vector_store.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Документ не найден"
            )
        
        return {"message": "Документ успешно удален"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка удаления документа: {str(e)}"
        )


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Получение документа по ID."""
    try:
        document = vector_store.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail="Документ не найден"
            )
        
        return {
            "id": str(document.id),
            "title": document.title,
            "content": document.content,
            "metadata": document.doc_metadata,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения документа: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Поиск документов по векторному сходству."""
    try:
        start_time = time.time()
        
        # Создаем эмбеддинг для запроса
        embedding_start = time.time()
        query_embedding = await embedding_service.get_embedding_async(request.query)
        query_embedding_time = time.time() - embedding_start
        
        # Выполняем поиск
        search_start = time.time()
        results = await vector_store.search_similar(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold
        )
        search_time = time.time() - search_start
        
        return SearchResponse(
            results=results,
            total_found=len(results),
            query_embedding_time=query_embedding_time,
            search_time=search_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка поиска: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния всех сервисов."""
    timestamp = datetime.utcnow().isoformat()
    services = {}
    
    try:
        # Проверяем LLaMA сервер
        llama_health = await llama_client.health_check()
        services["llama"] = llama_health["status"]
        
        # Проверяем векторное хранилище
        try:
            vector_stats = vector_store.get_stats()
            services["vector_store"] = "healthy"
        except Exception as e:
            services["vector_store"] = f"unhealthy: {str(e)}"
        
        # Проверяем сервис эмбеддингов
        try:
            embedding_info = embedding_service.get_model_info()
            if embedding_info.get("status") == "not_initialized":
                services["embedding"] = "unhealthy: not initialized"
            else:
                services["embedding"] = "healthy"
        except Exception as e:
            services["embedding"] = f"unhealthy: {str(e)}"
        
        # Определяем общий статус
        if all(status == "healthy" for status in services.values()):
            overall_status = "healthy"
        else:
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            services=services,
            timestamp=timestamp
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            services={"error": str(e)},
            timestamp=timestamp
        )


@router.get("/stats")
async def get_system_stats():
    """Получение статистики системы."""
    try:
        # Статистика векторного хранилища
        vector_stats = vector_store.get_stats()
        
        # Информация о модели эмбеддингов
        embedding_info = embedding_service.get_model_info()
        
        # Информация о LLaMA
        llama_info = await llama_client.get_model_info()
        
        return {
            "vector_store": vector_stats,
            "embedding_model": embedding_info,
            "llama_model": llama_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики: {str(e)}"
        )


@router.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "Mini RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    } 