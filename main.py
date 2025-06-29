"""Основной файл FastAPI приложения Mini RAG System."""

import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.api.routes import router
from app.core.config import settings
from app.core.database import create_tables


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    
    # Startup
    logger.info("Starting Mini RAG System...")
    
    try:
        # Создаем таблицы в базе данных
        logger.info("Creating database tables...")
        create_tables()
        logger.info("Database tables created successfully")
        
        # Здесь можно добавить другие инициализации
        logger.info("Initializing services...")
        
        # Проверяем доступность сервисов
        from app.services.embedding_service import get_embedding_service
        from app.services.llama_client import get_llama_client
        
        # Инициализируем сервис эмбеддингов
        try:
            embedding_service = get_embedding_service()
            logger.info("Embedding service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
        
        # Проверяем LLaMA сервер
        try:
            llama_client = get_llama_client()
            health = await llama_client.health_check()
            if health["status"] == "healthy":
                logger.info("LLaMA server is healthy")
            else:
                logger.warning(f"LLaMA server health check failed: {health.get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to check LLaMA server: {e}")
        
        logger.info("Mini RAG System started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Mini RAG System...")
    
    # Закрываем соединения
    try:
        llama_client = get_llama_client()
        await llama_client.client.aclose()
        logger.info("HTTP connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Mini RAG System shutdown complete")


# Создаем приложение FastAPI
app = FastAPI(
    title="Mini RAG System",
    description="Система вопросов и ответов с использованием RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настраиваем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования HTTP запросов."""
    start_time = time.time()
    
    # Логируем входящий запрос
    logger.info(f"[IN] {request.method} {request.url.path} - {request.client.host}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Логируем ответ
        logger.info(
            f"[OUT] {request.method} {request.url.path} - "
            f"{response.status_code} - {process_time:.3f}s"
        )
        
        # Добавляем заголовок с временем обработки
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"[ERROR] {request.method} {request.url.path} - "
            f"Error: {str(e)} - {process_time:.3f}s"
        )
        raise


# Обработчик исключений
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик HTTP исключений."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработчик общих исключений."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Произошла внутренняя ошибка сервера",
            "path": str(request.url.path)
        }
    )


# Подключаем роуты
app.include_router(router, prefix="/api/v1")


# Дополнительные роуты
@app.get("/")
async def root():
    """Корневая страница."""
    return {
        "message": "Mini RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api": "/api/v1"
    }


@app.get("/health")
async def health():
    """Проверка состояния."""
    return {"status": "healthy"}


@app.get("/ping")
async def ping():
    """Простая проверка доступности."""
    return {"status": "pong", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    # Запускаем сервер
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    ) 