"""Конфигурация приложения."""

import os

from pydantic import BaseModel
from pydantic_settings import BaseSettings


def _env(name: str, default):
    """Чтение переменной окружения с дефолтом (типобезопасно по дефолту)."""
    value = os.getenv(name)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    return value


class DatabaseSettings(BaseModel):
    """Настройки базы данных."""
    url: str = _env("DATABASE_URL", "postgresql://minirag:minirag123@localhost:5432/minirag")
    echo: bool = False
    pool_pre_ping: bool = True
    pool_recycle: int = 3600


class LlamaSettings(BaseModel):
    """Настройки LLaMA inference сервера (OpenAI-совместимый llama.cpp)."""
    server_url: str = _env("LLAMA_SERVER_URL", "http://localhost:8080")
    model_path: str = _env("MODEL_PATH", "./data/models/llama-3-8b-instruct.gguf")
    timeout: float = _env("LLAMA_TIMEOUT", 120.0)
    max_tokens: int = _env("LLAMA_MAX_TOKENS", 512)
    temperature: float = _env("LLAMA_TEMPERATURE", 0.2)


class VectorStoreSettings(BaseModel):
    """Настройки векторного хранилища."""
    type: str = _env("VECTOR_STORE_TYPE", "pgvector")
    embedding_model: str = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    dimension: int = _env("VECTOR_DIMENSION", 384)


class RedisSettings(BaseModel):
    """Настройки Redis-кэша эмбеддингов."""
    url: str = _env("REDIS_URL", "redis://localhost:6379")
    decode_responses: bool = True
    socket_timeout: float = _env("REDIS_SOCKET_TIMEOUT", 5.0)


class RAGSettings(BaseModel):
    """Настройки RAG-пайплайна."""
    max_context_chunks: int = _env("MAX_CONTEXT_CHUNKS", 5)
    chunk_size: int = _env("CHUNK_SIZE", 1000)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 200)
    similarity_threshold: float = _env("SIMILARITY_THRESHOLD", 0.7)
    max_search_results: int = _env("MAX_SEARCH_RESULTS", 10)
    # Лимиты агентного цикла (corrective RAG на LangGraph)
    max_query_transforms: int = _env("MAX_QUERY_TRANSFORMS", 2)
    max_generation_retries: int = _env("MAX_GENERATION_RETRIES", 2)


class Settings(BaseSettings):
    """Основные настройки приложения."""

    # API настройки
    api_host: str = _env("API_HOST", "0.0.0.0")
    api_port: int = _env("API_PORT", 8000)
    debug: bool = _env("DEBUG", True)

    # System Prompt
    system_prompt_file: str = _env("SYSTEM_PROMPT_FILE", "./prompts/mini-rag-system-prompt.txt")

    # Компоненты
    database: DatabaseSettings = DatabaseSettings()
    llama: LlamaSettings = LlamaSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    redis: RedisSettings = RedisSettings()
    rag: RAGSettings = RAGSettings()

    class Config:
        env_file = ".env"


# Глобальный объект настроек
settings = Settings()
