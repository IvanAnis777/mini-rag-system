"""Сервис для создания и управления векторными представлениями."""

import asyncio
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
import json
import hashlib
from functools import lru_cache

from app.core.config import settings


class EmbeddingService:
    """Сервис для работы с векторными представлениями текста."""
    
    def __init__(self):
        self.model = None
        self.redis_client = None
        self._initialize()
    
    def _initialize(self):
        """Инициализация модели и Redis клиента."""
        try:
            # Загружаем модель для эмбеддингов
            self.model = SentenceTransformer(settings.vector_store.embedding_model)
            
            # Подключаемся к Redis для кэширования
            self.redis_client = redis.from_url(
                settings.redis.url,
                decode_responses=settings.redis.decode_responses,
                socket_timeout=settings.redis.socket_timeout
            )
            
            print(f"✅ Embedding service initialized with model: {settings.vector_store.embedding_model}")
            
        except Exception as e:
            print(f"❌ Failed to initialize embedding service: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Создание ключа для кэширования эмбеддинга."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"embedding:{settings.vector_store.embedding_model}:{text_hash}"
    
    async def get_embedding_async(self, text: str) -> List[float]:
        """Асинхронное получение векторного представления текста с кэшированием."""
        cache_key = self._get_cache_key(text)
        
        # Проверяем кэш
        try:
            cached_embedding = self.redis_client.get(cache_key)
            if cached_embedding:
                return json.loads(cached_embedding)
        except Exception as e:
            print(f"⚠️ Redis cache error: {e}")
        
        # Создаем эмбеддинг в отдельном потоке
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._create_embedding, text)
        
        # Сохраняем в кэш
        try:
            self.redis_client.setex(
                cache_key, 
                3600,  # TTL 1 час
                json.dumps(embedding)
            )
        except Exception as e:
            print(f"⚠️ Failed to cache embedding: {e}")
        
        return embedding
    
    def get_embedding(self, text: str) -> List[float]:
        """Синхронное получение векторного представления текста."""
        cache_key = self._get_cache_key(text)
        
        # Проверяем кэш
        try:
            cached_embedding = self.redis_client.get(cache_key)
            if cached_embedding:
                return json.loads(cached_embedding)
        except Exception as e:
            print(f"⚠️ Redis cache error: {e}")
        
        embedding = self._create_embedding(text)
        
        # Сохраняем в кэш
        try:
            self.redis_client.setex(
                cache_key, 
                3600,  # TTL 1 час
                json.dumps(embedding)
            )
        except Exception as e:
            print(f"⚠️ Failed to cache embedding: {e}")
        
        return embedding
    
    def _create_embedding(self, text: str) -> List[float]:
        """Создание векторного представления для текста."""
        if not self.model:
            raise RuntimeError("Embedding model is not initialized")
        
        # Очищаем текст
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Создаем эмбеддинг
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    async def get_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Асинхронное создание эмбеддингов для списка текстов."""
        if not texts:
            return []
        
        embeddings = []
        
        # Проверяем кэш для каждого текста
        cache_keys = [self._get_cache_key(text) for text in texts]
        cached_embeddings = {}
        
        try:
            cached_values = self.redis_client.mget(cache_keys)
            for i, cached_value in enumerate(cached_values):
                if cached_value:
                    cached_embeddings[i] = json.loads(cached_value)
        except Exception as e:
            print(f"⚠️ Redis batch cache error: {e}")
        
        # Создаем эмбеддинги для текстов, которых нет в кэше
        texts_to_process = []
        indices_to_process = []
        
        for i, text in enumerate(texts):
            if i in cached_embeddings:
                embeddings.append(cached_embeddings[i])
            else:
                embeddings.append(None)  # Заполним позже
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        if texts_to_process:
            # Создаем эмбеддинги в отдельном потоке
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None, self._create_embeddings_batch, texts_to_process
            )
            
            # Заполняем результаты и кэшируем
            cache_data = {}
            for i, embedding in zip(indices_to_process, batch_embeddings):
                embeddings[i] = embedding
                cache_key = cache_keys[i]
                cache_data[cache_key] = json.dumps(embedding)
            
            # Сохраняем в кэш
            try:
                if cache_data:
                    pipeline = self.redis_client.pipeline()
                    for key, value in cache_data.items():
                        pipeline.setex(key, 3600, value)
                    pipeline.execute()
            except Exception as e:
                print(f"⚠️ Failed to cache batch embeddings: {e}")
        
        return embeddings
    
    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Создание эмбеддингов для батча текстов."""
        if not self.model:
            raise RuntimeError("Embedding model is not initialized")
        
        # Очищаем тексты
        cleaned_texts = [text.strip() for text in texts]
        
        if not all(cleaned_texts):
            raise ValueError("All texts must be non-empty")
        
        # Создаем эмбеддинги
        embeddings = self.model.encode(cleaned_texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]
    
    def get_model_info(self) -> dict:
        """Получение информации о модели эмбеддингов."""
        if not self.model:
            return {"status": "not_initialized"}
        
        return {
            "model_name": settings.vector_store.embedding_model,
            "dimension": settings.vector_store.dimension,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "device": str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
        }


# Глобальный экземпляр сервиса
@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Получение глобального экземпляра сервиса эмбеддингов."""
    return EmbeddingService() 