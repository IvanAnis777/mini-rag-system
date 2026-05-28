"""Сервис для работы с векторным хранилищем на основе PostgreSQL + pgvector."""

import time
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from functools import lru_cache

from app.core.database import get_db_session
from app.models.database import Document, DocumentChunk
from app.models.schemas import DocumentChunk as DocumentChunkSchema
from app.services.embedding_service import get_embedding_service
from app.core.config import settings


class VectorStoreService:
    """Сервис для работы с векторным хранилищем."""
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.similarity_threshold = settings.rag.similarity_threshold
        self.max_results = settings.rag.max_search_results
    
    async def add_document(
        self,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, int]:
        """Добавление документа в векторное хранилище."""
        
        if not content.strip():
            raise ValueError("Document content cannot be empty")
        
        # Разбиваем документ на чанки
        chunks = self._split_text(content)
        
        if not chunks:
            raise ValueError("Failed to create chunks from document")
        
        # Создаем эмбеддинги для всех чунков
        embeddings = await self.embedding_service.get_embeddings_batch_async(chunks)
        
        with get_db_session() as session:
            # Создаем документ
            document = Document(
                title=title,
                content=content,
                metadata=metadata or {}
            )
            session.add(document)
            session.flush()  # Получаем ID документа
            
            # Создаем чанки с эмбеддингами
            document_chunks = []
            for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=i,
                    embedding=embedding,
                    metadata={
                        "title": title,
                        "chunk_size": len(chunk_content),
                        **metadata or {}
                    }
                )
                document_chunks.append(chunk)
            
            session.add_all(document_chunks)
            session.commit()
            
            return str(document.id), len(document_chunks)
    
    async def search_similar(
        self,
        query: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[DocumentChunkSchema]:
        """Поиск похожих чунков по запросу."""
        
        limit = limit or self.max_results
        threshold = threshold or self.similarity_threshold
        
        start_time = time.time()
        
        # Создаем эмбеддинг для запроса
        query_embedding = await self.embedding_service.get_embedding_async(query)
        embedding_time = time.time() - start_time
        
        search_start = time.time()
        
        with get_db_session() as session:
            # Выполняем векторный поиск с использованием косинусного расстояния
            query_sql = text("""
                SELECT 
                    id,
                    document_id,
                    content,
                    chunk_index,
                    metadata,
                    created_at,
                    (1 - (embedding <=> :query_embedding)) as similarity_score
                FROM document_chunks
                WHERE (1 - (embedding <=> :query_embedding)) >= :threshold
                ORDER BY embedding <=> :query_embedding
                LIMIT :limit
            """)
            
            results = session.execute(
                query_sql,
                {
                    "query_embedding": str(query_embedding),
                    "threshold": threshold,
                    "limit": limit
                }
            ).fetchall()
            
            search_time = time.time() - search_start
            
            # Преобразуем результаты в схемы
            chunks = []
            for row in results:
                chunk = DocumentChunkSchema(
                    content=row.content,
                    similarity_score=float(row.similarity_score),
                    document_id=str(row.document_id),
                    metadata={
                        **row.metadata,
                        "chunk_index": row.chunk_index,
                        "created_at": row.created_at.isoformat(),
                        "query_embedding_time": embedding_time,
                        "search_time": search_time
                    }
                )
                chunks.append(chunk)
            
            return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Разбиение текста на чанки."""
        
        chunk_size = settings.rag.chunk_size
        chunk_overlap = settings.rag.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                # Последний чунк
                chunks.append(text[start:])
                break
            
            # Попробуем найти удобное место для разрыва (по предложениям или абзацам)
            chunk_text = text[start:end]
            
            # Ищем последнее предложение в чунке
            for delimiter in ['\n\n', '. ', '! ', '? ', '\n']:
                last_delimiter = chunk_text.rfind(delimiter)
                if last_delimiter > chunk_size // 2:  # Не слишком короткий чунк
                    end = start + last_delimiter + len(delimiter)
                    chunk_text = text[start:end]
                    break
            
            chunks.append(chunk_text.strip())
            
            # Следующий чунк начинается с перекрытием
            start = end - chunk_overlap
            
            # Убеждаемся, что не зацикливаемся
            if start <= 0:
                start = end
        
        # Убираем пустые чанки
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Получение документа по ID."""
        with get_db_session() as session:
            return session.query(Document).filter(Document.id == document_id).first()
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Получение всех чунков документа."""
        with get_db_session() as session:
            return session.query(DocumentChunk)\
                .filter(DocumentChunk.document_id == document_id)\
                .order_by(DocumentChunk.chunk_index)\
                .all()
    
    def delete_document(self, document_id: str) -> bool:
        """Удаление документа и всех его чунков."""
        with get_db_session() as session:
            # Удаляем чанки
            deleted_chunks = session.query(DocumentChunk)\
                .filter(DocumentChunk.document_id == document_id)\
                .delete()
            
            # Удаляем документ
            deleted_docs = session.query(Document)\
                .filter(Document.id == document_id)\
                .delete()
            
            session.commit()
            
            return deleted_docs > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики векторного хранилища."""
        with get_db_session() as session:
            documents_count = session.query(func.count(Document.id)).scalar()
            chunks_count = session.query(func.count(DocumentChunk.id)).scalar()
            
            # Получаем размер индекса
            index_size_query = text("""
                SELECT pg_size_pretty(pg_total_relation_size('document_chunks')) as table_size,
                       pg_size_pretty(pg_indexes_size('document_chunks')) as index_size
            """)
            
            try:
                size_result = session.execute(index_size_query).fetchone()
                table_size = size_result.table_size if size_result else "unknown"
                index_size = size_result.index_size if size_result else "unknown"
            except:
                table_size = "unknown"
                index_size = "unknown"
            
            return {
                "documents_count": documents_count,
                "chunks_count": chunks_count,
                "table_size": table_size,
                "index_size": index_size,
                "embedding_dimension": settings.vector_store.dimension,
                "similarity_threshold": self.similarity_threshold,
                "max_results": self.max_results
            }


# Глобальный экземпляр сервиса
@lru_cache()
def get_vector_store_service() -> VectorStoreService:
    """Получение глобального экземпляра сервиса векторного хранилища."""
    return VectorStoreService() 