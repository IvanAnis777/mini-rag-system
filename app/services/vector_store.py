"""Сервис для работы с векторным хранилищем на основе PostgreSQL + pgvector."""

import math
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
        self._fts_index_ready = False  # GIN-индекс полнотекста создаём лениво один раз
    
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
        
        # Создаем эмбеддинги для всех чанков
        embeddings = await self.embedding_service.get_embeddings_batch_async(chunks)
        
        with get_db_session() as session:
            # Создаем документ
            document = Document(
                title=title,
                content=content,
                doc_metadata=metadata or {}
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
                    doc_metadata={
                        "title": title,
                        "chunk_size": len(chunk_content),
                        **(metadata or {})
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
        """Гибридный поиск + reranking.

        1) векторный (pgvector, по смыслу) + ключевой (BM25/полнотекст Postgres,
           по точным словам) — каждый отдаёт пул кандидатов;
        2) слияние через RRF (Reciprocal Rank Fusion);
        3) cross-encoder reranking top-кандидатов;
        4) возвращаем top-`limit`.

        `threshold` оставлен для совместимости сигнатуры (в гибридном режиме
        ранжирование делают RRF/reranker, а не косинусный порог).
        """
        limit = limit or self.max_results

        start_time = time.time()
        query_embedding = await self.embedding_service.get_embedding_async(query)
        embedding_time = time.time() - start_time

        cand_k = max(limit * 4, 20)  # пул кандидатов с запасом для слияния/реранка
        search_start = time.time()
        with get_db_session() as session:
            self._ensure_fts_index(session)
            vector_rows = self._vector_search(session, query_embedding, cand_k)
            keyword_rows = self._keyword_search(session, query, cand_k)
        search_time = time.time() - search_start

        fused = self._rrf_merge(vector_rows, keyword_rows)

        # cross-encoder reranking (если включён)
        from app.services.reranker_service import get_reranker_service
        top = get_reranker_service().rerank(query, fused, top_n=limit)

        chunks = []
        for item in top:
            # similarity_score схемы ограничен [0,1]. Логит cross-encoder приводим
            # сигмоидой; без реранка берём косинусную близость (с клампом).
            if item.get("rerank_score") is not None:
                display_score = 1.0 / (1.0 + math.exp(-item["rerank_score"]))
            else:
                display_score = max(0.0, min(1.0, item.get("vector_score") or item.get("rrf_score") or 0.0))
            chunks.append(DocumentChunkSchema(
                content=item["content"],
                similarity_score=display_score,
                document_id=str(item["document_id"]),
                metadata={
                    **(item.get("metadata") or {}),
                    "chunk_index": item.get("chunk_index"),
                    "vector_score": item.get("vector_score"),
                    "keyword_rank": item.get("keyword_rank"),
                    "rrf_score": item.get("rrf_score"),
                    "rerank_score": item.get("rerank_score"),
                    "query_embedding_time": embedding_time,
                    "search_time": search_time,
                },
            ))
        return chunks

    def _ensure_fts_index(self, session):
        """Один раз создаём GIN-индекс полнотекста (idempotent)."""
        if self._fts_index_ready:
            return
        try:
            session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_fts "
                "ON document_chunks USING GIN (to_tsvector('russian', content))"
            ))
            self._fts_index_ready = True
        except Exception as e:
            print(f"⚠️ FTS index ensure skipped: {e}")

    def _vector_search(self, session, query_embedding, k):
        """Векторный поиск (косинусная близость pgvector). Топ-k кандидатов."""
        sql = text("""
            SELECT id, document_id, content, chunk_index, metadata,
                   (1 - (embedding <=> CAST(:qe AS vector))) AS score
            FROM document_chunks
            ORDER BY embedding <=> CAST(:qe AS vector)
            LIMIT :k
        """)
        rows = session.execute(sql, {"qe": str(query_embedding), "k": k}).fetchall()
        return [{
            "id": str(r.id), "document_id": r.document_id, "content": r.content,
            "chunk_index": r.chunk_index, "metadata": r.metadata,
            "vector_score": float(r.score),
        } for r in rows]

    def _keyword_search(self, session, query, k):
        """Ключевой поиск (BM25-подобный ts_rank по полнотексту Postgres, рус. словарь)."""
        sql = text("""
            SELECT id, document_id, content, chunk_index, metadata,
                   ts_rank(to_tsvector('russian', content),
                           plainto_tsquery('russian', :q)) AS rank
            FROM document_chunks
            WHERE to_tsvector('russian', content) @@ plainto_tsquery('russian', :q)
            ORDER BY rank DESC
            LIMIT :k
        """)
        rows = session.execute(sql, {"q": query, "k": k}).fetchall()
        return [{
            "id": str(r.id), "document_id": r.document_id, "content": r.content,
            "chunk_index": r.chunk_index, "metadata": r.metadata,
            "keyword_rank": float(r.rank),
        } for r in rows]

    @staticmethod
    def _rrf_merge(vector_rows, keyword_rows, k0=60):
        """Reciprocal Rank Fusion: score = Σ 1/(k0 + позиция_в_списке).

        Сливает два ранжированных списка по id чанка, не требуя калибровки шкал
        (вектор и ts_rank несравнимы напрямую) — учитывает только позиции.
        """
        fused = {}
        for rows in (vector_rows, keyword_rows):
            for pos, row in enumerate(rows):
                cid = row["id"]
                if cid not in fused:
                    fused[cid] = {**row, "rrf_score": 0.0}
                else:
                    for key in ("vector_score", "keyword_rank"):
                        if key in row:
                            fused[cid][key] = row[key]
                fused[cid]["rrf_score"] += 1.0 / (k0 + pos + 1)
        return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    
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
                # Последний чанк
                chunks.append(text[start:])
                break
            
            # Попробуем найти удобное место для разрыва (по предложениям или абзацам)
            chunk_text = text[start:end]
            
            # Ищем последнее предложение в чанке
            for delimiter in ['\n\n', '. ', '! ', '? ', '\n']:
                last_delimiter = chunk_text.rfind(delimiter)
                if last_delimiter > chunk_size // 2:  # Не слишком короткий чанк
                    end = start + last_delimiter + len(delimiter)
                    chunk_text = text[start:end]
                    break
            
            chunks.append(chunk_text.strip())
            
            # Следующий чанк начинается с перекрытием
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
        """Получение всех чанков документа."""
        with get_db_session() as session:
            return session.query(DocumentChunk)\
                .filter(DocumentChunk.document_id == document_id)\
                .order_by(DocumentChunk.chunk_index)\
                .all()
    
    def delete_document(self, document_id: str) -> bool:
        """Удаление документа и всех его чанков."""
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