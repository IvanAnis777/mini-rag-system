"""Основной RAG сервис, объединяющий все компоненты системы."""

import time
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.services.vector_store import get_vector_store_service
from app.services.llama_client import get_llama_client
from app.models.schemas import QueryRequest, QueryResponse, DocumentChunk
from app.core.config import settings
from app.core.database import get_db_session
from app.models.database import QueryLog


class RAGService:
    """Основной сервис RAG системы."""
    
    def __init__(self):
        self.vector_store = get_vector_store_service()
        self.llama_client = get_llama_client()
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Загрузка system prompt из файла."""
        try:
            prompt_path = Path(settings.system_prompt_file)
            if prompt_path.exists():
                return prompt_path.read_text(encoding='utf-8')
            else:
                print(f"⚠️ System prompt file not found: {settings.system_prompt_file}")
                return self._default_system_prompt()
        except Exception as e:
            print(f"⚠️ Failed to load system prompt: {e}")
            return self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Дефолтный system prompt на случай, если файл не найден."""
        return """You are MiniRAGBot, a helpful assistant that answers questions using retrieved context and your knowledge.

When answering:
1. Use both the provided context and your knowledge
2. Cite sources using ▲¹, ▲² format
3. Be concise and accurate
4. Answer in the same language as the question (default: Russian)

Format your response as:
**Ответ**
[Your answer here]

**Источник(и)**
▲1 [First source snippet]
▲2 [Second source snippet]
"""
    
    async def process_query(self, request: QueryRequest, user_ip: str = None, user_agent: str = None) -> QueryResponse:
        """Обработка пользовательского запроса."""
        start_time = time.time()
        
        try:
            # 1. Переформулируем запрос
            query_rewritten = await self._rewrite_query(request.question)
            
            # 2. Поиск релевантного контекста
            context_chunks = await self.vector_store.search_similar(
                query_rewritten,
                limit=request.max_chunks or settings.rag.max_context_chunks,
                threshold=settings.rag.similarity_threshold
            )
            
            # 3. Генерируем ответ
            answer, confidence = await self._generate_answer(
                request.question,
                query_rewritten,
                context_chunks,
                request.language
            )
            
            # 4. Форматируем источники
            sources = self._format_sources(context_chunks)
            
            processing_time = time.time() - start_time
            
            # 5. Создаем ответ
            response = QueryResponse(
                answer=answer,
                sources=sources,
                query_rewritten=query_rewritten,
                context_chunks=context_chunks,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # 6. Логируем запрос
            await self._log_query(
                original_query=request.question,
                rewritten_query=query_rewritten,
                response=answer,
                confidence=confidence,
                processing_time=processing_time,
                chunks_used=len(context_chunks),
                language=request.language,
                user_ip=user_ip,
                user_agent=user_agent
            )
            
            return response
            
        except Exception as e:
            # В случае ошибки возвращаем стандартный ответ
            processing_time = time.time() - start_time
            
            return QueryResponse(
                answer="Извините, не могу помочь с этим запросом из-за технической ошибки.",
                sources=[],
                query_rewritten=request.question,
                context_chunks=[],
                confidence=0.0,
                processing_time=processing_time
            )
    
    async def _rewrite_query(self, original_query: str) -> str:
        """Переформулирование запроса для улучшения поиска."""
        
        # Простая эвристика для переформулирования
        query = original_query.strip()
        
        # Убираем вопросительные слова в начале
        query = re.sub(r'^(что|как|где|когда|почему|зачем|какой|какая|какое|кто)\s+', '', query, flags=re.IGNORECASE)
        
        # Убираем пунктуацию в конце
        query = re.sub(r'[?!.]+$', '', query)
        
        # Если запрос слишком короткий, оставляем как есть
        if len(query.split()) < 2:
            return original_query
        
        return query
    
    async def _generate_answer(
        self,
        original_query: str,
        rewritten_query: str,
        context_chunks: List[DocumentChunk],
        language: str
    ) -> tuple[str, float]:
        """Генерация ответа с использованием LLaMA."""
        
        # Подготавливаем контекст
        context_text = self._prepare_context(context_chunks)
        
        # Формируем промпт
        user_prompt = self._build_user_prompt(
            original_query,
            rewritten_query,
            context_text,
            language
        )
        
        # Генерируем ответ через chat-эндпоинт: llama.cpp сам применяет родной
        # шаблон модели (ChatML у Qwen) и корректные стоп-токены. Raw /completions
        # с самодельными стопами оставлял хвостовой мусор вроде <|im_end|>/<|assistant|>.
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        result = await self.llama_client.generate_chat_completion(
            messages=messages,
            max_tokens=settings.llama.max_tokens,
            temperature=settings.llama.temperature
        )
        
        if not result["success"]:
            print(f"⚠️ LLaMA generation failed: {result.get('error', 'Unknown error')}")
            return self._fallback_answer(original_query, context_chunks, language), 0.1
        
        answer = result["content"]
        confidence = self._calculate_confidence(answer, context_chunks)
        
        # Проверяем качество ответа
        if confidence < 0.25:
            return "Извините, не могу помочь с этим.", confidence
        
        return answer, confidence
    
    def _prepare_context(self, context_chunks: List[DocumentChunk]) -> str:
        """Подготовка контекста из найденных чунков."""
        if not context_chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            # Берем первые 200 символов для контекста
            content = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            context_parts.append(f"({i}) {content}")
        
        return "\n\n".join(context_parts)
    
    def _build_user_prompt(
        self,
        original_query: str,
        rewritten_query: str,
        context: str,
        language: str
    ) -> str:
        """Создание промпта для пользователя."""
        
        context_section = f"\n\nКонтекст:\n{context}" if context else "\n\nКонтекст отсутствует."
        
        prompt = f"""Вопрос пользователя: {original_query}
Переформулированный запрос: {rewritten_query}
Язык ответа: {language}{context_section}

Пожалуйста, ответь на вопрос пользователя, используя предоставленный контекст и свои знания."""
        
        return prompt
    
    def _calculate_confidence(self, answer: str, context_chunks: List[DocumentChunk]) -> float:
        """Расчет уверенности в ответе."""
        confidence = 0.5  # Базовая уверенность
        
        # Увеличиваем уверенность, если есть контекст
        if context_chunks:
            confidence += 0.2 * min(len(context_chunks), 3) / 3
        
        # Увеличиваем уверенность для длинных ответов
        if len(answer) > 100:
            confidence += 0.1
        
        # Уменьшаем уверенность для коротких ответов или отказов
        if len(answer) < 50 or "не могу" in answer.lower() or "извините" in answer.lower():
            confidence -= 0.3
        
        # Увеличиваем уверенность, если есть цитирование
        citation_count = len(re.findall(r'▲\d+', answer))
        if citation_count > 0:
            confidence += 0.1 * min(citation_count, 3) / 3
        
        return max(0.0, min(1.0, confidence))
    
    def _format_sources(self, context_chunks: List[DocumentChunk]) -> List[str]:
        """Форматирование источников для ответа."""
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            # Берем первые 12 слов из чунка
            words = chunk.content.split()[:12]
            source_text = " ".join(words)
            if len(chunk.content.split()) > 12:
                source_text += "..."
            
            sources.append(f"▲{i} {source_text}")
        
        return sources
    
    def _fallback_answer(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
        language: str
    ) -> str:
        """Резервный ответ, если LLaMA недоступна."""
        
        if not context_chunks:
            return "Извините, не могу найти информацию по вашему запросу."
        
        # Формируем простой ответ на основе контекста
        context_summary = context_chunks[0].content[:200]
        if len(context_chunks[0].content) > 200:
            context_summary += "..."
        
        return f"""**Ответ**

На основе найденной информации: {context_summary}

**Источник(и)**
▲1 {" ".join(context_chunks[0].content.split()[:12])}..."""
    
    async def _log_query(
        self,
        original_query: str,
        rewritten_query: str,
        response: str,
        confidence: float,
        processing_time: float,
        chunks_used: int,
        language: str,
        user_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Логирование запроса."""
        try:
            with get_db_session() as session:
                log_entry = QueryLog(
                    original_query=original_query,
                    rewritten_query=rewritten_query,
                    response=response,
                    confidence=confidence,
                    processing_time=processing_time,
                    chunks_used=chunks_used,
                    language=language,
                    user_ip=user_ip,
                    user_agent=user_agent
                )
                session.add(log_entry)
                session.commit()
        except Exception as e:
            print(f"⚠️ Failed to log query: {e}")


# Глобальный экземпляр сервиса
def get_rag_service() -> RAGService:
    """Получение глобального экземпляра RAG сервиса."""
    return RAGService() 