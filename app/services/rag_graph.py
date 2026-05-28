"""Агентный RAG на LangGraph: corrective / self-RAG поверх существующих сервисов.

Линейный пайплайн `rag_service.process_query` (rewrite → search → generate) расширяется
графом с самокоррекцией:

    retrieve → grade_documents ─┬─(нет релевантных)→ transform_query ─┐
                                │                                     │
                                └─(есть)→ generate                    │
                                            │   ▲___________(retrieve)─┘
                                            ▼
                                    grade_generation ─┬─(галлюцинация)→ generate
                                                       ├─(не отвечает)→ transform_query
                                                       └─(ок)→ END

Граф не тянет тяжёлые зависимости на импорте: retriever/LLM передаются как async-колбэки,
поэтому логику маршрутизации можно гонять на стабах в юнит-тестах. Прод-обвязка
(`get_agentic_rag_graph`) лениво подключает vector_store + llama_client.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

# Тип ретривера: (query, limit) -> список документов вида {"content": str, "score": float, ...}
RetrieverFn = Callable[[str, int], Awaitable[List[dict]]]
# Тип LLM: (prompt, system_prompt) -> текст ответа
LLMFn = Callable[[str, Optional[str]], Awaitable[str]]


class AgentState(TypedDict, total=False):
    """Состояние, протекающее по графу."""
    question: str            # исходный вопрос пользователя
    query: str               # текущий (возможно переформулированный) поисковый запрос
    documents: List[dict]    # отфильтрованные релевантные документы
    generation: str          # текущий сгенерированный ответ
    transforms: int          # сколько раз переформулировали запрос
    generations: int         # сколько раз генерировали ответ
    grounded: bool           # ответ обоснован контекстом (не галлюцинация)
    answers_question: bool    # ответ действительно отвечает на вопрос
    trace: List[str]         # человекочитаемый лог шагов (для отладки/демо)


def _yes(text: str) -> bool:
    """Парсинг бинарного вердикта грейдера. Пусто/непонятно трактуем как 'no'."""
    if not text:
        return False
    head = text.strip().lower()
    return head.startswith("yes") or head.startswith("да") or "\"yes\"" in head or "'yes'" in head


class CorrectiveRagGraph:
    """Строит и компилирует corrective-RAG граф поверх инъектируемых retriever/LLM."""

    def __init__(
        self,
        retriever: RetrieverFn,
        llm: LLMFn,
        *,
        max_chunks: int = 5,
        max_transforms: int = 2,
        max_generations: int = 2,
        system_prompt: Optional[str] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_chunks = max_chunks
        self.max_transforms = max_transforms
        self.max_generations = max_generations
        self.system_prompt = system_prompt

    # ---------- узлы ----------

    async def retrieve(self, state: AgentState) -> AgentState:
        query = state.get("query") or state["question"]
        docs = await self.retriever(query, self.max_chunks)
        trace = state.get("trace", []) + [f"retrieve(query={query!r}) → {len(docs)} док."]
        return {"documents": docs, "query": query, "trace": trace}

    async def grade_documents(self, state: AgentState) -> AgentState:
        """LLM-грейдинг релевантности каждого документа вопросу. Отсев нерелевантных."""
        question = state["question"]
        kept: List[dict] = []
        for doc in state.get("documents", []):
            verdict = await self.llm(
                _GRADE_DOC_PROMPT.format(question=question, document=doc.get("content", "")),
                _GRADE_DOC_SYSTEM,
            )
            if _yes(verdict):
                kept.append(doc)
        trace = state.get("trace", []) + [
            f"grade_documents → релевантных {len(kept)}/{len(state.get('documents', []))}"
        ]
        return {"documents": kept, "trace": trace}

    async def transform_query(self, state: AgentState) -> AgentState:
        """Переформулирование запроса для повторного поиска."""
        question = state["question"]
        current = state.get("query") or question
        better = await self.llm(
            _TRANSFORM_PROMPT.format(question=question, current=current),
            _TRANSFORM_SYSTEM,
        )
        better = (better or "").strip() or current
        transforms = state.get("transforms", 0) + 1
        trace = state.get("trace", []) + [f"transform_query #{transforms} → {better!r}"]
        return {"query": better, "transforms": transforms, "trace": trace}

    async def generate(self, state: AgentState) -> AgentState:
        question = state["question"]
        context = _format_context(state.get("documents", []))
        answer = await self.llm(
            _GENERATE_PROMPT.format(question=question, context=context),
            self.system_prompt or _GENERATE_SYSTEM,
        )
        generations = state.get("generations", 0) + 1
        trace = state.get("trace", []) + [f"generate #{generations} ({len(answer or '')} симв.)"]
        return {"generation": (answer or "").strip(), "generations": generations, "trace": trace}

    async def grade_generation(self, state: AgentState) -> AgentState:
        """Двойная проверка ответа: обоснован ли контекстом и отвечает ли на вопрос."""
        context = _format_context(state.get("documents", []))
        generation = state.get("generation", "")
        grounded_raw = await self.llm(
            _HALLUCINATION_PROMPT.format(context=context, generation=generation),
            _GRADE_DOC_SYSTEM,
        )
        answers_raw = await self.llm(
            _ANSWERS_PROMPT.format(question=state["question"], generation=generation),
            _GRADE_DOC_SYSTEM,
        )
        grounded, answers = _yes(grounded_raw), _yes(answers_raw)
        trace = state.get("trace", []) + [
            f"grade_generation → grounded={grounded}, answers={answers}"
        ]
        return {"grounded": grounded, "answers_question": answers, "trace": trace}

    # ---------- рёбра (маршрутизация) ----------

    def _decide_after_grade(self, state: AgentState) -> str:
        if state.get("documents"):
            return "generate"
        # нет релевантных документов — пробуем переформулировать, пока есть бюджет
        if state.get("transforms", 0) < self.max_transforms:
            return "transform_query"
        return "generate"  # бюджет исчерпан — честно генерируем "не нашёл" по пустому контексту

    def _decide_after_generation(self, state: AgentState) -> str:
        if not state.get("grounded"):
            if state.get("generations", 0) < self.max_generations:
                return "generate"  # галлюцинация — перегенерировать
            return END
        if not state.get("answers_question"):
            if state.get("transforms", 0) < self.max_transforms:
                return "transform_query"  # обоснован, но мимо — расширить поиск
            return END
        return END  # обоснован и отвечает

    # ---------- сборка ----------

    def build(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self.retrieve)
        g.add_node("grade_documents", self.grade_documents)
        g.add_node("transform_query", self.transform_query)
        g.add_node("generate", self.generate)
        g.add_node("grade_generation", self.grade_generation)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_documents")
        g.add_conditional_edges(
            "grade_documents",
            self._decide_after_grade,
            {"generate": "generate", "transform_query": "transform_query"},
        )
        g.add_edge("transform_query", "retrieve")
        g.add_edge("generate", "grade_generation")
        g.add_conditional_edges(
            "grade_generation",
            self._decide_after_generation,
            {"generate": "generate", "transform_query": "transform_query", END: END},
        )
        return g.compile()


def _format_context(documents: List[dict]) -> str:
    if not documents:
        return "(контекст не найден)"
    return "\n\n".join(f"[{i}] {d.get('content', '')}" for i, d in enumerate(documents, 1))


# ---------- промпты грейдеров и генерации ----------

_GRADE_DOC_SYSTEM = "Ты строгий ассистент-оценщик. Отвечай РОВНО одним словом: yes или no."

_GRADE_DOC_PROMPT = (
    "Документ:\n{document}\n\n"
    "Вопрос пользователя:\n{question}\n\n"
    "Релевантен ли документ вопросу (содержит ключевые слова или смысл, помогающий ответить)? "
    "Ответь одним словом: yes или no."
)

_TRANSFORM_SYSTEM = "Ты улучшаешь поисковые запросы для семантического поиска. Верни только новый запрос."

_TRANSFORM_PROMPT = (
    "Исходный вопрос: {question}\n"
    "Текущий запрос, давший слабые результаты: {current}\n\n"
    "Переформулируй запрос так, чтобы он лучше попадал в семантический поиск: "
    "выдели ключевые сущности и термины, убери лишние слова. Верни только новый запрос."
)

_GENERATE_SYSTEM = (
    "Ты ассистент, отвечающий строго по предоставленному контексту. "
    "Если в контексте нет ответа — честно скажи, что информации недостаточно. Не выдумывай."
)

_GENERATE_PROMPT = (
    "Контекст:\n{context}\n\n"
    "Вопрос: {question}\n\n"
    "Дай точный ответ, опираясь ТОЛЬКО на контекст. Если ответа в контексте нет — так и скажи."
)

_HALLUCINATION_PROMPT = (
    "Контекст (факты):\n{context}\n\n"
    "Ответ модели:\n{generation}\n\n"
    "Полностью ли ответ обоснован контекстом, без выдуманных фактов? Ответь одним словом: yes или no."
)

_ANSWERS_PROMPT = (
    "Вопрос: {question}\n\n"
    "Ответ модели:\n{generation}\n\n"
    "Отвечает ли ответ по существу на вопрос? Ответь одним словом: yes или no."
)


# ---------- прод-обвязка ----------

def get_agentic_rag_graph():
    """Собирает граф на реальных сервисах (vector_store + llama_client).

    Импорты ленивые: модуль остаётся импортируемым без sentence-transformers/БД,
    чтобы граф можно было тестировать на стабах.
    """
    from app.core.config import settings
    from app.services.vector_store import get_vector_store_service
    from app.services.llama_client import get_llama_client

    vector_store = get_vector_store_service()
    llama_client = get_llama_client()

    async def retriever(query: str, limit: int) -> List[dict]:
        chunks = await vector_store.search_similar(
            query, limit=limit, threshold=settings.rag.similarity_threshold
        )
        return [
            {
                "content": c.content,
                "score": c.similarity_score,
                "document_id": c.document_id,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

    async def llm(prompt: str, system_prompt: Optional[str] = None) -> str:
        result = await llama_client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=settings.llama.max_tokens,
            temperature=settings.llama.temperature,
        )
        return result.get("content", "") if result.get("success") else ""

    return CorrectiveRagGraph(
        retriever=retriever,
        llm=llm,
        max_chunks=settings.rag.max_context_chunks,
        max_transforms=settings.rag.max_query_transforms,
        max_generations=settings.rag.max_generation_retries,
    ).build()
