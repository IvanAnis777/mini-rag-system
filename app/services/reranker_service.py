"""Cross-encoder reranker: точное переранжирование кандидатов после retrieval.

Bi-encoder (эмбеддинги) ищет быстро, но грубо. Cross-encoder читает пару
(вопрос, чанк) целиком и даёт точную оценку релевантности — им пересортировываем
top-N кандидатов перед подачей в LLM.

Модель мультиязычная (включая русский). Загрузка ленивая. Отключается
RERANKER_ENABLED=false (тогда rerank() возвращает вход как есть).
"""

import os
from functools import lru_cache
from typing import List


def _enabled() -> bool:
    return os.getenv("RERANKER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


class RerankerService:
    """Обёртка над sentence-transformers CrossEncoder."""

    def __init__(self):
        self.enabled = _enabled()
        self.model_name = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self._model = None  # ленивая загрузка при первом rerank

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            print(f"✅ Reranker initialized: {self.model_name}")
        return self._model

    def rerank(self, query: str, items: List[dict], top_n: int) -> List[dict]:
        """Пересортировать items по релевантности вопросу и вернуть top_n.

        items — список dict с ключом 'content'. К каждому добавляется 'rerank_score'.
        Если reranker выключен/нет кандидатов — возвращаем первые top_n как есть.
        """
        if not self.enabled or not items:
            return items[:top_n]
        model = self._get_model()
        pairs = [(query, it.get("content", "")) for it in items]
        scores = model.predict(pairs)
        for it, sc in zip(items, scores):
            it["rerank_score"] = float(sc)
        ranked = sorted(items, key=lambda it: it["rerank_score"], reverse=True)
        return ranked[:top_n]


@lru_cache()
def get_reranker_service() -> RerankerService:
    return RerankerService()
