"""Унифицированный LLM-провайдер за одним async-интерфейсом.

Один и тот же агентный граф (LangGraph) и Ragas-оценка работают на любом бэкенде:
  - llama     — локальный llama.cpp (OpenAI-совместимый сервер), без внешних API;
  - anthropic — Claude (требует ANTHROPIC_API_KEY);
  - openai    — GPT (требует OPENAI_API_KEY).

Выбор через переменную окружения LLM_BACKEND (см. app/core/config.py:LLMSettings).
SDK импортируются лениво — пакет нужен только для реально выбранного бэкенда.

Сигнатура колбэка совпадает с rag_graph.LLMFn:  async (prompt, system_prompt) -> str.
"""

from __future__ import annotations

from typing import Optional

from app.core.config import settings


async def _llama_complete(prompt: str, system_prompt: Optional[str] = None) -> str:
    from app.services.llama_client import get_llama_client

    client = get_llama_client()
    result = await client.generate_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=settings.llama.max_tokens,
        temperature=settings.llama.temperature,
    )
    return result.get("content", "") if result.get("success") else ""


async def _anthropic_complete(prompt: str, system_prompt: Optional[str] = None) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.llm.anthropic_api_key or None)
    message = await client.messages.create(
        model=settings.llm.anthropic_model,
        max_tokens=settings.llama.max_tokens,
        temperature=settings.llama.temperature,
        system=system_prompt or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )


async def _openai_complete(prompt: str, system_prompt: Optional[str] = None) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.llm.openai_api_key or None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = await client.chat.completions.create(
        model=settings.llm.openai_model,
        max_tokens=settings.llama.max_tokens,
        temperature=settings.llama.temperature,
        messages=messages,
    )
    return response.choices[0].message.content or ""


_BACKENDS = {
    "llama": _llama_complete,
    "anthropic": _anthropic_complete,
    "openai": _openai_complete,
}


def get_llm_fn(backend: Optional[str] = None):
    """Возвращает async-колбэк (prompt, system_prompt) -> str для выбранного бэкенда."""
    name = (backend or settings.llm.backend or "llama").lower()
    fn = _BACKENDS.get(name)
    if fn is None:
        raise ValueError(
            f"Неизвестный LLM_BACKEND={name!r}. Доступно: {', '.join(_BACKENDS)}"
        )
    return fn
