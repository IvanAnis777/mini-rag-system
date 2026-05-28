"""Юнит-тесты corrective-RAG графа на стабах (без БД и LLaMA-сервера).

Проверяем именно маршрутизацию графа:
 - happy path: релевантные документы → генерация без переформулировок;
 - corrective path: первый поиск пустой → transform_query → повторный retrieve → генерация;
 - hallucination path: первый ответ не обоснован → повторная генерация;
 - бюджет переформулировок не превышается.
"""

import pytest

from app.services.rag_graph import CorrectiveRagGraph


def make_llm(script):
    """LLM-стаб: отдаёт ответы по правилам в зависимости от содержимого промпта.

    `script` — словарь маркеров → функция(prompt) -> str. Первый сработавший маркер
    определяет ответ; иначе 'no'.
    """
    calls = []

    async def llm(prompt: str, system_prompt=None) -> str:
        calls.append(prompt)
        for marker, fn in script.items():
            if marker in prompt:
                return fn(prompt)
        return "no"

    llm.calls = calls
    return llm


@pytest.mark.asyncio
async def test_happy_path_no_transform():
    """Релевантный документ найден сразу → один проход, без переформулировок."""

    async def retriever(query, limit):
        return [{"content": "Парацетамол — анальгетик и антипиретик.", "score": 0.9}]

    llm = make_llm({
        "Релевантен ли документ": lambda p: "yes",
        "Дай точный ответ": lambda p: "Парацетамол — это анальгетик и жаропонижающее.",
        "Полностью ли ответ обоснован": lambda p: "yes",
        "Отвечает ли ответ": lambda p: "yes",
    })

    graph = CorrectiveRagGraph(retriever, llm, max_transforms=2, max_generations=2).build()
    out = await graph.ainvoke({"question": "Что такое парацетамол?"})

    assert out["generation"].startswith("Парацетамол")
    assert out.get("transforms", 0) == 0
    assert out.get("generations") == 1
    assert out["grounded"] and out["answers_question"]


@pytest.mark.asyncio
async def test_corrective_transforms_when_no_relevant_docs():
    """Первый поиск отдаёт мусор → грейдер отсеивает → transform_query → второй поиск релевантен."""
    state = {"n": 0}

    async def retriever(query, limit):
        state["n"] += 1
        if state["n"] == 1:
            return [{"content": "Прогноз погоды на завтра.", "score": 0.4}]
        return [{"content": "Ибупрофен — НПВС, противовоспалительное.", "score": 0.88}]

    def grade_doc(prompt):
        # релевантен только документ про ибупрофен
        return "yes" if "Ибупрофен" in prompt else "no"

    llm = make_llm({
        "Релевантен ли документ": grade_doc,
        "Переформулируй запрос": lambda p: "ибупрофен НПВС противовоспалительное",
        "Дай точный ответ": lambda p: "Ибупрофен — нестероидное противовоспалительное средство.",
        "Полностью ли ответ обоснован": lambda p: "yes",
        "Отвечает ли ответ": lambda p: "yes",
    })

    graph = CorrectiveRagGraph(retriever, llm, max_transforms=2, max_generations=2).build()
    out = await graph.ainvoke({"question": "Что за препарат ибупрофен?"})

    assert state["n"] == 2  # был повторный поиск
    assert out["transforms"] == 1
    assert "Ибупрофен" in out["generation"]
    assert out["grounded"] and out["answers_question"]


@pytest.mark.asyncio
async def test_regenerates_on_hallucination():
    """Первый ответ не обоснован контекстом → перегенерация (generations == 2)."""
    gen_state = {"n": 0}

    async def retriever(query, limit):
        return [{"content": "Метформин снижает уровень глюкозы в крови.", "score": 0.9}]

    def generate(prompt):
        gen_state["n"] += 1
        return "Первый выдуманный ответ." if gen_state["n"] == 1 else "Метформин снижает глюкозу."

    def hallucination(prompt):
        # обоснован только второй ответ
        return "yes" if "снижает глюкозу" in prompt else "no"

    llm = make_llm({
        "Релевантен ли документ": lambda p: "yes",
        "Дай точный ответ": generate,
        "Полностью ли ответ обоснован": hallucination,
        "Отвечает ли ответ": lambda p: "yes",
    })

    graph = CorrectiveRagGraph(retriever, llm, max_transforms=2, max_generations=2).build()
    out = await graph.ainvoke({"question": "Что делает метформин?"})

    assert out["generations"] == 2
    assert out["grounded"]


@pytest.mark.asyncio
async def test_transform_budget_is_respected():
    """Релевантных документов нет вовсе → граф не зацикливается, упирается в лимит."""
    retr_calls = {"n": 0}

    async def retriever(query, limit):
        retr_calls["n"] += 1
        return [{"content": "Нерелевантный текст.", "score": 0.1}]

    llm = make_llm({
        "Релевантен ли документ": lambda p: "no",
        "Переформулируй запрос": lambda p: "другой запрос",
        "Дай точный ответ": lambda p: "В контексте нет информации по вопросу.",
        "Полностью ли ответ обоснован": lambda p: "yes",
        "Отвечает ли ответ": lambda p: "no",
    })

    graph = CorrectiveRagGraph(retriever, llm, max_transforms=2, max_generations=2).build()
    out = await graph.ainvoke({"question": "Вопрос без ответа в корпусе?"})

    # 2 переформулировки максимум → не больше max_transforms
    assert out["transforms"] <= 2
    # ретрив вызывался стартово + по разу на каждую переформулировку
    assert retr_calls["n"] <= 3
