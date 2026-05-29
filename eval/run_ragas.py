#!/usr/bin/env python3
"""Оценка агентного RAG через Ragas (faithfulness, answer relevancy, context precision/recall).

Пайплайн:
  1) для каждого вопроса из eval/pharma_qa.jsonl прогоняем агентный граф (LangGraph);
  2) собираем (вопрос, ответ, retrieved_contexts, эталон) в датасет Ragas;
  3) считаем метрики; судьёй по умолчанию выступает ЛОКАЛЬНАЯ llama
     (OpenAI-совместимый эндпоинт llama.cpp) + локальные эмбеддинги — без внешних API;
  4) пишем отчёт в eval/report.md.

Судью можно переключить переменными окружения:
  RAGAS_JUDGE=openai   + OPENAI_API_KEY     — судить через GPT-4o-mini
  RAGAS_JUDGE=local    (по умолчанию)        — судить локальной llama

Предпосылки: подняты postgres+redis+llama-server, корпус загружен (python eval/ingest_corpus.py),
установлены зависимости оценки:  pip install -r requirements-eval.txt
Запуск:  python eval/run_ragas.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Внимание: app.* импортируется ТОЛЬКО в фазе collect (нужен рантайм приложения),
# а ragas/langchain — только в фазе evaluate. Их зависимости несовместимы в одном
# окружении (ragas тянет langchain-core<1, приложение — langchain-core>=1.4), поэтому
# фазы запускаются раздельно и общаются через eval/samples.json.

QA_PATH = Path(__file__).parent / "pharma_qa.jsonl"
SAMPLES_PATH = Path(__file__).parent / "samples.json"
REPORT_PATH = Path(__file__).parent / "report.md"


def load_qa():
    rows = []
    for line in QA_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def collect_samples():
    """Прогоняет агентный граф по каждому вопросу и собирает данные для оценки."""
    from app.services.rag_graph import get_agentic_rag_graph

    graph = get_agentic_rag_graph()
    samples = []
    for row in load_qa():
        result = await graph.ainvoke(
            {"question": row["question"], "query": row["question"]}
        )
        contexts = [d.get("content", "") for d in result.get("documents", [])]
        samples.append(
            {
                "user_input": row["question"],
                "response": result.get("generation", ""),
                "retrieved_contexts": contexts or ["(контекст не найден)"],
                "reference": row["ground_truth"],
                "transforms": result.get("transforms", 0),
                "grounded": result.get("grounded", False),
            }
        )
        print(f"  · {row['question'][:50]}… → transforms={result.get('transforms', 0)}")
    return samples


def build_judge():
    """Возвращает (llm, embeddings) для Ragas. По умолчанию — локальная llama."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # По умолчанию судья = LLM_BACKEND приложения, можно переопределить RAGAS_JUDGE.
    # В фазе evaluate настройки берём из env (app.core.config НЕ импортируем).
    emb_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    judge = os.getenv("RAGAS_JUDGE", os.getenv("LLM_BACKEND", "llama")).lower()
    if judge == "llama":
        judge = "local"

    if judge == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
    elif judge == "groq":
        # Groq (OpenAI-совместимый endpoint). Ключ — GROQ_API_KEY, модель — GROQ_MODEL
        # (по умолчанию llama-3.3-70b-versatile). Эмбеддинги — локальные HF.
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=emb_model)
    elif judge in ("gemini", "google"):
        # Google AI Studio (Gemini). Ключ читается из GOOGLE_API_KEY; модель — из
        # GEMINI_MODEL (по умолчанию gemini-2.0-flash). Эмбеддинги — локальные HF.
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=emb_model)
    elif judge == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), temperature=0)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=emb_model)
    else:
        # Локальная llama через OpenAI-совместимый эндпоинт llama.cpp
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="local",
            base_url=f"{os.getenv('LLAMA_SERVER_URL', 'http://llama-server:8080')}/v1",
            api_key="sk-no-key-required",
            temperature=0,
        )
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=emb_model)

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def run_eval(samples):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    llm, emb = build_judge()
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": s["user_input"],
                "response": s["response"],
                "retrieved_contexts": s["retrieved_contexts"],
                "reference": s["reference"],
            }
            for s in samples
        ]
    )
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb)


def write_report(result, samples):
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in
                   ("user_input", "response", "retrieved_contexts", "reference")]

    lines = ["# Ragas-отчёт: агентный RAG (фарма-корпус)", ""]
    lines.append(f"Вопросов: {len(samples)} · судья: `{os.getenv('RAGAS_JUDGE', 'local')}`")
    lines.append("")
    lines.append("## Средние метрики")
    lines.append("")
    lines.append("| Метрика | Среднее |")
    lines.append("|---|---|")
    for col in metric_cols:
        lines.append(f"| {col} | {df[col].mean():.3f} |")
    lines.append("")
    lines.append("## По вопросам")
    lines.append("")
    header = "| Вопрос | " + " | ".join(metric_cols) + " | transforms |"
    lines.append(header)
    lines.append("|" + "---|" * (len(metric_cols) + 2))
    for i, row in df.iterrows():
        q = str(row["user_input"])[:40]
        vals = " | ".join(f"{row[c]:.2f}" for c in metric_cols)
        lines.append(f"| {q}… | {vals} | {samples[i]['transforms']} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт записан: {REPORT_PATH}")
    print("\n".join(lines[2:8]))


def do_collect():
    """Фаза 1: прогон графа → eval/samples.json (нужен app, без ragas)."""
    print("collect: прогон агентного графа по вопросам…")
    samples = asyncio.run(collect_samples())
    SAMPLES_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"collect: сохранено {len(samples)} сэмплов → {SAMPLES_PATH}")


def do_evaluate():
    """Фаза 2: eval/samples.json → метрики Ragas → отчёт (нужен ragas, без app)."""
    if not SAMPLES_PATH.exists():
        sys.exit(f"Нет {SAMPLES_PATH} — сначала запусти фазу collect.")
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    print(f"evaluate: {len(samples)} сэмплов, судья={os.getenv('RAGAS_JUDGE', 'local')}…")
    result = run_eval(samples)
    write_report(result, samples)


def main():
    # Режимы: collect (app) | evaluate (ragas) | all (оба — нужен env с обеими зависимостями).
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("collect", "evaluate", "all"):
        sys.exit("Использование: run_ragas.py [collect|evaluate|all]")
    if mode in ("collect", "all"):
        do_collect()
    if mode in ("evaluate", "all"):
        do_evaluate()


if __name__ == "__main__":
    main()
