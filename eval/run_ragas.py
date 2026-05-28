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

from app.core.config import settings

QA_PATH = Path(__file__).parent / "pharma_qa.jsonl"
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

    judge = os.getenv("RAGAS_JUDGE", "local").lower()

    if judge == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Локальная llama через OpenAI-совместимый эндпоинт llama.cpp
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="local",
            base_url=f"{settings.llama.server_url}/v1",
            api_key="sk-no-key-required",
            temperature=0,
        )
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=settings.vector_store.embedding_model)

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


def main():
    print("1/3 Прогон агентного графа по вопросам…")
    samples = asyncio.run(collect_samples())
    print("2/3 Оценка через Ragas…")
    result = run_eval(samples)
    print("3/3 Запись отчёта…")
    write_report(result, samples)


if __name__ == "__main__":
    main()
