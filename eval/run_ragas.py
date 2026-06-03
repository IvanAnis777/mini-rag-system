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
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

# Внимание: app.* импортируется ТОЛЬКО в фазе collect (нужен рантайм приложения),
# а ragas/langchain — только в фазе evaluate. Их зависимости несовместимы в одном
# окружении (ragas тянет langchain-core<1, приложение — langchain-core>=1.4), поэтому
# фазы запускаются раздельно и общаются через eval/samples.json.

QA_PATH = Path(__file__).parent / "pharma_qa.jsonl"
SAMPLES_PATH = Path(__file__).parent / "samples.json"
REPORT_PATH = Path(__file__).parent / "report.md"
CORPUS_DIR = Path(__file__).parent / "corpus"


def load_qa():
    rows = []
    for line in QA_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------- retrieval-метрики (precision@k / recall@k) ----------
# Считаются БЕЗ LLM-судьи: чисто по тому, из «правильного» ли документа пришёл каждый
# найденный чанк. Поэтому их можно гонять офлайн, без ragas и без API-ключей.

def _normalize(text: str) -> str:
    """Схлопывает пробелы и приводит к нижнему регистру для устойчивого сравнения."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _load_corpus() -> dict:
    """{имя_документа: нормализованный_текст} по файлам eval/corpus/*.md."""
    return {f.stem: _normalize(f.read_text(encoding="utf-8")) for f in CORPUS_DIR.glob("*.md")}


def _chunk_source(chunk: str, corpus: dict) -> Optional[str]:
    """Из какого документа корпуса пришёл чанк. Чанк — подстрока ровно одного файла
    (так нарезали при ингесте), поэтому ищем включением. None — если ни в один не попал
    (например, плейсхолдер «(контекст не найден)»)."""
    c = _normalize(chunk)
    if not c:
        return None
    matches = [name for name, full in corpus.items() if c in full]
    return matches[0] if len(matches) == 1 else None


def compute_retrieval_metrics(samples) -> dict:
    """precision@k и recall (hit-rate) ретривера по gold-документу каждого вопроса.

    Для каждого вопроса известен gold-документ (`source`). Среди найденных чанков
    релевантными считаем те, что пришли из gold-документа:
      - precision@k = релевантных_чанков / всего_найденных_чанков (k = размер контекста);
      - recall (hit-rate) = попал ли gold-документ в контекст вообще (1/0), усреднённо.
    Gold берём из самих samples, а если его там нет (старый сбор) — по вопросу из pharma_qa.
    """
    corpus = _load_corpus()
    qa_source = {row["question"]: row.get("source") for row in load_qa()}

    per_question = []
    for i, s in enumerate(samples):
        gold = s.get("source") or qa_source.get(s["user_input"])
        chunks = s.get("retrieved_contexts", []) or []
        sources = [_chunk_source(c, corpus) for c in chunks]
        relevant = sum(1 for src in sources if src == gold)
        k = len(chunks)
        precision = relevant / k if k else 0.0
        hit = 1.0 if relevant > 0 else 0.0
        per_question.append({
            "idx": i, "gold": gold, "k": k,
            "relevant": relevant, "precision": precision, "hit": hit,
        })

    n = len(per_question) or 1
    return {
        "precision_at_k": sum(q["precision"] for q in per_question) / n,
        "recall_hit_rate": sum(q["hit"] for q in per_question) / n,
        "per_question": per_question,
    }


def format_retrieval_section(rm: dict) -> list:
    """Markdown-секция retrieval-метрик для отчёта."""
    lines = ["## Retrieval-метрики (без судьи)", ""]
    lines.append(f"- **precision@k** (доля релевантных среди найденных чанков): **{rm['precision_at_k']:.3f}**")
    lines.append(f"- **recall@k / hit-rate** (gold-документ попал в контекст): **{rm['recall_hit_rate']:.3f}**")
    lines.append("")
    lines.append("| # | gold-документ | чанков (k) | релевантных | precision@k | hit |")
    lines.append("|---|---|---|---|---|---|")
    for q in rm["per_question"]:
        lines.append(
            f"| {q['idx']} | {q['gold']} | {q['k']} | {q['relevant']} | "
            f"{q['precision']:.2f} | {int(q['hit'])} |"
        )
    lines.append("")
    return lines


def log_to_mlflow(params: dict, metrics: dict, experiment: str = "mini-rag-ragas") -> None:
    """Логирует параметры прогона и метрики в ЛОКАЛЬНЫЙ MLflow (eval/mlruns) — закрывает
    гэп «experiment tracking». Сервер поднимать не нужно (file-store); посмотреть прогоны:
    `mlflow ui --backend-store-uri eval/mlruns`.

    Мягкая зависимость: если mlflow не установлен — просто пропускаем, не ломая прогон.
    nan-метрики (судья не досчитал) отбрасываем, чтобы не засорять трекер.
    """
    try:
        # mlflow 3.x держит простой file-store в «maintenance mode» — для локального
        # учебного трекинга он нам подходит, явно разрешаем (можно переопределить из env).
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except ImportError:
        print("mlflow не установлен — трекинг пропущен (pip install -r requirements-eval.txt).")
        return
    mlflow.set_tracking_uri((Path(__file__).parent / "mlruns").resolve().as_uri())
    mlflow.set_experiment(experiment)
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(clean)
        if REPORT_PATH.exists():
            mlflow.log_artifact(str(REPORT_PATH))
    print(f"mlflow: {len(clean)} метрик → эксперимент '{experiment}' (смотреть: mlflow ui --backend-store-uri eval/mlruns)")


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
                "source": row.get("source"),  # gold-документ для retrieval-метрик
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
    from ragas.run_config import RunConfig

    # Умеренный параллелизм + ретраи с бэкоффом под TPM-лимит судьи.
    # Groq free = 12000 ток/мин; при свежей суточной квоте этого хватает на чистый
    # прогон. timeout держим небольшим (180с): зависшую на лимите задачу лучше
    # быстро ретраить, чем блокировать прогон на минуты.
    run_config = RunConfig(max_workers=4, timeout=180, max_retries=10, max_wait=30)

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
    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb, run_config=run_config)


def write_report(result, samples):
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in
                   ("user_input", "response", "retrieved_contexts", "reference")]

    lines = ["# Ragas-отчёт: агентный RAG (фарма-корпус)", ""]
    lines.append(f"Вопросов: {len(samples)} · судья: `{os.getenv('RAGAS_JUDGE', 'local')}`")
    lines.append("")
    lines.append("## Средние метрики (LLM-судья)")
    lines.append("")
    lines.append("| Метрика | Среднее |")
    lines.append("|---|---|")
    for col in metric_cols:
        lines.append(f"| {col} | {df[col].mean():.3f} |")
    lines.append("")
    # retrieval-метрики считаем тут же — они не зависят от судьи
    rm = compute_retrieval_metrics(samples)
    lines += format_retrieval_section(rm)
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

    # метрики для трекинга: средние по судье + retrieval
    metrics = {col: float(df[col].mean()) for col in metric_cols}
    metrics["precision_at_k"] = rm["precision_at_k"]
    metrics["recall_hit_rate"] = rm["recall_hit_rate"]
    return metrics


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
    judge = os.getenv("RAGAS_JUDGE", os.getenv("LLM_BACKEND", "local"))
    print(f"evaluate: {len(samples)} сэмплов, судья={judge}…")
    result = run_eval(samples)
    metrics = write_report(result, samples)
    log_to_mlflow(
        {"phase": "evaluate", "judge": judge, "n_questions": len(samples)},
        metrics,
    )


def do_retrieval():
    """Только retrieval-метрики (precision@k / recall@k) по готовым samples.json.

    Не требует ни ragas, ни LLM-судьи, ни API-ключей — чистый stdlib. Удобно гонять
    после каждого collect, чтобы быстро увидеть качество поиска без затрат на судью.
    """
    if not SAMPLES_PATH.exists():
        sys.exit(f"Нет {SAMPLES_PATH} — сначала запусти фазу collect.")
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    rm = compute_retrieval_metrics(samples)
    print("\n".join(format_retrieval_section(rm)))
    log_to_mlflow(
        {"phase": "retrieval", "n_questions": len(samples),
         "backend": os.getenv("LLM_BACKEND", "llama")},
        {"precision_at_k": rm["precision_at_k"], "recall_hit_rate": rm["recall_hit_rate"]},
    )
    return rm


def main():
    # Режимы: collect (app) | evaluate (ragas+судья) | retrieval (только поиск, без судьи) | all.
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("collect", "evaluate", "retrieval", "all"):
        sys.exit("Использование: run_ragas.py [collect|evaluate|retrieval|all]")
    if mode == "retrieval":
        do_retrieval()
        return
    if mode in ("collect", "all"):
        do_collect()
    if mode in ("evaluate", "all"):
        do_evaluate()


if __name__ == "__main__":
    main()
