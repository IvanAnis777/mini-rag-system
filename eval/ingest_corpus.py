#!/usr/bin/env python3
"""Загрузка фарма-корпуса (eval/corpus/*.md) в векторное хранилище.

Требует поднятых Postgres+pgvector и Redis (docker-compose up -d postgres redis).
Запуск:  python eval/ingest_corpus.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.services.vector_store import get_vector_store_service

CORPUS_DIR = Path(__file__).parent / "corpus"


async def main():
    vector_store = get_vector_store_service()
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        print(f"Нет файлов в {CORPUS_DIR}")
        return

    total_chunks = 0
    for path in files:
        content = path.read_text(encoding="utf-8")
        title = content.splitlines()[0].lstrip("# ").strip() or path.stem
        doc_id, chunks = await vector_store.add_document(
            title=title,
            content=content,
            metadata={"source": path.name, "domain": "pharma"},
        )
        total_chunks += chunks
        print(f"  + {path.name}: doc_id={doc_id}, чанков={chunks}")

    print(f"Загружено документов: {len(files)}, чанков: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(main())
