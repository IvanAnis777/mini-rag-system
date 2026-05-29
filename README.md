# Mini RAG System

[![CI](https://github.com/IvanAnis777/mini-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/IvanAnis777/mini-rag-system/actions/workflows/ci.yml)

Система вопросов и ответов с использованием RAG (Retrieval-Augmented Generation). Проект разработан для создания умного помощника, который может отвечать на вопросы на основе загруженных документов.

## Возможности

- **RAG Pipeline**: Система поиска и генерации ответов
- **Агентный RAG (LangGraph)**: corrective / self-RAG с грейдингом документов, переформулировкой запроса и самопроверкой ответа на галлюцинации
- **Оценка качества (Ragas)**: faithfulness, answer relevancy, context precision/recall на фарма-корпусе
- **Векторное хранилище**: PostgreSQL + pgvector для семантического поиска
- **LLaMA интеграция**: локальный инференс через llama.cpp (OpenAI-совместимый сервер)
- **Кэширование**: Redis для оптимизации эмбеддингов
- **REST API**: FastAPI с документацией
- **Docker**: Контейнеризированное решение
- **Мониторинг**: Логирование запросов

## Архитектура

```mermaid
graph TB
    Client[Клиент] --> API[FastAPI API]
    API --> RAG[RAG Service]
    RAG --> Vector[Vector Store]
    RAG --> LLama[LLaMA Server]
    Vector --> PG[PostgreSQL + pgvector]
    RAG --> Embed[Embedding Service]
    Embed --> Redis[Redis Cache]
    Embed --> ST[SentenceTransformers]
```



## Агентный RAG (LangGraph)

Кроме линейного `/query` есть агентный эндпоинт `/query/agentic` — граф самокоррекции
(`app/services/rag_graph.py`), реализующий corrective / self-RAG поверх тех же
vector_store и llama_client:

```mermaid
graph TD
    A[retrieve] --> B[grade_documents]
    B -->|есть релевантные| D[generate]
    B -->|нет релевантных| C[transform_query]
    C --> A
    D --> E[grade_generation]
    E -->|галлюцинация| D
    E -->|не отвечает| C
    E -->|ок| F[END]
```



Что это даёт против линейного пайплайна:

- **grade_documents** — LLM отсеивает нерелевантные чанки до генерации;
- **transform_query** — при слабом поиске запрос переформулируется и поиск повторяется (с лимитом);
- **grade_generation** — ответ проверяется на обоснованность контекстом (анти-галлюцинация) и на то, отвечает ли он по существу.

Ответ возвращает `trace` шагов, число переформулировок и флаги `grounded` / `answers_question`.

```bash
curl -X POST "http://localhost:8000/api/v1/query/agentic" \
     -H "Content-Type: application/json" \
     -d '{"question": "Какая максимальная суточная доза парацетамола?"}'
```

Логика маршрутизации покрыта юнит-тестами на стабах (без БД и LLaMA):

```bash
pytest tests/test_rag_graph.py -o asyncio_mode=auto -q
```

## Оценка качества (Ragas)

Численная оценка агентного RAG на доменном фарма-корпусе (`eval/corpus/`, QA в
`eval/pharma_qa.jsonl`). Метрики: **faithfulness**, **answer relevancy**,
**context precision**, **context recall**. Судьёй по умолчанию выступает локальная
llama (без внешних API).

```bash
pip install -r requirements-eval.txt        # ragas + langchain-openai/anthropic + langchain-huggingface
python eval/ingest_corpus.py                  # загрузить корпус в pgvector
python eval/run_ragas.py                      # прогон + отчёт в eval/report.md
# судья по умолчанию = LLM_BACKEND; переопределить: RAGAS_JUDGE=openai|anthropic|local
```

## Выбор модели и бэкенда

Система **model-agnostic**: `api` ходит к LLM по HTTP/OpenAI-совместимому интерфейсу и
не зависит от конкретной модели. Переключение — без правок кода.

### Локальная модель (GGUF) — реестр

```bash
make download-model MODEL=qwen      # qwen | mistral | llama3 | llama2 (по умолч.)
make restart                        # подхватит MODEL_FILE из .env
```

`download-model` качает GGUF и прописывает `MODEL_FILE` в `.env`; `docker-compose`
монтирует нужный файл в `llama-server`. Доступно: Llama-2-7B, Qwen2.5-7B,
Mistral-7B-v0.3, Llama-3.1-8B (все Q4_K_M).

### Бэкенд: локально ↔ облако

Один и тот же агентный граф работает на локальной llama или в облаке —
переменная `LLM_BACKEND` (`app/services/llm_provider.py`):

```bash
LLM_BACKEND=llama                                   # локально, по умолчанию, без ключей
LLM_BACKEND=anthropic  ANTHROPIC_API_KEY=sk-ant-... # Claude
LLM_BACKEND=openai     OPENAI_API_KEY=sk-...         # GPT
```

## Быстрый старт

### Предварительные требования

- Docker и Docker Compose
- Минимум 8GB RAM (для LLaMA-8B)
- 20GB свободного места

### Установка

1. **Клонируйте репозиторий**

```bash
git clone <your-repo-url>
cd mini-rag-system
```

1. **Скачайте LLaMA модель**

```bash
# Создайте директорию для моделей
mkdir -p data/models

# Скачайте модель (пример)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf \
     -O data/models/llama-3-8b-instruct.gguf
```

1. **Настройте окружение**

```bash
cp config.env.example .env
# Отредактируйте .env при необходимости
```

1. **Запустите систему**

```bash
docker-compose up -d
```

1. **Проверьте статус**

```bash
curl http://localhost:8000/api/v1/health
```

## Использование

### Веб-интерфейс

Откройте в браузере:

- **API документация**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Примеры API запросов

#### Загрузка документа

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Документ о машинном обучении",
       "content": "Машинное обучение - это подмножество искусственного интеллекта...",
       "metadata": {"source": "example"}
     }'
```

#### Задать вопрос

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Что такое машинное обучение?",
       "language": "ru",
       "max_chunks": 3
     }'
```

#### Поиск документов

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "искусственный интеллект",
       "limit": 5,
       "threshold": 0.7
     }'
```

## Пошаговый туториал

> **Всё работает в Docker.** Команды `make init-db`, `make load-test-data`, `make test`
> выполняются **внутри контейнера `api`** (там Python 3.11 и установленные зависимости),
> поэтому требуют поднятого стека — сначала `make up`. Запускать их на хосте не нужно.
>
> ⚠️ Пины в `requirements.txt` рассчитаны на **Python 3.11**: на Python 3.13+ часть
> пакетов (`numpy==1.24.4` и др.) не собирается. Если хочется работать с хоста — используйте
> Python 3.11 (`python3.11 -m venv .venv`). Иначе просто работайте через контейнер.

### Запуск с нуля за 5 минут

#### Шаг 1: Подготовка среды

```bash
# Настраиваем проект (создает .env, папки)
make dev-setup

# Проверяем что создалось
ls -la
```

#### Шаг 2: Запуск сервисов (без LLaMA)

```bash
# Запускаем PostgreSQL, Redis (быстро)
make up

# Проверяем статус
make status
```

#### Шаг 3: Инициализация базы данных

```bash
# Создаем таблицы для векторного хранилища
make init-db
```

#### Шаг 4: Загрузка тестовых данных

```bash
# Загружаем примеры документов о ML/AI
make load-test-data
```

#### Шаг 5: Тестируем систему

```bash
# Открываем веб-интерфейс
open http://localhost:8000/docs

# Или тестируем через curl
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Что такое RAG?", "language": "ru"}'
```

### Добавление LLaMA (опционально)

```bash
# Скачиваем модель ~3.8GB (займет время)
make download-model

# Перезапускаем с LLaMA сервером
make restart
```

### Способы использования

#### 1. Веб-интерфейс (самый простой)

- **API документация**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Альтернативная документация**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

В веб-интерфейсе можно:

- Загружать документы через форму
- Задавать вопросы и получать ответы
- Искать по базе знаний
- Смотреть статистику системы

#### 2. Через Python код

```python
import requests

# Базовая настройка
BASE_URL = "http://localhost:8000/api/v1"

# Загружаем документ
doc_response = requests.post(f"{BASE_URL}/documents", json={
    "title": "Руководство по Python",
    "content": "Python - высокоуровневый язык программирования...",
    "metadata": {"category": "programming", "author": "Ivan"}
})

print("Документ загружен:", doc_response.json())

# Задаем вопрос
query_response = requests.post(f"{BASE_URL}/query", json={
    "question": "Что такое Python?",
    "language": "ru",
    "max_chunks": 3
})

result = query_response.json()
print("Ответ:", result["answer"])
print("Источники:", result["sources"])
print("Уверенность:", result["confidence"])
```

#### 3. Интеграция в чат-бот

```python
class RAGChatBot:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def ask(self, question, language="ru"):
        response = requests.post(f"{self.base_url}/query", json={
            "question": question,
            "language": language
        })
        return response.json()
    
    def add_knowledge(self, title, content, metadata=None):
        return requests.post(f"{self.base_url}/documents", json={
            "title": title,
            "content": content,
            "metadata": metadata or {}
        })

# Использование
bot = RAGChatBot()
answer = bot.ask("Расскажи про машинное обучение")
print(answer["answer"])
```

### Практические сценарии

#### Корпоративная база знаний

```bash
# 1. Загружаем документы компании
curl -X POST "http://localhost:8000/api/v1/documents" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Политика отпусков",
       "content": "Сотрудники имеют право на 28 календарных дней отпуска...",
       "metadata": {"department": "HR", "version": "2024"}
     }'

# 2. Сотрудники задают вопросы
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Сколько дней отпуска у меня есть?", "language": "ru"}'
```

#### Образовательный помощник

```python
# Загружаем учебные материалы
materials = [
    {"title": "Лекция 1: Введение в ML", "content": "..."},
    {"title": "Лекция 2: Линейная регрессия", "content": "..."},
    {"title": "Практика: Sklearn", "content": "..."}
]

for material in materials:
    requests.post(f"{BASE_URL}/documents", json=material)

# Студенты задают вопросы
answer = requests.post(f"{BASE_URL}/query", json={
    "question": "Как работает линейная регрессия?",
    "language": "ru"
})
```

#### Техподдержка

```python
# Загружаем FAQ и инструкции
faq_data = {
    "title": "FAQ: Проблемы с авторизацией",
    "content": "Если не можете войти: 1) Проверьте пароль...",
    "metadata": {"category": "auth", "priority": "high"}
}

requests.post(f"{BASE_URL}/documents", json=faq_data)

# Автоматические ответы на вопросы пользователей
user_question = "Не могу войти в систему"
auto_response = requests.post(f"{BASE_URL}/query", json={
    "question": user_question,
    "language": "ru"
})
```

### Полезные команды для работы

```bash
# Мониторинг работы
make monitor          # Системная статистика
make logs            # Все логи
make logs-api        # Только логи API
make status          # Статус сервисов

# Управление данными
make backup          # Резервная копия БД
make clean           # Полная очистка
make restart         # Перезапуск сервисов

# Разработка
make test            # Запуск тестов
make lint            # Проверка кода
make format          # Форматирование кода
```

### Мониторинг и метрики

```bash
# Проверка здоровья системы
curl http://localhost:8000/api/v1/health

# Получение статистики
curl http://localhost:8000/api/v1/stats

# Пример ответа статистики:
{
  "vector_store": {
    "documents_count": 5,
    "chunks_count": 23,
    "embedding_dimension": 384
  },
  "embedding_model": {
    "model_name": "all-MiniLM-L6-v2",
    "status": "healthy"
  }
}
```

### Устранение проблем

```bash
# Если что-то не работает:

# 1. Проверить статус всех сервисов
docker-compose ps

# 2. Посмотреть логи ошибок
make logs | grep -i error

# 3. Перезапустить проблемный сервис
docker-compose restart api

# 4. Полный перезапуск
make down && make up

# 5. Проверить доступность портов
lsof -i :8000  # API
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
```

### Результат

После выполнения туториала у вас будет:

- **Работающая RAG система** с веб-интерфейсом  
- **База знаний** с тестовыми документами  
- **REST API** для интеграции в приложения  
- **Примеры использования** на Python  
- **Мониторинг и логирование**

Система готова для:

- Создания корпоративных баз знаний
- Образовательных платформ  
- Умных чат-ботов
- Систем техподдержки
- Продуктовых решений

## Конфигурация

### Переменные окружения (.env)

```bash
# База данных
DATABASE_URL=postgresql://minirag:minirag123@localhost:5432/minirag

# LLaMA сервер
LLAMA_SERVER_URL=http://localhost:8080
MODEL_PATH=./data/models/llama-3-8b-instruct.gguf

# System Prompt
SYSTEM_PROMPT_FILE=./prompts/mini-rag-system-prompt.txt

# Векторное хранилище
VECTOR_STORE_TYPE=pgvector
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DIMENSION=384

# Redis
REDIS_URL=redis://localhost:6379

# RAG настройки
MAX_CONTEXT_CHUNKS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.7
```

### System Prompt

Система использует system prompt из файла `prompts/mini-rag-system-prompt.txt` со следующими функциями:

- Чёткое разделение ролей
- Переменные для RAG-контекста  
- Форматирование ответа с цитированием
- Правила отказа и самопроверка
- Многязычность (русский/английский)

## Docker Services


| Сервис       | Порт | Описание               |
| ------------ | ---- | ---------------------- |
| api          | 8000 | FastAPI приложение     |
| llama-server | 8080 | LLaMA inference сервер |
| postgres     | 5432 | PostgreSQL + pgvector  |
| redis        | 6379 | Кэш для эмбеддингов    |


## Мониторинг

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Системная статистика

```bash
curl http://localhost:8000/api/v1/stats
```

### Логи

```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f api
```

## Тестирование

```bash
# Запуск тестов
python -m pytest tests/ -v

# Тестирование API
python -m pytest tests/test_api.py -v

# Тестирование RAG pipeline
python -m pytest tests/test_rag.py -v
```

## Разработка

### Локальная разработка (на хосте)

> Нужен **Python 3.11** — пины `requirements.txt` не ставятся на 3.13+. Если на хосте
> другая версия, просто работайте через контейнер (см. ниже), хост-шаги не обязательны.

1. **Установите зависимости** (Python 3.11)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Запустите зависимости в Docker**

```bash
docker compose up -d postgres redis llama-server
```

1. **Создайте таблицы**

```bash
python -c "from app.core.database import create_tables; create_tables()"
```

1. **Запустите API**

```bash
python main.py
```

### Запуск через контейнер (рекомендуется)

Ничего не ставя на хост — всё уже есть в образе `api`:

```bash
make up                    # поднять стек
make init-db               # создать таблицы (в контейнере)
make load-test-data        # загрузить тестовые данные (в контейнере)
make test                  # прогнать тесты (в контейнере): pytest tests/

# разовые команды напрямую:
docker compose exec api python -c "from app.core.database import create_tables; create_tables()"
docker compose exec api python -m pytest tests/ -o asyncio_mode=auto -v
```

### Добавление новых компонентов

1. **Сервисы**: Добавьте в `app/services/`
2. **API роуты**: Расширьте `app/api/routes.py`
3. **Модели данных**: Обновите `app/models/`
4. **Конфигурация**: Дополните `app/core/config.py`

## Производительность

### Рекомендуемые ресурсы:

- **Минимум**: 8GB RAM, 4 CPU cores
- **Рекомендуется**: 16GB RAM, 8 CPU cores
- **Для продакшена**: 32GB+ RAM, GPU поддержка

### Оптимизация:

- Увеличьте `N_PARALLEL` для LLaMA при большей нагрузке
- Настройте размер PostgreSQL shared_buffers
- Используйте SSD для векторных индексов
- Рассмотрите GPU-ускорение для эмбеддингов

## 🚨 Устранение неполадок

### Частые проблемы:

1. **LLaMA сервер не запускается**
  - Проверьте наличие модели в `data/models/`
  - Убедитесь в достатке RAM
2. **Медленные эмбеддинги**
  - Проверьте подключение к Redis
  - Рассмотрите GPU-ускорение
3. **Ошибки PostgreSQL**
  - Убедитесь, что pgvector расширение установлено
  - Проверьте настройки памяти

### Логи и отладка:

```bash
# Подробные логи API
docker-compose logs -f api

# Состояние сервисов
docker-compose ps

# Перезапуск сервиса
docker-compose restart api
```

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 🙏 Благодарности

- [LangChain](https://github.com/langchain-ai/langchain) за RAG фреймворк
- [llama.cpp](https://github.com/ggerganov/llama.cpp) за оптимизированный inference
- [pgvector](https://github.com/pgvector/pgvector) за векторное расширение PostgreSQL
- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers) за эмбеддинги

---

**Создано для демонстрации лучших практик MLOps и RAG-систем 2025 года** 🚀 