# 🤖 Mini RAG System

Самодостаточная система вопросов и ответов с использованием RAG (Retrieval-Augmented Generation), демонстрирующая лучшие практики 2025 года в области MLOps, prompt-design и RAG-pipeline.

## 🌟 Возможности

- **RAG Pipeline**: Полноценная система поиска и генерации ответов
- **Векторное хранилище**: PostgreSQL + pgvector для быстрого семантического поиска
- **LLaMA интеграция**: Поддержка LLaMA-3 8B/70B моделей
- **Кэширование**: Redis для оптимизации эмбеддингов
- **REST API**: FastAPI с автоматической документацией
- **Docker**: Полностью контейнеризированное решение
- **Мониторинг**: Логирование запросов и системные метрики

## 🏗️ Архитектура

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

## 🚀 Быстрый старт

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

2. **Скачайте LLaMA модель**
```bash
# Создайте директорию для моделей
mkdir -p data/models

# Скачайте модель (пример)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf \
     -O data/models/llama-3-8b-instruct.gguf
```

3. **Настройте окружение**
```bash
cp config.env.example .env
# Отредактируйте .env при необходимости
```

4. **Запустите систему**
```bash
docker-compose up -d
```

5. **Проверьте статус**
```bash
curl http://localhost:8000/api/v1/health
```

## 📋 Использование

### Веб-интерфейс

Откройте в браузере:
- **API документация**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

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

## 📖 Пошаговый туториал

### 🚀 Запуск с нуля за 5 минут

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

### 🤖 Добавление LLaMA (опционально)

```bash
# Скачиваем модель ~3.8GB (займет время)
make download-model

# Перезапускаем с LLaMA сервером
make restart
```

### 📱 Способы использования

#### 1️⃣ Веб-интерфейс (самый простой)
- **API документация**: http://localhost:8000/docs
- **Альтернативная документация**: http://localhost:8000/redoc

В веб-интерфейсе можно:
- 📝 Загружать документы через форму
- ❓ Задавать вопросы и получать ответы
- 🔍 Искать по базе знаний
- 📊 Смотреть статистику системы

#### 2️⃣ Через Python код
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

#### 3️⃣ Интеграция в чат-бот
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

### 🎯 Практические сценарии

#### 📚 Корпоративная база знаний
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

#### 🎓 Образовательный помощник
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

#### 🔧 Техподдержка
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

### 🔧 Полезные команды для работы

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

### 📊 Мониторинг и метрики

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

### 🚨 Устранение проблем

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

### 🎉 Результат

После выполнения туториала у вас будет:

✅ **Работающая RAG система** с веб-интерфейсом  
✅ **База знаний** с тестовыми документами  
✅ **REST API** для интеграции в приложения  
✅ **Примеры использования** на Python  
✅ **Мониторинг и логирование**  

Система готова для:
- 📚 Создания корпоративных баз знаний
- 🎓 Образовательных платформ  
- 🤖 Умных чат-ботов
- 🔧 Систем техподдержки
- 💼 Продуктовых решений

## 🔧 Конфигурация

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

Система использует продвинутый system prompt из файла `prompts/mini-rag-system-prompt.txt` со следующими функциями:

- ✅ Чёткое разделение ролей
- ✅ Переменные для RAG-контекста  
- ✅ Форматирование ответа с цитированием
- ✅ Правила отказа и самопроверка
- ✅ Многязычность (русский/английский)

## 🐳 Docker Services

| Сервис | Порт | Описание |
|--------|------|----------|
| api | 8000 | FastAPI приложение |
| llama-server | 8080 | LLaMA inference сервер |
| postgres | 5432 | PostgreSQL + pgvector |
| redis | 6379 | Кэш для эмбеддингов |

## 📊 Мониторинг

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

## 🧪 Тестирование

```bash
# Запуск тестов
python -m pytest tests/ -v

# Тестирование API
python -m pytest tests/test_api.py -v

# Тестирование RAG pipeline
python -m pytest tests/test_rag.py -v
```

## 🛠️ Разработка

### Локальная разработка

1. **Установите зависимости**
```bash
pip install -r requirements.txt
```

2. **Запустите зависимости в Docker**
```bash
docker-compose up -d postgres redis llama-server
```

3. **Создайте таблицы**
```bash
python -c "from app.core.database import create_tables; create_tables()"
```

4. **Запустите API**
```bash
python main.py
```

### Добавление новых компонентов

1. **Сервисы**: Добавьте в `app/services/`
2. **API роуты**: Расширьте `app/api/routes.py`
3. **Модели данных**: Обновите `app/models/`
4. **Конфигурация**: Дополните `app/core/config.py`

## 📈 Производительность

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