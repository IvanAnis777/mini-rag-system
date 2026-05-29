# Makefile для Mini RAG System

# Переменные
PYTHON = python3
PIP = pip3
DOCKER_COMPOSE = docker-compose
PROJECT_NAME = mini-rag-system

# Цвета для вывода
BLUE = \033[34m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
NC = \033[0m # No Color

.PHONY: help install dev-setup build up down restart logs clean test lint format

# Помощь
help: ## Показать справку
	@echo "$(BLUE)Mini RAG System - Команды управления$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Установка зависимостей
install: ## Установить Python зависимости
	@echo "$(BLUE)📦 Устанавливаем зависимости...$(NC)"
	$(PIP) install -r requirements.txt

# Настройка среды разработки
dev-setup: ## Настроить среду разработки
	@echo "$(BLUE)🔧 Настраиваем среду разработки...$(NC)"
	cp config.env.example .env
	mkdir -p data/models
	mkdir -p logs
	@echo "$(GREEN)✅ Среда разработки готова!$(NC)"
	@echo "$(YELLOW)⚠️ Не забудьте скачать LLaMA модель в data/models/$(NC)"

# Сборка Docker образов
build: ## Собрать Docker образы
	@echo "$(BLUE)🔨 Собираем Docker образы...$(NC)"
	$(DOCKER_COMPOSE) build

# Запуск всех сервисов
up: ## Запустить все сервисы
	@echo "$(BLUE)🚀 Запускаем сервисы...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Сервисы запущены!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

# Остановка сервисов
down: ## Остановить все сервисы
	@echo "$(BLUE)🛑 Останавливаем сервисы...$(NC)"
	$(DOCKER_COMPOSE) down

# Перезапуск сервисов
restart: ## Перезапустить сервисы
	@echo "$(BLUE)🔄 Перезапускаем сервисы...$(NC)"
	$(DOCKER_COMPOSE) restart

# Просмотр логов
logs: ## Показать логи всех сервисов
	$(DOCKER_COMPOSE) logs -f

# Логи конкретного сервиса
logs-api: ## Показать логи API
	$(DOCKER_COMPOSE) logs -f api

logs-llama: ## Показать логи LLaMA сервера
	$(DOCKER_COMPOSE) logs -f llama-server

logs-db: ## Показать логи базы данных
	$(DOCKER_COMPOSE) logs -f postgres

logs-redis: ## Показать логи Redis
	$(DOCKER_COMPOSE) logs -f redis

# Проверка состояния
status: ## Проверить состояние сервисов
	@echo "$(BLUE)📊 Состояние сервисов:$(NC)"
	$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "$(BLUE)🏥 Health check:$(NC)"
	-curl -s http://localhost:8000/api/v1/health | jq '.' || echo "API недоступен"

# Создание таблиц БД (внутри контейнера api — там зависимости и доступ к postgres)
init-db: ## Создать таблицы в базе данных (нужен make up)
	@echo "$(BLUE)📊 Создаём таблицы базы данных...$(NC)"
	$(DOCKER_COMPOSE) exec -T api python -c "from app.core.database import create_tables; create_tables()"
	@echo "$(GREEN)✅ Таблицы созданы!$(NC)"

# Загрузка тестовых данных (внутри контейнера api)
load-test-data: ## Загрузить тестовые данные (нужен make up)
	@echo "$(BLUE)📝 Загружаем тестовые данные...$(NC)"
	$(DOCKER_COMPOSE) exec -T api python scripts/load_test_data.py
	@echo "$(GREEN)✅ Тестовые данные загружены!$(NC)"

# Тестирование
# Тесты гоняем ВНУТРИ контейнера api: там Python 3.11 + установленные зависимости
# (на хосте старые пины requirements не ставятся на новый Python). Нужен поднятый стек: make up
PYTEST = $(DOCKER_COMPOSE) exec -T api python -m pytest -o asyncio_mode=auto

test: ## Запустить тесты (в контейнере api; нужен make up)
	@echo "$(BLUE)🧪 Запускаем тесты...$(NC)"
	$(PYTEST) tests/ -v --color=yes

test-api: ## Тестировать только API
	$(PYTEST) tests/test_api.py -v --color=yes

test-rag: ## Тестировать RAG граф (corrective/self-RAG)
	$(PYTEST) tests/test_rag_graph.py -v --color=yes

# Линтинг
lint: ## Проверить код линтером
	@echo "$(BLUE)🔍 Проверяем код...$(NC)"
	flake8 app/ main.py --max-line-length=120
	black --check app/ main.py
	isort --check-only app/ main.py

# Форматирование кода
format: ## Отформатировать код
	@echo "$(BLUE)✨ Форматируем код...$(NC)"
	black app/ main.py
	isort app/ main.py
	@echo "$(GREEN)✅ Код отформатирован!$(NC)"

# Очистка
clean: ## Очистить временные файлы и контейнеры
	@echo "$(BLUE)🧹 Очищаем проект...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Проект очищен!$(NC)"

# Мониторинг
monitor: ## Показать системную статистику
	@echo "$(BLUE)📈 Системная статистика:$(NC)"
	-curl -s http://localhost:8000/api/v1/stats | jq '.' || echo "API недоступен"

# Скачивание модели (реестр GGUF). Пример: make download-model MODEL=qwen
MODEL ?= llama3
download-model: ## Скачать GGUF-модель из реестра: MODEL=llama2|qwen|mistral|llama3 (по умолч. llama3)
	@echo "$(BLUE)⬇️ Скачиваем модель: $(MODEL)$(NC)"
	@mkdir -p data/models
	@case "$(MODEL)" in \
	  llama2)  url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"; file="llama-2-7b-chat.q4_k_m.gguf";; \
	  qwen)    url="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"; file="qwen2.5-7b-instruct-q4_k_m.gguf";; \
	  mistral) url="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"; file="mistral-7b-instruct-v0.3-q4_k_m.gguf";; \
	  llama3)  url="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"; file="meta-llama-3.1-8b-instruct-q4_k_m.gguf";; \
	  *) echo "$(RED)❌ Неизвестная MODEL=$(MODEL). Доступно: llama2|qwen|mistral|llama3$(NC)"; exit 1;; \
	esac; \
	echo "$(YELLOW)Файл: $$file (докачка поддерживается, можно прервать и повторить)$(NC)"; \
	curl -L -C - -o "data/models/$$file" "$$url"; \
	touch .env; \
	if grep -q '^MODEL_FILE=' .env; then \
	  sed -i.bak "s|^MODEL_FILE=.*|MODEL_FILE=$$file|" .env && rm -f .env.bak; \
	else \
	  echo "MODEL_FILE=$$file" >> .env; \
	fi; \
	echo "$(GREEN)✅ Модель скачана, в .env прописан MODEL_FILE=$$file$(NC)"; \
	echo "$(BLUE)Теперь: make restart$(NC)"

# Резервное копирование
backup: ## Создать резервную копию данных
	@echo "$(BLUE)💾 Создаём резервную копию...$(NC)"
	mkdir -p backups
	docker exec mini-rag-system_postgres_1 pg_dump -U minirag minirag > backups/db_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Резервная копия создана!$(NC)"

# Восстановление из резервной копии
restore: ## Восстановить из последней резервной копии
	@echo "$(BLUE)📥 Восстанавливаем из резервной копии...$(NC)"
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)❌ Укажите файл: make restore BACKUP_FILE=backups/file.sql$(NC)"; \
		exit 1; \
	fi
	docker exec -i mini-rag-system_postgres_1 psql -U minirag minirag < $(BACKUP_FILE)
	@echo "$(GREEN)✅ Данные восстановлены!$(NC)"

# Обновление зависимостей
update-deps: ## Обновить зависимости
	@echo "$(BLUE)🔄 Обновляем зависимости...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✅ Зависимости обновлены!$(NC)"

# Shell в контейнере
shell-api: ## Открыть shell в API контейнере
	$(DOCKER_COMPOSE) exec api bash

shell-db: ## Открыть psql в контейнере БД
	$(DOCKER_COMPOSE) exec postgres psql -U minirag minirag

# Полная установка проекта
install-full: dev-setup build up init-db ## Полная установка проекта
	@echo "$(GREEN)🎉 Проект полностью установлен и запущен!$(NC)"
	@echo "$(BLUE)Следующие шаги:$(NC)"
	@echo "1. Скачайте модель: make download-model"
	@echo "2. Загрузите тестовые данные: make load-test-data"
	@echo "3. Откройте API документацию: http://localhost:8000/docs"

# По умолчанию показываем справку
.DEFAULT_GOAL := help 