# Dockerfile для LLaMA сервера
# Образ llama.cpp переехал из ggerganov в ggml-org; бинарь теперь /app/llama-server
FROM ghcr.io/ggml-org/llama.cpp:server

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем скрипт запуска
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Создаем директорию для моделей
RUN mkdir -p /models

# Устанавливаем curl для health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Переменные окружения
ENV MODEL_PATH=/models/llama-3-8b-instruct.gguf
ENV HOST=0.0.0.0
ENV PORT=8080
ENV N_GPU_LAYERS=0
ENV N_CTX=4096
ENV N_PARALLEL=4

# Открываем порт
EXPOSE 8080

# Проверка здоровья
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Запускаем сервер
ENTRYPOINT ["/app/entrypoint.sh"] 