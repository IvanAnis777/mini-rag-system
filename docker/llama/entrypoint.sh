#!/bin/bash

# Скрипт запуска LLaMA сервера

echo "🚀 Starting LLaMA server..."

# Проверяем наличие модели
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo "Please download a LLaMA model and place it in the models directory"
    echo "Example: wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O $MODEL_PATH"
    exit 1
fi

echo "✅ Model found: $MODEL_PATH"
echo "🔧 Server configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Context size: $N_CTX"
echo "   GPU layers: $N_GPU_LAYERS"
echo "   Parallel requests: $N_PARALLEL"

# Запускаем LLaMA сервер (в ggml-org образе бинарь — /app/llama-server)
exec /app/llama-server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$N_CTX" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --parallel "$N_PARALLEL" \
    --threads "$(nproc)" \
    --batch-size 512 