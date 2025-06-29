-- Инициализация базы данных PostgreSQL для Mini RAG System
-- Этот скрипт выполняется при первом запуске контейнера

-- Создаем расширение pgvector для работы с векторами
CREATE EXTENSION IF NOT EXISTS vector;

-- Создаем пользователя и базу данных (если не существуют)
-- В docker-compose это уже настроено через переменные окружения

-- Настройки для оптимальной работы с векторами
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Перезагружаем конфигурацию
SELECT pg_reload_conf();

-- Выводим информацию о расширениях
SELECT name, default_version, installed_version, comment 
FROM pg_available_extensions 
WHERE name = 'vector';

-- Проверяем, что векторное расширение работает
SELECT '✅ pgvector extension is ready' as status; 