"""Конфигурация приложения."""

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseModel):
    """Настройки базы данных."""
    url: str = "postgresql://minirag:minirag123@localhost:5432/minirag"
    echo: bool = False
    pool_pre_ping: bool = True
    pool_recycle: int = 3600


class Settings(BaseSettings):
    """Основные настройки приложения."""
    
    # API настройки
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # System Prompt
    system_prompt_file: str = "./prompts/mini-rag-system-prompt.txt"
    
    # Компоненты
    database: DatabaseSettings = DatabaseSettings()
    
    class Config:
        env_file = ".env"


# Глобальный объект настроек
settings = Settings() 