"""Конфигурация базы данных."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator

from app.core.config import settings
from app.models.database import Base


# Синхронный движок
engine = create_engine(
    settings.database.url,
    echo=settings.database.echo,
    pool_pre_ping=settings.database.pool_pre_ping,
    pool_recycle=settings.database.pool_recycle,
)

# Асинхронный движок (для будущего использования)
async_database_url = settings.database.url.replace("postgresql://", "postgresql+asyncpg://")
async_engine = create_async_engine(
    async_database_url,
    echo=settings.database.echo,
    pool_pre_ping=settings.database.pool_pre_ping,
    pool_recycle=settings.database.pool_recycle,
)

# Фабрики сессий
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


def create_tables():
    """Создание всех таблиц."""
    Base.metadata.create_all(bind=engine)


async def create_tables_async():
    """Асинхронное создание всех таблиц."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Контекстный менеджер для получения сессии базы данных."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Асинхронный контекстный менеджер для получения сессии базы данных."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_db() -> Generator[Session, None, None]:
    """Dependency для FastAPI - получение сессии базы данных."""
    with get_db_session() as session:
        yield session


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Асинхронная dependency для FastAPI."""
    async with get_async_db_session() as session:
        yield session 