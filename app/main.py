"""
Точка входа в приложение FastAPI.

Этот файл отвечает за инициализацию приложения FastAPI, настройку жизненного цикла
(lifespan), подключение маршрутов и определение базовых эндпоинтов.
"""

import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.session import engine
from app.db.init_db import init_database
from app.core.config import settings
from app.core.logger_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления жизненным циклом приложения.

    Выполняет инициализацию при запуске и очистку ресурсов при завершении.
    """
    # Startup
    print("🚀 Стартуем приложение...")

    # Настраиваем логирование с уровнем из настроек
    setup_logging(settings.LOG_LEVEL)

    # Получаем логгер для текущего модуля
    logger = logging.getLogger(__name__)
    logger.info("Приложение запускается...")

    # Инициализируем базу данных
    init_database(engine)

    print("✅ База данных готова")

    yield  # Приложение работает

    # Shutdown
    # ОБУЧЕНИЕ: При завершении приложения освобождаем все соединения с БД
    # Это предотвращает утечки соединений
    print("🛑 Завершаем приложение...")
    engine.dispose()
    print("👋 Все ресурсы освобождены")


# ОБУЧЕНИЕ: Создаем экземпляр FastAPI приложения с указанием lifespan функции
# и названием приложения "Hybrid RAG"
app = FastAPI(title="Hybrid RAG", lifespan=lifespan)

# ОБУЧЕНИЕ: Импортируем маршруты из модулей API
# Это делается после создания app для избежания циклических импортов
from app.api import documents, search, monitoring

# ОБУЧЕНИЕ: Подключаем маршруты к основному приложению
# Это позволяет использовать эндпоинты из documents и search модулей
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(monitoring.router)
# app.include_router(ask.router)
# app.include_router(admin.router)
# app.include_router(user.router)


if __name__ == "__main__":
    # ОБУЧЕНИЕ: Запуск приложения с помощью uvicorn при прямом вызове файла
    # Используем настройки из конфигурации приложения
    import uvicorn

    uvicorn.run(
        app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.SERVER_RELOAD,
    )
