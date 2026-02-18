"""
Модуль для работы с сессией базы данных.

Этот файл отвечает за создание движка SQLAlchemy, настройку сессий
и предоставление зависимости для FastAPI.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

# ОБУЧЕНИЕ: Создаем движок базы данных SQLAlchemy
# Используем строку подключения из настроек приложения
# Для PostgreSQL не нужны дополнительные параметры connect_args
# (в отличие от SQLite, где нужен check_same_thread=False)
logger.info("Создание движка базы данных")
engine = create_engine(settings.DATABASE_URL)

# ОБУЧЕНИЕ: Создаем фабрику сессий SQLAlchemy
# autocommit=False: отключаем автоматическую фиксацию транзакций
# autoflush=False: отключаем автоматическую отправку изменений в БД
# bind=engine: связываем сессии с нашим движком базы данных
logger.debug("Создание фабрики сессий")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Зависимость для FastAPI
def get_db():
    """
    Зависимость FastAPI для получения сессии базы данных.

    Создает новую сессию базы данных для каждого запроса и
    автоматически закрывает её после завершения обработки запроса.

    Yields:
        Session: Сессия базы данных SQLAlchemy
    """
    # ОБУЧЕНИЕ: Создаем новую сессию для каждого HTTP запроса
    # Это обеспечивает изоляцию транзакций между запросами
    logger.debug("Создание новой сессии базы данных")
    db = SessionLocal()
    try:
        # ОБУЧЕНИЕ: Передаем сессию в обработчик запроса
        # yield позволяет использовать сессию как генератор
        # FastAPI автоматически вызовет next() для получения сессии
        yield db
    finally:
        # ОБУЧЕНИЕ: Закрываем сессию после обработки запроса
        # Это освобождает соединение с базой данных и предотвращает утечки
        logger.debug("Закрытие сессии базы данных")
        db.close()
