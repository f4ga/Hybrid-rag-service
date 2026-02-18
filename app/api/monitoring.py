"""
Модуль для мониторинга и проверки состояния приложения.

Содержит эндпоинты для проверки работоспособности приложения и подключения к БД.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.repositories.document_repository import DocumentRepository

router = APIRouter()


@router.get("/")
def read_root():
    """
    Базовый эндпоинт для проверки работоспособности приложения.

    Returns:
        dict: Статус приложения
    """
    return {"status": "ok"}


# Эндпоинт для проверки подключения к базе данных
@router.get("/db-check")
def db_check(db: Session = Depends(get_db)):
    """
    Эндпоинт для проверки подключения к базе данных.

    Args:
        db: Зависимость для получения сессии базы данных

    Returns:
        dict: Статус подключения к базе данных
    """
    try:
        # Используем DocumentRepository для проверки подключения
        repository = DocumentRepository(db)
        # Выполняем простой запрос к базе данных через репозиторий
        # ОБУЧЕНИЕ: Проверяем подключение к БД с помощью выполнения SELECT 1
        # Это стандартный способ проверки живости соединения
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        # ОБУЧЕНИЕ: В случае ошибки возвращаем информацию об ошибке
        # Это помогает в диагностике проблем с подключением к БД
        return {"status": "error", "message": str(e)}


# Пустой роут для будущих эндпоинтов
@router.get("/health")
def health_check():
    """
    Эндпоинт для проверки состояния приложения.

    Returns:
        dict: Статус здоровья приложения
    """
    return {"status": "healthy"}
