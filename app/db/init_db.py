"""
Модуль для инициализации базы данных.

Содержит функции для настройки расширений, таблиц и индексов в PostgreSQL.
"""

from sqlalchemy import text, Engine
from app.db.models import Base


def init_database(engine: Engine) -> None:
    """
    Инициализирует базу данных: создает расширения, таблицы и индексы.

    Args:
        engine: Движок SQLAlchemy для подключения к БД
    """
    # ОБУЧЕНИЕ: Создаем расширение pgvector в PostgreSQL для работы с векторами
    # Это необходимо для хранения и поиска эмбеддингов
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # ОБУЧЕНИЕ: Создаем все таблицы в базе данных на основе SQLAlchemy моделей
    # Это происходит при каждом запуске, но SQLAlchemy не пересоздает существующие таблицы
    Base.metadata.create_all(bind=engine)

    # Подготовка для полнотекстового поиска
    # ОБУЧЕНИЕ: Настраиваем полнотекстовый поиск с использованием tsvector
    # Удаляем старый индекс и столбец, если они существуют (для пересоздания)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_chunks_text_search_vector"))
        conn.execute(
            text("ALTER TABLE chunks DROP COLUMN IF EXISTS text_search_vector")
        )

        # ОБУЧЕНИЕ: Добавляем столбец text_search_vector, который автоматически
        # генерируется из текста с использованием to_tsvector для английского языка
        # Это позволяет эффективно выполнять полнотекстовый поиск
        conn.execute(
            text(
                """
            ALTER TABLE chunks ADD COLUMN text_search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        """
            )
        )

        # ОБУЧЕНИЕ: Создаем GIN индекс для ускорения полнотекстового поиска
        # GIN (Generalized Inverted Index) хорошо подходит для tsvector
        conn.execute(
            text(
                """
            CREATE INDEX idx_chunks_text_search_vector
            ON chunks USING GIN(text_search_vector)
        """
            )
        )
