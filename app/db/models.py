"""
Модели данных для SQLAlchemy.

Этот файл определяет структуру таблиц в базе данных с использованием SQLAlchemy ORM.
Основные сущности: Document (документ) и Chunk (фрагмент текста).
"""

import logging
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

# ОБУЧЕНИЕ: Создаем базовый класс для всех моделей SQLAlchemy
# От него будут наследоваться все наши модели
logger.debug("Создание базового класса для моделей SQLAlchemy")
Base = declarative_base()


class Document(Base):
    """
    Модель документа.

    Представляет собой файл, загруженный в систему. Один документ может
    содержать множество фрагментов (Chunk).
    """

    __tablename__ = "documents"

    # ОБУЧЕНИЕ: Определяем поля таблицы документов
    # id - уникальный идентификатор с индексом для быстрого поиска
    id = Column(Integer, primary_key=True)
    # filename - имя файла (не может быть пустым)
    filename = Column(String, nullable=False, index=True)
    # file_type - тип файла (например, 'pdf', 'txt')
    file_type = Column(String, nullable=False, index=True)

    # Связь с чанками: один документ → много чанков
    # ОБУЧЕНИЕ: Определяем связь один-ко-многим с моделью Chunk
    # back_populates="document" создает двустороннюю связь с полем document в Chunk
    # cascade="all, delete-orphan" обеспечивает каскадное удаление чанков
    # при удалении документа (если чанк не связан с другими документами)
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """
    Модель фрагмента текста (чанка).

    Представляет собой часть документа, которая хранится отдельно для
    возможности поиска и обработки. Каждый чанк связан с документом.
    """

    __tablename__ = "chunks"

    # ... существующие поля
    # ОБУЧЕНИЕ: Столбец для хранения векторного представления текста
    # Vector(384) означает, что вектор будет иметь 384 компонента
    # Это соответствует размерности эмбеддингов модели all-MiniLM-L6-v2
    # TODO: Fix dimension mismatch - возможно, нужно изменить размерность
    # в зависимости от используемой модели эмбеддингов
    embedding = Column(Vector(384))  # TODO: Fix dimension mismatch

    # ОБУЧЕНИЕ: Основные поля чанка
    # id - уникальный идентификатор с индексом
    id = Column(Integer, primary_key=True, index=True)
    # document_id - внешний ключ, связывающий чанк с документом
    # ondelete="CASCADE" обеспечивает автоматическое удаление чанков
    # при удалении родительского документа
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    # text - текстовое содержимое чанка (не может быть пустым)
    text = Column(String, nullable=False)
    # position - позиция чанка в оригинальном документе
    position = Column(Integer, nullable=False)
    # length - длина текста чанка в символах
    length = Column(Integer, nullable=False)

    # ОБУЧЕНИЕ: Определяем обратную связь с моделью Document
    # back_populates="chunks" создает двустороннюю связь с полем chunks в Document
    document = relationship("Document", back_populates="chunks")
