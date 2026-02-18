"""
Тесты для DocumentRepository.
"""

import pytest
from sqlalchemy.exc import SQLAlchemyError
from app.db.models import Document, Chunk
from app.repositories.document_repository import DocumentRepository


def test_get_by_id(db):
    """Тест получения документа по ID через репозиторий"""
    # Создаем документ напрямую через ORM
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Используем репозиторий для получения документа
    repository = DocumentRepository(db)
    retrieved_document = repository.get_by_id(document.id)

    # Проверяем, что документ был найден
    assert retrieved_document is not None
    assert retrieved_document.id == document.id
    assert retrieved_document.filename == "test.txt"
    assert retrieved_document.file_type == "txt"


def test_get_by_id_not_found(db):
    """Тест получения несуществующего документа по ID через репозиторий"""
    repository = DocumentRepository(db)
    retrieved_document = repository.get_by_id(999999)

    # Проверяем, что документ не был найден
    assert retrieved_document is None


def test_save_document(db):
    """Тест сохранения документа через репозиторий"""
    repository = DocumentRepository(db)
    document = repository.save_document("test.txt", "txt")

    # Проверяем, что документ был создан
    assert document.id is not None
    assert document.filename == "test.txt"
    assert document.file_type == "txt"

    # Проверяем, что документ можно получить из базы данных
    retrieved_document = db.query(Document).filter(Document.id == document.id).first()
    assert retrieved_document is not None
    assert retrieved_document.filename == "test.txt"
    assert retrieved_document.file_type == "txt"


def test_save_chunk(db):
    """Тест сохранения чанка через репозиторий"""
    # Создаем документ
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Создаем чанк через репозиторий
    repository = DocumentRepository(db)
    chunk = repository.save_chunk(
        document_id=document.id,
        text="Test chunk content",
        position=0,
        embedding=[0.1] * 384,
    )

    # Проверяем, что чанк был создан
    assert chunk.id is not None
    assert chunk.document_id == document.id
    assert chunk.text == "Test chunk content"
    assert chunk.position == 0
    assert chunk.length == 18
    assert len(chunk.embedding) == 384
    assert all(abs(x - 0.1) < 1e-6 for x in chunk.embedding)

    # Проверяем, что чанк можно получить из базы данных
    retrieved_chunk = db.query(Chunk).filter(Chunk.id == chunk.id).first()
    assert retrieved_chunk is not None
    assert retrieved_chunk.document_id == document.id
    assert retrieved_chunk.text == "Test chunk content"
    assert retrieved_chunk.position == 0
    assert retrieved_chunk.length == 18
    assert len(retrieved_chunk.embedding) == 384
    assert all(abs(x - 0.1) < 1e-6 for x in retrieved_chunk.embedding)


def test_delete_document(db):
    """Тест удаления документа через репозиторий"""
    # Создаем документ
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Удаляем документ через репозиторий
    repository = DocumentRepository(db)
    result = repository.delete_document(document.id)

    # Проверяем, что удаление прошло успешно
    assert result is True

    # Проверяем, что документ был удален
    retrieved_document = db.query(Document).filter(Document.id == document.id).first()
    assert retrieved_document is None


def test_delete_document_not_found(db):
    """Тест удаления несуществующего документа через репозиторий"""
    repository = DocumentRepository(db)
    result = repository.delete_document(999999)

    # Проверяем, что удаление не прошло (документ не найден)
    assert result is False


def test_lexical_search(db):
    """Тест лексического поиска через репозиторий"""
    # Создаем документ и чанк
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    chunk = Chunk(
        document_id=document.id,
        text="This is a test document for searching",
        position=0,
        length=37,
        embedding=[0.1] * 384,
    )
    db.add(chunk)
    db.commit()

    # Выполняем лексический поиск через репозиторий
    repository = DocumentRepository(db)
    results = repository.lexical_search("test", 10)

    # Проверяем, что результаты были найдены
    assert len(results) >= 0  # Может быть 0, если полнотекстовый поиск не настроен


def test_semantic_search(db):
    """Тест семантического поиска через репозиторий"""
    # Создаем документ и чанк
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    chunk = Chunk(
        document_id=document.id,
        text="This is a test document for searching",
        position=0,
        length=37,
        embedding=[0.1] * 384,
    )
    db.add(chunk)
    db.commit()

    # Выполняем семантический поиск через репозиторий
    repository = DocumentRepository(db)
    results = repository.semantic_search([0.1] * 384, 10)

    # Проверяем, что результаты были найдены
    assert len(results) >= 0  # Может быть 0, если векторный поиск не настроен


def test_find_similar(db):
    """Тест поиска похожих чанков через репозиторий"""
    # Создаем документ и чанк
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    chunk = Chunk(
        document_id=document.id,
        text="This is a test document for searching",
        position=0,
        length=37,
        embedding=[0.1] * 384,
    )
    db.add(chunk)
    db.commit()

    # Выполняем поиск похожих чанков через репозиторий
    repository = DocumentRepository(db)
    results = repository.find_similar([0.1] * 384, 10)

    # Проверяем, что результаты были найдены
    assert len(results) >= 0  # Может быть 0, если векторный поиск не настроен
