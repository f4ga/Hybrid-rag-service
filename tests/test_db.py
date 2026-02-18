import pytest
from sqlalchemy.exc import SQLAlchemyError
from app.db.models import Document, Chunk
from app.db.session import get_db


def test_document_creation(db):
    """Тест создания документа в базе данных"""
    # Создаем новый документ
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Проверяем, что документ был создан
    assert document.id is not None
    assert document.filename == "test.txt"
    assert document.file_type == "txt"

    # Проверяем, что можно получить документ из базы данных
    retrieved_document = db.query(Document).filter(Document.id == document.id).first()
    assert retrieved_document is not None
    assert retrieved_document.filename == "test.txt"
    assert retrieved_document.file_type == "txt"


def test_chunk_creation(db):
    """Тест создания чанка в базе данных"""
    # Создаем документ, к которому будет привязан чанк
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Создаем чанк
    chunk = Chunk(
        document_id=document.id,
        text="Test chunk content",
        position=0,
        length=18,
        embedding=[0.1] * 384,  # Fix dimension to match Vector(384)
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)

    # Проверяем, что чанк был создан
    assert chunk.id is not None
    assert chunk.document_id == document.id
    assert chunk.text == "Test chunk content"
    assert chunk.position == 0
    assert chunk.length == 18
    # Проверяем, что embedding имеет правильную длину
    assert len(chunk.embedding) == 384
    # Проверяем, что все значения равны 0.1
    assert all(abs(x - 0.1) < 1e-6 for x in chunk.embedding)

    # Проверяем, что можно получить чанк из базы данных
    retrieved_chunk = db.query(Chunk).filter(Chunk.id == chunk.id).first()
    assert retrieved_chunk is not None
    assert retrieved_chunk.document_id == document.id
    assert retrieved_chunk.text == "Test chunk content"
    assert retrieved_chunk.position == 0
    assert retrieved_chunk.length == 18
    # Проверяем, что embedding имеет правильную длину
    assert len(retrieved_chunk.embedding) == 384
    # Проверяем, что все значения равны 0.1
    assert all(abs(x - 0.1) < 1e-6 for x in retrieved_chunk.embedding)


def test_document_cascade_delete(db):
    """Тест каскадного удаления чанков при удалении документа"""
    # Создаем документ
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Создаем чанки, привязанные к документу
    chunk1 = Chunk(
        document_id=document.id,
        text="Test chunk content 1",
        position=0,
        length=20,
        embedding=[0.1] * 384,  # Fix dimension to match Vector(384)
    )
    chunk2 = Chunk(
        document_id=document.id,
        text="Test chunk content 2",
        position=1,
        length=20,
        embedding=[0.4] * 384,  # Fix dimension to match Vector(384)
    )
    db.add_all([chunk1, chunk2])
    db.commit()

    # Проверяем, что чанки были созданы
    chunks = db.query(Chunk).filter(Chunk.document_id == document.id).all()
    assert len(chunks) == 2

    # Удаляем документ
    db.delete(document)
    db.commit()

    # Проверяем, что чанки также были удалены
    chunks = db.query(Chunk).filter(Chunk.document_id == document.id).all()
    assert len(chunks) == 0


def test_document_relationships(db):
    """Тест связей между документами и чанками"""
    # Создаем документ
    document = Document(filename="test.txt", file_type="txt")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Создаем чанки, привязанные к документу
    chunk1 = Chunk(
        document_id=document.id,
        text="Test chunk content 1",
        position=0,
        length=20,
        embedding=[0.1] * 384,  # Fix dimension to match Vector(384)
    )
    chunk2 = Chunk(
        document_id=document.id,
        text="Test chunk content 2",
        position=1,
        length=20,
        embedding=[0.4] * 384,  # Fix dimension to match Vector(384)
    )
    db.add_all([chunk1, chunk2])
    db.commit()
    db.refresh(document)

    # Проверяем, что у документа есть связь с чанками
    assert len(document.chunks) == 2
    assert document.chunks[0].text == "Test chunk content 1"
    assert document.chunks[1].text == "Test chunk content 2"

    # Проверяем, что у чанков есть связь с документом
    assert chunk1.document.filename == "test.txt"
    assert chunk2.document.filename == "test.txt"


def test_db_connection(db):
    """Тест подключения к базе данных"""
    # Проверяем, что подключение к базе данных работает
    try:
        from sqlalchemy import text

        db.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        pytest.fail(f"Database connection failed: {e}")
