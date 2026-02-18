import pytest
from unittest.mock import patch, MagicMock
from collections import namedtuple
from app.main import app
from app.db.models import Document, Chunk


def test_semantic_search_success(client):
    """Тест успешного семантического поиска"""
    # Создаем тестовые данные
    test_chunks = [
        {
            "id": 1,
            "text": "This is a test document content.",
            "filename": "test.txt",
            "similarity": 0.95,
        },
        {
            "id": 2,
            "text": "Another test document content.",
            "filename": "test2.txt",
            "similarity": 0.85,
        },
    ]

    # Мокаем функцию создания эмбеддинга для запроса
    with patch("app.services.embedder.Embedder.embed_query") as mock_embed_query:

        # Настраиваем моки
        mock_embed_query.return_value = [0.1] * 384

        # Создаём мок сессии и подменяем зависимость get_db
        mock_db = MagicMock()
        mock_result = MagicMock()
        Row = namedtuple("Row", ["id", "text", "filename", "similarity"])
        mock_result.__iter__.return_value = [
            Row(
                id=1,
                text="This is a test document content.",
                filename="test.txt",
                similarity=0.95,
            ),
            Row(
                id=2,
                text="Another test document content.",
                filename="test2.txt",
                similarity=0.85,
            ),
        ]
        mock_db.execute.return_value = mock_result

        def override_get_db():
            yield mock_db

        from app.db.session import get_db

        app.dependency_overrides[get_db] = override_get_db

        # Отправляем GET запрос на семантический поиск
        response = client.get("/search/semantic?query=test&limit=5")

        app.dependency_overrides.clear()

        # Проверяем успешный ответ
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "results" in data
        results = data["results"]
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["text"] == "This is a test document content."
        assert results[0]["filename"] == "test.txt"
        assert results[0]["similarity"] == 0.95
        assert results[1]["id"] == 2
        assert results[1]["text"] == "Another test document content."
        assert results[1]["filename"] == "test2.txt"
        assert results[1]["similarity"] == 0.85

        # Проверяем, что моки были вызваны с правильными аргументами
        mock_embed_query.assert_called_once_with("test")


def test_semantic_search_empty_query(client):
    """Тест семантического поиска с пустым запросом"""
    # Отправляем GET запрос с пустым запросом
    response = client.get("/search/semantic?query=&limit=5")

    # Проверяем, что получили ошибку 422 (валидация)
    assert response.status_code == 422


def test_semantic_search_no_results(client):
    """Тест семантического поиска без результатов"""
    # Мокаем функцию создания эмбеддинга для запроса
    with patch("app.services.embedder.Embedder.embed_query") as mock_embed_query:

        # Настраиваем моки
        mock_embed_query.return_value = [0.1] * 384

        # Создаём мок сессии и подменяем зависимость get_db
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__.return_value = []
        mock_db.execute.return_value = mock_result

        def override_get_db():
            yield mock_db

        from app.db.session import get_db

        app.dependency_overrides[get_db] = override_get_db

        # Отправляем GET запрос на семантический поиск
        response = client.get("/search/semantic?query=nonexistent&limit=5")

        app.dependency_overrides.clear()

        # Проверяем успешный ответ с пустым списком
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "results" in data
        assert len(data["results"]) == 0


def test_lexical_search_success(client, db):
    """Тест успешного лексического поиска"""
    # Создаем документ
    doc = Document(filename="test.txt", file_type="txt")
    db.add(doc)
    db.flush()

    # Чанки
    chunk1 = Chunk(
        document_id=doc.id,
        text="Python programming language",
        position=0,
        length=10,
        embedding=[0.1] * 384,
    )
    chunk2 = Chunk(
        document_id=doc.id,
        text="Java programming language",
        position=1,
        length=10,
        embedding=[0.1] * 384,
    )
    db.add_all([chunk1, chunk2])
    db.commit()

    response = client.get("/search/lexical?query=python&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "results" in data
    results = data["results"]
    assert len(results) > 0
    assert any("python" in item["text"].lower() for item in results)


def test_lexical_search_empty(client):
    """Тест лексического поиска с отсутствующим словом"""
    response = client.get("/search/lexical?query=nonexistentword&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "results" in data
    assert len(data["results"]) == 0


def test_hybrid_search_success(client, db):
    """Тест успешного гибридного поиска"""
    # Создаем документ
    doc = Document(filename="test.txt", file_type="txt")
    db.add(doc)
    db.flush()

    # Чанки
    chunk1 = Chunk(
        document_id=doc.id,
        text="Python programming language",
        position=0,
        length=10,
        embedding=[0.1] * 384,
    )
    chunk2 = Chunk(
        document_id=doc.id,
        text="Java programming language",
        position=1,
        length=10,
        embedding=[0.1] * 384,
    )
    db.add_all([chunk1, chunk2])
    db.commit()

    response = client.get("/search/hybrid?query=python&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "results" in data
    results = data["results"]
    # Проверяем структуру ответа
    if len(results) > 0:
        assert "id" in results[0]
        assert "text" in results[0]
        assert "filename" in results[0]
        assert "similarity" in results[0]


def test_hybrid_search_no_results(client):
    """Тест гибридного поиска без результатов"""
    response = client.get("/search/hybrid?query=nonexistentquery&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "results" in data
    assert len(data["results"]) == 0
