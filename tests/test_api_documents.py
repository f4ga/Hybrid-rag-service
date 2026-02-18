import pytest
from unittest.mock import patch, MagicMock
import io


def test_upload_document_success(client):
    """Тест успешной загрузки документа"""
    # Создаем тестовый файл
    file_content = b"Test document content for testing purposes."
    test_file = io.BytesIO(file_content)
    test_file.name = "test.txt"

    # Мокаем функции извлечения текста, чанкинга и эмбеддинга
    with patch("app.api.documents.extract_text") as mock_extract, patch(
        "app.api.documents.chunker.split"
    ) as mock_split, patch("app.api.documents.embedder.embed_documents") as mock_embed:

        # Настраиваем возвращаемые значения для моков
        mock_extract.return_value = "Test document content for testing purposes."
        mock_split.return_value = ["Test document content", "for testing purposes."]
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        # Отправляем POST запрос на загрузку документа
        response = client.post(
            "/documents/upload",
            files={"file": (test_file.name, test_file, "text/plain")},
        )

        # Проверяем успешный ответ
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["filename"] == test_file.name
        assert data["chunks_count"] == 2

        # Проверяем, что моки были вызваны с правильными аргументами
        mock_extract.assert_called_once()
        mock_split.assert_called_once_with(
            "Test document content for testing purposes."
        )
        mock_embed.assert_called_once_with(
            ["Test document content", "for testing purposes."]
        )


def test_upload_document_invalid_file_type(client):
    """Тест загрузки файла неподдерживаемого типа"""
    # Создаем тестовый файл неподдерживаемого типа
    file_content = b"\x00\x01\x02\x03"
    test_file = io.BytesIO(file_content)
    test_file.name = "test.bin"

    # Мокаем функцию определения MIME-типа
    with patch("magic.from_buffer") as mock_magic:
        mock_magic.return_value = "application/octet-stream"

        # Отправляем POST запрос на загрузку документа
        response = client.post(
            "/documents/upload",
            files={"file": (test_file.name, test_file, "application/octet-stream")},
        )

        # Проверяем, что получили ошибку 400
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Неподдерживаемый тип файла" in data["detail"]


def test_upload_document_db_error(client):
    """Тест ошибки базы данных при загрузке документа"""
    # Создаем тестовый файл
    file_content = b"Test document content for testing purposes."
    test_file = io.BytesIO(file_content)
    test_file.name = "test.txt"

    # Мокаем функции сервисов
    with patch("app.services.parser.extract_text") as mock_extract, patch(
        "app.services.chuncker.Chunker.split"
    ) as mock_split, patch(
        "app.services.embedder.Embedder.embed_documents"
    ) as mock_embed:

        # Настраиваем возвращаемые значения для моков
        mock_extract.return_value = "Test document content for testing purposes."
        mock_split.return_value = ["Test document content", "for testing purposes."]
        # Настраиваем мок эмбеддинга для выброса исключения
        mock_embed.side_effect = Exception("Database connection error")

        # Отправляем POST запрос на загрузку документа
        response = client.post(
            "/documents/upload",
            files={"file": (test_file.name, test_file, "text/plain")},
        )

        # Проверяем, что получили ошибку 500
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection error" in data["detail"]
