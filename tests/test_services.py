import pytest
import numpy as np
from app.services.parser import extract_text, clean_text
from app.services.chuncker import Chunker
from app.services.embedder import Embedder
from unittest.mock import patch, MagicMock


def test_extract_text_pdf():
    """Тест извлечения текста из PDF файла"""
    # Создаем тестовое содержимое PDF файла
    pdf_content = b"%PDF-1.4\x0a%\x0a1 0 obj\x0a<<\x0a/Type /Catalog\x0a/Pages 2 0 R\x0a>>\x0aendobj\x0a2 0 obj\x0a<<\x0a/Type /Pages\x0a/Kids [3 0 R]\x0a/Count 1\x0a>>\x0aendobj\x0a3 0 obj\x0a<<\x0a/Type /Page\x0a/Parent 2 0 R\x0a/MediaBox [0 0 612 792]\x0a/Contents 4 0 R\x0a/Resources <<\x0a/ProcSet [/PDF /Text]\x0a>>\x0a>>\x0aendobj\x0a4 0 obj\x0a<<\x0a/Length 44\x0a>>\x0astream\x0aBT\x0a/F1 12 Tf\x0a72 720 Td\x0a(Test PDF content) Tj\x0aET\x0aendstream\x0aendobj\x0axref\x0a0 5\x0a0000000000 65535 f \x0a0000000015 00000 n \x0a0000000060 00000 n \x0a0000000111 00000 n \x0a0000000231 00000 n \x0atrailer\x0a<<\x0a/Size 5\x0a/Root 1 0 R\x0a>>\x0astartxref\x0a325\x0a%%EOF"

    # Мокаем PdfReader для возврата тестовых страниц
    with patch("app.services.parser.pypdf.PdfReader") as mock_pdf_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf_reader.return_value.pages = [mock_page]

        # Вызываем функцию извлечения текста
        result = extract_text(pdf_content, "test.pdf")

        # Проверяем результат
        assert result == "Test PDF content"


def test_extract_text_docx():
    """Тест извлечения текста из DOCX файла"""
    # Создаем тестовое содержимое DOCX файла
    docx_content = b"PK\x03\x04\x14\x00\x00\x00\x08\x00"

    # Мокаем Document для возврата тестовых параграфов
    with patch("app.services.parser.docx.Document") as mock_document:
        mock_paragraph = MagicMock()
        mock_paragraph.text = "Test DOCX content"
        mock_document.return_value.paragraphs = [mock_paragraph]

        # Вызываем функцию извлечения текста
        result = extract_text(docx_content, "test.docx")

        # Проверяем результат
        assert result == "Test DOCX content\n"


def test_extract_text_txt():
    """Тест извлечения текста из TXT файла"""
    # Создаем тестовое содержимое TXT файла
    txt_content = b"Test TXT content with special characters: \xc3\xa9\xc3\xa0\xc3\xb6"

    # Вызываем функцию извлечения текста
    result = extract_text(txt_content, "test.txt")

    # Проверяем результат
    assert result == "Test TXT content with special characters: éàö"


def test_extract_text_invalid_type():
    """Тест извлечения текста из файла неподдерживаемого типа"""
    # Создаем тестовое содержимое файла неподдерживаемого типа
    invalid_content = b"\x00\x01\x02\x03"

    # Мокаем magic.from_buffer для возврата неподдерживаемого MIME-типа
    with patch("app.services.parser.magic.from_buffer") as mock_magic:
        mock_magic.return_value = "application/octet-stream"

        # Проверяем, что функция выбрасывает исключение
        with pytest.raises(ValueError) as exc_info:
            extract_text(invalid_content, "test.bin")

        # Проверяем сообщение об ошибке
        assert "Неподдерживаемый тип файла" in str(exc_info.value)


def test_clean_text():
    """Тест функции очистки текста"""
    # Создаем тестовый текст с ANSI escape последовательностями и управляющими символами
    dirty_text = "\x1b[31mRed text\x1b[0m\x01\x02\x03\n\tValid text\x04\x05"

    # Вызываем функцию очистки текста
    result = clean_text(dirty_text)

    # Проверяем результат
    assert result == "Red text\n\tValid text"


def test_chunker_split():
    """Тест функции разбиения текста на чанки"""
    # Создаем тестовый текст
    text = "This is a test document. It contains multiple sentences. " * 100

    # Создаем экземпляр Chunker
    chunker = Chunker(chunk_size=100, overlap=20)

    # Мокаем encoder для возврата фиксированного количества токенов
    with patch.object(chunker, "encoder") as mock_encoder:
        mock_encoder.encode.return_value = list(range(50))  # 50 токенов
        mock_encoder.decode.return_value = "overlap text"

        # Вызываем функцию разбиения
        result = chunker.split(text)

        # Проверяем результат
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)


def test_embedder_embed_documents():
    """Тест функции создания эмбеддингов для документов"""
    # Создаем тестовые данные
    texts = ["Test document 1", "Test document 2"]

    # Создаем экземпляр Embedder
    embedder = Embedder()

    # Мокаем модель для возврата тестовых эмбеддингов
    with patch.object(embedder, "model") as mock_model:
        mock_model.encode.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]

        # Вызываем функцию создания эмбеддингов
        result = embedder.embed_documents(texts)

        # Проверяем результат
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(
            isinstance(value, float) for embedding in result for value in embedding
        )


@patch("app.services.embedder.SentenceTransformer")
def test_embedder_embed_query(mock_transformer):
    """Тест функции создания эмбеддинга для запроса"""
    # Создаем тестовые данные
    query = "Test query"

    # Настраиваем мок для SentenceTransformer
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])
    mock_transformer.return_value = mock_model

    # Создаем экземпляр Embedder
    embedder = Embedder()

    # Вызываем функцию создания эмбеддинга
    result = embedder.embed_query(query)

    # Проверяем результат
    assert isinstance(result, list)
    assert result == [0.7, 0.8, 0.9]
