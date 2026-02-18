"""
Модуль API для работы с документами.

Этот файл содержит эндпоинты для загрузки документов и их обработки:
извлечение текста, разбиение на чанки, создание эмбеддингов.
"""

import asyncio
import logging
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.models import Document, Chunk
from app.db.session import get_db
from app.services.parser import extract_text
from app.services.chuncker import Chunker
from app.services.embedder import Embedder
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentUploadResponse

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

# ОБУЧЕНИЕ: Создаем глобальные экземпляры сервисов чанкинга и эмбеддингов
# Эти экземпляры создаются один раз при запуске приложения и используются
# во всех запросах, что позволяет избежать повторной инициализации моделей
chunker = Chunker()
embedder = Embedder()

# ОБУЧЕНИЕ: Создаем роутер FastAPI с префиксом "/documents" и тегом "documents"
# Это позволяет группировать связанные эндпоинты и упрощает навигацию в документации
router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.post(
    "/upload",
    description="Загружает документ, извлекает из него текст, разбивает на чанки, создает эмбеддинги и сохраняет все в базе данных.",
)
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
):
    """
    Загружает документ, извлекает из него текст, разбивает на чанки,
    создает эмбеддинги и сохраняет все в базе данных.

    :param file: Загружаемый файл
    :param db: Сессия базы данных
    :return: Информация о загруженном документе
    """
    try:
        logger.info(f"Загрузка файла: {file.filename}")
        # ОБУЧЕНИЕ: Читаем содержимое файла в память
        # Для больших файлов это может быть проблемой памяти
        # Альтернатива: обрабатывать файл по частям или использовать потоковую передачу
        content = await file.read()

        # ОБУЧЕНИЕ: Извлекаем текст из файла с помощью соответствующего парсера
        # extract_text автоматически определяет тип файла и использует подходящий метод
        # asyncio.to_thread позволяет выполнять CPU-интенсивную операцию в отдельном потоке
        # чтобы не блокировать event loop FastAPI
        text = await asyncio.to_thread(extract_text, content, file.filename)

        # Создаем репозиторий для работы с документами
        repository = DocumentRepository(db)

        # ОБУЧЕНИЕ: Создаем запись документа в базе данных
        # file_type определяется по расширению файла
        doc = repository.save_document(file.filename, file.filename.split(".")[-1])

        # ОБУЧЕНИЕ: Разбиваем текст на чанки с помощью Chunker сервиса
        # Размер чанка и стратегия разбиения определяются в Chunker классе
        chunks = await asyncio.to_thread(chunker.split, text)

        # ОБУЧЕНИЕ: Создаем эмбеддинги для всех чанков
        # embedder.embed_documents может обрабатывать список текстов за один вызов
        # что более эффективно, чем создавать эмбеддинги по одному
        embeddings = await asyncio.to_thread(embedder.embed_documents, chunks)

        # ОБУЧЕНИЕ: Создаем записи чанков в базе данных через репозиторий
        chunk_records = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_record = repository.save_chunk(
                document_id=doc.id,
                text=chunk_text,
                position=i,
                embedding=embedding,
            )
            chunk_records.append(chunk_record)

        # ОБУЧЕНИЕ: Коммитим транзакцию
        # commit() завершает транзакцию и сохраняет все изменения в БД
        db.commit()

        # ОБУЧЕНИЕ: Возвращаем информацию о загруженном документе
        # Это позволяет клиенту узнать ID документа и количество созданных чанков
        logger.info(
            f"Документ {doc.filename} успешно загружен. Создано чанков: {len(chunks)}"
        )
        return DocumentUploadResponse(
            id=doc.id, filename=doc.filename, chunks_count=len(chunks)
        )

    except ValueError as e:
        # ОБУЧЕНИЕ: Обрабатываем ошибки парсинга (например, неподдерживаемый формат файла)
        # Преобразуем их в HTTP 400 ошибку, которая указывает на проблему с запросом клиента
        logger.error(f"Ошибка при парсинге файла {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # ОБУЧЕНИЕ: Обрабатываем другие ошибки (например, проблемы с БД)
        # rollback() отменяет все изменения в текущей транзакции
        # Это предотвращает сохранение частично обработанных данных
        logger.error(f"Ошибка при загрузке файла {file.filename}: {str(e)}")
        db.rollback()
        # ОБУЧЕНИЕ: Преобразуем внутренние ошибки в HTTP 500 ошибку
        # Это указывает на проблему на стороне сервера
        raise HTTPException(status_code=500, detail=str(e))
