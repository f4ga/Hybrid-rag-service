import logging
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.embedder import Embedder
from app.services.hybrid_search import HybridSearcher
from app.repositories.document_repository import DocumentRepository
from app.schemas.search import (
    SearchResult,
    LexicalSearchResponse,
    SemanticSearchResponse,
    HybridSearchResponse,
)

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

# Создаем глобальный экземпляр эмбеддера
embedder = Embedder()

router = APIRouter(
    prefix="/search",
    tags=["search"],
)


@router.get(
    "/lexical",
    description="Выполняет лексический (полнотекстовый) поиск по чанкам документов.",
)
def lexical_search(
    query: str = Query(..., min_length=1, description="Поисковый запрос"),
    limit: int = Query(
        5, ge=1, le=100, description="Максимальное количество результатов"
    ),
    db: Session = Depends(get_db),
):
    """
    Выполняет лексический (полнотекстовый) поиск по чанкам документов.

    :param query: Поисковый запрос
    :param limit: Максимальное количество результатов
    :param db: Сессия базы данных
    :return: Список релевантных чанков
    """
    logger.info(f"Выполнение лексического поиска: {query}")

    if not query.strip():
        logger.warning("Пустой поисковый запрос")
        return []

    # Используем DocumentRepository для выполнения поиска
    repository = DocumentRepository(db)
    results = repository.lexical_search(query, limit)

    logger.info(f"Лексический поиск завершен. Найдено результатов: {len(results)}")

    # Преобразуем результаты в Pydantic модели
    search_results = [SearchResult(**result) for result in results]
    return LexicalSearchResponse(results=search_results)


@router.get("/hybrid", description="Выполняет гибридный поиск по чанкам документов.")
def hybrid_search(
    query: str = Query(..., min_length=1, description="Поисковый запрос"),
    limit: int = Query(
        10, ge=1, le=100, description="Максимальное количество результатов"
    ),
    db: Session = Depends(get_db),
):
    """
    Выполняет гибридный поиск по чанкам документов.

    :param query: Поисковый запрос
    :param limit: Максимальное количество результатов
    :param db: Сессия базы данных
    :return: Список релевантных чанков
    """
    logger.info(f"Выполнение гибридного поиска: {query}")

    searcher = HybridSearcher(db, embedder)
    results = searcher.search(query, limit)

    logger.info(f"Гибридный поиск завершен. Найдено результатов: {len(results)}")

    # Преобразуем результаты в Pydantic модели
    search_results = [SearchResult(**result) for result in results]
    return HybridSearchResponse(results=search_results)


@router.get(
    "/semantic", description="Выполняет семантический поиск по чанкам документов."
)
def semantic_search(
    query: str = Query(..., min_length=1, description="Поисковый запрос"),
    limit: int = Query(
        5, ge=1, le=100, description="Максимальное количество результатов"
    ),
    db: Session = Depends(get_db),
):
    """
    Выполняет семантический поиск по чанкам документов.

    :param query: Поисковый запрос
    :param limit: Максимальное количество результатов
    :param db: Сессия базы данных
    :return: Список релевантных чанков
    """
    logger.info(f"Выполнение семантического поиска: {query}")

    # Получаем эмбеддинг запроса
    query_vec = embedder.embed_query(query)

    # Используем DocumentRepository для выполнения поиска
    repository = DocumentRepository(db)
    results = repository.semantic_search(query_vec, limit)

    logger.info(f"Семантический поиск завершен. Найдено результатов: {len(results)}")

    # Преобразуем результаты в Pydantic модели
    search_results = [SearchResult(**result) for result in results]
    return SemanticSearchResponse(results=search_results)
