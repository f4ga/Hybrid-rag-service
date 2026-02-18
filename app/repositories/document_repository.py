"""
Репозиторий для работы с документами и чанками в базе данных.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db.models import Document, Chunk

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Репозиторий для работы с документами и чанками.

    Предоставляет методы для поиска, получения, сохранения и удаления документов и чанков.
    """

    def __init__(self, db: Session):
        """
        Инициализирует репозиторий с сессией базы данных.

        :param db: Сессия базы данных SQLAlchemy
        """
        logger.debug("Инициализация DocumentRepository")
        self.db = db

    def get_by_id(self, doc_id: int) -> Optional[Document]:
        """
        Получает документ по его идентификатору.

        :param doc_id: Идентификатор документа
        :return: Объект документа или None, если не найден
        """
        logger.debug(f"Получение документа по ID: {doc_id}")
        return self.db.query(Document).filter(Document.id == doc_id).first()

    def find_similar(
        self, embedding: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Находит чанки с похожими эмбеддингами.

        :param embedding: Вектор эмбеддинга для поиска
        :param limit: Максимальное количество результатов
        :return: Список похожих чанков с информацией о документах
        """
        logger.debug(f"Поиск похожих чанков. Limit: {limit}")

        sql = text(
            """
            SELECT c.id, c.text, d.filename,
                   1 - (c.embedding <=> CAST(:query_vec AS vector)) as similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> CAST(:query_vec AS vector)
            LIMIT :limit
            """
        )

        result = self.db.execute(sql, {"query_vec": embedding, "limit": limit})

        chunks = []
        for row in result:
            chunks.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "filename": row.filename,
                    "similarity": row.similarity,
                }
            )

        logger.debug(f"Найдено {len(chunks)} похожих чанков")
        return chunks

    def semantic_search(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Выполняет семантический поиск чанков.

        :param query_embedding: Вектор эмбеддинга запроса
        :param limit: Максимальное количество результатов
        :return: Список релевантных чанков с оценками схожести
        """
        logger.debug(f"Выполнение семантического поиска. Limit: {limit}")

        sql = text(
            """
            SELECT c.id, c.text, d.filename,
                   1 - (c.embedding <=> CAST(:query_vec AS vector)) as similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> CAST(:query_vec AS vector)
            LIMIT :limit
            """
        )

        result = self.db.execute(sql, {"query_vec": query_embedding, "limit": limit})

        chunks = []
        for row in result:
            chunks.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "filename": row.filename,
                    "similarity": float(row.similarity),
                }
            )

        logger.debug(f"Семантический поиск завершен. Найдено {len(chunks)} результатов")
        return chunks

    def lexical_search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Выполняет лексический (полнотекстовый) поиск чанков.

        :param query_text: Текстовый запрос для поиска
        :param limit: Максимальное количество результатов
        :return: Список релевантных чанков с оценками ранга
        """
        logger.debug(f"Выполнение лексического поиска: {query_text}. Limit: {limit}")

        sql = text(
            """
            SELECT c.id, c.text, d.filename,
                   ts_rank(c.text_search_vector, plainto_tsquery('english', :query)) as rank
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.text_search_vector @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
            """
        )

        result = self.db.execute(sql, {"query": query_text, "limit": limit})

        chunks = []
        for row in result:
            chunks.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "filename": row.filename,
                    "similarity": float(row.rank),
                }
            )

        logger.debug(f"Лексический поиск завершен. Найдено {len(chunks)} результатов")
        return chunks

    def hybrid_search_rrf(
        self,
        lexical_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        top_k: int = 10,
        alpha: float = 0.5,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Объединяет результаты лексического и семантического поиска с помощью RRF.

        :param lexical_results: Результаты лексического поиска
        :param semantic_results: Результаты семантического поиска
        :param top_k: Максимальное количество результатов
        :param alpha: Вес семантического поиска (0-1)
        :param k: Константа сглаживания для RRF
        :return: Объединенный список результатов с оценками релевантности
        """
        logger.debug(
            f"Объединение результатов поиска. Лексических: {len(lexical_results)}, семантических: {len(semantic_results)}"
        )

        # Словарь для хранения объединенных оценок релевантности
        scores = {}

        # Добавляем веса для лексического поиска
        for rank, item in enumerate(lexical_results, 1):
            # Вычисляем взвешенную оценку по формуле RRF
            score = (1 - alpha) * (1 / (k + rank))
            # Суммируем оценки для каждого чанка
            scores[item["id"]] = scores.get(item["id"], 0) + score

        # Добавляем веса для семантического поиска
        for rank, item in enumerate(semantic_results, 1):
            # Вычисляем взвешенную оценку по формуле RRF
            score = alpha * (1 / (k + rank))
            # Суммируем оценки для каждого чанка
            scores[item["id"]] = scores.get(item["id"], 0) + score

        # Сортируем по убыванию оценки и берем top_k результатов
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Получаем полную информацию о чанках из базы данных
        results = []
        for chunk_id, score in sorted_ids:
            # Запрашиваем чанк из базы данных по его id
            chunk = self.db.query(Chunk).filter(Chunk.id == chunk_id).first()
            if chunk:
                # Добавляем полную информацию о чанке в результаты
                results.append(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "filename": chunk.document.filename,
                        "similarity": score,
                    }
                )

        logger.debug(
            f"Объединение результатов завершено. Итоговое количество: {len(results)}"
        )
        return results

    def save_document(self, filename: str, file_type: str) -> Document:
        """
        Сохраняет новый документ в базе данных.

        :param filename: Имя файла
        :param file_type: Тип файла
        :return: Созданный объект документа
        """
        logger.debug(f"Сохранение документа: {filename} (тип: {file_type})")

        document = Document(filename=filename, file_type=file_type)
        self.db.add(document)
        self.db.flush()  # Получаем ID документа без коммита

        logger.debug(f"Документ сохранен с ID: {document.id}")
        return document

    def save_chunk(
        self, document_id: int, text: str, position: int, embedding: List[float]
    ) -> Chunk:
        """
        Сохраняет новый чанк в базе данных.

        :param document_id: Идентификатор документа
        :param text: Текст чанка
        :param position: Позиция чанка в документе
        :param embedding: Векторное представление чанка
        :return: Созданный объект чанка
        """
        logger.debug(
            f"Сохранение чанка для документа ID {document_id}. Позиция: {position}"
        )

        chunk = Chunk(
            document_id=document_id,
            text=text,
            position=position,
            length=len(text),
            embedding=embedding,
        )
        self.db.add(chunk)
        self.db.flush()  # Получаем ID чанка без коммита

        logger.debug(f"Чанк сохранен с ID: {chunk.id}")
        return chunk

    def delete_document(self, doc_id: int) -> bool:
        """
        Удаляет документ и все связанные с ним чанки.

        :param doc_id: Идентификатор документа
        :return: True, если документ был удален, False если не найден
        """
        logger.debug(f"Удаление документа с ID: {doc_id}")

        document = self.db.query(Document).filter(Document.id == doc_id).first()
        if document:
            self.db.delete(document)
            logger.debug(f"Документ с ID {doc_id} удален")
            return True
        logger.warning(f"Документ с ID {doc_id} не найден для удаления")
        return False
