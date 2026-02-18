import logging
from sqlalchemy.orm import Session
from app.db.models import Chunk
from app.services.embedder import Embedder
from app.repositories.document_repository import DocumentRepository
from typing import List, Dict, Any


# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(self, db: Session, embedder: Embedder):
        """
        Инициализирует гибридный поиск с заданными параметрами.
        Гибридный поиск объединяет лексический (полнотекстовый) и семантический поиск
        для повышения качества результатов поиска.

        :param db: Сессия базы данных SQLAlchemy
        :param embedder: Экземпляр эмбеддера для создания векторных представлений текста
        """
        logger.info("Инициализация HybridSearcher")

        self.db = db
        self.embedder = embedder
        # Константа сглаживания для алгоритма RRF (Reciprocal Rank Fusion)
        # Используется для предотвращения доминирования рангов с высокими значениями
        self.k = 60
        # Вес семантического поиска в объединении результатов (от 0 до 1)
        # 0.5 означает равный вес лексического и семантического поиска
        self.alpha = 0.5

        logger.debug(f"Параметры поиска: k={self.k}, alpha={self.alpha}")

    def _lexical_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Выполняет лексический (полнотекстовый) поиск в базе данных.
        Использует PostgreSQL полнотекстовый поиск с ранжированием по релевантности.

        :param query: Поисковый запрос пользователя
        :param top_k: Максимальное количество результатов
        :return: Список словарей с id чанков и их рангами релевантности
        """
        logger.debug(f"Выполнение лексического поиска: {query}")

        # Используем DocumentRepository для выполнения поиска
        repository = DocumentRepository(self.db)
        results = repository.lexical_search(query, top_k)

        logger.debug(f"Лексический поиск завершен. Найдено {len(results)} результатов")

        # Преобразуем результаты в формат, ожидаемый методом _rrf_merge
        return [{"id": r["id"], "rank": r["similarity"]} for r in results]

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Выполняет семантический поиск и возвращает список словарей с ключами id и similarity.
        Семантический поиск находит документы с похожим смыслом, а не только с совпадающими словами.

        :param query: Поисковый запрос пользователя
        :param top_k: Максимальное количество результатов
        :return: Список словарей с id чанков и их семантической близостью
        """
        logger.debug(f"Выполнение семантического поиска: {query}")

        # Создаем векторное представление запроса с помощью эмбеддера
        query_vec = self.embedder.embed_query(query)

        # Используем DocumentRepository для выполнения поиска
        repository = DocumentRepository(self.db)
        results = repository.semantic_search(query_vec, top_k)

        logger.debug(
            f"Семантический поиск завершен. Найдено {len(results)} результатов"
        )

        # Преобразуем результаты в формат, ожидаемый методом _rrf_merge
        return [{"id": r["id"], "similarity": r["similarity"]} for r in results]

    def _rrf_merge(
        self, lexical: List[Dict[str, Any]], semantic: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Реализует объединение результатов по алгоритму RRF (Reciprocal Rank Fusion).
        RRF объединяет ранги из разных поисковых систем без необходимости калибровки.
        Формула: score = 1 / (k + rank), где k - константа сглаживания.

        :param lexical: Результаты лексического поиска
        :param semantic: Результаты семантического поиска
        :param top_k: Максимальное количество результатов в финальном выводе
        :return: Объединенный список результатов с оценками релевантности
        """
        logger.debug(
            f"Объединение результатов поиска. Лексических: {len(lexical)}, семантических: {len(semantic)}"
        )

        # Словарь для хранения объединенных оценок релевантности
        scores = {}

        # Добавляем веса для лексического поиска
        # enumerate(lexical, 1) дает нам ранг, начиная с 1
        for rank, item in enumerate(lexical, 1):
            # Вычисляем взвешенную оценку по формуле RRF
            # (1 - self.alpha) - вес лексического поиска
            score = (1 - self.alpha) * (1 / (self.k + rank))
            # Суммируем оценки для каждого чанка
            scores[item["id"]] = scores.get(item["id"], 0) + score

        # Добавляем веса для семантического поиска
        for rank, item in enumerate(semantic, 1):
            # Вычисляем взвешенную оценку по формуле RRF
            # self.alpha - вес семантического поиска
            score = self.alpha * (1 / (self.k + rank))
            # Суммируем оценки для каждого чанка
            scores[item["id"]] = scores.get(item["id"], 0) + score

        # Сортируем по убыванию оценки и берем top_k результатов
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Получаем полную информацию о чанках из базы данных через репозиторий
        repository = DocumentRepository(self.db)
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

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Основной метод гибридного поиска.
        Объединяет результаты лексического и семантического поиска.

        :param query: Поисковый запрос пользователя
        :param top_k: Максимальное количество результатов (по умолчанию 10)
        :return: Список наиболее релевантных чанков с оценками релевантности
        """
        logger.info(f"Выполнение гибридного поиска: {query}")

        # Получаем результаты обоих поисков
        # Запрашиваем в 2 раза больше результатов для лучшего объединения
        lexical_results = self._lexical_search(query, top_k * 2)
        semantic_results = self._semantic_search(query, top_k * 2)

        # Объединяем результаты с помощью алгоритма RRF
        results = self._rrf_merge(lexical_results, semantic_results, top_k)

        logger.info(f"Гибридный поиск завершен. Найдено {len(results)} результатов")
        return results
