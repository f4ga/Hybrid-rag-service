import logging
from sentence_transformers import SentenceTransformer


# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        """
        Инициализирует эмбеддер с предобученной моделью.
        Эмбеддер используется для преобразования текста в числовые векторы (эмбеддинги),
        которые могут быть использованы для поиска и сравнения семантической близости текстов.
        Модель paraphrase-MiniLM-L3-v2 выбрана за свою эффективность и качество
        для задач семантического поиска.
        """
        logger.info("Инициализация модели SentenceTransformer")

        # Загружаем предобученную модель SentenceTransformer
        # Эта модель специализируется на создании эмбеддингов для задач перефразирования
        # и семантического поиска
        self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        # Получаем размерность вектора эмбеддинга
        # Это необходимо для конфигурации векторной базы данных
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Модель инициализирована. Размерность эмбеддингов: {self.dimension}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Создает эмбеддинги для списка документов.
        Используется для индексации документов в векторной базе данных.
        Префикс "passage: " добавляется для улучшения качества поиска
        в моделях, обученных с использованием префиксов.

        :param texts: Список текстов документов для преобразования в эмбеддинги
        :return: Список эмбеддингов (численных векторов) для каждого документа
        """
        logger.debug(f"Создание эмбеддингов для {len(texts)} документов")

        # Добавляем префикс "passage: " к каждому тексту
        # Это помогает модели лучше понимать, что мы индексируем документы
        prefixed_texts = ["passage: " + text for text in texts]

        # Создаем эмбеддинги для всех текстов одновременно
        # normalize_embeddings=True нормализует векторы до единичной длины,
        # что улучшает качество поиска
        embeddings = self.model.encode(prefixed_texts, normalize_embeddings=True)

        # Преобразуем результат в список списков float
        # Модель может возвращать разные типы данных в зависимости от окружения
        result = (
            embeddings.tolist()
            if hasattr(embeddings, "tolist")
            else [embedding.tolist() for embedding in embeddings]
        )

        logger.debug(
            f"Эмбеддинги созданы. Размерность: {len(result)} x {len(result[0])}"
        )
        return result

    def embed_query(self, text: str) -> list[float]:
        """
        Создает эмбеддинг для поискового запроса.
        Используется для преобразования пользовательского запроса в вектор
        для поиска по векторной базе данных.
        Префикс "query: " добавляется для согласования с префиксами документов.

        :param text: Текст поискового запроса пользователя
        :return: Эмбеддинг (численный вектор) запроса
        """
        logger.debug(f"Создание эмбеддинга для запроса: {text}")

        # Добавляем префикс "query: " к тексту запроса
        # Это помогает модели понять, что мы обрабатываем поисковый запрос
        prefixed_text = "query: " + text

        # Создаем эмбеддинг для запроса
        # Передаем список, так как encode ожидает итерируемый объект
        embedding = self.model.encode([prefixed_text], normalize_embeddings=True)

        # Возвращаем первый (и единственный) вектор как список
        # [0] извлекает вектор из списка, tolist() преобразует в список float
        result = embedding[0].tolist()

        logger.debug(f"Эмбеддинг запроса создан. Размерность: {len(result)}")
        return result
