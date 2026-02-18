import logging
import tiktoken


# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Инициализирует чанкер с заданными параметрами.
        Чанкер используется для разделения большого текста на более мелкие части (чанки),
        что необходимо для обработки больших документов в системах машинного обучения.
        Перекрытие между чанками помогает сохранить контекст между частями текста.

        :param chunk_size: Максимальный размер чанка в токенах (по умолчанию 1000)
        :param overlap: Количество токенов перекрытия между чанками (по умолчанию 200)
        """
        logger.info(
            f"Инициализация Chunker с chunk_size={chunk_size}, overlap={overlap}"
        )

        # Сохраняем параметры чанкера
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Инициализируем энкодер для работы с токенами
        # cl100k_base - это кодировка, используемая в моделях OpenAI
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def split(self, text: str) -> list[str]:
        """
        Разделяет текст на чанки с учетом перекрытия.
        Алгоритм работает по абзацам, чтобы сохранить логическую структуру текста.
        Если абзац не помещается в текущий чанк, он начинает новый чанк,
        при этом сохраняя перекрытие с предыдущим чанком.

        :param text: Входной текст для разделения на чанки
        :return: Список чанков (строк), на которые был разделен текст
        """
        logger.debug("Начало разбиения текста на чанки")

        # Разделяем текст на абзацы по двойным переносам строк
        # Это помогает сохранить логическую структуру документа
        paragraphs = text.split("\n\n")
        logger.debug(f"Текст разделен на {len(paragraphs)} абзацев")

        chunks = []
        current_chunk = ""
        current_tokens = 0

        # Обрабатываем каждый абзац
        for paragraph in paragraphs:
            # Подсчитываем количество токенов в абзаце
            # Токены - это минимальные единицы текста, которые понимает модель
            paragraph_tokens = len(self.encoder.encode(paragraph))

            # Проверяем, помещается ли абзац в текущий чанк
            if current_tokens + paragraph_tokens <= self.chunk_size:
                # Если абзац помещается, добавляем его к текущему чанку
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    # Если это первый абзац в чанке, просто присваиваем его
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
            else:
                # Если абзац не помещается, сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)

                # Начинаем новый чанк с перекрытием
                if current_chunk:
                    # Получаем последние overlap токенов из текущего чанка
                    # Это обеспечивает плавный переход между чанками
                    current_chunk_tokens = self.encoder.encode(current_chunk)
                    # Выбираем последние токены для перекрытия
                    overlap_tokens = (
                        current_chunk_tokens[-self.overlap :]
                        if len(current_chunk_tokens) > self.overlap
                        else current_chunk_tokens
                    )
                    # Декодируем токены обратно в текст
                    overlap_text = self.encoder.decode(overlap_tokens)
                    # Новый чанк начинается с перекрытия и нового абзаца
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    # Если текущий чанк пуст, начинаем с нового абзаца
                    current_chunk = paragraph

                # Подсчитываем токены в новом чанке
                current_tokens = len(self.encoder.encode(current_chunk))

        # Добавляем последний чанк, если он не пуст
        # Это необходимо, так как последний чанк не обрабатывается в основном цикле
        if current_chunk:
            chunks.append(current_chunk)

        logger.debug(f"Разбиение завершено. Получено {len(chunks)} чанков")
        return chunks
