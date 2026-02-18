"""
Модуль конфигурации логирования.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log"):
    """
    Настраивает корневой логгер:
    - вывод в консоль (уровень из log_level)
    - вывод в файл (уровень DEBUG, чтобы записывать всё)

    :param log_level: Уровень логирования для консоли
    :param log_file: Путь к файлу логов
    """
    # Создаём папку для логов, если её нет
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Уровень в числовом виде
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(
        logging.DEBUG
    )  # корневой логгер принимает всё, а обработчики фильтруют

    # Формат логов
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Файловый обработчик (все уровни)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Если у корневого логгера уже есть обработчики (например, при перезагрузке), удалим их
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Отключаем лишние логи от библиотек (по желанию)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
