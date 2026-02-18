"""
Модуль конфигурации приложения.

Этот файл отвечает за загрузку и управление настройками приложения,
включая параметры подключения к базе данных и другие конфигурационные переменные.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """
    Класс для хранения и управления конфигурацией приложения.

    Использует Pydantic для валидации и загрузки настроек из переменных окружения.
    """

    # Параметры подключения к базе данных PostgreSQL
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "hybrid_rag")

    # Параметры сервера
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    SERVER_RELOAD: bool = os.getenv("SERVER_RELOAD", "True").lower() == "true"

    # Параметры логирования
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def DATABASE_URL(self) -> str:
        """
        Свойство для формирования строки подключения к базе данных.

        Returns:
            str: Строка подключения к PostgreSQL
        """
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


# Создаем экземпляр Settings для использования в приложении
settings = Settings()
