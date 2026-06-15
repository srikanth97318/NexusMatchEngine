import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "NexusMatch Engine"
    ENVIRONMENT: str = "development"

    # Databases
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/nexusmatch"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "nexusmatch_profiles"

    # Celery & Cache
    REDIS_URL: str = "redis://localhost:6379/0"

    # LLM Settings
    VLLM_API_URL: Optional[str] = None
    VLLM_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = "mock-key-for-development"

    # Embedding Model Settings
    BGE_MODEL_NAME: str = "BAAI/bge-m3"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
