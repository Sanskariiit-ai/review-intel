from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "change_me_to_a_random_secret"

    DATABASE_URL: str = (
        "postgresql+asyncpg://review_intel:review_intel_pw@localhost:5432/review_intel"
    )

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "review_embeddings"

    REDIS_URL: str = "redis://localhost:6379/0"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # matches all-MiniLM-L6-v2 output size

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""


settings = Settings()
