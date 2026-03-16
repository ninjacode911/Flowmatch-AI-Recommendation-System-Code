"""
Centralized configuration using pydantic-settings.

All services import Settings from here. Values are loaded from environment
variables (or .env file). This ensures a single source of truth for config
across the entire monorepo.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── General ──
    app_name: str = "flowmatch-recsys"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ── PostgreSQL ──
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "recsys"
    postgres_password: str = "recsys_dev"
    postgres_db: str = "recsys"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── Qdrant ──
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "item_embeddings"

    # ── Model Serving ──
    embedding_dim: int = 384  # MiniLM default output dimension
    candidate_top_k: int = 500  # ANN retrieval count
    final_top_k: int = 10  # Items returned to client

    # ── LLM (Phase 3) ──
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
