"""Unit tests for shared configuration."""

from shared.config.settings import Settings


def test_default_settings() -> None:
    """Settings should load with sensible defaults even without .env."""
    s = Settings()
    assert s.app_name == "flowmatch-recsys"
    assert s.embedding_dim == 384
    assert s.candidate_top_k == 500
    assert s.final_top_k == 10


def test_postgres_url() -> None:
    s = Settings()
    assert "postgresql+asyncpg://" in s.postgres_url
    assert "recsys" in s.postgres_url


def test_redis_url() -> None:
    s = Settings()
    assert s.redis_url.startswith("redis://")
