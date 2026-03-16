"""Health check endpoints for the API gateway and downstream services."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Basic liveness check — returns 200 if the gateway is running."""
    return {"status": "healthy", "service": "api-gateway"}


@router.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    """
    Readiness check — verifies downstream dependencies are reachable.
    TODO (Phase 1): Check PostgreSQL, Redis, and Qdrant connectivity.
    """
    return {"status": "ready"}
