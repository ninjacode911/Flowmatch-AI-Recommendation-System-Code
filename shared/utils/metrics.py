"""
Prometheus metrics for monitoring service health and performance.

Exposes standard RED metrics (Rate, Errors, Duration) for every endpoint:
  - request_count: total requests by method, path, status
  - request_duration: latency histogram by method, path
  - request_in_progress: currently active requests

Plus recommendation-specific metrics:
  - recommendation_latency: end-to-end recommendation time
  - candidates_retrieved: how many candidates the retrieval stage found
  - model_inference_duration: time spent in model scoring

Usage:
  from shared.utils.metrics import PrometheusMiddleware, METRICS_ROUTE
  app.add_middleware(PrometheusMiddleware)
  app.add_route(METRICS_ROUTE, metrics_endpoint)
"""

import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# ── Standard RED Metrics ──────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

REQUEST_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Currently active HTTP requests",
    ["method", "path"],
)

# ── Recommendation-Specific Metrics ───────────────────────────────────────────

RECOMMENDATION_LATENCY = Histogram(
    "recommendation_latency_seconds",
    "End-to-end recommendation generation time",
    ["strategy"],  # query_based, history_based, popular_fallback
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

CANDIDATES_RETRIEVED = Histogram(
    "candidates_retrieved",
    "Number of candidates from retrieval stage",
    buckets=[10, 25, 50, 100, 200, 500],
)

MODEL_INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Time spent in model scoring",
    ["model"],  # two_tower, ncf, ltr
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5],
)

EVENT_INGESTION_COUNT = Counter(
    "events_ingested_total",
    "Total events ingested",
    ["event_type"],
)

FEATURE_STORE_LATENCY = Histogram(
    "feature_store_latency_seconds",
    "Feature store read latency",
    ["operation"],  # get_user, get_item, get_session
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05],
)


# ── Middleware ────────────────────────────────────────────────────────────────


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Automatically tracks request metrics for all endpoints."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        path = request.url.path

        # Skip metrics endpoint itself
        if path == "/metrics":
            return await call_next(request)

        REQUEST_IN_PROGRESS.labels(method=method, path=path).inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.perf_counter() - start
            REQUEST_COUNT.labels(method=method, path=path, status_code=status).inc()
            REQUEST_DURATION.labels(method=method, path=path).observe(duration)
            REQUEST_IN_PROGRESS.labels(method=method, path=path).dec()

        return response


# ── Metrics Endpoint ──────────────────────────────────────────────────────────

METRICS_ROUTE = "/metrics"


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus scrape endpoint. Returns all metrics in Prometheus text format."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
