"""
Event Collector Service -- ingests real-time user interaction events.

Endpoints:
  POST /events        -- ingest a single event
  POST /events/batch  -- ingest multiple events at once
  GET  /events/stats  -- producer statistics and health
"""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException

from services.event_collector.app.producer import EventProducer
from services.event_collector.app.schemas import (
    EventBatch,
    EventResponse,
    UserEvent,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: initialize Kafka producer. Shutdown: flush and close."""
    logger.info("event_collector.starting")
    producer = EventProducer(local_mode=True)  # Set to False when Kafka is available
    app.state.producer = producer
    logger.info("event_collector.ready")
    yield
    producer.close()
    logger.info("event_collector.shutdown")


app = FastAPI(
    title="Event Collector Service",
    description="Ingests real-time user interaction events for the recommendation system",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/events", response_model=EventResponse)
async def collect_event(event: UserEvent) -> EventResponse:
    """Ingest a single user event."""
    producer: EventProducer = app.state.producer

    # Generate event_id if not provided
    if not event.event_id:
        event.event_id = str(uuid.uuid4())

    event_dict = event.model_dump()
    event_dict["timestamp"] = event.timestamp.isoformat()

    success = producer.send_event(event_dict)
    if not success:
        raise HTTPException(status_code=503, detail="Event ingestion temporarily unavailable")

    logger.info(
        "event.received",
        event_type=event.event_type.value,
        user_id=event.user_id,
        item_id=event.item_id,
    )

    return EventResponse(status="accepted", events_received=1)


@app.post("/events/batch", response_model=EventResponse)
async def collect_batch(batch: EventBatch) -> EventResponse:
    """Ingest a batch of user events."""
    producer: EventProducer = app.state.producer

    event_dicts = []
    for event in batch.events:
        if not event.event_id:
            event.event_id = str(uuid.uuid4())
        d = event.model_dump()
        d["timestamp"] = event.timestamp.isoformat()
        event_dicts.append(d)

    accepted = producer.send_batch(event_dicts)

    logger.info("event_batch.received", count=accepted, total=len(batch.events))

    return EventResponse(
        status="accepted",
        events_received=accepted,
        message=f"Accepted {accepted}/{len(batch.events)} events",
    )


@app.get("/events/stats")
async def get_stats() -> dict:
    """Return producer statistics."""
    producer: EventProducer = app.state.producer
    return producer.get_stats()
