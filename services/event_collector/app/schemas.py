"""
Event schemas for the event collection pipeline.

These define the contract between clients (web/mobile apps) and our
event ingestion system. Every user action (view, click, purchase) is
captured as an event and streamed through Kafka for real-time processing.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """All trackable user actions."""

    VIEW = "view"
    CLICK = "click"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PURCHASE = "purchase"
    SEARCH = "search"
    RATE = "rate"
    SHARE = "share"


class UserEvent(BaseModel):
    """A single user interaction event."""

    event_id: str = Field(default="", description="Unique event ID (generated server-side if empty)")
    user_id: str = Field(..., description="User who performed the action")
    item_id: str = Field(default="", description="Item involved (empty for search events)")
    event_type: EventType = Field(..., description="Type of interaction")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional context
    session_id: str = Field(default="", description="Browser/app session ID")
    query: str = Field(default="", description="Search query (for search events)")
    position: int = Field(default=0, description="Position in the list where user clicked")
    source: str = Field(default="unknown", description="Where the event originated (homepage, search, recs)")

    # Optional metadata
    device: str = Field(default="unknown", description="Device type: web, ios, android")
    price: float = Field(default=0.0, description="Item price at time of event")


class EventBatch(BaseModel):
    """Batch of events for bulk ingestion."""

    events: list[UserEvent]


class EventResponse(BaseModel):
    """Response after event ingestion."""

    status: str = "accepted"
    events_received: int = 0
    message: str = ""
