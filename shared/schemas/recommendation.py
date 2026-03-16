"""
Shared request/response schemas for the recommendation API.

These Pydantic models define the data contract between the client and the
api-gateway, and between internal microservices. Every service speaks the
same language because they all import from here.
"""

from pydantic import BaseModel, Field


class SessionEvent(BaseModel):
    """A single user interaction event within the current session."""

    item_id: str
    event_type: str = Field(description="click | view | add_to_cart | purchase")
    timestamp: float | None = None


class RecommendationRequest(BaseModel):
    """
    What the client sends to POST /api/v1/recommend.

    - user_id: Who to recommend for (empty string = anonymous/cold-start)
    - session_events: Recent interactions in this session (real-time context)
    - query: Optional natural language query ("something relaxing for Sunday")
    - top_k: How many items to return
    """

    user_id: str = ""
    session_events: list[SessionEvent] = Field(default_factory=list)
    query: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendedItem(BaseModel):
    """A single recommended item with its score and explanation."""

    item_id: str
    score: float = Field(description="Final relevance score after ranking")
    title: str = ""
    category: str = ""
    explanation: str = Field(default="", description="Why this item was recommended")


class RecommendationResponse(BaseModel):
    """What the API returns to the client."""

    user_id: str
    items: list[RecommendedItem]
    model_version: str = ""
    explanation: str = Field(default="", description="Overall recommendation context")
