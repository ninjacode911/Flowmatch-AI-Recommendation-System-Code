"""
User schema — represents a user profile with their features.
"""

from pydantic import BaseModel, Field


class User(BaseModel):
    """A user in the system."""

    user_id: str
    age: int | None = None
    gender: str | None = None
    location: str | None = None
    preferred_categories: list[str] = Field(default_factory=list)
    interaction_count: int = 0
