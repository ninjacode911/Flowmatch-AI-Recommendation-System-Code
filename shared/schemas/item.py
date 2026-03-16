"""
Item schema — represents a product/content item in the catalogue.

This is the core entity that gets embedded, indexed, and recommended.
"""

from pydantic import BaseModel, Field


class Item(BaseModel):
    """An item in the catalogue (product, article, video, etc.)."""

    item_id: str
    title: str
    description: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    price: float | None = None
    image_url: str = ""
    rating: float = 0.0
    popularity_score: float = 0.0
    created_at: float | None = None  # Unix timestamp
