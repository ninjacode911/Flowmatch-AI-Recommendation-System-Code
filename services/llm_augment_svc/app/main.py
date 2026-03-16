"""
LLM Augmentation Service -- adds natural language intelligence to recommendations.

Features:
  1. Explanation Generation: "Why was this recommended to you?"
  2. Query Understanding: Parse natural language queries into structured intent
  3. Conversational Recommendations: Multi-turn dialogue for refining preferences

This service calls an external LLM API (Claude, GPT, etc.) to generate
human-readable explanations and understand user intent.

Endpoints:
  POST /explain           -- generate explanations for recommendations
  POST /parse-query       -- parse a natural language query into structured intent
  GET  /health            -- health check
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from pydantic import BaseModel, Field

from shared.config.settings import settings

logger = structlog.get_logger()


# ── Request/Response Models ──────────────────────────────────────────────────


class RecommendedItemInput(BaseModel):
    item_id: str
    title: str
    category: str
    score: float
    price: float = 0.0


class ExplanationRequest(BaseModel):
    user_id: str
    items: list[RecommendedItemInput]
    user_context: str = Field(default="", description="Optional context about the user's preferences")


class ItemExplanation(BaseModel):
    item_id: str
    title: str
    explanation: str


class ExplanationResponse(BaseModel):
    user_id: str
    explanations: list[ItemExplanation]
    model_used: str


class QueryParseRequest(BaseModel):
    query: str
    user_id: str = ""


class ParsedQuery(BaseModel):
    original_query: str
    intent: str  # "search", "recommendation", "comparison", "question"
    categories: list[str]  # detected categories
    price_range: str  # "budget", "mid", "premium", "any"
    attributes: list[str]  # detected attributes like "wireless", "organic", "premium"
    refined_query: str  # cleaned-up query for embedding


class QueryParseResponse(BaseModel):
    parsed: ParsedQuery
    model_used: str


# ── LLM Client ───────────────────────────────────────────────────────────────


class LLMClient:
    """
    Wrapper for LLM API calls.

    Supports multiple backends:
      - Claude (Anthropic API)
      - Local/mock mode for development (no API calls)
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514", local_mode: bool = True) -> None:
        self.api_key = api_key
        self.model = model
        self.local_mode = local_mode
        self._client = None

        if not local_mode and api_key:
            self._init_client()
        else:
            logger.info("llm_client.local_mode", msg="Using template-based responses")

    def _init_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("llm_client.connected", model=self.model)
        except ImportError:
            logger.warning("llm_client.anthropic_not_installed", msg="Falling back to local mode")
            self.local_mode = True
        except Exception as e:
            logger.error("llm_client.init_failed", error=str(e))
            self.local_mode = True

    def generate_explanations(self, items: list[dict], user_context: str = "") -> list[str]:
        """Generate natural language explanations for recommended items."""
        if self.local_mode:
            return self._template_explanations(items, user_context)

        prompt = self._build_explanation_prompt(items, user_context)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            explanations = text.strip().split("\n")
            # Pad if fewer explanations than items
            while len(explanations) < len(items):
                explanations.append(f"Recommended based on your preferences")
            return explanations[: len(items)]
        except Exception as e:
            logger.error("llm_client.generate_failed", error=str(e))
            return self._template_explanations(items, user_context)

    def parse_query(self, query: str) -> dict:
        """Parse a natural language query into structured intent."""
        if self.local_mode:
            return self._template_parse(query)

        prompt = self._build_parse_prompt(query)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            import json

            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error("llm_client.parse_failed", error=str(e))
            return self._template_parse(query)

    def _template_explanations(self, items: list[dict], user_context: str) -> list[str]:
        """Generate rule-based explanations (no LLM needed)."""
        explanations = []
        for i, item in enumerate(items):
            category = item.get("category", "")
            score = item.get("score", 0)
            price = item.get("price", 0)

            if i == 0:
                reason = "Top pick based on your browsing history"
            elif score > 0.8:
                reason = f"Highly relevant to your interest in {category}"
            elif price < 20:
                reason = f"Great value find in {category}"
            else:
                reason = f"Popular in {category} -- others with similar taste loved this"

            explanations.append(reason)
        return explanations

    def _template_parse(self, query: str) -> dict:
        """Rule-based query parsing (no LLM needed)."""
        query_lower = query.lower()

        # Detect intent
        if any(w in query_lower for w in ["compare", "vs", "versus", "difference"]):
            intent = "comparison"
        elif any(w in query_lower for w in ["recommend", "suggest", "best", "top"]):
            intent = "recommendation"
        elif query_lower.endswith("?"):
            intent = "question"
        else:
            intent = "search"

        # Detect categories
        known_categories = ["electronics", "clothing", "food", "beauty", "sports", "home", "toys", "books"]
        categories = [c for c in known_categories if c in query_lower]

        # Detect price range
        if any(w in query_lower for w in ["cheap", "budget", "affordable", "under"]):
            price_range = "budget"
        elif any(w in query_lower for w in ["premium", "expensive", "luxury", "high-end"]):
            price_range = "premium"
        else:
            price_range = "any"

        # Detect attributes
        attribute_keywords = ["wireless", "organic", "premium", "compact", "lightweight", "waterproof", "eco", "smart"]
        attributes = [a for a in attribute_keywords if a in query_lower]

        return {
            "original_query": query,
            "intent": intent,
            "categories": categories,
            "price_range": price_range,
            "attributes": attributes,
            "refined_query": query.strip(),
        }

    def _build_explanation_prompt(self, items: list[dict], user_context: str) -> str:
        """Build prompt for explanation generation."""
        items_text = "\n".join(
            f"- {item.get('title', '')} ({item.get('category', '')}, ${item.get('price', 0):.2f}, score={item.get('score', 0):.2f})"
            for item in items
        )
        ctx = f"\nUser context: {user_context}" if user_context else ""
        return (
            f"Generate a short, friendly one-line explanation for why each item was recommended to this user.{ctx}\n\n"
            f"Items:\n{items_text}\n\n"
            f"Return exactly {len(items)} explanations, one per line. Keep each under 80 characters."
        )

    def _build_parse_prompt(self, query: str) -> str:
        """Build prompt for query parsing."""
        return (
            f'Parse this search query into structured JSON:\n\nQuery: "{query}"\n\n'
            "Return JSON with keys: intent (search/recommendation/comparison/question), "
            "categories (list), price_range (budget/mid/premium/any), "
            "attributes (list), refined_query (cleaned query string)."
        )


# ── FastAPI App ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("llm_augment_svc.starting")
    api_key = settings.llm_api_key if hasattr(settings, "llm_api_key") else ""
    local_mode = not bool(api_key)
    client = LLMClient(api_key=api_key, local_mode=local_mode)
    app.state.llm = client
    logger.info("llm_augment_svc.ready", local_mode=local_mode)
    yield
    logger.info("llm_augment_svc.shutdown")


app = FastAPI(
    title="LLM Augmentation Service",
    description="Adds natural language explanations and query understanding to recommendations",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/explain", response_model=ExplanationResponse)
async def explain_recommendations(body: ExplanationRequest) -> ExplanationResponse:
    """Generate natural language explanations for recommended items."""
    llm: LLMClient = app.state.llm

    items_dicts = [item.model_dump() for item in body.items]
    explanations = llm.generate_explanations(items_dicts, body.user_context)

    return ExplanationResponse(
        user_id=body.user_id,
        explanations=[
            ItemExplanation(item_id=item.item_id, title=item.title, explanation=exp)
            for item, exp in zip(body.items, explanations)
        ],
        model_used="template" if llm.local_mode else llm.model,
    )


@app.post("/parse-query", response_model=QueryParseResponse)
async def parse_query(body: QueryParseRequest) -> QueryParseResponse:
    """Parse a natural language query into structured intent."""
    llm: LLMClient = app.state.llm
    parsed = llm.parse_query(body.query)

    return QueryParseResponse(
        parsed=ParsedQuery(**parsed),
        model_used="template" if llm.local_mode else llm.model,
    )


@app.get("/health")
async def health() -> dict:
    llm: LLMClient = app.state.llm
    return {
        "status": "healthy",
        "local_mode": llm.local_mode,
        "model": llm.model if not llm.local_mode else "template",
    }
