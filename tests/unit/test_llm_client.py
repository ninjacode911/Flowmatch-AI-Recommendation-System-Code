"""Unit tests for the LLM client (template/local mode)."""

from services.llm_augment_svc.app.main import LLMClient


def test_llm_client_local_mode() -> None:
    client = LLMClient(local_mode=True)
    assert client.local_mode is True
    assert client._client is None


def test_template_explanations() -> None:
    client = LLMClient(local_mode=True)
    items = [
        {"title": "Headphones", "category": "electronics", "score": 0.95, "price": 79.99},
        {"title": "Sneakers", "category": "sports", "score": 0.85, "price": 120.0},
        {"title": "Tea Set", "category": "food", "score": 0.5, "price": 12.99},
    ]
    explanations = client.generate_explanations(items)
    assert len(explanations) == 3
    assert "Top pick" in explanations[0]  # first item always gets "Top pick"
    assert "sports" in explanations[1].lower() or "relevant" in explanations[1].lower()


def test_template_explanations_low_price() -> None:
    client = LLMClient(local_mode=True)
    items = [
        {"title": "First", "category": "toys", "score": 0.95, "price": 10.0},
        {"title": "Cheap Item", "category": "food", "score": 0.3, "price": 5.99},
    ]
    explanations = client.generate_explanations(items)
    assert "value" in explanations[1].lower() or "food" in explanations[1].lower()


def test_parse_query_search() -> None:
    client = LLMClient(local_mode=True)
    result = client.parse_query("blue running shoes")
    assert result["intent"] == "search"
    assert result["original_query"] == "blue running shoes"
    assert result["refined_query"] == "blue running shoes"


def test_parse_query_recommendation() -> None:
    client = LLMClient(local_mode=True)
    result = client.parse_query("recommend me the best electronics")
    assert result["intent"] == "recommendation"
    assert "electronics" in result["categories"]


def test_parse_query_comparison() -> None:
    client = LLMClient(local_mode=True)
    result = client.parse_query("compare these two products vs each other")
    assert result["intent"] == "comparison"


def test_parse_query_question() -> None:
    client = LLMClient(local_mode=True)
    result = client.parse_query("what are good organic snacks?")
    assert result["intent"] == "question"
    assert "organic" in result["attributes"]


def test_parse_query_budget_detection() -> None:
    client = LLMClient(local_mode=True)
    result = client.parse_query("cheap affordable electronics")
    assert result["price_range"] == "budget"
    assert "electronics" in result["categories"]


def test_parse_query_premium_detection() -> None:
    client = LLMClient(local_mode=True)
    result = client.parse_query("premium luxury headphones")
    assert result["price_range"] == "premium"
    assert "premium" in result["attributes"]


def test_parse_query_attribute_detection() -> None:
    client = LLMClient(local_mode=True)
    result = client.parse_query("wireless waterproof compact speaker")
    assert "wireless" in result["attributes"]
    assert "waterproof" in result["attributes"]
    assert "compact" in result["attributes"]
