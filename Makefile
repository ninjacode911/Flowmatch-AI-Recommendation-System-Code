.PHONY: help install dev test lint format docker-up docker-down run clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ========================
# Local Development
# ========================

install: ## Install production dependencies
	pip install -r requirements.txt

dev: ## Install all dependencies (production + dev)
	pip install -r requirements-dev.txt
	pre-commit install

run: ## Run the API gateway locally (no Docker)
	uvicorn services.api_gateway.app.main:app --reload --port 8000

# ========================
# Code Quality
# ========================

lint: ## Run linter (ruff)
	ruff check .

format: ## Auto-format code (ruff)
	ruff format .
	ruff check --fix .

typecheck: ## Run type checker (mypy)
	mypy shared/ services/ --ignore-missing-imports

# ========================
# Testing
# ========================

test: ## Run all tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=shared --cov=services --cov-report=term-missing

test-unit: ## Run only unit tests
	pytest tests/unit/ -v

test-integration: ## Run only integration tests
	pytest tests/integration/ -v

# ========================
# Docker
# ========================

docker-up: ## Start all services with Docker Compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-build: ## Rebuild all Docker images
	docker compose build --no-cache

docker-logs: ## Tail logs from all services
	docker compose logs -f

# ========================
# Cleanup
# ========================

clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
