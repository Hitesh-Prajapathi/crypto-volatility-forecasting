SYMBOL ?= BTCUSDT
START  ?= 2017-08-17

.DEFAULT_GOAL := help

.PHONY: help setup lint format typecheck test test-all cov data quality backtest report \
	up down kind-up kind-down helm-lint hooks clean

help: ## Show this help
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install all dependencies (uv sync --all-extras)
	uv sync --all-extras

lint: ## ruff check + ruff format --check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and apply safe lint fixes
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy (strict)
	uv run mypy

test: ## Fast tests (no network, not slow)
	uv run pytest -m "not network and not slow"

test-all: ## All tests, including network and slow
	uv run pytest

cov: ## Fast tests with coverage (term-missing + coverage.xml)
	uv run pytest -m "not network and not slow" --cov --cov-report=term-missing --cov-report=xml

data: ## Ingest 5-minute bars: make data SYMBOL=BTCUSDT START=2017-08-17
	uv run python -m volforecast.data ingest --symbol $(SYMBOL) --start $(START)

quality: lint typecheck test ## lint + typecheck + test

backtest: ## Walk-forward backtest (placeholder)
	@echo "make backtest: available after M2"

report: ## Generate reports/tables.md and README results (placeholder)
	@echo "make report: available after M2"

up: ## Start the local stack (docker compose)
	docker compose up -d --build

down: ## Stop the local stack
	docker compose down

kind-up: ## Create the local kind cluster
	kind create cluster --config deploy/kind/cluster.yaml --name volforecast

kind-down: ## Delete the local kind cluster
	kind delete cluster --name volforecast

helm-lint: ## helm lint the chart (helm runs in docker)
	docker run --rm -v $(PWD):/w -w /w alpine/helm:3.16.4 lint deploy/helm/volforecast

hooks: ## Install pre-commit git hooks
	uv run pre-commit install

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
