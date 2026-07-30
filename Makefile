.PHONY: help install dev-install test lint format clean build

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Hugo (CPU-only PyTorch, sufficient for PTQ)
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e .

dev-install: ## Install Hugo with all dev + training dependencies
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,train]"

test: ## Run the test suite
	pytest -v

test-cov: ## Run tests with coverage report
	pytest -v --cov=src/hugo --cov-report=term-missing

lint: ## Run ruff linter
	ruff check src tests

format: ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

typecheck: ## Run mypy type checking
	mypy src/hugo

clean: ## Remove build artifacts and cache
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: ## Build distribution packages
	python -m build

all: lint test ## Run lint + tests (same as CI)
