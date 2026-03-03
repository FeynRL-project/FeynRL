PYTHON ?= python3

.PHONY: lint format test test-unit test-integration ci

lint:
	ruff check .

format:
	ruff format .

test:
	PYTHONPATH=. pytest tests

test-unit:
	PYTHONPATH=. pytest tests/unit

test-integration:
	PYTHONPATH=. pytest tests/integration

ci: lint test-unit
