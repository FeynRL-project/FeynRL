PYTHON ?= python3

.PHONY: lint format test test-unit test-integration ci

lint:
	ruff check .

format:
	ruff format .

test:
	PYTHONPATH=. pytest -c .github/pytest.ini unit_tests

test-unit:
	PYTHONPATH=. pytest -c .github/pytest.ini unit_tests/unit

test-integration:
	PYTHONPATH=. pytest -c .github/pytest.ini unit_tests/integration

ci: lint test-unit
