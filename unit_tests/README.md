### Install Dependencies
```bash
pip install pytest pytest-cov pytest-mock hypothesis
```

### Running Tests

To run all tests:
```bash
PYTHONPATH=. pytest
```

To run only unit tests:
```bash
PYTHONPATH=. pytest -c .github/pytest.ini unit_tests/unit
```

To run only integration tests:
```bash
PYTHONPATH=. pytest -c .github/pytest.ini unit_tests/integration
```

Tests are designed to run on CPU and use mocks for heavy dependencies like Ray and DeepSpeed.
