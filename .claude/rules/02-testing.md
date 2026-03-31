# Testing Rules

## Framework
- pytest for all tests
- pytest-asyncio for async endpoint tests
- pytest-cov for coverage reporting

## Structure
- tests/unit/ — unit tests for individual functions
- tests/integration/ — API endpoint tests
- tests/conftest.py — shared fixtures

## Coverage Targets
- Overall: 80%+
- Critical paths (agents, tools, engine): 90%+
- Utility functions: 100%

## Commands
- Run all: `python -m pytest`
- With coverage: `python -m pytest --cov=src --cov-report=term-missing`
- Single file: `python -m pytest tests/unit/test_db.py`
- With race detection: `python -m pytest -x -v`

## Rules
- Every bug fix gets a regression test FIRST
- Every new function gets at least 1 test (happy path + 1 error case)
- Mock external APIs (Anthropic, N8N) — never call real services in tests
- Use fixtures for database setup/teardown
- Test files mirror source structure: src/db.py → tests/unit/test_db.py
