# Project Conventions

## Language & Framework
- Python 3.11+ with type annotations on all function signatures
- FastAPI for HTTP endpoints
- SQLite for data storage (via src/db.py)
- Anthropic Claude API for LLM calls

## Code Style
- PEP 8 compliance enforced by black + isort + ruff
- Max line length: 88 (black default)
- Import order: stdlib → third-party → local
- Docstrings: Google style for public APIs only

## File Organization
- `src/` — all application code
- `src/agents/` — agent definitions (system prompt + tools)
- `src/tools/` — business logic tools
- `server.py` — FastAPI entry point
- `n8n/` — N8N workflow JSONs
- `tests/` — all tests (mirrors src/ structure)

## Naming
- Files: snake_case
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: SCREAMING_SNAKE_CASE
- Agent tools return dicts, engine serializes to JSON

## Data
- All dates in UTC ISO format
- Conversation state stored in SQLite, keyed by conversation_id
- Database files in data/ directory

## Error Handling
- Custom exception classes per domain
- Never bare `except:` — always catch specific exceptions
- Wrap errors with context for debugging
- Never expose internal errors to API consumers
