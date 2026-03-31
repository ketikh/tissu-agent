# Tissu Agent System

Local-first AI agent system for business. Two connected agents:

1. **Support + Sales** (`/api/support`) — handles customer inquiries, qualifies leads
2. **Marketing + Content** (`/api/marketing`) — generates content, analyzes data

## Tech Stack
- Python 3.11+ / FastAPI / SQLite / Anthropic Claude API
- N8N for workflow orchestration (localhost:5678)

## Run
```bash
cp .env.example .env  # Add your ANTHROPIC_API_KEY
pip install -r requirements.txt
python server.py      # Starts on localhost:8000
```

## Architecture
- `src/engine.py` — Agent loop (LLM → Tool → Result → LLM)
- `src/agents/` — Agent definitions (system prompt + tools)
- `src/tools/` — Business logic tools (DB operations)
- `src/llm.py` — LLM client abstraction
- `src/db.py` — SQLite database layer
- `server.py` — FastAPI entry point
- `n8n/` — Importable N8N workflow JSONs

## API Endpoints
- `POST /api/support` — Chat with support+sales agent
- `POST /api/marketing` — Chat with marketing agent
- `GET /api/leads` — List leads
- `GET /api/tickets` — List tickets
- `GET /api/content` — List content
- `GET /api/conversations` — List conversations
- `GET /api/health` — Health check

## Conventions
- All dates in UTC ISO format
- Agent tools return dicts, engine serializes to JSON
- Conversation state stored in SQLite, keyed by conversation_id

## Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py

# Run tests
python -m pytest
python -m pytest --cov=src --cov-report=term-missing

# Code quality
python -m black src/ tests/ server.py
python -m isort src/ tests/ server.py
python -m ruff check src/ tests/ server.py
python -m mypy src/

# Dependency audit
pip audit
```

## Project Structure
```
├── server.py           # FastAPI entry point
├── src/
│   ├── engine.py       # Agent loop (LLM → Tool → Result → LLM)
│   ├── llm.py          # LLM client abstraction
│   ├── db.py           # SQLite database layer
│   ├── config.py       # Configuration management
│   ├── channels.py     # Communication channels
│   ├── models.py       # Data models
│   ├── agents/         # Agent definitions
│   │   ├── support_sales.py
│   │   └── marketing.py
│   └── tools/          # Business logic tools
│       ├── support.py
│       └── marketing.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── n8n/                # N8N workflow JSONs
├── data/               # SQLite databases (gitignored)
├── static/             # Static assets
├── docs/               # Documentation
│   └── decisions/      # Architecture Decision Records
└── .github/            # CI/CD and templates
```

## Environment Variables
- `ANTHROPIC_API_KEY` — Required. Claude API key.
- `DATABASE_PATH` — Optional. SQLite database path (default: data/tissu.db)
- `PORT` — Optional. Server port (default: 8000)
- `LOG_LEVEL` — Optional. Logging level (default: INFO)
