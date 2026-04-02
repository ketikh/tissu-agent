# Project Conventions

## Language & Framework
- Python 3.11+ with type annotations on all function signatures
- FastAPI for HTTP endpoints
- SQLite for data storage (via src/db.py)
- Google Gemini API for LLM calls (gemini-2.5-flash)
- Gemini Vision AI for image analysis
- Cloudinary for image CDN

## Code Style
- PEP 8 compliance enforced by black + isort + ruff
- Max line length: 88 (black default)
- Import order: stdlib → third-party → local
- Docstrings: Google style for public APIs only

## File Organization
- `src/` — all application code
- `src/agents/` — agent definitions (system prompt + tools)
- `src/tools/` — business logic tools
- `server.py` — FastAPI entry point + webhooks
- `admin.html` — Admin panel
- `seed_inventory.json` — Product data with Cloudinary URLs
- `tests/` — all tests

## Naming
- Files: snake_case
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: SCREAMING_SNAKE_CASE
- Product codes: FP, TP, FD, TD + number

## Data
- All dates in UTC ISO format
- Conversation state stored in SQLite, keyed by conversation_id
- Product images on Cloudinary CDN
- Database recreated on each Railway deploy from seed_inventory.json

## Error Handling
- Never bare `except:` — always catch specific exceptions
- WhatsApp notification to owner on critical errors
- Fallback messages for empty LLM responses
- Retry logic for Gemini API rate limits (429, 503)
