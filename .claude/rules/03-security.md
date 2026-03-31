# Security Rules for This Project

## API Keys & Secrets
- NEVER hardcode ANTHROPIC_API_KEY or any secret in source code
- All secrets via environment variables loaded from .env
- .env is in .gitignore — never commit it
- Validate all config at startup — fail fast if missing

## Input Validation
- Validate all API request bodies with Pydantic models
- Sanitize user input before passing to LLM prompts
- Never pass raw user input to shell commands or SQL
- Use parameterized queries for all database operations

## API Security
- Rate limit authentication endpoints
- Return consistent error format — never expose stack traces
- Log errors server-side with request context
- Never log PII, passwords, or API keys

## Database
- Use parameterized queries only (src/db.py handles this)
- Never concatenate user input into SQL strings
- Database files stored in data/ directory, excluded from git
