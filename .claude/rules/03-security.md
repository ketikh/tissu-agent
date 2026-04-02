# Security Rules for This Project

## API Keys & Secrets
- NEVER hardcode API keys in source code
- All secrets via Railway environment variables
- .env is in .gitignore — never commit it
- Permanent tokens via Meta System User (never expires)

## Bot Security
- Never reveal stock quantities to customers
- Never reveal customer personal data (name, phone, address)
- Never reveal internal business info (orders count, revenue, admin panel)
- Never answer non-Tissu questions (weather, politics, etc.)
- Confidential info response: "ეს კონფიდენციალური ინფორმაციაა ✨"
- [SYSTEM:] tags never shown to customers

## Input Validation
- Validate all API request bodies with Pydantic models
- Sanitize user input before passing to LLM prompts
- Never pass raw user input to shell commands or SQL
- Use parameterized queries for all database operations

## Image Security
- Gemini Vision analyzes customer photos before processing
- Payment receipts forwarded to owner for manual confirmation
- Product photos compared with inventory automatically

## API Security
- Return consistent error format — never expose stack traces
- Log errors server-side with request context
- Never log PII, passwords, or API keys
- Anti-duplicate message processing (mid tracking)

## Database
- Use parameterized queries only (src/db.py handles this)
- Never concatenate user input into SQL strings
- Database files excluded from git
