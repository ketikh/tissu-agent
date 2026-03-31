# N8N Workflows

Import these workflows into your local n8n instance (http://localhost:5678).

## Workflows

### 1. support-webhook.json
**Trigger**: Webhook (POST to n8n)
**Flow**: Receive customer message → Call Agent API → Return response
**Use case**: Connect to website chat, Telegram bot, or any messaging platform

### 2. content-scheduler.json
**Trigger**: Cron (daily at 9 AM)
**Flow**: Ask Marketing Agent to generate daily content → Save to DB
**Use case**: Automated content generation for social media

### 3. lead-monitor.json
**Trigger**: Cron (every hour)
**Flow**: Check for hot leads → Send notification
**Use case**: Never miss a hot sales opportunity

## How to import
1. Open n8n at http://localhost:5678
2. Click "Add workflow" → "Import from file"
3. Select the JSON file
4. Activate the workflow

## Agent API endpoints (all at http://localhost:8000)
- POST /api/support — Chat with Support+Sales agent
- POST /api/marketing — Chat with Marketing agent
- GET /api/leads — List all leads
- GET /api/tickets — List all tickets
- GET /api/content — List all content
