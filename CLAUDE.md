# Tissu Agent System

AI-powered sales agent for Tissu Shop — handmade laptop sleeves. Runs on Railway, uses Facebook Messenger for customer communication and WhatsApp for owner notifications.

## Tech Stack
- Python 3.11+ / FastAPI / SQLite / Google Gemini API (gemini-2.5-flash)
- Gemini Vision AI for photo analysis (product identification + payment receipt detection)
- Cloudinary CDN for product images
- Railway for hosting
- Facebook Messenger API for customer chat
- WhatsApp Business API for owner notifications

## Architecture
- `src/engine.py` — Agent loop (LLM → Tool → Result → LLM)
- `src/agents/support_sales.py` — Sales agent (system prompt + tools)
- `src/agents/marketing.py` — Marketing agent
- `src/tools/support.py` — Business logic tools (inventory, orders, notifications)
- `src/llm.py` — Gemini LLM client
- `src/db.py` — SQLite database layer
- `server.py` — FastAPI entry point + Facebook/WhatsApp webhooks
- `seed_inventory.json` — Product inventory with Cloudinary image URLs
- `admin.html` — Admin panel (inventory, orders, leads)

## Live URLs
- **Bot API**: https://tissu-agent-production.up.railway.app
- **Admin Panel**: https://tissu-agent-production.up.railway.app/admin
- **Health Check**: https://tissu-agent-production.up.railway.app/api/health

## API Endpoints
- `POST /api/support` — Chat with sales agent
- `POST /api/marketing` — Chat with marketing agent
- `POST /webhook` — Facebook Messenger webhook
- `POST /wa-webhook` — WhatsApp webhook (owner responses)
- `GET /api/inventory` — List products
- `GET /api/orders` — List orders
- `GET /api/leads` — List leads
- `GET /api/conversations` — List conversations
- `GET /admin` — Admin panel

## Environment Variables (Railway)
- `GEMINI_API_KEY` — Google Gemini API key
- `FB_PAGE_TOKEN` — Facebook Page access token (permanent, via System User)
- `WA_TOKEN` — WhatsApp Business API token (permanent, via System User)
- `WA_PHONE_ID` — WhatsApp phone number ID
- `OWNER_WHATSAPP` — Owner's WhatsApp number (for notifications)
- `LLM_MODEL` — Gemini model name (gemini-2.5-flash)
- `LLM_PROVIDER` — LLM provider (gemini)
- `PUBLIC_URL` — Railway public URL

## Key Features
- Facebook Messenger bot for Tissu Shop
- Gemini Vision AI: analyzes customer photos (product vs payment receipt)
- Cloudinary CDN for product images
- WhatsApp notifications to owner for order confirmations
- Admin panel for inventory/order management
- Automatic order creation with create_order tool
- Product code system (FP, TP, FD, TD)

## Product Codes
- FP = ფხრიწიანი პატარა (FP1-FP16)
- TP = თასმიანი პატარა (TP1-TP15)
- FD = ფხრიწიანი დიდი (FD1-FD6)
- TD = თასმიანი დიდი (TD1)

## Development
```bash
pip install -r requirements.txt
python server.py  # Starts on localhost:8000
```

## Deployment
- Push to GitHub → Railway auto-deploys
- Product images stored on Cloudinary CDN
- SQLite DB recreated on each deploy (seed_inventory.json)
- All tokens are permanent (System User tokens)
