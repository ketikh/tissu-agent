# Tissu Bot — System Architecture

## Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                        TISSU BOT SYSTEM                         │
│                                                                 │
│  Facebook Messenger → Server → Agent Engine → Gemini LLM        │
│                                    ↕                            │
│                              Tool System                        │
│                          ↙    ↓     ↓    ↘                      │
│                    Inventory Orders  WA   Vision                │
│                      (SQLite)      Notify  (Gemini)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Message Flow (კლიენტის შეტყობინება → პასუხი)

```
კლიენტი (Messenger)
    │
    ▼
┌──────────────┐     GET/POST
│  Facebook    │ ◄──────────── Facebook Webhook Verification
│  Webhook     │
│  /webhook    │
└──────┬───────┘
       │
       │  ფოტო?  ──────────────────┐
       │  ტექსტი? ──────┐          │
       │  ლინკი? ───┐   │          │
       │             │   │          │
       ▼             ▼   ▼          ▼
┌──────────────────────────────────────────┐
│           _process_message()             │
│                                          │
│  1. ლინკი? → [system tag: ბმული]        │
│  2. ფოტო? → Vision AI ანალიზი           │
│  3. customer_name → [SYSTEM: tag]        │
│  4. run_agent() → Gemini LLM            │
│  5. reply წმენდა (URLs, [Image], etc)    │
│  6. FB API → ტექსტი კლიენტს             │
│  7. _send_product_images() → ფოტოები     │
└──────────────────────────────────────────┘
```

---

## Agents (აგენტები)

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│              🤖 ORCHESTRATOR (მთავარი)                   │
│              src/agents/support_sales.py                 │
│              .claude/agents/orchestrator.md              │
│                                                         │
│  ფუნქცია: კლიენტთან საუბარი, flow მართვა              │
│  LLM: Gemini 2.5 Flash (ქართული + Vision)             │
│                                                         │
│  ეძახის tool-ებს:                                      │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │check_       │ │create_order  │ │notify_owner  │     │
│  │inventory    │ │              │ │              │     │
│  │(მარაგი)     │ │(შეკვეთა)     │ │(WA alert)    │     │
│  └─────────────┘ └──────────────┘ └──────────────┘     │
│  ┌─────────────┐ ┌──────────────┐                      │
│  │save_lead    │ │search_       │                      │
│  │             │ │knowledge     │                      │
│  │(ლიდი)      │ │(FAQ)         │                      │
│  └─────────────┘ └──────────────┘                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              👁 VISION AGENT (ავტომატური)                │
│              src/vision.py                              │
│              .claude/agents/vision-agent.md              │
│                                                         │
│  ფუნქცია: ფოტოს ანალიზი (Gemini Vision AI)            │
│  - გადახდის ქვითარი vs პროდუქტის ფოტო                 │
│  - მსგავსი პროდუქტების მოძებნა მარაგში                │
│  - ფოტოს description + similar_codes                   │
│                                                         │
│  გამოძახება: facebook.py → _handle_image()             │
│  LLM არ იძახებს — server-ის კოდი პირდაპირ             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              📦 CATALOG AGENT (tool-ების შიგნით)        │
│              src/tools/support.py → check_inventory     │
│              .claude/agents/catalog-agent.md             │
│                                                         │
│  ფუნქცია: მარაგის შემოწმება, კოდის ვალიდაცია          │
│  - ფილტრი: ზომა, მოდელი, ფერი, ტეგები                 │
│  - მარაგში > 0 მხოლოდ                                  │
│  - ფოტოს URL-ები (Cloudinary CDN)                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              💳 PAYMENT AGENT (prompt + tool)            │
│              src/tools/support.py → create_order        │
│              .claude/agents/payment-agent.md             │
│                                                         │
│  ფუნქცია: გადახდის ინფო + შეკვეთის შექმნა            │
│  - ბანკის ანგარიშები (prompt-ში)                       │
│  - კალკულაცია: ფასი + 6₾ საკურიერო                    │
│  - create_order → DB + WA notification                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              🔔 ESCALATION AGENT (tool)                  │
│              src/tools/support.py → notify_owner        │
│              .claude/agents/escalation-agent.md          │
│                                                         │
│  ფუნქცია: მფლობელის შეტყობინება WhatsApp-ით          │
│  - პროდუქტის ფოტო + კოდი                              │
│  - Confirm/Deny ლინკები                                │
│  - გაუგებარი სიტუაციების ესკალაცია                    │
└─────────────────────────────────────────────────────────┘
```

---

## Owner Flow (მფლობელის მხარე)

```
მფლობელი (WhatsApp)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Notification Types:                         │
│                                              │
│  📷 გადახდის ქვითარი                         │
│  ├── ✅ ვადასტურებ (clickable link)          │
│  │   └── /api/owner-confirm/{conv_id}        │
│  │       └── კლიენტს: "გადახდა დადასტურდა"  │
│  │           └── ეკითხება მისამართს           │
│  └── ❌ არ ვადასტურებ (clickable link)       │
│      └── /api/owner-deny/{conv_id}           │
│          └── კლიენტს: "ეს მოდელი არ არის"   │
│                                              │
│  🔔 ესკალაცია (გაუგებარი კითხვა)             │
│  └── მფლობელი წერს "უპასუხე: ..." →         │
│      პირდაპირ ეგზავნება კლიენტს              │
│                                              │
│  📷 პროდუქტის ძებნა (ვერ მოიძებნა)          │
│  └── მფლობელი ხედავს რას ეძებს კლიენტი     │
└──────────────────────────────────────────────┘
```

---

## Database (SQLite)

```
┌─────────────────────────────────────────────────────────┐
│                    SQLite Database                       │
│                    data/tissu.db                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📦 inventory          │  🛒 orders                     │
│  ├── id                │  ├── id                        │
│  ├── code (FP1, TD2)   │  ├── customer_name             │
│  ├── model (ფხრიწ/თასმ)│  ├── customer_phone            │
│  ├── size (პატარა/დიდი)│  ├── customer_address          │
│  ├── price (69/74)     │  ├── items (კოდი)              │
│  ├── stock             │  ├── total                     │
│  ├── color, style, tags│  ├── payment_method            │
│  ├── image_url (CDN)   │  ├── status                    │
│  └── image_url_back    │  └── created_at                │
│                        │                                │
│  💬 conversations      │  📨 messages                   │
│  ├── id (conv_id)      │  ├── conversation_id           │
│  ├── agent_type        │  ├── role (user/assistant)     │
│  └── updated_at        │  ├── content                   │
│                        │  └── created_at                │
│                        │                                │
│  👤 leads              │  🎫 tickets                    │
│  ├── name, phone       │  ├── subject                   │
│  ├── source            │  ├── description               │
│  ├── score             │  ├── status, priority          │
│  └── conversation_id   │  └── conversation_id           │
│                        │                                │
│  📚 knowledge_base     │                                │
│  ├── question          │                                │
│  ├── answer            │                                │
│  └── category          │                                │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

```
┌────────────────────┬──────────────────────────────────────┐
│ Layer              │ Technology                           │
├────────────────────┼──────────────────────────────────────┤
│ Runtime            │ Python 3.11+ / FastAPI / Uvicorn     │
│ LLM               │ Google Gemini 2.5 Flash              │
│ Vision AI          │ Gemini Vision (photo analysis)       │
│ Database           │ SQLite (aiosqlite)                   │
│ Image CDN          │ Cloudinary                           │
│ Customer Channel   │ Facebook Messenger API               │
│ Owner Channel      │ WhatsApp Business API                │
│ Hosting            │ Railway (auto-deploy from GitHub)    │
│ Development        │ Claude Code (architecture + code)    │
└────────────────────┴──────────────────────────────────────┘
```

---

## File Structure

```
tissu-agent/
├── server.py                    ← FastAPI entry + webhooks mount
├── CLAUDE.md                    ← Bot shared context (პროდუქტი, ფასები)
├── seed_inventory.json          ← პროდუქტების data + Cloudinary URLs
├── admin.html                   ← Admin panel (მარაგი, შეკვეთები)
├── chat.html                    ← ლოკალური ტესტი
│
├── src/
│   ├── engine.py                ← Agent loop (LLM → Tool → Result → LLM)
│   ├── llm.py                   ← Gemini API client
│   ├── vision.py                ← Vision AI (ფოტო ანალიზი)
│   ├── notifications.py         ← WhatsApp send helpers
│   ├── db.py                    ← SQLite layer
│   ├── config.py                ← Environment config
│   ├── models.py                ← Pydantic models
│   ├── channels.py              ← Channel adapters
│   │
│   ├── agents/
│   │   ├── support_sales.py     ← Orchestrator prompt + entry point
│   │   └── marketing.py         ← Marketing agent
│   │
│   ├── tools/
│   │   ├── support.py           ← check_inventory, create_order, notify_owner
│   │   └── marketing.py         ← Marketing tools
│   │
│   └── webhooks/
│       ├── facebook.py          ← Messenger webhook + image handling
│       └── whatsapp.py          ← WA webhook (owner confirm/deny)
│
├── .claude/
│   ├── agents/
│   │   ├── orchestrator.md      ← მთავარი ბოტის სრული ინსტრუქცია
│   │   ├── vision-agent.md      ← ფოტო ანალიზის ლოგიკა
│   │   ├── catalog-agent.md     ← მარაგის მართვა
│   │   ├── payment-agent.md     ← გადახდა/შეკვეთა
│   │   └── escalation-agent.md  ← WA ესკალაცია
│   │
│   ├── skills/
│   │   └── tissu-skills.md      ← ზომის გზამკვლევი, კალკულატორი
│   │
│   └── rules/                   ← Development rules
│
├── data/                        ← SQLite DB (gitignored)
├── static/products/             ← ფოტოები (local fallback)
└── docs/
    └── architecture.md          ← ეს ფაილი
```

---

## Order Flow (სრული შეკვეთის ციკლი)

```
კლიენტი                    ბოტი                     მფლობელი
   │                        │                          │
   │  "გამარჯობა"           │                          │
   │ ──────────────────────>│                          │
   │                        │  "გამარჯობა ✨            │
   │  <─────────────────────│   ორი ზომა გვაქვს..."    │
   │                        │                          │
   │  "პატარა"              │                          │
   │ ──────────────────────>│                          │
   │                        │  "ფხრიწიანი თუ           │
   │  <─────────────────────│   თასმიანი?"             │
   │                        │                          │
   │  "ფხრიწიანი"           │                          │
   │ ──────────────────────>│                          │
   │                        │  check_inventory()       │
   │                        │  [ფოტოები + კოდები]      │
   │  <─────────────────────│  "აი რა გვაქვს ✨"       │
   │                        │                          │
   │  "FP3"                 │                          │
   │ ──────────────────────>│                          │
   │                        │  "თიბისი თუ              │
   │  <─────────────────────│   საქართველოს ბანკი?"    │
   │                        │                          │
   │  "თიბისი"              │                          │
   │ ──────────────────────>│                          │
   │                        │  "ანგარიში: GE58...      │
   │  <─────────────────────│   სულ: 75₾"             │
   │                        │                          │
   │  [სქრინი/ქვითარი]      │                          │
   │ ──────────────────────>│                          │
   │                        │──Vision AI──>│           │
   │                        │  "receipt"   │           │
   │                        │              │  📷 + ✅/❌ │
   │                        │              │──────────>│
   │  <─────────────────────│                          │
   │  "გადავამოწმებ ✨"      │                          │
   │                        │              │  ✅ click  │
   │                        │<─────────────│           │
   │                        │                          │
   │  <─────────────────────│  "გადახდა დადასტურდა!   │
   │  "მისამართი + ტელ"     │   მისამართი?"            │
   │ ──────────────────────>│                          │
   │                        │  create_order()          │
   │                        │  notify_owner()          │
   │                        │              │  📦 order  │
   │                        │              │──────────>│
   │  <─────────────────────│                          │
   │  "შეკვეთა გაფორმდა ✨"  │                          │
   │                        │                          │
```
