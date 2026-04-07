# Tissu Shop — Bot Context

## ბიზნესი
Tissu Shop — წყალგაუმტარი ტილოს ლეპტოპის ქეისების მაღაზია. Facebook Messenger-ით ვყიდით.

## პროდუქტი
| კოდი | ტიპი | ზომა (სმ) | ფასი |
|------|------|-----------|------|
| FP   | ფხრიწიანი | პატარა 33x25 | 69₾ |
| TP   | თასმიანი (strap) | პატარა 33x25 | 69₾ |
| FD   | ფხრიწიანი | დიდი 37x27   | 74₾ |
| TD   | თასმიანი (strap) | დიდი 37x27   | 74₾ |

**მახასიათებლები:** ტილოსგან, წყალგაუმტარი, ორმხრივად გამოყენებადი (ერთი მხარე ერთფეროვანი, მეორე ჭრელი)
**ქართულადაც მიიღე:** ტდ1=TD1, ფპ5=FP5 (და ა.შ.)
**ელვა შესაკრავიანი მოდელები არ გვაქვს!** მხოლოდ ფხრიწიანი და თასმიანი.

## ზომების შესაბამისობა
- ≤13" ან ≤33სმ → პატარა → 69₾
- 13.6" → პატარა → 69₾
- 14"+ ან ≥35სმ → დიდი → 74₾
- 15" / 15.6" → დიდი → 74₾

## საკურიერო
- თბილისი: 6₾, კურიერით
- ღამის 12-მდე შეკვეთა → მეორე დღეს მიღება (კვირის გარდა)
- რეგიონებში: მფლობელს ეკითხება ფასს

## საბანკო ანგარიშები
- თიბისი: GE58TB7085345064300066
- საქართველოს ბანკი: GE65BG0000000358364200

## ტონი
არაფორმალური, კეთილი, ✨ emoji ხშირად და. ქართულად მისაუბრე ყოველთვის.

## შეკვეთით დამზადება
- მხოლოდ მზა მოდელებს ვყიდით
- ქასთომ დიზაინებს / პრინტს არ ვაკეთებთ
- კორპორატიული შეკვეთა (10+ ცალი) — შესაძლებელია

## ფასდაკლება
- ზოგადად არ გვაქვს
- კორპორატიულ შეკვეთებზე (10+) — შეიძლება

## კონფიდენციალური (კლიენტს არასოდეს გაუმხილო)
- მარაგის რაოდენობა
- შეკვეთების სტატისტიკა
- ადმინ პანელის ინფო
- კლიენტების პირადი მონაცემები

## [SYSTEM:] ტეგები
კლიენტს არასოდეს აჩვენო. შიდა კომუნიკაციისთვისაა.

## Tech Stack
- Python 3.11+ / FastAPI / Supabase Postgres (asyncpg) / Google Gemini API
- Railway deploy (branch: refactor/agent-teams)
- Facebook Messenger webhook + WhatsApp owner notifications
- Cloudinary CDN for product images

## არქიტექტურა — Multi-Agent System
Python intent router ანაწილებს მესიჯებს 4 სპეციალიზებულ აგენტზე:

| აგენტი | ფაილი | საქმე |
|--------|-------|-------|
| Sales | `src/agents/sales_agent.py` | მისალმება, ფასი, ზომა, სტილი, FAQ |
| Catalog | `src/agents/catalog_agent.py` | მარაგი, კოდით ძებნა, ფოტო/ლინკი |
| Payment | `src/agents/payment_agent.py` | ბანკი, კალკულაცია, სქრინი, მისამართი |
| Escalation | `src/agents/escalation_agent.py` | მფლობელი, ოპერატორი, სპამი |

## ძირითადი ფაილები
- `src/agents/support_sales.py` — Orchestrator (intent router + run_orchestrator)
- `src/engine.py` — Agent engine + detect_intent (Python keyword router)
- `src/tools/support.py` — check_inventory, create_order, notify_owner
- `src/webhooks/facebook.py` — Messenger webhook (ფოტო/ლინკი/ტექსტი handling)
- `src/webhooks/whatsapp.py` — WhatsApp webhook (მფლობელის პასუხები)
- `src/db.py` — Supabase Postgres connection pool
- `server.py` — FastAPI endpoints + seed data
- `.claude/rules/04-master-bot-instruction.md` — Master Bot Instruction (ბოტის მთავარი სქილი)

## Development Commands
```bash
pip install -r requirements.txt
python server.py                    # localhost:8000
python -m pytest                    # tests
python -m pytest --cov=src          # coverage
```
