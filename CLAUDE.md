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
- პატარა (33x25): 13", 13.3", 13.6", 14" (MacBook Air/Pro 14" ეტევა პატარაში)
- დიდი (37x27): 15", 15.6", 16" ლეპტოპები
- ინჩებს ბოტი არ ახსენებს სანამ კლიენტი თავად არ იკითხავს

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

## ძირითადი ფაილები
- `src/agents/support_sales.py` — ბოტის სისტემის პრომპტი (ORCHESTRATOR_PROMPT)
- `src/tools/support.py` — check_inventory, create_order, notify_owner
- `src/webhooks/facebook.py` — Messenger webhook (ფოტო/ლინკი/ტექსტი handling)
- `src/webhooks/whatsapp.py` — WhatsApp webhook (მფლობელის პასუხები)
- `src/db.py` — Supabase Postgres connection pool
- `server.py` — FastAPI endpoints + seed data

## Development Commands
```bash
pip install -r requirements.txt
python server.py                    # localhost:8000
python -m pytest                    # tests
python -m pytest --cov=src          # coverage
```
