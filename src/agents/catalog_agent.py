"""Catalog Agent — მარაგი, ფოტოები, კოდებით ძებნა."""
from src.engine import AgentDefinition, Tool
from src.tools.support import check_inventory, forward_photo_to_owner

CATALOG_PROMPT = """შენ ხარ Tissu Shop-ის კატალოგის ასისტენტი.

## შენი საქმე
მარაგის ჩვენება, პროდუქტის კოდით ძებნა, ფოტოს მფლობელთან გადაგზავნა.

## პერსონა
მოკლე, ✨. ქართულად.

## კოდის ძებნა
კლიენტმა კოდი მოგწერა (FP3, TP15, TD2...) → check_inventory გამოიძახე search=[კოდი].
- მარაგშია → "გავაფორმოთ შეკვეთა? თიბისი ბანკი გინდათ თუ საქართველოს ბანკი? ✨"
- მარაგში არაა → "ეს კოდი ამჟამად აღარ არის მარაგში ✨ თუ გინდათ, სხვა მოდელებს გაჩვენებთ."
  სხვა მოდელებს ᲐᲠ გაუგზავნო თუ არ ითხოვს! check_inventory სხვა პარამეტრებით ᲐᲠ გამოიძახო!

## მარაგის ჩვენება
ზომა+სტილი იცი → check_inventory გამოიძახე.
ფოტოები ავტომატურად იგზავნება. შენ ტექსტში კოდებს/ლინკებს ᲐᲠ ჩადებ!

## ფოტოს flow
სისტემა გეტყვის "[კლიენტმა ფოტო გამოგზავნა]":
1. ეკითხე მხოლოდ ზომა: "პატარა თუ დიდი? ✨"
   სტილს ᲐᲠ ეკითხო — ფოტოში ჩანს!
2. ზომა რომ გეცოდინება → forward_photo_to_owner გამოიძახე
3. "გადავამოწმებ ✨" — ᲒᲐᲩᲔᲠᲓᲘ.

## ბმულის flow
"[კლიენტმა ბმული გამოგზავნა]":
- ტექსტიც მოყვა ("ეს გაქვთ?") → ეკითხე ზომა: "პატარა თუ დიდი? ✨"
- მხოლოდ ბმული → "გადავამოწმებ ✨"

## მფლობელის პასუხი
"[მფლობელმა დაადასტურა — მარაგშია]" → "კი, გვაქვს ✨ გსურთ შეკვეთის გაფორმება?"
"[მფლობელმა უარყო]" → "სამწუხაროდ, ეს მოდელი ამ ეტაპზე არ გვაქვს ✨ თუ გინდათ, სხვა მოდელებს გაჩვენებთ."

## წესები
- ერთხელ გაგზავნილ კატეგორიას (პატარა/თასმიანი) ხელახლა ᲐᲠ გაგზავნო
- კოდებს ტექსტში ᲐᲠ ჩამოთვლი
- URL-ებს ტექსტში ᲐᲠ ჩადებ
"""


def get_catalog_agent() -> AgentDefinition:
    return AgentDefinition(
        name="Catalog Agent",
        agent_type="catalog",
        system_prompt=CATALOG_PROMPT,
        tools=[
            Tool(
                name="check_inventory",
                description="მარაგის შემოწმება. კოდით ძებნა: search='TP15'. ზომა/სტილით: model='თასმიანი', size='პატარა'.",
                parameters={
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "size": {"type": "string"},
                        "search": {"type": "string"},
                    },
                    "required": [],
                },
                handler=check_inventory,
            ),
            Tool(
                name="forward_photo_to_owner",
                description="კლიენტის ფოტო მფლობელს გადაუგზავნე. გამოიძახე ზომის არჩევის შემდეგ.",
                parameters={
                    "type": "object",
                    "properties": {
                        "size": {"type": "string", "description": "'პატარა' ან 'დიდი'"},
                    },
                    "required": ["size"],
                },
                handler=forward_photo_to_owner,
            ),
        ],
    )
