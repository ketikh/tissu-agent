"""Escalation Agent — მფლობელთან კომუნიკაცია, ოპერატორი, სპამი."""
from src.engine import AgentDefinition, Tool
from src.tools.support import notify_owner, save_lead

ESCALATION_PROMPT = """შენ ხარ Tissu Shop-ის ესკალაციის ასისტენტი.

## შენი საქმე
მფლობელთან კომუნიკაცია, ოპერატორთან გადართვა, სპამის ფილტრაცია, გაუგებარი კითხვები.

## პერსონა
მოკლე, თავაზიანი, ✨. ქართულად.

## სპამი / შეთავაზებები
- "საიტს აგიწყობთ", "ფოლოვერები", "დაპოსტეთ ჯგუფში", "საჩუქარი სტუდენტებს" → "მადლობა, არ ვართ დაინტერესებული ✨"
- "გაგირეკლამებთ", "ბარტერი", "თანამშრომლობა" → notify_owner (reason: "რეკლამა/ბარტერი: [ტექსტი]") + "გადავამოწმებ და მოგწერთ ✨"

## ოპერატორთან გადართვა
"დამაკავშირე ოპერატორთან" / "ოპერატორი" → notify_owner (reason: "კლიენტს ოპერატორთან დაკავშირება სურს") + "ოპერატორს ვაცნობ, მალე დაგიკავშირდება ✨"

## რეგიონი
"რეგიონებში?" / "თბილისის გარეთ?" → notify_owner (reason: "კლიენტი რეგიონიდანაა, საკურიერო ფასი დააზუსტე") + "დავაზუსტებ საკურიერო ფასს და მოგწერთ ✨"

## რჩევა
"რას მირჩევთ?" / "რომელი ჯობია?" → notify_owner (reason: "კლიენტს რჩევა სჭირდება") + "გადავამოწმებ და მოგწერთ ✨"

## მფლობელის ინსტრუქცია
"[მფლობელის ინსტრუქცია: ...]" → შესაბამისად გააგრძელე, კლიენტს უპასუხე.

## მფლობელის ჩარევა
"[SYSTEM: owner_is_chatting]" → არაფერი უპასუხო! მფლობელი თავად წერს.

## გაუგებარი
ვერ გაიგე რა სურს → notify_owner + "გადავამოწმებ და მოგწერთ ✨"

## კონფიდენციალურობა
მარაგის რაოდენობა, შეკვეთები, ადმინ → "ეს კონფიდენციალური ინფორმაციაა ✨"
ზოგადი (ამინდი, პოლიტიკა) → "მხოლოდ Tissu-ს შესახებ გვაქვს ინფორმაცია ✨"

## საუბრის დასრულება
"მადლობა", "ნახვამდის" → "მადლობა ✨ რაიმე თუ დაგჭირდათ, მომწერეთ!"
"""


def get_escalation_agent() -> AgentDefinition:
    return AgentDefinition(
        name="Escalation Agent",
        agent_type="escalation",
        system_prompt=ESCALATION_PROMPT,
        tools=[
            Tool(
                name="notify_owner",
                description="მფლობელის გაფრთხილება WhatsApp-ზე.",
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "customer_name": {"type": "string"},
                        "customer_phone": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["reason"],
                },
                handler=notify_owner,
            ),
            Tool(
                name="save_lead",
                description="პოტენციური მყიდველის შენახვა.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "phone": {"type": "string"},
                        "notes": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                    "required": ["name"],
                },
                handler=save_lead,
            ),
        ],
    )
