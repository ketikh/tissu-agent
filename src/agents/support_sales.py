"""Tissu Shop — Multi-Agent Orchestrator.

Routes incoming messages to specialized agents via Python intent detection.
No LLM call for routing — Python keywords determine the agent.
"""
from src.engine import AgentDefinition, detect_intent, run_agent
from src.agents.sales_agent import get_sales_agent
from src.agents.catalog_agent import get_catalog_agent
from src.agents.payment_agent import get_payment_agent
from src.agents.escalation_agent import get_escalation_agent
from src.db import get_conversation_messages

# Agent registry
_AGENTS = {
    "sales": get_sales_agent,
    "catalog": get_catalog_agent,
    "payment": get_payment_agent,
    "escalation": get_escalation_agent,
}


async def run_orchestrator(user_message: str, conversation_id: str) -> dict:
    """Route message to the right agent and return its response."""
    # Get history for context-aware routing
    history = await get_conversation_messages(conversation_id)

    # Python-based intent detection (no LLM call)
    intent = detect_intent(user_message, history)
    print(f"[ROUTER] Intent: {intent} | Message: {user_message[:60]}...", flush=True)

    # Get the specialized agent
    agent_factory = _AGENTS.get(intent, get_sales_agent)
    agent = agent_factory()

    # Run agent with full conversation history
    result = await run_agent(agent, user_message, conversation_id)
    result["routed_to"] = intent
    return result


# Backward compatibility — old code calls get_support_sales_agent()
def get_support_sales_agent() -> AgentDefinition:
    """Return sales agent for backward compatibility. Use run_orchestrator() for routing."""
    return get_sales_agent()
