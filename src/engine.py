from __future__ import annotations
import re
import uuid
import json
from dataclasses import dataclass, field
from typing import Callable, Any
from src.llm import chat_with_tools
from src.db import ensure_conversation, save_message, get_conversation_messages


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]


@dataclass
class AgentDefinition:
    name: str
    agent_type: str
    system_prompt: str
    tools: list[Tool] = field(default_factory=list)


def parse_agent_metadata(text: str) -> dict | None:
    """Extract structured metadata block from agent response."""
    match = re.search(r"---AGENT_METADATA---\s*(.+?)\s*---END_METADATA---", text, re.DOTALL)
    if not match:
        return None
    block = match.group(1)
    metadata = {}
    for line in block.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            metadata[key] = value
    return metadata


def strip_metadata_block(text: str) -> str:
    """Remove the metadata block from the customer-facing reply."""
    # Try multiple patterns the LLM might use
    text = re.sub(r"\s*-{2,}AGENT_METADATA-{2,}.*?-{2,}END_METADATA-{2,}\s*", "", text, flags=re.DOTALL).strip()
    # Also catch if LLM writes it without exact dashes
    text = re.sub(r"\s*AGENT_METADATA\s*\n.*?END_METADATA\s*", "", text, flags=re.DOTALL).strip()
    # Remove any leftover lines that look like metadata
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("intent:") or stripped.startswith("customer_stage:") or stripped.startswith("hot_lead:") or stripped.startswith("next_action:") or stripped.startswith("notes:"):
            continue
        if stripped in ("---", "---AGENT_METADATA---", "---END_METADATA---"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def tool_to_schema(tool: Tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


async def run_agent(agent: AgentDefinition, user_message: str, conversation_id: str | None = None) -> dict:
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    await ensure_conversation(conversation_id, agent.agent_type)

    history = await get_conversation_messages(conversation_id)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": user_message})

    await save_message(conversation_id, "user", user_message)

    tool_schemas = [tool_to_schema(t) for t in agent.tools]
    tool_map = {t.name: t for t in agent.tools}
    tools_used = []
    tool_results_data = {}  # Store actual tool return values

    max_iterations = 10
    for _ in range(max_iterations):
        response = await chat_with_tools(
            system_prompt=agent.system_prompt,
            messages=messages,
            tools=tool_schemas,
        )

        # response is now a dict: {"stop_reason": ..., "content": [...]}
        stop_reason = response["stop_reason"]
        content_blocks = response["content"]

        if stop_reason == "end_turn":
            final_text = ""
            for block in content_blocks:
                if block["type"] == "text":
                    final_text += block["text"]

            metadata = parse_agent_metadata(final_text)
            clean_reply = strip_metadata_block(final_text)

            await save_message(conversation_id, "assistant", clean_reply)
            result = {
                "reply": clean_reply,
                "conversation_id": conversation_id,
                "agent_type": agent.agent_type,
                "tool_calls_made": tools_used,
                "tool_results_data": tool_results_data,
            }
            if metadata:
                result["agent_metadata"] = metadata
            return result

        if stop_reason == "tool_use":
            assistant_content = []
            tool_results = []

            for block in content_blocks:
                if block["type"] == "text":
                    assistant_content.append({"type": "text", "text": block["text"]})
                elif block["type"] == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": block["input"],
                    })

                    tool = tool_map.get(block["name"])
                    if tool:
                        try:
                            # Inject conversation_id for tools that need it
                            tool_input = dict(block["input"])
                            if block["name"] in ("notify_owner", "forward_photo_to_owner") and "conversation_id" not in tool_input:
                                tool_input["conversation_id"] = conversation_id
                            result = await tool.handler(**tool_input)
                            result_str = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
                            tool_results_data[block["name"]] = result
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)})
                        tools_used.append(block["name"])
                    else:
                        result_str = json.dumps({"error": f"Unknown tool: {block['name']}"})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "tool_name": block["name"],
                        "content": result_str,
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        final_text = ""
        for block in content_blocks:
            if block["type"] == "text":
                final_text += block["text"]
        await save_message(conversation_id, "assistant", final_text or "[no response]")
        return {
            "reply": final_text or "[no response]",
            "conversation_id": conversation_id,
            "agent_type": agent.agent_type,
            "tool_calls_made": tools_used,
        }

    await save_message(conversation_id, "assistant", "[max iterations reached]")
    return {
        "reply": "[max iterations reached]",
        "conversation_id": conversation_id,
        "agent_type": agent.agent_type,
        "tool_calls_made": tools_used,
    }
