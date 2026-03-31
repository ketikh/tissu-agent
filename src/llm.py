from __future__ import annotations
import asyncio as _asyncio
import logging
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, LLM_MODEL

_logger = logging.getLogger(__name__)
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def convert_tools_to_gemini(tools: list[dict]) -> types.Tool | None:
    """Convert our tool schemas to Gemini function declarations."""
    if not tools:
        return None
    declarations = []
    for tool in tools:
        declarations.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        })
    return types.Tool(function_declarations=declarations)


def convert_messages_to_gemini(system_prompt: str, messages: list[dict]) -> tuple[str, list[types.Content]]:
    """Convert our message format to Gemini contents format."""
    contents = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif isinstance(content, list):
                # Tool results
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        parts.append(types.Part.from_function_response(
                            name=item.get("tool_name", "unknown"),
                            response={"result": item.get("content", "")},
                        ))
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
        elif role == "assistant":
            if isinstance(content, str):
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(types.Part(text=item["text"]))
                        elif item.get("type") == "tool_use":
                            parts.append(types.Part.from_function_call(
                                name=item["name"],
                                args=item["input"],
                            ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))

    return system_prompt, contents


async def chat_with_tools(
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
    model: str = LLM_MODEL,
    max_tokens: int = 4096,
) -> dict:
    """Call Gemini with tools and return a normalized response."""
    client = get_client()
    gemini_tools = convert_tools_to_gemini(tools)
    _, contents = convert_messages_to_gemini(system_prompt, messages)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
    )
    if gemini_tools:
        config.tools = [gemini_tools]

    # Retry up to 5 times if Gemini returns empty or rate limited
    response = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                wait = (attempt + 1) * 5
                _logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1}/5)")
                await _asyncio.sleep(wait)
                continue
            raise

    # Normalize response to our internal format
    result_content = []
    has_tool_calls = False

    if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                has_tool_calls = True
                fc = part.function_call
                result_content.append({
                    "type": "tool_use",
                    "id": fc.id if hasattr(fc, 'id') and fc.id else f"call_{fc.name}",
                    "name": fc.name,
                    "input": dict(fc.args) if fc.args else {},
                })
            elif part.text:
                result_content.append({
                    "type": "text",
                    "text": part.text,
                })

    return {
        "stop_reason": "tool_use" if has_tool_calls else "end_turn",
        "content": result_content,
    }
