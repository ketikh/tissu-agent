from __future__ import annotations
import asyncio as _asyncio
import logging
import os
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, LLM_MODEL

_logger = logging.getLogger(__name__)
_client: genai.Client | None = None
_sleep = _asyncio.sleep  # indirection so tests can skip the retry back-off
_RETRYABLE_MARKERS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")


class LLMUnavailableError(RuntimeError):
    """Gemini produced no usable answer (outage, quota, safety block, token limit)."""


def _has_parts(response) -> bool:
    return bool(
        response and response.candidates
        and response.candidates[0].content
        and response.candidates[0].content.parts
    )


def _empty_reason(response) -> str:
    """Why did Gemini return nothing? Human-readable, for logs and owner alerts."""
    if response is None:
        return "no response"
    if not response.candidates:
        block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        return f"no candidates (block_reason={getattr(block, 'name', block)})"
    fr = getattr(response.candidates[0], "finish_reason", None)
    return f"finish_reason={getattr(fr, 'name', fr)}"


def _is_intentional_silence(response) -> bool:
    """A normal STOP finish with no parts means the model chose to say nothing."""
    if not response or not response.candidates:
        return False
    fr = getattr(response.candidates[0], "finish_reason", None)
    return (getattr(fr, "name", None) or str(fr)).endswith("STOP")

# Vertex AI config — when set, uses Vertex AI instead of AI Studio API key
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT", "")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if VERTEX_PROJECT:
            # Use Vertex AI (service account auth, billed to project's billing account)
            _client = genai.Client(
                vertexai=True,
                project=VERTEX_PROJECT,
                location=VERTEX_LOCATION,
            )
            print(f"[LLM] Using Vertex AI — project={VERTEX_PROJECT}, location={VERTEX_LOCATION}")
        else:
            # Fallback to AI Studio API key
            _client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[LLM] Using AI Studio API key")
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

    # Retry up to 3 times if Gemini returns empty or rate limited
    response = None
    last_error = ""
    for attempt in range(3):
        try:
            print(f"[LLM] Calling Gemini (attempt {attempt+1}/3)...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if _has_parts(response):
                print(f"[LLM] Got response, {len(response.candidates[0].content.parts)} parts")
                break
            last_error = _empty_reason(response)
            print(f"[LLM] Empty response ({last_error}), retrying...")
        except Exception as e:
            last_error = str(e)
            print(f"[LLM] Error: {last_error[:200]}")
            if any(marker in last_error for marker in _RETRYABLE_MARKERS):
                wait = (attempt + 1) * 3
                print(f"[LLM] Rate limited, waiting {wait}s")
                await _sleep(wait)
                response = None
                continue
            raise

    if not _has_parts(response) and not _is_intentional_silence(response):
        # An outage must not look like a deliberate silence: the caller sends
        # a holding message to the customer and alerts the owner with the reason.
        raise LLMUnavailableError(f"Gemini returned no content after 3 attempts: {last_error[:300]}")

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
