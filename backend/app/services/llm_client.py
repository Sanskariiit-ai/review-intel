"""
Thin wrapper around the Anthropic SDK. This is the ONLY place the LLM gets
called from in the whole pipeline — per the spec, everything upstream
(cleaning, embedding, clustering, keyword matching, VADER sentiment) is
traditional ML/NLP specifically so this expensive step only runs a handful
of times per product, not once per review.

Two call sites use this: topic labeling/summarization (batched — all of a
product's topics in ONE call) and executive summary generation (one call,
built from the topic summaries rather than raw reviews).
"""

import json
from functools import lru_cache

from anthropic import Anthropic

from app.core.config import settings

DEFAULT_MODEL = "claude-sonnet-5"


class LLMGenerationError(Exception):
    """Raised when the LLM call fails or returns output we can't parse as
    the JSON shape we asked for. Callers should surface this clearly rather
    than silently writing garbage into the DB."""


@lru_cache(maxsize=1)
def get_anthropic_client() -> Anthropic:
    if not settings.ANTHROPIC_API_KEY:
        raise LLMGenerationError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file to use "
            "topic labeling, summaries, or Ask AI."
        )
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def generate_text(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> str:
    """Like generate_json but for prose answers (Ask AI) rather than structured data."""
    client = get_anthropic_client()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    return "".join(text_parts).strip()


def generate_json(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> dict | list:
    """
    Calls the LLM expecting a JSON-only response, and parses it. Raises
    LLMGenerationError with the raw text on failure so callers can log/debug
    rather than get a bare JSONDecodeError.
    """
    client = get_anthropic_client()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    raw_text = "".join(text_parts).strip()

    # Models occasionally wrap JSON in markdown fences despite instructions not to.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise LLMGenerationError(
            f"LLM response was not valid JSON: {e}\nRaw response: {raw_text[:500]}"
        ) from e
