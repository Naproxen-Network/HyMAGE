"""Utility functions: API key loading and LLM client initialization."""

import os
import openai
from typing import Optional


def load_api_keys(filename: str = "api-key.txt") -> str:
    """Load OpenAI-compatible API key from file."""
    possible_paths = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line
    raise FileNotFoundError(f"API key file not found: {filename}")


def get_llm_client(model: str, base_url: str = "https://api.openai.com/v1",
                   local_endpoints: Optional[dict] = None):
    """Return an OpenAI-compatible client based on model name.

    Args:
        model: Model identifier (e.g. 'gpt-3.5-turbo', 'qwen2.5-3b').
        base_url: Base URL for the OpenAI-compatible API.
        local_endpoints: Optional dict mapping model prefixes to local URLs.
    """
    if local_endpoints and model.startswith("qwen"):
        if "3b" in model.lower():
            url = local_endpoints.get("3b", base_url)
        else:
            url = local_endpoints.get("7b", base_url)
        return openai.OpenAI(api_key="EMPTY", base_url=url)
    else:
        api_key = load_api_keys("api-key.txt")
        return openai.OpenAI(api_key=api_key, base_url=base_url)


# Mapping from CLI model names to vLLM-served model identifiers
QWEN_MODEL_MAP = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def resolve_model_name(model: str) -> str:
    """Resolve a CLI model name to the actual model identifier."""
    return QWEN_MODEL_MAP.get(model, model)
