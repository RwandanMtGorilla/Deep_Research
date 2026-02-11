"""Centralized configuration for the Deep Research system.

Loads settings from .env file with fallback to system environment variables.
All model instances and system constants are configured here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load .env file from project root, with override=False to respect existing env vars
try:
    _project_root = Path(__file__).resolve().parent.parent
except NameError:
    _project_root = Path.cwd()
load_dotenv(_project_root / ".env", override=False)

# --- API Keys ---
# Read implicitly by langchain (OPENAI_API_KEY) and tavily (TAVILY_API_KEY).
# Exposed here for optional explicit validation.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# --- Model Names ---
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "openai:gpt-5")
SUPERVISOR_MODEL: str = os.getenv("SUPERVISOR_MODEL", "openai:gpt-5")
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "openai:gpt-5")
COMPRESS_MODEL: str = os.getenv("COMPRESS_MODEL", "openai:gpt-5")
WRITER_MODEL: str = os.getenv("WRITER_MODEL", "openai:gpt-5")

# --- Model Token Limits ---
COMPRESS_MODEL_MAX_TOKENS: int = int(os.getenv("COMPRESS_MODEL_MAX_TOKENS", "32000"))
REPORT_WRITER_MODEL_MAX_TOKENS: int = int(os.getenv("REPORT_WRITER_MODEL_MAX_TOKENS", "40000"))
DRAFT_WRITER_MODEL_MAX_TOKENS: int = int(os.getenv("DRAFT_WRITER_MODEL_MAX_TOKENS", "32000"))

# --- API Base URLs (optional, None means use official default endpoint) ---
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL") or None
ANTHROPIC_BASE_URL: str | None = os.getenv("ANTHROPIC_BASE_URL") or None

# --- Research Parameters ---
MAX_RESEARCHER_ITERATIONS: int = int(os.getenv("MAX_RESEARCHER_ITERATIONS", "15"))
MAX_CONCURRENT_RESEARCHERS: int = int(os.getenv("MAX_CONCURRENT_RESEARCHERS", "3"))
MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "250000"))


# --- Model Instance Factory ---

def _parse_provider(model_name: str) -> str | None:
    """Extract the provider name from a model string.

    Supports both "provider:model" and "provider/model" formats.

    Args:
        model_name: Model name string.

    Returns:
        The provider name, or None if no provider prefix found.
    """
    if ":" in model_name:
        return model_name.split(":", 1)[0]
    if "/" in model_name:
        return model_name.split("/", 1)[0]
    return None


def _get_base_url(model_name: str) -> str | None:
    """Determine the base_url based on the model's provider prefix.

    Args:
        model_name: Model name string.

    Returns:
        The base_url for the provider, or None if not set.
    """
    provider = _parse_provider(model_name)
    if provider == "openai":
        return OPENAI_BASE_URL
    if provider == "anthropic":
        return ANTHROPIC_BASE_URL
    return None


def create_chat_model(model_name: str, **kwargs):
    """Create a chat model instance with centralized configuration.

    Supports both "provider:model" (LangChain native) and "provider/model"
    (OpenRouter) formats. For "provider/model" format, automatically sets
    model_provider and passes the full string as the model name.

    Args:
        model_name: Model name in "provider:model" or "provider/model" format.
        **kwargs: Additional arguments passed to init_chat_model.

    Returns:
        A configured BaseChatModel instance.
    """
    base_url = _get_base_url(model_name)
    if base_url is not None:
        kwargs.setdefault("base_url", base_url)
    # "provider/model" format (e.g. OpenRouter): extract provider explicitly
    if ":" not in model_name and "/" in model_name:
        provider = model_name.split("/", 1)[0]
        kwargs.setdefault("model_provider", provider)
    return init_chat_model(model=model_name, **kwargs)


# --- Pre-built Model Instances ---
# Module-level singletons imported by other modules.

default_model = create_chat_model(DEFAULT_MODEL)
supervisor_model = create_chat_model(SUPERVISOR_MODEL)
summarization_model = create_chat_model(SUMMARIZATION_MODEL)
compress_model = create_chat_model(COMPRESS_MODEL, max_tokens=COMPRESS_MODEL_MAX_TOKENS)
report_writer_model = create_chat_model(WRITER_MODEL, max_tokens=REPORT_WRITER_MODEL_MAX_TOKENS)
draft_writer_model = create_chat_model(WRITER_MODEL, max_tokens=DRAFT_WRITER_MODEL_MAX_TOKENS)
