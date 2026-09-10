"""Environment-only configuration for the optional live generator."""

import os
from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when required live-demo configuration is missing."""


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.1


def load_openai_config() -> OpenAIConfig:
    api_key = os.getenv("HEALTH_COPILOT_API_KEY", "").strip()
    model = os.getenv("HEALTH_COPILOT_MODEL", "").strip()
    base_url = os.getenv("HEALTH_COPILOT_BASE_URL", "").strip() or None

    missing = []
    if not api_key:
        missing.append("HEALTH_COPILOT_API_KEY")
    if not model:
        missing.append("HEALTH_COPILOT_MODEL")
    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(
            f"Live generation requires {names}. Set them before running the CLI."
        )

    return OpenAIConfig(api_key=api_key, model=model, base_url=base_url)
