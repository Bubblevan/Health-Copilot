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


def load_openai_config(*, model_override: str | None = None) -> OpenAIConfig:
    api_key = _env_value("HEALTH_COPILOT_API_KEY")
    model = (
        model_override
        or _env_value("HEALTH_COPILOT_MODEL")
        or _env_value("HEALTH_COPILOT_MODEL_ID")
    )
    model = _strip_quotes(model)
    base_url = _env_value("HEALTH_COPILOT_BASE_URL") or None

    missing = []
    if not api_key:
        missing.append("HEALTH_COPILOT_API_KEY")
    if not model:
        missing.append("HEALTH_COPILOT_MODEL (or HEALTH_COPILOT_MODEL_ID)")
    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(
            f"Live generation requires {names}. Set them before running the CLI."
        )

    return OpenAIConfig(api_key=api_key, model=model, base_url=base_url)


def _env_value(name: str) -> str:
    return _strip_quotes(os.getenv(name, "").strip())


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value
