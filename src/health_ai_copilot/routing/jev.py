"""Small, opt-in client for TypeSafe's typed System One API."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://api.typesafe.ai"
HOSTED_BASE_URL = "https://jevtypesafeai.com"
DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Base error for the optional Jev integration."""


class JevConfigurationError(JevError):
    """Raised when Jev configuration is missing or invalid."""


class JevAPIError(JevError):
    """Raised when the TypeSafe API request fails or returns malformed data."""


@dataclass(frozen=True)
class JevResult:
    model: str
    answers: Mapping[str, Mapping[str, Any]]
    input_tokens: int
    output_tokens: int | None
    latency_ms: int
    cost_usd: float | None = None
    credits_remaining_usd: float | None = None


@dataclass(frozen=True)
class JevConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 20.0
    api_mode: str = "typesafe"

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise JevConfigurationError("JEV_API_KEY is required")
        if not self.model.strip():
            raise JevConfigurationError("JEV_MODEL must be non-empty")
        if not self.base_url.startswith("https://"):
            raise JevConfigurationError("JEV_BASE_URL must use HTTPS")
        if self.api_mode not in {"hosted", "typesafe"}:
            raise JevConfigurationError("JEV_API_MODE must be 'hosted' or 'typesafe'")
        if self.timeout_seconds <= 0:
            raise JevConfigurationError("JEV_TIMEOUT_SECONDS must be positive")

    @classmethod
    def from_env(cls, *, dotenv_path: Path | None = None) -> JevConfig:
        file_values = _read_jev_dotenv(dotenv_path or _project_root() / ".env")

        def setting(name: str, default: str = "") -> str:
            return _strip_quotes(os.environ.get(name, "").strip() or file_values.get(name, default))

        key = setting("JEV_API_KEY")
        if not key:
            raise JevConfigurationError(
                "JEV_API_KEY is required; set it in the environment or project .env"
            )
        try:
            timeout = float(setting("JEV_TIMEOUT_SECONDS", "20"))
        except ValueError as exc:
            raise JevConfigurationError("JEV_TIMEOUT_SECONDS must be a number") from exc
        api_mode = setting("JEV_API_MODE").lower()
        if not api_mode:
            api_mode = "hosted" if key.startswith("jv_live_") else "typesafe"
        if api_mode not in {"hosted", "typesafe"}:
            raise JevConfigurationError("JEV_API_MODE must be 'hosted' or 'typesafe'")
        default_base_url = HOSTED_BASE_URL if api_mode == "hosted" else DEFAULT_BASE_URL
        return cls(
            api_key=key,
            model=setting("JEV_MODEL", DEFAULT_MODEL),
            base_url=setting("JEV_BASE_URL", default_base_url).rstrip("/"),
            timeout_seconds=timeout,
            api_mode=api_mode,
        )


class JevClient:
    """Calls either the hosted decision API or TypeSafe's direct System One API."""

    def __init__(self, config: JevConfig | None = None) -> None:
        self.config = config or JevConfig.from_env()

    async def evaluate(
        self,
        *,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> JevResult:
        if not questions:
            raise ValueError("at least one named Jev question is required")
        payload = {"state": state, "model": self.config.model}
        payload["questions"] = {
            str(name): dict(question) for name, question in questions.items()
        }
        started = time.perf_counter()
        response = await asyncio.to_thread(self._post, payload)
        latency_ms = round((time.perf_counter() - started) * 1000)
        answers = response.get("answers")
        usage = response.get("usage")
        if not isinstance(answers, Mapping) or not isinstance(usage, Mapping):
            raise JevAPIError("Jev response is missing answers or usage")
        if any(not isinstance(answer, Mapping) for answer in answers.values()):
            raise JevAPIError("Jev response contains an invalid answer")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (input_tokens,)
        ):
            raise JevAPIError("Jev response contains invalid input-token usage")
        if output_tokens is None and self.config.api_mode == "typesafe":
            raise JevAPIError("TypeSafe response contains no output-token usage")
        if output_tokens is not None and (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
        ):
            raise JevAPIError("Jev response contains invalid output-token usage")
        cost_usd = _optional_nonnegative_number(usage.get("cost_usd"), "cost_usd")
        credits_remaining_usd = _optional_nonnegative_number(
            usage.get("credits_remaining_usd"), "credits_remaining_usd"
        )
        if self.config.api_mode == "hosted" and cost_usd is None:
            raise JevAPIError("hosted Jev response contains no actual cost")
        return JevResult(
            model=str(response.get("model", self.config.model)),
            answers={str(name): dict(answer) for name, answer in answers.items()},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            credits_remaining_usd=credits_remaining_usd,
        )

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        endpoint_path = "/api/v1/decide" if self.config.api_mode == "hosted" else "/v1/systemone"
        request = Request(
            f"{self.config.base_url}{endpoint_path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise JevAPIError(f"Jev API returned HTTP {exc.code}") from None
        except (URLError, TimeoutError) as exc:
            raise JevAPIError(f"Jev API request failed: {type(exc).__name__}") from None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JevAPIError(f"Jev API returned invalid JSON: {type(exc).__name__}") from None
        if not isinstance(parsed, Mapping):
            raise JevAPIError("Jev API response must be a JSON object")
        return parsed


def _optional_nonnegative_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise JevAPIError(f"Jev response contains invalid {field}")
    return float(value)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_jev_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, separator, value = stripped.partition("=")
        if separator and name.strip() in {
            "JEV_API_KEY",
            "JEV_API_MODE",
            "JEV_MODEL",
            "JEV_BASE_URL",
            "JEV_TIMEOUT_SECONDS",
        }:
            values[name.strip()] = _dotenv_value(value.strip())
    return values


def _dotenv_value(value: str) -> str:
    if value and value[0] in {"'", '"'}:
        closing_quote = value.find(value[0], 1)
        if closing_quote >= 0:
            return value[1:closing_quote].strip()
        return value
    return value.split("#", maxsplit=1)[0].strip()


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "JevAPIError",
    "JevClient",
    "JevConfig",
    "JevConfigurationError",
    "JevError",
    "JevResult",
]
