"""OpenAI-compatible grounded generator for the optional live demo."""

import json
from collections.abc import Sequence
from dataclasses import asdict

from ..config import ConfigurationError, load_openai_config
from ..contracts import Evidence, GenerationDraft
from ..runtime import (
    OpenAICompatibleProviderExecutor,
    ProviderCallKind,
    ProviderExecutor,
    ProviderFailure,
    ProviderRequest,
    RunContext,
)
from .base import GenerationError

_SYSTEM_PROMPT = """你是患者教育原型中的回答生成器。

规则：
- 只能提供患者教育信息，不得诊断、处方或替代医生决策。
- 只能使用用户问题下方 supplied_evidence 中的信息；不要把模型预训练知识当作证据。
- 如果 supplied_evidence 不足以可靠回答，必须 abstain=true。
- citation_ids 只能填写 supplied_evidence 中出现的 source_id。
- 只输出 JSON，不要输出 Markdown、解释文字或代码围栏。

JSON 格式：
{"answer":"...","citation_ids":["source-id"],"abstain":false}
"""


class OpenAICompatibleGenerator:
    """Call an OpenAI-compatible chat endpoint with low-temperature JSON output."""

    def __init__(
        self,
        *,
        provider_executor: ProviderExecutor | None = None,
        model: str | None = None,
        temperature: float | None = None,
        runtime: RunContext | None = None,
    ) -> None:
        if provider_executor is None:
            try:
                config = load_openai_config()
                provider_executor = OpenAICompatibleProviderExecutor(config)
            except (ConfigurationError, ProviderFailure) as exc:
                raise GenerationError(str(exc)) from exc
            model = config.model
            temperature = config.temperature
        self._provider_executor = provider_executor
        self._model = model or "injected-provider-model"
        self._temperature = 0.1 if temperature is None else temperature
        self._runtime = runtime

    def _prompt(self, question: str, evidence: Sequence[Evidence]) -> str:
        evidence_json = json.dumps(
            [asdict(item) for item in evidence], ensure_ascii=False, indent=2
        )
        return (
            f"用户问题：\n{question}\n\n"
            f"supplied_evidence（仅允许使用这些证据）：\n{evidence_json}"
        )

    @staticmethod
    def _parse(content: str | None) -> GenerationDraft:
        if not content or not content.strip():
            raise GenerationError("generator returned an empty response")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise GenerationError("generator returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise GenerationError("generator JSON must be an object")

        answer = payload.get("answer")
        citation_ids = payload.get("citation_ids")
        if "abstain" not in payload:
            raise GenerationError("generator JSON field 'abstain' is required")
        abstain = payload["abstain"]
        if not isinstance(answer, str):
            raise GenerationError("generator JSON field 'answer' must be a string")
        if not isinstance(citation_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in citation_ids
        ):
            raise GenerationError(
                "generator JSON field 'citation_ids' must be a list of non-empty strings"
            )
        if not isinstance(abstain, bool):
            raise GenerationError("generator JSON field 'abstain' must be boolean")
        if not abstain and not answer.strip():
            raise GenerationError("generator returned an empty answer")

        return GenerationDraft(
            answer=answer.strip(),
            citation_ids=[item.strip() for item in citation_ids],
            abstain=abstain,
        )

    def generate(self, question: str, evidence: Sequence[Evidence]) -> GenerationDraft:
        try:
            response = self._provider_executor.execute(
                ProviderRequest.create(
                    kind=ProviderCallKind.GENERATOR,
                    model=self._model,
                    messages=(
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": self._prompt(question, evidence)},
                    ),
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                    timeout_seconds=30.0,
                ),
                self._runtime or RunContext.create("m0"),
            )
            content = response.content
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError("generation request failed") from exc
        return self._parse(content)
