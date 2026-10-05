import asyncio
import sys
import types
from types import SimpleNamespace

from health_ai_copilot.providers.model import ModelRequest, VllmModelProvider


def test_vllm_provider_sends_frozen_top_p_and_thinking_settings(monkeypatch) -> None:
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion),
            )

        async def create_completion(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
                model="qwen3-8b-system-v1",
                _request_id="request-1",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="A"), finish_reason="stop",
                )],
            )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    provider = VllmModelProvider(
        base_url="http://127.0.0.1:8001/v1",
        model="qwen3-8b-system-v1",
        default_temperature=0.0,
        default_top_p=1.0,
        max_output_tokens=2048,
        chat_template_kwargs={"enable_thinking": True},
    )

    reply = asyncio.run(provider.complete(ModelRequest(
        messages=({"role": "user", "content": "question"},),
        max_output_tokens=2048,
        json_schema={
            "name": "team_recruitment",
            "strict": True,
            "schema": {"type": "object", "required": ["teams"]},
        },
    )))

    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 1.0
    assert captured["max_tokens"] == 2048
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["name"] == "team_recruitment"
    assert reply.finish_reason == "stop"
