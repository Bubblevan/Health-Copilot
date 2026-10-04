"""Resume Memora LoCoMo after a local-Qwen judge-format compatibility amendment.

The upstream judge prompt and model are unchanged. This adapter accepts an
unambiguous official label when Qwen appends rationale text and gives the same
judge request a larger output allowance so the label is not truncated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


RUNNER_PATH = Path(__file__).with_name("run_memora_qwen_locomo.py")
JUDGE_MAX_TOKENS = 256
JUDGE_FORMAT_RETRIES = 1
READER_PROVIDER_RETRIES = 1


def call_reader_with_provider_retry(call, retries: int = READER_PROVIDER_RETRIES):
    result = call()
    retry_count = 0
    while (
        isinstance(result, tuple)
        and result
        and str(result[0]).startswith("ERROR: LLM call failed.")
        and retry_count < retries
    ):
        retry_count += 1
        result = call()
    return result, retry_count


def parse_local_judge_output(text: str, extract_json) -> tuple[str, bool, str]:
    candidate = extract_json(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        label = str(payload.get("label", "")).upper()
        if label in {"CORRECT", "WRONG"}:
            return label, False, str(payload.get("reason", ""))

    decoder = json.JSONDecoder()
    decoded_labels: list[tuple[str, str]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            label = str(value.get("label", "")).upper()
            if label in {"CORRECT", "WRONG"}:
                decoded_labels.append((label, str(value.get("reason", ""))))
    unique_labels = {label for label, _ in decoded_labels}
    if len(unique_labels) == 1:
        label = unique_labels.pop()
        explanation = next((reason for value, reason in decoded_labels if value == label), "")
        return label, True, explanation
    if len(unique_labels) > 1:
        raise RuntimeError("Local Qwen judge returned conflicting JSON labels")

    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    normalized_last_line = last_line.strip("`*_ \"'.,!?;:").upper()
    if normalized_last_line in {"CORRECT", "WRONG"}:
        return normalized_last_line, True, ""
    labeled_line = re.fullmatch(r"label\s*:\s*(CORRECT|WRONG)", last_line.strip(), re.IGNORECASE)
    if labeled_line:
        return labeled_line.group(1).upper(), True, ""
    terminal_label = re.search(r"\b(CORRECT|WRONG)\b[\s`*_\"'.,!?;:]*$", last_line, re.IGNORECASE)
    if terminal_label:
        explicit_labels = re.findall(r"\b(CORRECT|WRONG)\b", text)
        label = terminal_label.group(1).upper()
        if explicit_labels and {value.upper() for value in explicit_labels} == {label}:
            return label, True, ""
    raise RuntimeError(f"Local Qwen judge output contains no unambiguous label: {text[:200]!r}")


def _load_runner():
    spec = importlib.util.spec_from_file_location("memora_qwen_locomo_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the pinned Memora LoCoMo runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_judge_compat(runner: Any) -> None:
    def compatible_judge(searcher, helpers, qa, response, question_id, strategy):
        prompt = helpers["ACCURACY_PROMPT"].format(
            question=qa["question"],
            gold_answer=qa["answer"],
            generated_answer=response,
        )
        last_error = None
        for attempt in range(JUDGE_FORMAT_RETRIES + 1):
            with helpers["call_context"](f"Memora-{strategy}", question_id, "judge_local"):
                result = searcher.llm_client.chat.completions.create(
                    model=runner.READER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    seed=42,
                    max_tokens=JUDGE_MAX_TOKENS,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            content = result.choices[0].message.content or ""
            try:
                label, format_fallback, explanation = parse_local_judge_output(
                    content, helpers["extract_json"]
                )
            except RuntimeError as error:
                last_error = error
                continue
            return {
                "label": label,
                "correct": int(label == "CORRECT"),
                "explanation": explanation or content.strip(),
                "format_fallback": format_fallback,
                "format_retry_count": attempt,
                "judge_max_tokens": JUDGE_MAX_TOKENS,
            }
        raise RuntimeError(f"Local Qwen judge remained unparseable after retry: {last_error}")

    runner._judge = compatible_judge


def _install_reader_provider_retry(runner: Any) -> None:
    original_run_strategy = runner._run_strategy

    def run_strategy_with_reader_retry(dataset, questions, run_root, identity, cfg, strategy, helpers):
        searcher_class = helpers["MemoraSearch"]
        original_answer_question = searcher_class.answer_question

        def answer_question_with_retry(searcher, *args, **kwargs):
            result, retry_count = call_reader_with_provider_retry(
                lambda: original_answer_question(searcher, *args, **kwargs)
            )
            if retry_count:
                active = helpers["current_call_context"]()
                runner._append_jsonl(
                    run_root / "provider_retry_events.jsonl",
                    {
                        "question_id": active.question_id,
                        "strategy": strategy,
                        "stage": "reader_answer",
                        "retry_count": retry_count,
                        "condition": "explicit ERROR: LLM call failed. response",
                        "run_identity": identity,
                    },
                )
            return result

        searcher_class.answer_question = answer_question_with_retry
        try:
            return original_run_strategy(dataset, questions, run_root, identity, cfg, strategy, helpers)
        finally:
            searcher_class.answer_question = original_answer_question

    runner._run_strategy = run_strategy_with_reader_retry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", choices=("semantic", "prompt"), default=["semantic", "prompt"])
    args = parser.parse_args()
    runner = _load_runner()
    _install_judge_compat(runner)

    run_root = runner.RUN_ROOT
    manifest_path = run_root / "protocol_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Judge compatibility mode requires the existing full-run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    amendment = {
        "run_identity": manifest["identity"]["identity_sha256"],
        "original_runner_sha256": manifest["runner_sha256"],
        "compatibility_wrapper_sha256": runner._sha256_file(Path(__file__).resolve()),
        "judge_model": "same local Qwen3-8B Q4_K_M via loopback llama.cpp",
        "judge_prompt": "unchanged pinned Microsoft Memora LoCoMo ACCURACY_PROMPT",
        "judge_response_format": "json_object unchanged",
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "format_retries": JUDGE_FORMAT_RETRIES,
        "reader_provider_retries": READER_PROVIDER_RETRIES,
        "parsing": "accept one unambiguous JSON label object anywhere, a strict final label/label line, or one unambiguous terminal label token on the final nonempty line",
        "reader_retry_condition": "retry once only on the explicit generic ERROR: LLM call failed. response; a second failure remains INFRA_FAILURE",
        "scope": "evaluation formatting compatibility plus one reader transport retry on explicit provider failure; no failed response is scored and answer prompts, memory stores, retrieval, and metrics remain unchanged",
    }
    runner._write_json(run_root / "judge_format_amendment.json", amendment)
    _install_reader_provider_retry(runner)
    summary = runner.run(argparse.Namespace(phase="full", strategies=args.strategies))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
