"""Optional local LongMemEval semantic judge for already-frozen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from mem1_artifacts import (
    append_jsonl,
    canonical_json,
    read_jsonl,
    sha256_file,
    verify_hash_sidecar,
    write_hash_sidecar,
)

ROOT = Path(__file__).resolve().parents[3]
RESULT_NAME = "local_judge_predictions.jsonl"
SIDECAR_NAME = "local_judge_predictions.sha256"


def build_longmemeval_judge_prompt(
    question: str,
    expected: str,
    predicted: str,
    category: str,
    question_id: str,
) -> str:
    if question_id.endswith("_abs"):
        template = (
            "I will give you an unanswerable question, an explanation, and "
            "a response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            "Question: {q}\n\nExplanation: {a}\n\nModel Response: {r}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    elif category == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. In addition, do not penalize off-by-one errors for the number "
            "of days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still "
            "correct.\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\n"
            "Model Response: {r}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif category == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response "
            "should be considered as correct as long as the updated answer is "
            "the required answer.\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\n"
            "Model Response: {r}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif category == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\nQuestion: {q}\n\n"
            "Rubric: {a}\n\nModel Response: {r}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no.\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\n"
            "Model Response: {r}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    return template.format(q=question, a=expected, r=predicted)


def _identity(prediction_sha256: str, row: dict[str, Any], prompt_sha256: str) -> dict[str, str]:
    material = {
        "prediction_sha256": prediction_sha256,
        "system": row["system"],
        "question_id": row["question_id"],
        "judge_prompt_sha256": prompt_sha256,
    }
    return {**material, "identity_sha256": hashlib.sha256(canonical_json(material)).hexdigest()}


def run(run_dir: Path) -> dict[str, Any]:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.pop("OPENAI_API_KEY", None)
    manifest_path = run_dir / "run_manifest.json"
    prediction_path = run_dir / "predictions.jsonl"
    prediction_sidecar = run_dir / "predictions.sha256"
    if not manifest_path.is_file() or not prediction_sidecar.is_file():
        raise RuntimeError("Local judge requires a frozen predictions artifact")
    if not verify_hash_sidecar(prediction_path, prediction_sidecar):
        raise RuntimeError("Prediction hash is not valid; local judge will not run")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "DEV" or manifest.get("test_access") is not False:
        raise RuntimeError("Local judge only accepts the frozen DEV artifact")
    if manifest.get("track") == "main_local_only_context_controlled":
        bundles_sidecar = run_dir / "context_bundles.sha256"
        if not bundles_sidecar.is_file() or not verify_hash_sidecar(
            run_dir / "context_bundles.jsonl", bundles_sidecar
        ):
            raise RuntimeError("ContextBundle artifact must also be frozen before judging")

    prediction_sha256 = prediction_sidecar.read_text(encoding="ascii").split()[0]
    systems = manifest["roles"]["memory_system"]["systems"]
    question_ids = manifest["question_ids"]
    predictions = {}
    for row in read_jsonl(prediction_path):
        predictions[(row.get("system"), row.get("question_id"))] = row
    expected = {(system, qid) for system in systems for qid in question_ids}
    if set(predictions) != expected or any(
        row.get("quality_status") != "OK" or not isinstance(row.get("predicted"), str)
        for row in predictions.values()
    ):
        raise RuntimeError("Local judge requires a complete, successful frozen prediction matrix")

    model_protocol = json.loads((ROOT / "docs/research/memory/model_protocol.json").read_text(encoding="utf-8"))
    reader_sha = model_protocol["main_track"]["sha256"]
    from agents_memory.healthcopilot_provider import (
        ProviderConfig,
        answer_request_kwargs,
        call_context,
        configure_call_ledger,
        reader_client,
        release_local_embedding_runtimes,
    )

    config = ProviderConfig.from_env()
    if config.reader_model != manifest["roles"]["reader_answer_model"]["model"]:
        raise RuntimeError("Local judge reader differs from the frozen prediction reader")
    prompt_sha = hashlib.sha256(
        b"longmemeval-category-prompts-v1;question_id.endswith(_abs)"
    ).hexdigest()
    judge_manifest = {
        "schema_version": 1,
        "metric": "local_qwen_judge_accuracy",
        "provider": "local_qwen",
        "model": config.reader_model,
        "artifact_sha256": reader_sha,
        "temperature": 0,
        "seed": 42,
        "enable_thinking": False,
        "max_new_tokens": 10,
        "prediction_sha256": prediction_sha256,
        "judge_prompt_sha256": prompt_sha,
        "official_gpt4o_accuracy": False,
        "manual_calibration": "REQUIRED_BEFORE_HEADLINE_USE",
    }
    judge_manifest_path = run_dir / "local_judge_manifest.json"
    encoded_manifest = json.dumps(judge_manifest, indent=2, sort_keys=True) + "\n"
    if judge_manifest_path.exists():
        existing = json.loads(judge_manifest_path.read_text(encoding="utf-8"))
        if existing != judge_manifest:
            raise FileExistsError("Refusing to change a frozen local judge protocol")
    else:
        judge_manifest_path.write_text(encoded_manifest, encoding="utf-8")

    result_path = run_dir / RESULT_NAME
    sidecar_path = run_dir / SIDECAR_NAME
    if sidecar_path.exists():
        if not verify_hash_sidecar(result_path, sidecar_path):
            raise RuntimeError("Frozen local judge prediction hash mismatch")
        return json.loads((run_dir / "local_judge_metrics.json").read_text(encoding="utf-8"))

    configure_call_ledger(run_dir / "local_judge_call_ledger.jsonl")
    completed = {
        (row.get("system"), row.get("question_id")): row
        for row in read_jsonl(result_path)
    }
    for key in sorted(expected):
        prediction = predictions[key]
        prompt = build_longmemeval_judge_prompt(
            question=prediction["question"],
            expected=str(prediction["ground_truth"]),
            predicted=prediction["predicted"],
            category=str(prediction["category"]),
            question_id=prediction["question_id"],
        )
        identity = _identity(prediction_sha256, prediction, prompt_sha)
        prior = completed.get(key)
        if prior is not None and prior.get("cache_identity") == identity and prior.get("judge_status") == "OK":
            continue
        result = None
        error = None
        try:
            client = reader_client("judge_local", prediction["system"], prediction["question_id"], config)
            with call_context(prediction["system"], prediction["question_id"], "judge_local"):
                response = client.chat.completions.create(
                    **answer_request_kwargs(
                        config,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                    )
                )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("Local judge returned no text")
            first = content.strip().lower().split(maxsplit=1)[0].strip(".,!?:;")
            if first not in {"yes", "no"}:
                raise RuntimeError("Local judge response was not a binary yes/no")
            result = first == "yes"
        except Exception as caught:
            error = caught
        row = {
            "system": prediction["system"],
            "question_id": prediction["question_id"],
            "prediction_sha256": prediction_sha256,
            "local_qwen_judge_correct": result,
            "judge_status": "OK" if result is not None else "INFRA_FAILURE",
            "error_type": type(error).__name__ if error else None,
            "cache_identity": identity,
        }
        append_jsonl(result_path, row)
        completed[key] = row

    if not verify_hash_sidecar(prediction_path, prediction_sidecar):
        raise RuntimeError("Prediction artifact changed while local judging was running")
    result_rows = list(completed.values())
    valid = [row for row in result_rows if row.get("judge_status") == "OK"]
    accuracy = (
        sum(bool(row["local_qwen_judge_correct"]) for row in valid) / len(valid)
        if valid else None
    )
    metrics = {
        "metric": "local_qwen_judge_accuracy",
        "accuracy": accuracy,
        "n": len(valid),
        "infra_failure_n": len(result_rows) - len(valid),
        "prediction_sha256": prediction_sha256,
        "manual_calibration": "REQUIRED_BEFORE_HEADLINE_USE",
        "headline_eligible": False,
    }
    if len(result_rows) == len(expected) and len(valid) == len(expected):
        write_hash_sidecar(result_path, sidecar_path)
    (run_dir / "local_judge_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    release_local_embedding_runtimes()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
