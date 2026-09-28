import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools" / "research" / "memory"
sys.path.insert(0, str(TOOLS))
import final_reader_contract as reader_contract


def test_final_contract_is_locked_and_v3_adds_only_the_official_date():
    contract, contract_sha, system_template, user_template = reader_contract.load_final_reader_contract()

    assert contract["locked_for_mem2_plus"] is True
    assert contract["decision"]["selected_by_protocol_not_dev_score"] is True
    assert contract_sha == reader_contract._sha256_file(reader_contract.CONTRACT_PATH)
    assert user_template.replace("Current Date: <<QUESTION_DATE>>\n", "") == reader_contract.V1_USER_TEMPLATE
    assert contract["template"]["v1_canonical_message_template_sha256"] == "0ff70b000bd4b43db85ae587731fea35b691febc815cfbb57c2acb1c8b295b2b"

    messages = reader_contract.build_reader_messages(
        "What happened yesterday?",
        "2023/06/14 (Wed) 13:43",
        "The event happened on June 13.",
        system_template=system_template,
        user_template=user_template,
    )
    assert messages == [
        {"role": "system", "content": reader_contract.SYSTEM_V1},
        {
            "role": "user",
            "content": (
                "Memory context:\nThe event happened on June 13.\n\n"
                "Current Date: 2023/06/14 (Wed) 13:43\nQuestion: What happened yesterday?"
            ),
        },
    ]


def test_contract_binding_rejects_hash_or_template_drift():
    contract, contract_sha, _, _ = reader_contract.load_final_reader_contract()
    binding = reader_contract.contract_binding(contract, contract_sha)
    reader_contract.verify_contract_binding(binding, contract, contract_sha)

    with pytest.raises(RuntimeError, match="not bound"):
        reader_contract.verify_contract_binding(
            {**binding, "sha256": "0" * 64}, contract, contract_sha
        )


def test_modified_template_fails_closed(tmp_path, monkeypatch):
    temp_root = tmp_path / "repo"
    docs = temp_root / "docs" / "research" / "memory"
    docs.mkdir(parents=True)
    contract_path = docs / "final_reader_contract.json"
    sidecar_path = docs / "final_reader_contract.json.sha256"
    template_path = docs / "shared_reader_v3_final.txt"
    shutil.copyfile(reader_contract.CONTRACT_PATH, contract_path)
    shutil.copyfile(reader_contract.CONTRACT_SIDECAR, sidecar_path)
    shutil.copyfile(reader_contract.TEMPLATE_PATH, template_path)
    template_path.write_text(template_path.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")

    monkeypatch.setattr(reader_contract, "ROOT", temp_root)
    monkeypatch.setattr(reader_contract, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(reader_contract, "CONTRACT_SIDECAR", sidecar_path)
    monkeypatch.setattr(reader_contract, "TEMPLATE_PATH", template_path)
    with pytest.raises(RuntimeError, match="template"):
        reader_contract.load_final_reader_contract()


def test_contract_and_sidecar_cannot_be_changed_together(tmp_path, monkeypatch):
    temp_root = tmp_path / "repo"
    docs = temp_root / "docs" / "research" / "memory"
    docs.mkdir(parents=True)
    contract_path = docs / "final_reader_contract.json"
    sidecar_path = docs / "final_reader_contract.json.sha256"
    template_path = docs / "shared_reader_v3_final.txt"
    modified_contract = reader_contract.CONTRACT_PATH.read_bytes() + b" "
    contract_path.write_bytes(modified_contract)
    sidecar_path.write_text(
        f"{reader_contract._sha256_bytes(modified_contract)}  {contract_path.name}\n",
        encoding="ascii",
    )
    shutil.copyfile(reader_contract.TEMPLATE_PATH, template_path)

    monkeypatch.setattr(reader_contract, "ROOT", temp_root)
    monkeypatch.setattr(reader_contract, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(reader_contract, "CONTRACT_SIDECAR", sidecar_path)
    monkeypatch.setattr(reader_contract, "TEMPLATE_PATH", template_path)
    with pytest.raises(RuntimeError, match="permanently pinned"):
        reader_contract.load_final_reader_contract()


def test_missing_question_date_is_rejected():
    _, _, system_template, user_template = reader_contract.load_final_reader_contract()
    with pytest.raises(ValueError, match="official question_date"):
        reader_contract.build_reader_messages(
            "Question?", "", "Context", system_template=system_template, user_template=user_template
        )


def test_mem1_shared_reader_uses_the_frozen_contract_date():
    runner_path = TOOLS / "run_mem1.py"
    spec = importlib.util.spec_from_file_location("mem1_final_reader_contract_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)

    messages = runner._shared_reader_messages("Question?", "Context", "2023/06/14 (Wed) 13:43")
    assert messages[1]["content"].endswith(
        "Current Date: 2023/06/14 (Wed) 13:43\nQuestion: Question?"
    )
