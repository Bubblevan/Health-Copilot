from __future__ import annotations

from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1 as engine
from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1r3 as v1r3


def test_v1r3_scoring_adapter_wraps_only_v1_requests() -> None:
    with v1r3._configured_engine():
        pack, requests, scoring = engine._build_inputs()

    assert len(pack["cases"]) == 3
    assert set(requests) == set(engine.CASE_IDS)
    assert all(set(record) == {"request"} for record in scoring.values())
    assert all(record["request"]["response_format"]["type"] == "json_schema" for record in scoring.values())


def test_v1r3_lock_caps_new_posts_and_pins_prior_response() -> None:
    lock = v1r3.build_lock_payload()

    assert lock["max_completion_posts"] == 2
    assert lock["run_id"] == v1r3.OUTPUT_ROOT.name
    assert lock["requests"] == v1r3.v1r2.build_lock_payload()["requests"]
    assert "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/EVTCONF-01/response_body.bin" in lock["dependencies"]
