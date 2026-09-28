import json
from collections import Counter
from pathlib import Path

from e2_support import capability

from health_ai_copilot.capabilities import (
    CapabilityEligibility,
    CapabilitySourceCatalog,
    E2WorkerRole,
    production_worker_capabilities,
)
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import load_knowledge_scope
from health_ai_copilot.runtime.components import ComponentManifest


def test_capability_contract_canonical_hash_is_order_independent_and_round_trips():
    left = capability(
        E2WorkerRole.PUBLIC_HEALTH,
        "public_health",
        "risk_factors",
        families=("public_health", "reviewed_public_health"),
        domains=("risk_factors", "measurement"),
    )
    right = capability(
        E2WorkerRole.PUBLIC_HEALTH,
        "public_health",
        "risk_factors",
        families=("reviewed_public_health", "public_health"),
        domains=("measurement", "risk_factors"),
    )
    assert left.canonical_json == right.canonical_json
    assert left.contract_hash == right.contract_hash
    assert type(left).from_dict(left.to_dict()).contract_hash == left.contract_hash


def test_e2_contract_hashes_enter_component_manifest_without_changing_legacy_bytes():
    spec = capability(E2WorkerRole.PUBLIC_HEALTH, "public_health", "facts")
    legacy = ComponentManifest("legacy-profile", ())
    extended = ComponentManifest(
        "e2-profile",
        (),
        capability_contracts=(spec.manifest_ref(),),
    )
    assert "capability_contracts" not in legacy.to_dict()
    assert extended.to_dict()["capability_contracts"] == [spec.manifest_ref().to_dict()]
    assert legacy.manifest_hash == ComponentManifest("legacy-profile", ()).manifest_hash
    assert extended.manifest_hash != legacy.manifest_hash


def test_current_reviewed_corpus_only_qualifies_public_health_surface():
    cards = load_knowledge_cards("data/knowledge_cards")
    scope = load_knowledge_scope("data/knowledge_scope.json", cards)
    catalog = CapabilitySourceCatalog.from_reviewed_knowledge(
        cards,
        scope,
        publisher_families={
            "World Health Organization": "public_health",
            "Centers for Disease Control and Prevention": "public_health",
            "国家卫生健康委员会": "public_health",
        },
    )
    family_counts = Counter(item.source_family for item in catalog.sources)
    capabilities = {item.role: item for item in production_worker_capabilities()}
    assert len(cards) == 30
    assert family_counts == {"public_health": 30}
    assert capabilities[E2WorkerRole.PUBLIC_HEALTH].eligibility == CapabilityEligibility.ELIGIBLE
    assert capabilities[E2WorkerRole.GUIDELINE].eligibility == CapabilityEligibility.NOT_YET_ELIGIBLE
    assert capabilities[E2WorkerRole.LITERATURE].eligibility == CapabilityEligibility.NOT_YET_ELIGIBLE


def test_machine_readable_capability_registry_matches_current_canonical_contracts():
    cards = load_knowledge_cards("data/knowledge_cards")
    scope = load_knowledge_scope("data/knowledge_scope.json", cards)
    catalog = CapabilitySourceCatalog.from_reviewed_knowledge(
        cards,
        scope,
        publisher_families={
            "World Health Organization": "public_health",
            "Centers for Disease Control and Prevention": "public_health",
            "国家卫生健康委员会": "public_health",
        },
    )
    payload = json.loads(
        Path("docs/research/multi_agent/e2_worker_capabilities.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["source_catalog_hash"] == catalog.catalog_hash
    assert payload["source_catalog"] == catalog.to_dict()
    assert payload["capabilities"] == [
        {**item.to_dict(), "contract_hash": item.contract_hash}
        for item in production_worker_capabilities()
    ]
    assert payload["test_accessed"] is False


def test_ineligible_contract_stays_hashed_but_cannot_be_treated_as_live_capability():
    guideline = next(
        item for item in production_worker_capabilities() if item.role == E2WorkerRole.GUIDELINE
    )
    assert guideline.contract_hash
    assert guideline.eligibility_reason
    assert guideline.allowed_source_families == ("reviewed_guideline",)
