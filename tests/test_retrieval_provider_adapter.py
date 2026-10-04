import asyncio
from types import SimpleNamespace

from health_ai_copilot.providers.retrieval import CommonMedicalKBManifest, FrozenMedicalRAGProvider


class Search:
    def search(self, query, top_k=5):
        return [SimpleNamespace(source_id="doc-1", title="Guidance", excerpt="Observed.",
                                source_url="https://example.test", score=0.5)]


def test_frozen_rag_provider_returns_manifest_and_stable_evidence_identity() -> None:
    manifest = CommonMedicalKBManifest(
        "kb-v1", ("source-a",), "research-use", ("a" * 64,), "b" * 64,
        {"k1": 1.2}, "encoder@sha", {"rrf_k": 60}, 3,
    )
    provider = FrozenMedicalRAGProvider(Search(), manifest)
    result = asyncio.run(provider.retrieve(query="question", context=SimpleNamespace(request_id="r")))
    assert result.corpus_id == "kb-v1"
    assert len(result.evidence_sha256) == 64
    assert result.evidence[0].evidence_id == "doc-1"
