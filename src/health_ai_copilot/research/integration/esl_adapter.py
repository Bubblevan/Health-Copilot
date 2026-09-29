"""Schema-only ESL-like mapping proposal; no ESL query or gold loader exists here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ESLFieldMapping:
    source_surface: str
    target_surface: str
    mapping_status: str
    note: str

    def to_dict(self) -> dict[str, str]:
        return {"source_surface": self.source_surface, "target_surface": self.target_surface,
                "mapping_status": self.mapping_status, "note": self.note}


@dataclass(frozen=True)
class ESLAdapterFeasibility:
    adapter_id: str
    status: str
    mappings: tuple[ESLFieldMapping, ...]
    constraints: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"adapter_id": self.adapter_id, "status": self.status,
                "mappings": [item.to_dict() for item in self.mappings],
                "constraints": list(self.constraints)}


def esl_adapter_feasibility() -> ESLAdapterFeasibility:
    return ESLAdapterFeasibility(
        "esl-like-schema-v1", "SCHEMA_PROPOSAL_ONLY",
        (
            ESLFieldMapping("profile fields", "PatientStateRecord(PROFILE)", "PROPOSED", "Preserve subject scope and provenance."),
            ESLFieldMapping("timeline events", "PatientStateRecord(EVENT)", "PROPOSED", "Keep source timestamps; apply decision-time cutoff."),
            ESLFieldMapping("exam records", "PatientStateRecord(EXAM)", "PROPOSED", "Do not infer missing exams."),
            ESLFieldMapping("measurements", "PatientStateRecord(MEASUREMENT)", "PROPOSED", "Preserve units and time metadata in content/provenance."),
            ESLFieldMapping("conversation history", "PatientStateRecord(CONVERSATION)", "PROPOSED", "Do not treat assistant output as patient fact."),
        ),
        (
            "No ESL query, answer, evaluation/test row, or gold content was loaded.",
            "This adapter proposal does not make ESL data training-eligible.",
            "Dataset and split licenses/provenance must be reviewed before any row assignment.",
        ),
    )
