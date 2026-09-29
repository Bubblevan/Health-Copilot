"""Layer-B deterministic natural-language surfaces for latent owned facts."""

from __future__ import annotations

from datetime import datetime

from .schema import OwnedEvidenceRecord, OwnedPatientRecord


def realize_patient_record(
    *, record_id: str, subject_id: str, timestamp: datetime, fact_id: str,
    value: str, key: str, record_type: str = "EVENT", authority: str = "SYNTHETIC_RECORD",
    conversation_kind: str = "USER_CONFIRMED_STATE",
) -> OwnedPatientRecord:
    if record_type == "CONVERSATION":
        conversation_prefix = {
            "USER_PREFERENCE": "the user's confirmed preference",
            "PREVIOUS_INSTRUCTION": "the user's previous confirmed instruction",
            "PREVIOUS_PLAN_CODE": "the prior user-confirmed plan code",
            "PREVIOUS_CORRECTION": "the user's confirmed correction",
            "USER_CONFIRMED_STATE": "user-confirmed synthetic state",
        }
        if conversation_kind not in conversation_prefix:
            raise ValueError("unsupported synthetic conversation-memory kind")
        content = (f"Synthetic conversation records {conversation_prefix[conversation_kind]} "
                   f"{value} for {key}.")
        authority = "USER_CONFIRMED_SYNTHETIC"
    else:
        content = f"Synthetic {record_type.lower()} encodes {value} for {key}."
    return OwnedPatientRecord(
        record_id=record_id, subject_id=subject_id, timestamp=timestamp,
        record_type=record_type, latent_fact_ids=(fact_id,),
        natural_language_content=content, retrieval_terms=(key,), authority=authority,
    )


def realize_evidence_record(
    *, source_id: str, source_family: str, publication_time: datetime,
    fact_id: str, value: str, key: str,
    effective_time: datetime | None = None,
    effective_until: datetime | None = None,
) -> OwnedEvidenceRecord:
    return OwnedEvidenceRecord(
        source_id=source_id, source_family=source_family,
        publication_time=publication_time, effective_time=effective_time,
        latent_fact_ids=(fact_id,),
        natural_language_content=f"Synthetic {source_family.lower()} record maps {key} to {value}.",
        retrieval_terms=(key,), effective_until=effective_until,
    )
