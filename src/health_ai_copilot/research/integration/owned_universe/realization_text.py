"""Layer-B deterministic natural-language surfaces for latent owned facts."""

from __future__ import annotations

from datetime import datetime

from .schema import OwnedEvidenceRecord, OwnedPatientRecord


def realize_patient_record(
    *, record_id: str, subject_id: str, timestamp: datetime, fact_id: str,
    value: str, key: str, record_type: str = "EVENT",
) -> OwnedPatientRecord:
    return OwnedPatientRecord(
        record_id=record_id, subject_id=subject_id, timestamp=timestamp,
        record_type=record_type, latent_fact_ids=(fact_id,),
        natural_language_content=f"Synthetic {record_type.lower()} encodes {value} for {key}.",
        retrieval_terms=(key,),
    )


def realize_evidence_record(
    *, source_id: str, source_family: str, publication_time: datetime,
    fact_id: str, value: str, key: str,
    effective_time: datetime | None = None,
) -> OwnedEvidenceRecord:
    return OwnedEvidenceRecord(
        source_id=source_id, source_family=source_family,
        publication_time=publication_time, effective_time=effective_time,
        latent_fact_ids=(fact_id,),
        natural_language_content=f"Synthetic {source_family.lower()} record maps {key} to {value}.",
        retrieval_terms=(key,),
    )
