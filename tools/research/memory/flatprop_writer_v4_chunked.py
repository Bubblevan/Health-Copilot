"""Chunked v4 execution adapter retaining the minimal v3 proposition semantics."""

from __future__ import annotations

from typing import Any

try:
    from . import flat_proposition_writer_v3 as writer_v3
except ImportError:
    import flat_proposition_writer_v3 as writer_v3


CONTRACT_ID = "flat-proposition-extractor-v4-chunked"
MAX_COMPLETION_TOKENS = 8192


def writer_request(
    *,
    session_date: str,
    catalog: list[dict[str, Any]],
    system_prompt: str,
    model_alias: str,
) -> dict[str, Any]:
    request = writer_v3.writer_request(
        session_date=session_date,
        catalog=catalog,
        system_prompt=system_prompt,
        model_alias=model_alias,
        max_tokens=4096,
    )
    request["max_tokens"] = MAX_COMPLETION_TOKENS
    request["response_format"]["json_schema"]["name"] = "flat_proposition_packet_v4_chunked"
    return request


dynamic_output_schema = writer_v3.dynamic_output_schema
schema_sha256 = writer_v3.schema_sha256
validate_packet = writer_v3.validate_packet
