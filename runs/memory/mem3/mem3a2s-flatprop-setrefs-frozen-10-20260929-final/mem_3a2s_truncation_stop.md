# MEM-3A.2S Stop Record

Status: `STOPPED`; completion gate: `MEM3A2S_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO`.

The run revalidated and imported all 102 historical MEM-3A.2R outcomes without provider replay. It then obtained 148 successful new local writer outcomes. The next local request, for source session `de43030f_1` (preflight ordinal 257), returned HTTP 200 but exhausted the frozen 16,384-token completion cap with `finish_reason=length`. The response was retained in the local ignored write-ahead cache; its hashes and request identity are recorded in `writer_truncation_stop.json`.

The protocol requires an immediate stop for this condition. No cap increase, retry, repair prompt, skip, or subsequent writer request was issued. The attempt totals are 102 historical imports, 148 successful new calls, and one truncated new call; 226 source identities remain unattempted. `writer_call_ledger.partial.jsonl` contains the 250 normalized outcomes (SHA-256 `979a779822115d2268fcb7cf03c980b31f8fe3ef56050961df4e802fe1423a16`); the failed transport event is separately recorded because the writer API raises before returning a ledger row.

No materialization, embedding, Dense top-8, projection, reader, or label-join stage started. Labels, 102 DEV, TEST, and MedMemoryBench were not accessed. Therefore no frozen-ten quality or retrieval metrics exist for this run, and `MEM3A2S_FLAT_NO_REVISION` is not established. This stop record is operational evidence only, not a benchmark result or performance claim.

The historical routing and MEM-3A.2R artifacts remain unchanged. The exact-ref set normalizer and its tests remain valid independently of this incomplete corpus run.
