# Factorized Event-Slot Confirmation v1r2 Preflight Failure

Status: invalid run attempt; zero completion POSTs.

The v1r2 registry mapping set the shared engine's `CASE_IDS` to `EVTPROJ-*`
before the legacy builder was called for its prior-pack overlap audit. That
builder validates its own frozen `EVTCONF-*` IDs against the shared global and
aborted with `confirmation_pack_identity_mismatch` during lock input
construction. The failure occurred before run reservation and before model
access.

v1r3 will isolate the legacy builder's case-ID global for the duration of that
read-only overlap audit, then restore the factorized registry for the frozen
case loop. No dataset, request, prompt, projection, scorer, or runtime change is
intended.
