# MEM-3B0Q R4 Read-Only Runtime Preflight - 2026-10-01

Status: `PASS_READ_ONLY_PREFLIGHT`. This evidence records GET-only inspection; no inference was performed and the sampler-init request remains unrun.

## Runtime

- Repository head: `08ca0d45ce5ea352b691755f0f030dea7b75c0be`
- Standalone preflight SHA-256: `f71813101221e781b10b29afe0a65675f35e00e536773991b26c3ffb0b7b6273`
- Frozen protocol manifest SHA-256: `278dd2079d0196a5d7b58111b3604a5a11ebe92bfdc22bc5fe868e8d3b547e1f`
- Runtime-source amendment SHA-256: `57564c2b834d4f798cf6415aacf200cf8b6bef9c3d3c1aeddb7ad0b2a79a2955`
- Existing listener: loopback `127.0.0.1:8081`, PID `73952`; no additional server was started and port `8092` was not touched.
- Server: llama.cpp `b10068-571d0d540`, binary SHA-256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`.
- Model: Qwen3-8B Q4_K_M, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Launch/runtime settings observed: context `131072`, `99` GPU layers, Flash Attention `on`, Q4_0 K/V cache, one parallel slot.

## Endpoint Evidence

Only `GET /health`, `GET /props`, `GET /v1/models`, and `GET /slots` were issued. All returned HTTP 200 and passed the strict preflight validator.

`/props.default_generation_settings.params` reported `top_k=40`, `top_p=0.949999988079071`, `min_p=0.05000000074505806`, and `repeat_penalty=1.0`; values pass the frozen tolerance. The one `/slots` entry was idle with `n_ctx=131072`. Its response also contained historical `task_prev.params`; these were not treated as global defaults.

| Endpoint | Raw response SHA-256 |
|---|---|
| `/health` | `a29ee2b15c494311c52521766e44af56a3ad2248e7a8ab465e5206463c13d288` |
| `/props` | `45510a3f068d32140d27954cc10d6bd196098b3666d908a20cdb782fd4a0f1f1` |
| `/v1/models` | `ad941c0f5d96f1a414b13bd98e584cf2d453cd300a8e606c4f487db708316516` |
| `/slots` | `7e5599646377e51eb59935a99616f2337d921c5a88918ea245cfdc1edf597d04` |

## Gate State

- Read-only runtime preflight: `PASS`
- Completion POSTs by this preflight: `0`
- R4-P01 sampler-init gate: `NOT_RUN`
- R4 identity qualification: `PENDING`
- `MEM3B0Q_MEM3B1_READY=NO`

The independent code review approved only readiness for this read-only preflight. It did not authorize inference. No LongMemEval/MedMemoryBench scoring, MemoryStore operation, or B1 work occurred.
