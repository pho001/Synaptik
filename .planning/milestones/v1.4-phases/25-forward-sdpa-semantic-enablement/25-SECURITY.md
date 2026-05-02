---
phase: 25
slug: forward-sdpa-semantic-enablement
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 25 - Security

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-25-01 | Integrity | SDPA semantics | mitigate | Semantics and parity tests lock scale, rank, mask/no-mask, dtype, and backward interaction before support. | closed |
| T-25-02 | Spoofing / integrity | Metal SDPA support claim | mitigate | Metal direct SDPA is admitted only for verified unmasked FLOAT32 rank-3/rank-4 primitive DAG cases. | closed |
| T-25-03 | Spoofing / integrity | CUDA SDPA support claim | mitigate | CUDA direct SDPA remains visible `CAPABILITY_MISSING` instead of being reported as supported. | closed |
| T-25-04 | Repudiation | Hidden SDPA CPU fallback | mitigate | Coverage gate tests fail hidden tensor-array/CPU fallback for supported Metal SDPA. | closed |
| T-25-05 | Supply chain / integrity | Local profile artifacts | mitigate | Phase validation and verification record local profile artifacts as unstaged and not evidence. | closed |

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-02 | 5 | 5 | 0 | Codex inline security audit |

## Sign-Off

- [x] All threats have a disposition.
- [x] `threats_open: 0` confirmed.
- [x] `status: verified` set in frontmatter.

**Approval:** verified 2026-05-02
