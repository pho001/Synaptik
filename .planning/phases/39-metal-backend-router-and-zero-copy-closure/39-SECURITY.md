---
phase: 39
slug: metal-backend-router-and-zero-copy-closure
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 39 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Prepared execution to Metal route router | Already-selected Metal regions are routed to MPSGraph, custom-kernel seam, or explicit fallback. | Route decision, backend path, rejected reason |
| MPSGraph native result to caller buffer | Native result storage may be copied into caller output buffer. | Device result handle, output handle, copy timing stats |
| Coverage reports to users | Route and copy claims must be backed by trace/report evidence. | Route counts, copy strategy, fallback counters |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-39-01 | Integrity | Execution router | mitigate | Router scope remains inside already-selected `GPU_METAL` regions and tests assert backend selection traces. | closed |
| T-39-02 | Repudiation | Custom-kernel route seam | mitigate | `CUSTOM_KERNEL` remains rejected with explicit capability evidence until real kernels and parity tests exist. | closed |
| T-39-03 | Integrity | Zero-copy claims | mitigate | `TRUE_OUTPUT_BUFFER_WRITE` is reserved for sentinel/alias proof; current behavior is `MPSGRAPH_RESULT_COPY`. | closed |
| T-39-04 | Repudiation | Milestone audit evidence | mitigate | Final verification maps every router and copy claim to trace/report fields and focused tests. | closed |

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-02 | 4 | 4 | 0 | Codex |

## Sign-Off

- [x] All threats have a disposition.
- [x] Accepted risks documented.
- [x] `threats_open: 0` confirmed.
- [x] `status: verified` set in frontmatter.

**Approval:** verified 2026-05-02
