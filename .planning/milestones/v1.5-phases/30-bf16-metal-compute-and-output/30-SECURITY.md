---
phase: 30
slug: bf16-metal-compute-and-output
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 30 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Java BF16 tensors to native Metal shim | BF16 raw storage and descriptors cross the FFM/native boundary. | Raw BF16 words, dtype codes, shape/layout metadata |
| Planner to coverage gates | BF16 operation admission is reported as native or rejected. | Operation family, dtype evidence, fallback counters |
| Repository to local profiles | Autotune output exists outside canonical evidence. | Generated profile artifacts |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-30-01 | Integrity | BF16 executable signature/cache | mitigate | Output dtype metadata is part of signatures; raw `short[]` storage and dtype codes are tested. | closed |
| T-30-02 | Integrity | BF16 native semantics | mitigate | Support is admitted by operation family only after parity/tolerance tests; dtype-aware constants are required or rejected. | closed |
| T-30-03 | Repudiation | Coverage gates | mitigate | BF16 gates require native path evidence and forbid hidden CPU exits for supported targets. | closed |
| T-30-04 | Integrity | Docs and commit hygiene | mitigate | Docs publish scoped support/fallback tables; local benchmark artifacts remain unstaged. | closed |

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
