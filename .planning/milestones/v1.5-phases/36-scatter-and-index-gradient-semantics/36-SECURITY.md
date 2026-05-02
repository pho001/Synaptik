---
phase: 36
slug: scatter-and-index-gradient-semantics
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 36 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Index-write semantics to planner | Scatter and index-gradient operations are analyzed before native admission. | Index values, duplicate-index status, bounds proof |
| CPU fallback to trace/reporting | Unsupported write-add/gradient paths must remain explicit CPU boundaries. | Rejection reasons, backend ownership, fallback counters |
| Coverage reports to users | Forward gather/take support must not imply write/gradient support. | Target names, truth rows, unsupported reasons |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-36-01 | Integrity | Native scatter admission | mitigate | Matrix, planner, docs, and tests fail unsupported or unproven duplicate-index claims. | closed |
| T-36-02 | Integrity | Duplicate-index accumulation | mitigate | Operations remain unsupported unless duplicate-index parity tests pass; static bounds reuse Phase 32 proof rules. | closed |
| T-36-03 | Repudiation | Backward ownership | mitigate | Backward-specific fixtures and prepared-execution checks expose CPU boundaries and rejection reasons. | closed |
| T-36-04 | Repudiation | Reports and profiles | mitigate | Separate target names/truth rows avoid counting forward index support as gradient support; local artifacts remain unstaged. | closed |

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
