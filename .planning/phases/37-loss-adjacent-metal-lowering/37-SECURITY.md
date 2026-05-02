---
phase: 37
slug: loss-adjacent-metal-lowering
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 37 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Public loss ops to Metal lowering | Dense and index-target loss variants are classified before native admission. | Labels/targets, rank/shape, reduction mode, dtype/layout |
| Training traces to coverage reports | Loss boundaries must show backend ownership and unsupported reasons. | Backend path, CPU step, rejection reason, target name |
| Docs to users | Dense loss support must not imply index-target or gradient support. | Coverage tables and residual-scope statements |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-37-01 | Integrity | Dense loss formula | mitigate | Contract tests compare CPU formulas, output shape, and class-axis behavior before support rows change. | closed |
| T-37-02 | Integrity | Index-target semantics | mitigate | Ignore-index, class weights, denominators, and gradients reject until exact parity fixtures exist. | closed |
| T-37-03 | Repudiation | Training CPU exits | mitigate | Prepared-execution and trace tests assert visible backend ownership and CPU/rejection boundaries. | closed |
| T-37-04 | Repudiation | Coverage rows | mitigate | Dense and index-target targets have separate names, policies, and expected evidence checks. | closed |

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
