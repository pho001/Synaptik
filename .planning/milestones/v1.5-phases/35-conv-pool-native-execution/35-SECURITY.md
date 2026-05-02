---
phase: 35
slug: conv-pool-native-execution
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 35 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Synaptik tensor layout to MPSGraph conv/pool | NCHW/OIHW public tensors map to MPSGraph descriptors. | Shape, layout, stride, padding, dilation, kernel metadata |
| Planner to native execution | Conv/pool variants are admitted only after native parity is proven. | Operation variant, dtype/layout/rank legality, reason codes |
| Coverage reports to users | Forward support must not imply backward/grouped/dilated support. | Coverage rows, blocker rows, fallback counters |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-35-01 | Integrity | Conv2D admission | mitigate | Planner diagnostics stay capability-missing until execution exists; direct conv and GEMM forms remain separate. | closed |
| T-35-02 | Integrity | MPSGraph layout and bias semantics | mitigate | Descriptor/transpose logic and asymmetric bias parity tests catch layout inversion and broadcast errors. | closed |
| T-35-03 | Integrity | Pooling semantics | mitigate | `countIncludePad` is gated precisely; docs keep forward support separate from backward tie handling. | closed |
| T-35-04 | Repudiation | Coverage and profiles | mitigate | Hard gates flip only with native parity tests; local tuning output remains unstaged. | closed |

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
