---
phase: 22
slug: coverage-truth-and-semantics-lock
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 22 - Security

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-22-01 | Spoofing / integrity | Native coverage truth | mitigate | `GpuTargetCoverageTruth` separates native-executable proof from matrix support and fallback/rejection rows. | closed |
| T-22-02 | Integrity | Target semantics | mitigate | `GpuTargetSemanticsContractTest` verifies dtype/rank/layout/shape/axis/mask/index contracts before backend admission. | closed |
| T-22-03 | Repudiation | Representative coverage gates | mitigate | `GpuHotPathCoverageTargetsTest` and `StandardWorkloadsTest` verify deterministic v1.4 target workloads. | closed |
| T-22-04 | Integrity | False support claims | mitigate | Phase 22 intentionally locks truth and does not mark unsupported families as native-executable. | closed |

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-02 | 4 | 4 | 0 | Codex inline security audit |

## Sign-Off

- [x] All threats have a disposition.
- [x] `threats_open: 0` confirmed.
- [x] `status: verified` set in frontmatter.

**Approval:** verified 2026-05-02
