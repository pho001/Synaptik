---
phase: 32
slug: int32-index-tensor-and-gather-take-path
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 32 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Java index tensors to native Metal shim | INT32 index buffers and axis metadata cross the native boundary. | Index values, dtype codes, shape/rank/axis metadata |
| Planner to native execution | Bounds/layout legality must be proven before native dispatch. | Static bounds decisions, layout classes, rejection reasons |
| Coverage reports to users | Forward index support must not imply scatter or generic INT32 compute. | DType role evidence, operation support rows |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-32-01 | Integrity | INT32 role capability | mitigate | Role-specific methods and tests keep index input residency separate from generic compute/output. | closed |
| T-32-02 | Integrity | Gather/take semantics | mitigate | Planner legality and native descriptor validation share rank/axis constraints; parity tests cover ordering and shape. | closed |
| T-32-03 | Tampering | Index bounds and layout | mitigate | Unsafe bounds and unsupported layouts reject before native execution or route through explicit Phase 33 materialization. | closed |
| T-32-04 | Repudiation | Coverage and commit hygiene | mitigate | Reports gate on role-specific `dtypeResidency`; local benchmark artifacts remain unstaged. | closed |

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
