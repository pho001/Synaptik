---
phase: 38
slug: metal-training-backward-coverage
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 38 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Autograd/backward graph to accelerator truth | Backward ops are classified independently from forward rows. | Op type, gradient metadata, backend ownership |
| Native backward execution to traces | Supported backward rows must prove buffer binding rather than tensor-array/CPU replay. | Trace path, native bridge status, parity evidence |
| Coverage policy to users | Gradient publication is separated from hidden internal materialization. | Materialization reason, policy budget, blocker target |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-38-01 | Integrity | Backward truth rows | mitigate | Tests assert backward rows separately from forward target rows. | closed |
| T-38-02 | Repudiation | Native path evidence | mitigate | Tests inspect trace/backend path evidence, not only numerical parity. | closed |
| T-38-03 | Integrity | Materialization policy | mitigate | Gate policy separates gradient publication from unexpected internal CPU exits. | closed |
| T-38-04 | Repudiation | Phase closure | mitigate | Verification lists supported, unsupported, and residual backward families explicitly. | closed |

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
