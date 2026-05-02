---
phase: 34
slug: masked-and-causal-sdpa
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 34 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Public BOOL mask to native SDPA | Public mask semantics are converted into native MPSGraph SDPA behavior. | BOOL mask storage, effective causal mask, SDPA scale metadata |
| Planner to native execution | Masked/causal SDPA is admitted only when executable semantics are proven. | Mask mode, dtype/layout/rank legality, rejection reasons |
| Coverage reports to users | Hot-path closure is based on backend trace fields. | Native route, CPU materialization counts, unsupported rows |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-34-01 | Integrity | SDPA mask classifier | mitigate | Planner names effective mask mode and keeps unsupported decisions until executable semantics are enabled. | closed |
| T-34-02 | Integrity | Mask polarity and CPU exits | mitigate | Asymmetric native parity fixtures prove polarity; REQUIRE-mode tests assert no `CPU_CONSUMER` materialization. | closed |
| T-34-03 | Integrity | Causal generation | mitigate | Fixtures prove diagonal convention and backend-internal causal/effective mask handling. | closed |
| T-34-04 | Repudiation | Reports and docs | mitigate | Gates use trace fields and docs separate supported mask modes from unsupported dtype/layout/rank/CUDA gaps. | closed |

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
