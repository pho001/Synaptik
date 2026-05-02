---
phase: 29
slug: metal-dtype-abi-and-capability-truth
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 29 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Java runtime to native Metal shim | Java capability discovery calls optional native dtype ABI v3 symbols. | ABI version, dtype descriptors, capability booleans |
| Planner/reporting to users | Capability truth is rendered into coverage reports and docs. | Support status, reason codes, dtype roles |
| Repository to local profiles | Local autotune outputs exist but are not milestone evidence. | Generated profile artifacts |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-29-01 | Integrity | DType capability model | mitigate | Role-specific decisions and tests keep storage/residency separate from compute/output support. | closed |
| T-29-02 | Tampering | Native dtype ABI discovery | mitigate | Optional symbol/version checks default widened dtype support to unavailable while preserving FLOAT32. | closed |
| T-29-03 | Repudiation | Coverage reports and docs | mitigate | Reports render concise role/dtype/reason triples; docs state Phase 29 is capability truth only. | closed |
| T-29-04 | Integrity | Commit hygiene | mitigate | Source/docs/tests/planning files were staged explicitly; local profile artifacts stayed unstaged. | closed |

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
