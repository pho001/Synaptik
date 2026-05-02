---
phase: 31
slug: bool-producing-metal-compute
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 31 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Java BOOL tensors to native Metal shim | BOOL descriptors and raw predicate/output bytes cross the native boundary. | BOOL storage bytes, dtype codes, DAG node codes |
| Planner to region execution | BOOL producers may feed downstream GPU operations without CPU materialization. | Region ownership, dtype residency, materialization counters |
| Docs to users | Supported BOOL compute must not imply masked SDPA support before Phase 34. | Support matrix and deferred-scope statements |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-31-01 | Integrity | BOOL capability and ABI | mitigate | Operation-specific BOOL decisions and non-colliding DAG enum values are tested. | closed |
| T-31-02 | Integrity | Native BOOL semantics | mitigate | Native tests assert raw BOOL bytes; reduction edge cases remain rejected unless parity-covered. | closed |
| T-31-03 | Information Disclosure | Region materialization | mitigate | Gates assert no hidden CPU exit between BOOL producers and `WHERE`. | closed |
| T-31-04 | Repudiation | Docs and coverage | mitigate | Masked SDPA remains explicitly deferred to Phase 34; local artifacts are excluded. | closed |

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
