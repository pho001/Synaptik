---
phase: 23
slug: forward-reductions-native-execution
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 23 - Security

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-23-01 | Integrity | Reduction DAG metadata | mitigate | Lowerer tests verify op kind, axis, keep-dims, and output shape metadata. | closed |
| T-23-02 | Integrity | Backend native execution claims | mitigate | Matrix/truth rows moved to supported/native only with backend execution and tests. | closed |
| T-23-03 | Repudiation | Hidden CPU materialization | mitigate | Prepared execution and coverage tests keep CPU exits visible and fail hidden fallback in later Phase 28 gates. | closed |
| T-23-04 | Native boundary | Metal/CUDA reduction ABI | mitigate | Metal native gate passed; CUDA behavior remains portable/capability-gated where native hardware is unavailable. | closed |

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
