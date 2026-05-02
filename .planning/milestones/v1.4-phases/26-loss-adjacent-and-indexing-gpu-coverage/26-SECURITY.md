---
phase: 26
slug: loss-adjacent-and-indexing-gpu-coverage
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 26 - Security

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-26-01 | Integrity | Loss/index support claims | mitigate | Loss/index rows remain explicit support-or-rejection evidence and no unsupported native compute is claimed. | closed |
| T-26-02 | Integrity | Duplicate-index semantics | mitigate | CPU parity tests cover duplicate-index scatter/gather gradient behavior; GPU support remains blocked without native accumulation semantics. | closed |
| T-26-03 | Integrity | Ignore-index and reduction semantics | mitigate | Index-target loss tests cover ignore-index, weights, and reduction denominator behavior. | closed |
| T-26-04 | Repudiation | Adjacent GPU region hidden by fallback | mitigate | Prepared execution proves legal preceding GPU region selection remains visible before CPU-owned index/loss boundaries. | closed |
| T-26-05 | Supply chain / integrity | Local profile artifacts | mitigate | Phase verification records profile artifacts as local and unstaged. | closed |

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-02 | 5 | 5 | 0 | Codex inline security audit |

## Sign-Off

- [x] All threats have a disposition.
- [x] `threats_open: 0` confirmed.
- [x] `status: verified` set in frontmatter.

**Approval:** verified 2026-05-02
