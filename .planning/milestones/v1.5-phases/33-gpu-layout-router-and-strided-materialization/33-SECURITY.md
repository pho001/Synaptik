---
phase: 33
slug: gpu-layout-router-and-strided-materialization
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 33 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Logical tensor layout to accelerator buffer layout | Views/broadcasts/strides are classified before buffer binding. | Shape, strides, storage offset, physical span |
| Runtime materializer to Metal buffers | GPU-side materialization allocates and registers destination buffers. | Source handles, destination handles, currentness metadata |
| Coverage reports to users | Materialization/fallback detail is exposed as audit evidence. | Route kind, reason code, materialization counters |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-33-01 | Integrity | Layout router labels | mitigate | Availability reasons remain separate from materializer capability; backend id checks prevent CUDA inheriting Metal support. | closed |
| T-33-02 | Tampering | GPU materialization buffers | mitigate | Storage offset, strides, logical shape, and physical byte span are validated; dense repair allocates a distinct destination. | closed |
| T-33-03 | Integrity | Runtime currentness | mitigate | Repaired bindings validate target `AcceleratorBufferLayout` and emit trace layout metadata. | closed |
| T-33-04 | Repudiation | Benchmark gates and profiles | mitigate | Gates require buffer binding and zero tensor-array fallback; local profile output remains unstaged. | closed |

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
