---
phase: 40
slug: cuda-parity-gap-triage-and-capability-baseline
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 40 - Security

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| CUDA capability report boundary | Java-side reports consume native CUDA bridge discovery state and expose it to docs/reports. | Local native capability metadata, no secrets. |
| Coverage report boundary | Benchmark/triage renderers expose CUDA support, fallback, and blocker classifications. | Trace/report metadata, no user data. |
| Documentation boundary | Docs translate implementation state into support claims. | Public project documentation. |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T40-01 | Spoofing / integrity | `GpuBackendParityReporter` | mitigate | `GpuBackendParityReportTest` proves CUDA `CAPABILITY_MISSING` rows are not support and shared supported rows are not gaps. | closed |
| T40-02 | Integrity | `CudaCapabilityReport` | mitigate | `CudaCapabilityReportTest` proves `capabilitySkipCountsAsSupport()` is false, unsupported dtype roles remain explicit, and cuBLAS/cuDNN route is `NOT_INTEGRATED`. | closed |
| T40-03 | Integrity / auditability | `CudaHotPathBlockerPolicy` and triage renderers | mitigate | `GpuCoverageTriageReportTest` and `GpuHotPathCoverageTargetsTest` prove blocker classes render in text/JSON and accepted gaps stay separate from blockers. | closed |
| T40-04 | Repudiation / auditability | Docs and verification artifacts | mitigate | `docs/cuda-backend.md`, `docs/gpu-lowering-coverage.md`, and `40-VERIFICATION.md` state that capability skip is evidence, not support. | closed |

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
