---
phase: 28
slug: coverage-regression-closure
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-02
---

# Phase 28 - Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Coverage truth -> target policy | `GpuTargetCoverageTruth` decides whether a target requires hard native evidence or visible fallback evidence. | Operation family, execution status, backend, target workload |
| Target policy -> regression gate | Gate thresholds must fail hidden tensor-array bridge, CPU fallback, CPU materialization, and lost native buffer evidence. | Coverage thresholds, backend path counters, failure messages |
| Trace/report evidence -> milestone proof | Benchmark reports must prove device ownership with counters, not timing-only claims. | Region length, lowered primitive count, native buffer/tensor-array/CPU fallback counts, materialization counts |
| Unsupported target -> visible blocker | Unsupported/capability-gated families must expose stable reasons and must not pass as native coverage. | Reason codes, fallback reasons, target names, backend names |
| Local profile outputs -> source control | Machine-local benchmark/profile artifacts must not become canonical milestone evidence. | `profiles/platform/...` calibration/autotune outputs |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-28-01 | Spoofing / integrity | Native coverage claims | mitigate | `GpuHotPathCoverageTargetsTest` cross-checks hard native policies for supported targets, and `GpuTargetCoverageTruth` remains the source for native-executable status. | closed |
| T-28-02 | Repudiation | Hidden tensor-array bridge | mitigate | `GpuCoverageRegressionGateTest` fails supported targets on hidden tensor-array bridge and lost native buffer binding. | closed |
| T-28-03 | Repudiation | CPU fallback masquerading as GPU | mitigate | Supported target tests fail CPU fallback and unexpected CPU materialization with stable gate failures. | closed |
| T-28-04 | Integrity | Unsupported target passes without reason | mitigate | Visible-blocker expectations require stable reason evidence and failure messages name workload/backend when reason evidence is missing. | closed |
| T-28-05 | Integrity | Timing-only milestone proof | mitigate | Suite report renderers include `coverageDeltaVsBaseline` and trace-derived counters; docs state timing is secondary evidence. | closed |
| T-28-06 | Information disclosure / provenance | Local profile artifacts | mitigate | Verification and validation record `profiles/platform/...` artifacts remain dirty local files and are not staged Phase 28 evidence. | closed |
| T-28-07 | Availability / diagnosability | Missing target coverage summary | mitigate | Suite-level gate failures now include workload and backend context for missing target coverage summaries. | closed |
| T-28-08 | Native boundary | CUDA native proof unavailable locally | mitigate | CUDA native gate remains explicit manual-only evidence because `nvcc` is unavailable; portable Java gates prove report and fallback semantics. | closed |

*Status: open - closed*
*Disposition: mitigate (implementation required) - accept (documented risk) - transfer (third-party)*

## Accepted Risks Log

No accepted risks.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-02 | 8 | 8 | 0 | Codex inline security audit |

## Audit Evidence

| Evidence | Result |
|----------|--------|
| `GpuHotPathCoverageTargetsTest` | Supported targets require hard native/buffer policies; unsupported targets require visible blocker reasons. |
| `GpuCoverageRegressionGateTest` | Hidden tensor-array, CPU fallback, CPU materialization, lost native buffer binding, lost region evidence, and lost lowered primitives fail supported targets. |
| `BenchmarkSuiteSessionTest`, `BenchmarkSessionTest` | Text/JSON suite reports expose policy thresholds, observed counters, native evidence, and `coverageDeltaVsBaseline`. |
| `docs/gpu-lowering-coverage.md`, `docs/compute-flow.md`, `docs/testing.md`, `docs/calibration-autotune.md` | Docs state coverage proof is trace/report evidence, not timing-only, and local profile artifacts are not proof. |
| `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest` | Passed. |
| `./gradlew metalTest` | Passed. |
| `git diff --check` | Passed. |
| `command -v nvcc` | Not available locally; CUDA native gate not run. |

## Security Audit 2026-05-02

| Metric | Count |
|--------|-------|
| Threats found | 8 |
| Closed | 8 |
| Open | 0 |

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-02
