---
phase: 13
slug: coverage-benchmark-and-regression-gate
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
updated: 2026-05-01
---

# Phase 13 - Validation Strategy

Per-phase validation contract for coverage benchmark and regression gate work.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit 5 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest` |
| Full suite command | `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest` |
| Native Metal gate | `./gradlew metalTest` |
| Native CUDA gate | `./gradlew buildCudaGraphShim cudaTest` |
| Estimated runtime | ~1-180 seconds depending on Gradle cache and native capability gates |

---

## Sampling Rate

- After every task commit: run the task's focused `./gradlew test --tests ...` command from the plan verify block.
- After every plan wave: run the quick run command or the plan-level focused command.
- Before `$gsd-verify-work`: run the full suite command plus available native gates.
- Max feedback latency: one focused Gradle invocation per task; optional native gates may be skipped by local capability probes.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | GPUCOV-01, GPUCOV-03 | T-13-01 / T-13-02 | Coverage fields cannot silently disappear from report schema. | unit | `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | W0 | green |
| 13-01-02 | 01 | 1 | GPUCOV-01 | T-13-03 | Rejected candidate and selected region metrics remain tied to prepare trace source data. | unit | `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | W0 | green |
| 13-01-03 | 01 | 1 | GPUCOV-01 | T-13-04 | CPU materialization and device handoff counts are visible and reason-coded. | unit | `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | W0 | green |
| 13-02-01 | 02 | 2 | GPUCOV-01, GPUCOV-02 | T-13-05 / T-13-06 | Representative workload coverage is compared to a deterministic baseline contract, not local timing artifacts. | integration | `./gradlew test --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest` | W0 | green |
| 13-02-02 | 02 | 2 | GPUCOV-02 | T-13-07 | Transformer, MLP, and conv/normalization workload names are present in coverage reports. | integration | `./gradlew test --tests BenchmarkSuiteSessionTest` | W0 | green |
| 13-03-01 | 03 | 3 | GPUCOV-03 | T-13-08 / T-13-09 | Supported GPU target paths fail on lost coverage or unexpected CPU materialization. | regression | `./gradlew test --tests GpuCoverageRegressionGateTest --tests CompiledGraphTraceTest` | W0 | green |
| 13-03-02 | 03 | 3 | GPUCOV-03 | T-13-10 | Tensor-array fallback cannot masquerade as native GPU coverage. | regression | `./gradlew test --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest` | W0 | green |
| 13-04-01 | 04 | 4 | GPUCOV-01, GPUCOV-02, GPUCOV-03 | T-13-11 / T-13-12 | Docs and native gates separate portable evidence, native evidence, and local artifacts. | docs/native | `./gradlew classes`; `./gradlew metalTest`; `./gradlew buildCudaGraphShim cudaTest`; `git status --short` | W0 | green |

Status: pending = plan-time contract; green/red recorded during execution or validation audit.

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements:

- `build.gradle` provides Gradle/JUnit execution.
- Benchmark report tests exist in `BenchmarkSessionTest` and `BenchmarkSuiteSessionTest`.
- Trace/materialization tests exist in `CompiledGraphTraceTest`.
- Workload specs exist in `StandardWorkloads` and related workload classes.
- Native Metal/CUDA tasks are already represented by `metalTest` and `buildCudaGraphShim cudaTest`.

No generated test files are required before Plan 13-01 starts.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native CUDA execution on a CUDA-capable host | GPUCOV-01, GPUCOV-02, GPUCOV-03 | Local host may not have CUDA runtime/toolchain or GPU. Portable Java tests prove schema and fallback; native execution evidence requires hardware. | Run `./gradlew buildCudaGraphShim cudaTest` on a CUDA-capable host and require task success without capability skips before claiming local native CUDA execution. |

Native Metal can be validated with `./gradlew metalTest` on supported macOS hosts.

## Validation Audit 2026-04-30

Requirements audited: GPUCOV-01, GPUCOV-02, GPUCOV-03.

Task rows audited: 8 total, 8 green, 0 red, 0 pending.

Automated evidence:

- `./gradlew classes` passed.
- `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest` passed.
- `./gradlew metalTest` passed and built the local Metal shim under `build/native/apple/`.
- `./gradlew buildCudaGraphShim cudaTest` was capability-skipped after the sandbox lock issue was rerun with Gradle wrapper access.
- Documentation `rg` checks found `GPU coverage summary`, `gpuCoverageRatio`, `hidden tensor-array fallback`, and `GPU coverage regression checks`.
- `git status --short` showed only docs/planning edits plus existing unstaged `profiles/platform/.../tuning/abc/*` tuning profile changes.

Gaps found: 0.

Resolved: 0.

Escalated: 0.

## Validation Audit 2026-05-01

Requirements audited: GPUCOV-01, GPUCOV-02, GPUCOV-03.

Task rows audited: 8 total, 8 green, 0 red, 0 pending.

Cross-reference result:

- GPUCOV-01 is covered by `GpuCoverageSummaryTest`, `BenchmarkSessionTest`, text/JSON renderer assertions, and report-field documentation.
- GPUCOV-02 is covered by `GpuCoverageSummaryTest`, `BenchmarkSuiteSessionTest`, representative workload name assertions, and deterministic baseline comparison tests.
- GPUCOV-03 is covered by `GpuCoverageRegressionGateTest`, `BenchmarkSessionTest`, stable failure-string assertions, and docs for hidden tensor-array fallback.

Automated evidence:

- `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest` passed.
- `./gradlew metalTest` passed.
- `./gradlew buildCudaGraphShim cudaTest` was capability-skipped after rerunning with Gradle wrapper access.
- Test-name `rg` checks found all expected Phase 13 coverage, baseline, workload, and regression-gate tests.
- Security cross-check: `13-SECURITY.md` has `threats_open: 0`.
- `git status --short` still shows only existing unstaged `profiles/platform/.../tuning/abc/*` tuning profile changes.

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands or capability-gated native commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency bounded by focused Gradle filters.
- [x] `nyquist_compliant: true` set in frontmatter after execution validation.

**Approval:** verified 2026-05-01
