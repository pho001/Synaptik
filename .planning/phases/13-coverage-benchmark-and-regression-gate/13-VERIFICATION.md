---
phase: 13-coverage-benchmark-and-regression-gate
status: passed
score: 5/5
requirements_verified: [GPUCOV-01, GPUCOV-02, GPUCOV-03]
human_verification: []
gaps: []
verified: 2026-05-01
---

# Phase 13 Verification: Coverage Benchmark And Regression Gate

## Verdict

Passed. Phase 13 achieved its roadmap goal: GPU coverage is now a checked-in trace and benchmark report contract,
representative workloads have deterministic coverage/materialization baseline comparison semantics, and fail-fast
regression gates catch lost GPU coverage, unexpected CPU materialization, hidden tensor-array fallback, and device
handoff regressions.

## Requirement Coverage

| Requirement | Status | Evidence |
|---|---|---|
| GPUCOV-01 | Passed | `GpuCoverageSummary` derives `gpuCoverageRatio`, selected region length, rejected candidate reasons, fallback counts, CPU materialization reason counts, copy timing, storage residency, and device handoff counts for `GPU_METAL` and `GPU_CUDA`; text and JSON benchmark renderers expose the fields. |
| GPUCOV-02 | Passed | `GpuCoverageBaseline`, `GpuCoverageComparison`, `BenchmarkSuiteReport.coverageSummaries()`, and `bestCoverageByBackend()` compare longer GPU regions and fewer CPU exits without raw timing thresholds; suite tests cover `transformer_block_hot_path`, `mlp_classifier_small`, and `conv2d_resnet_3x3`. |
| GPUCOV-03 | Passed | `GpuCoverageRegressionGate` and `GpuCoverageGatePolicy` fail on lost GPU coverage, unexpected CPU materialization, hidden tensor-array fallback, unexpected device handoff, and missing coverage summary with stable failure strings. |

## Success Criteria

| # | Status | Evidence |
|---|---|---|
| 1. Trace and benchmark reports expose GPU coverage fields for Metal and CUDA | Passed | Plan 13-01 added `GpuCoverageSummary`, per-candidate `coverage:` text output, `"coverage"` JSON output, and synthetic Metal/CUDA summary tests. |
| 2. Representative workloads show coverage/materialization behavior versus v1.1 baseline | Passed | Plan 13-02 added deterministic baseline comparison and suite-level coverage aggregation/rendering for transformer, MLP, and conv coverage workloads. |
| 3. Regression gates fail on lost coverage or hidden CPU exits | Passed | Plan 13-03 added fail-fast policy/result/gate classes and tests for lost coverage, unexpected materialization, hidden tensor-array fallback, device handoff budget, and missing summary. |
| 4. Native checks remain capability-gated while portable Java gates prove contracts | Passed | Plan 13-04 recorded portable Java tests passing, `./gradlew metalTest` passing, and CUDA native tasks capability-skipping on this host. |
| 5. Docs and hygiene distinguish evidence contracts from local artifacts | Passed | `docs/compute-flow.md`, `docs/gpu-lowering-coverage.md`, `docs/development.md`, and `docs/testing.md` document coverage fields, gate semantics, native capability-gated evidence, and the rule to leave `profiles/platform/.../tuning/abc/*` unstaged. |

## Plan Completion

| Plan | Status | Summary |
|---|---|---|
| 13-01 Coverage Metric Contract And Report Schema | Complete | Added backend-neutral GPU coverage summary fields and per-workload text/JSON rendering. |
| 13-02 Representative Workload Coverage Benchmarks | Complete | Added deterministic baseline comparison and suite coverage summary evidence for representative workloads. |
| 13-03 Regression Gate And Hidden Exit Failures | Complete | Added fail-fast coverage gates for hidden CPU exits and coverage regressions. |
| 13-04 Docs Native Gates And Hygiene Closure | Complete | Documented evidence contracts, ran portable/native gates, updated validation, and confirmed tuning artifacts remain unstaged. |

## Automated Checks

- `./gradlew classes` - passed.
- `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest` - passed.
- `./gradlew metalTest` - passed.
- `./gradlew buildCudaGraphShim cudaTest` - capability-skipped after rerun with Gradle wrapper access.
- `rg -n "GPU coverage summary|gpuCoverageRatio|hidden tensor-array fallback|GPU coverage regression checks" docs/compute-flow.md docs/gpu-lowering-coverage.md docs/development.md docs/testing.md` - passed.
- `test -f .planning/phases/13-coverage-benchmark-and-regression-gate/13-04-SUMMARY.md` - passed.
- `rg -n "Phase 13 final verification|GPUCOV-01|GPUCOV-02|GPUCOV-03|native Metal status|native CUDA status|profiles/platform/.../tuning/abc/\* remained unstaged|nyquist_compliant: true|Validation Audit 2026-04-30" .planning/phases/13-coverage-benchmark-and-regression-gate/13-04-SUMMARY.md .planning/phases/13-coverage-benchmark-and-regression-gate/13-VALIDATION.md` - passed.

## Code Review

`13-REVIEW.md` is clean with 0 critical, 0 warning, and 0 info findings.

## Validation

`13-VALIDATION.md` is verified with `nyquist_compliant: true`, 8/8 task rows green, 0 gaps found, 0 resolved, and 0
escalated. Native CUDA execution remains capability-gated on this host.

## Human Verification

None required. Phase 13 changes are trace/report infrastructure, deterministic benchmark report contracts, tests, and
documentation.

## Gaps Summary

No gaps found. Phase goal achieved. Security enforcement should run next via `$gsd-secure-phase 13` before advancing the
milestone.

## Residual Risk

Native CUDA execution was not exercised on CUDA-capable hardware in this run because `buildCudaGraphShim` and `cudaTest`
were capability-skipped locally. A CUDA-capable host must run `./gradlew buildCudaGraphShim cudaTest` successfully before
claiming local native CUDA execution.

## Git Hygiene

Existing local profile tuning changes under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/abc/*`
remain unstaged and are not part of Phase 13.
