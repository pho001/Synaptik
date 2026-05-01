# Phase 14 Research: Coverage Gap Triage And Hot Path Targets

**Phase:** 14 - Coverage Gap Triage And Hot Path Targets
**Date:** 2026-05-01
**Status:** Complete

## RESEARCH COMPLETE

## Planning Question

What needs to be known to plan Phase 14 well?

Phase 14 should turn the v1.2 coverage report contract into a deterministic triage input for the rest of v1.3. The phase should not broaden GPU execution yet. It should identify, rank, and persist the top reasons GPU regions leave Metal/CUDA execution so later phases can close measured gaps instead of adding operation support opportunistically.

## Current Architecture

### Coverage Evidence Already Available

- `GpuCoverageSummary.fromTrace(ExecutionTrace)` computes backend coverage from prepare and run traces.
- `GpuCoverageSummary.BackendCoverage` already exposes `gpuCoverageRatio`, selected region counts and lengths, rejected candidate reason counts, buffer-binding/tensor-array/CPU-fallback counts, CPU materialization reason counts, copy duration, storage residency counts, reason codes, fallback reasons, and device handoff count.
- `GpuCoverageBaseline` and `GpuCoverageComparison` compare current coverage against named baselines without using raw latency thresholds.
- `GpuCoverageRegressionGate`, `GpuCoverageGatePolicy`, and `GpuCoverageGateResult` already fail on lost GPU coverage, unexpected CPU materialization, hidden tensor-array fallback, unexpected device handoff, and missing coverage summary.
- Per-workload text/JSON renderers expose the full `coverage` object.
- Suite-level text/JSON renderers expose `coverageSummary` using `BenchmarkSuiteReport.bestCoverageByBackend()`.

### Workload Inputs Already Available

`StandardWorkloads.defaultCatalog()` contains the representative workload names needed by Phase 14:

- `transformer_block_hot_path`
- `mlp_classifier_small`
- `conv2d_resnet_3x3`
- `layer_norm_small`

The existing Phase 13 tests already prove deterministic suite requests for `transformer_block_hot_path`, `mlp_classifier_small`, and `conv2d_resnet_3x3`. Phase 14 should formalize the target list and include `layer_norm_small` as the normalization-side proxy for the conv/normalization flow.

### Reporting And Test Patterns

- Report-domain records live under `src/main/java/tuning/benchmark/report`.
- Report records use Java records with null-normalizing compact constructors.
- Static helpers such as `GpuCoverageSummary.fromTrace(...)` keep derivation out of mutable benchmark execution code.
- Unit tests can use synthetic `ExecutionTrace` fixtures from `GpuCoverageSummaryTest.traceFor(...)` for portable Metal/CUDA evidence.
- Integration-style suite tests should reuse `StandardWorkloads.benchmarkSuite(...)` to prove workload names and request shape without requiring native hardware.

## Gaps To Close

### Gap 1: Coverage Metrics Are Measurable But Not Ranked

Phase 13 reports expose coverage fields, but there is no stable triage object that ranks:

- rejected candidate reasons,
- CPU materialization reasons,
- tensor-array fallback counts,
- CPU fallback counts,
- device handoff counts,
- low selected-region length,
- low GPU coverage ratio,
- storage residency signals.

Phase 14 should add deterministic `GpuCoverageGap` records and a `GpuCoverageGapTriage` utility that converts `BenchmarkReport` and `BenchmarkSuiteReport` into ordered coverage gaps.

### Gap 2: Representative Hot Paths Are Names, Not A Contract

The workload names exist in `StandardWorkloads`, but the v1.3 hot-path target list should be a checked-in contract that later phases can cite. The target record should include the workload name, category, expected focus families, and why the workload matters.

### Gap 3: No Source-Of-Truth Target List For Later Phases

Later phases need a durable artifact that says which CPU exits or region blockers matter first. Phase 14 should generate or maintain `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` with a stable table of target workloads, top gap reasons, mapped requirement families, and downstream owner phases.

### Gap 4: Reports Must Distinguish Evidence From Local Artifacts

Triage must use checked-in tests, deterministic synthetic traces, and report contracts as evidence. Local files under `profiles/platform/.../tuning/abc/*` remain machine-local tuning output and should not be staged unless intentionally promoted.

## Recommended Implementation Shape

### Triage Model

Add report-domain values under `tuning.benchmark.report`:

- `GpuCoverageGapCategory`
- `GpuCoverageGap`
- `GpuCoverageGapTriage`

The triage should produce exact category names:

- `REJECTED_CANDIDATE`
- `CPU_MATERIALIZATION`
- `TENSOR_ARRAY_FALLBACK`
- `CPU_FALLBACK`
- `DEVICE_HANDOFF`
- `LOW_REGION_LENGTH`
- `LOW_GPU_COVERAGE`
- `STORAGE_RESIDENCY`

Each `GpuCoverageGap` should include:

- workload name,
- candidate name,
- backend,
- category,
- reason,
- count,
- severity score,
- selected region length,
- CPU materialization count,
- fallback count,
- device handoff count,
- requirement family.

### Hot-Path Targets

Add a stable target list, likely:

- `GpuHotPathCoverageTarget`
- `GpuHotPathCoverageTargets`

The default target list should include exactly:

- `transformer_block_hot_path`
- `mlp_classifier_small`
- `conv2d_resnet_3x3`
- `layer_norm_small`

The targets should map to downstream work families:

- `GPUDAG`
- `GPUSTORAGE`
- `GPUNORM`
- `GPUFUSEX`
- `GPUMULTI`
- `GPUHARDEN`

### Triage Report

Add a report object and renderers:

- `GpuCoverageTriageReport`
- `TextGpuCoverageTriageReportRenderer`
- `JsonGpuCoverageTriageReportRenderer`

The report should expose:

- `hotPathTargets`,
- `topGaps`,
- `gapCountsByCategory`,
- `gapCountsByRequirementFamily`,
- `downstreamPhaseTargets`.

The text renderer should include stable headings:

- `GPU Coverage Gap Triage`
- `Hot Path Targets`
- `Top Coverage Gaps`
- `Requirement Family Ranking`
- `Downstream Phase Targets`

### Checked-In Planning Artifact

Create `14-HOT-PATH-TARGETS.md` during execution with:

- target workload table,
- top coverage gap table,
- downstream phase owner table,
- evidence command list,
- artifact hygiene note.

This file is the handoff contract for Phases 15-20.

## Validation Architecture

### Automated Sampling

- After triage model changes, run `./gradlew test --tests GpuCoverageGapTriageTest --tests GpuCoverageSummaryTest`.
- After hot-path target changes, run `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest`.
- After triage report rendering changes, run `./gradlew test --tests GpuCoverageTriageReportTest --tests BenchmarkSuiteSessionTest`.
- Before phase verification, run `./gradlew classes` and the focused Phase 14 test slice.

### Nyquist Targets

- Every plan has automated verification.
- No three consecutive tasks rely only on documentation or manual inspection.
- Portable Java tests prove the CUDA triage contract with synthetic traces when native CUDA is unavailable.
- Verification must include a `git status --short` check showing local tuning profiles remain unstaged.

## Research Risks

- Ranking weights can become arbitrary. Mitigation: define deterministic severity score constants in tests and keep raw counts in every gap record.
- Workload names can drift. Mitigation: tests must assert all default targets exist in `StandardWorkloads.defaultCatalog()`.
- Triage can become too timing-oriented. Mitigation: do not score by raw median latency in Phase 14; score by region length, materialization, fallback, handoff, and rejected reason counts.
- Reports can hide local machine state. Mitigation: checked-in target list and tests are canonical; local benchmark/profile files remain unstaged.
