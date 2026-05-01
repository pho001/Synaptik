# Phase 13 Research: Coverage Benchmark And Regression Gate

**Phase:** 13 - Coverage Benchmark And Regression Gate
**Date:** 2026-04-30
**Status:** Complete

## RESEARCH COMPLETE

## Planning Question

What needs to be known to plan Phase 13 well?

Phase 13 should not add broad new accelerator operation support. The work is to make the coverage delivered by Phases 9-12 measurable, comparable, and regression-gated. The implementation should extend existing trace and benchmark surfaces instead of creating a separate reporting subsystem.

## Current Architecture

### Trace Sources

- `MeasurementResult.trace()` carries `ExecutionTrace`, including compile, prepare, and run traces.
- `PrepareTrace.backendSelection()` carries `BackendSelectionTrace`, which records total, selected, and rejected candidates plus `BackendSelectionDecisionTrace` rows.
- `BackendSelectionDecisionTrace.nodeIds()` already provides selected region length source data.
- `BackendSelectionDecisionTrace.reason()` and `finalists()` already provide rejected candidate reason source data.
- `RunTrace.steps()` carries `ExecutionStepTrace` rows with backend, kernel, duration, operation label, and `StepExecutionMetadata`.
- Accelerator runtime evidence already appears in step metadata through keys such as `acceleratorBufferBackend`, `acceleratorBufferExecutionPath`, `acceleratorBufferReasonCode`, `acceleratorBufferReason`, copy timings, input/output bytes, and storage residency attributes.
- `RunTrace.cpuMaterializations()` carries `CpuMaterializationTrace` entries with reason, source residency, bytes, duration, completion state, and source backend.

### Benchmark Report Surfaces

- `BenchmarkReport` and `BenchmarkCandidateReport` are the per-workload report contracts.
- `BenchmarkSuiteReport` groups workload reports and computes candidate summaries/hotspots.
- `TextBenchmarkReportRenderer` and `JsonBenchmarkReportRenderer` already expose trace, accelerator summary, backend selection cost, CPU materializations, steps, and hot steps.
- `TextBenchmarkSuiteReportRenderer` and `JsonBenchmarkSuiteReportRenderer` expose suite-level summaries but do not yet aggregate coverage across workloads.
- Existing tests in `BenchmarkSessionTest` already assert that accelerator trace evidence is rendered for Metal and CUDA.

### Workload Sources

`StandardWorkloads.defaultCatalog()` already contains representative targets for Phase 13:

- `transformer_block_hot_path`
- `transformer_hot_path`
- `mlp_classifier_small`
- `mlp_classifier_blas_heavy`
- `conv2d_resnet_3x3`
- `layer_norm_small`

The workload selection should remain deterministic and small for automated tests. Longer local benchmarks can produce richer reports, but they should not be required in default verification.

## Gaps To Close

### Gap 1: Coverage Metrics Are Implicit

`AcceleratorTraceSummary` currently reports path counts and copy timings per backend. It does not compute:

- GPU coverage ratio
- selected region count
- max selected region length
- average selected region length
- rejected candidate reason counts
- CPU materialization reason counts
- device handoff counts
- storage residency counts

These should become explicit report fields so tests and benchmark reviews can assert coverage behavior without parsing free-form trace text.

### Gap 2: Reports Lack Workload-Level Coverage Comparison

Reports expose per-candidate trace details, but suite-level reports do not summarize whether representative workloads improved over a baseline contract. Phase 13 needs a lightweight, deterministic baseline comparison that focuses on fewer CPU exits or longer GPU-covered regions, not raw timing thresholds.

### Gap 3: Hidden CPU Exits Need Fail-Fast Gates

Phase 13 needs regression gates that fail when a target workload:

- loses selected accelerator coverage,
- adds unexpected CPU materialization boundaries,
- executes a supported target through tensor-array fallback,
- or hides fallback by reporting only generic accelerator presence.

The gate should run as portable Java tests. Native Metal/CUDA checks can add evidence when available but must remain capability-gated.

### Gap 4: Evidence Artifacts Need Hygiene Rules

Local profile and tuning outputs under `profiles/platform/.../tuning/abc/*` are machine-local. They should remain unstaged unless explicitly promoted to canonical fixtures. Phase 13 docs should distinguish checked-in evidence contracts from local measurement artifacts.

## Recommended Implementation Shape

### Shared Coverage Summary

Add a backend-neutral coverage summary under `tuning.benchmark.report`, for example `GpuCoverageSummary`, that can be computed from `ExecutionTrace`:

- source selected region metrics from `PrepareTrace.backendSelection().decisions()`,
- source executed path metrics from `RunTrace.steps()` and `StepExecutionMetadata`,
- source CPU materialization metrics from `RunTrace.cpuMaterializations()`,
- preserve existing `AcceleratorTraceSummary` output for backward compatibility.

The report schema should use stable field names:

- `gpuCoverageRatio`
- `selectedRegionCount`
- `maxSelectedRegionLength`
- `averageSelectedRegionLength`
- `rejectedCandidateReasonCounts`
- `fallbackCount`
- `tensorArrayStepCount`
- `cpuFallbackStepCount`
- `cpuMaterializationCount`
- `cpuMaterializationReasonCounts`
- `cpuMaterializationBytes`
- `cpuMaterializationDurationNs`
- `copyDurationNs`
- `deviceHandoffCount`
- `storageResidencyCounts`

### Baseline Comparison

Add small report/test helpers that compare current candidate coverage to a named baseline contract:

- current candidate must have equal-or-higher selected region length, or
- current candidate must have equal-or-lower CPU materialization count, fallback count, and device handoff count.

This avoids brittle wall-clock thresholds and matches the roadmap success criterion.

### Regression Gate

Add a focused assertion helper, for example `GpuCoverageRegressionGate`, that fails with explicit messages:

- `lost GPU coverage`
- `unexpected CPU materialization`
- `hidden tensor-array fallback`
- `unexpected device handoff`
- `missing coverage summary`

The helper should be used by JUnit tests against synthetic traces and at least one small representative workload path.

### Documentation

Update docs with the exact distinction:

- evidence contract: checked-in trace/report fields and deterministic tests,
- local artifact: machine-local benchmark/profile outputs,
- native gate: capability-gated Metal/CUDA tasks,
- portable gate: Java tests proving schema, fallback, materialization, and report contracts.

## Validation Architecture

### Automated Sampling

- After coverage summary schema changes, run renderer/report tests focused on `BenchmarkSessionTest`, `BenchmarkSuiteSessionTest`, and any new coverage summary test.
- After workload comparison changes, run tests that execute small deterministic StandardWorkloads and synthetic comparison fixtures.
- After regression gate changes, run fail-fast tests that intentionally construct hidden tensor-array fallback and unexpected CPU materialization traces.
- Before phase verification, run `./gradlew classes` and focused report/trace suites. Run `./gradlew metalTest` and `./gradlew buildCudaGraphShim cudaTest` as capability-gated native evidence.

### Nyquist Targets

- Every plan must include automated verification.
- No three consecutive tasks should rely only on documentation or manual checks.
- Native CUDA execution may be capability-skipped on non-CUDA hosts, but portable Java tests must still prove the CUDA report/fallback contract.
- Source hygiene must be checked with `git status --short` and no staged `profiles/platform/.../tuning/abc/*` paths.

## Research Risks

- Device handoff counting can become ambiguous if derived only from adjacent step backend strings. The plan should define a conservative count: transitions between adjacent traced steps where either side is an accelerator backend and the backend value changes, plus CPU materialization boundaries from device-owned residency.
- GPU coverage ratio can be ambiguous across selected candidates versus executed steps. The plan should define both region selection metrics and run-step coverage metrics, then expose `gpuCoverageRatio` as executed accelerator steps divided by total traced steps.
- Synthetic trace tests are necessary because native CUDA may not run locally.
