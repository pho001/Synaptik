---
phase: 15-gpu-region-internal-lowered-dag-contract
plan: "03"
status: complete
subsystem: trace-reporting
tags: [gpu, lowered-dag, manifest, trace, benchmark]
requires:
  - phase: 15-gpu-region-internal-lowered-dag-contract
    provides: Plan 15-02 selected Metal/CUDA plan manifests
provides:
  - Structured backend-selection manifest metadata
  - Compact lowered-region text rendering
  - Benchmark JSON manifest rendering
  - Run trace gpuLoweredRegionId evidence
affects: [phase-15, phase-20]
completed: 2026-05-01
---

# Phase 15 Plan 03: Trace And Report Manifest Contract Summary

## Trace and report manifest contract

Selected GPU backend decisions now carry `GpuLoweredRegionManifest` metadata through `BackendSelectionDecisionTrace`. Metal and CUDA partition plans expose the selected manifest via `PartitionPlan.gpuLoweredRegionManifest()`, and `DefaultBackendSelectionPolicy` attaches it only to accepted GPU decisions.

Benchmark reports now expose the selected manifest in both stable forms:

- Text reports render a compact `GPU Lowered Region` block with `Original Ops`, `Lowered Primitives`, `Value Assumptions`, `Fused Subpatterns`, and `Rejections`.
- JSON reports include `gpuLoweredRegionManifest` with stable keys `regionId`, `backend`, `selectedRegionLength`, `originalOps`, `loweredPrimitives`, `valueAssumptions`, `fusedSubpatterns`, `rejections`, and `candidateSpan`.
- Run-step metadata records only `gpuLoweredRegionId`; it does not duplicate the full `gpuLoweredRegionManifest`.

## Task Commits

1. **Tasks 1-5: Trace/report manifest implementation and tests** - `deba01f` (`feat(15-03): expose gpu lowered region manifests`)
2. **Task 6: Record trace/report summary and hygiene evidence** - this metadata commit

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest` | Passed |
| `./gradlew test --tests PreparedExecutionBuildTest` | Passed |
| `rg -n "gpuLoweredRegionManifestTraceContainsOriginalOpsAndPrimitives\|gpuLoweredRegionRunTraceReferencesRegionIdOnly\|benchmarkTextReportRendersGpuLoweredRegionManifest\|benchmarkJsonReportRendersGpuLoweredRegionManifest\|coverageSummaryIgnoresManifestWhenCountingSelectedRegions\|GPU Lowered Region\|gpuLoweredRegionManifest\|gpuLoweredRegionId" src/test/java/CompiledGraphTraceTest.java src/test/java/BenchmarkSessionTest.java src/test/java/GpuCoverageSummaryTest.java` | Passed |
| `rg -n "GpuLoweredRegionManifestRenderer\|gpuLoweredRegionManifest\|regionId\|selectedRegionLength\|originalOps\|loweredPrimitives\|valueAssumptions\|fusedSubpatterns\|candidateSpan" src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java src/test/java/BenchmarkSessionTest.java` | Passed |

## Requirement Coverage

- `GPUDAG-02`: Trace/report metadata exposes original operation IDs/types, lowered primitive IDs/types, fused subpattern summaries, backend ID, dtype/layout assumptions, and selected region length.
- `GPUDAG-03`: Rejection, fallback, materialization, and candidate-shortening evidence is represented in stable manifest fields and reason codes for report consumers.

## Decision Coverage

Covered decisions: `D-05`, `D-06`, `D-08`, `D-09`, `D-10`, `D-11`, `D-12`, `D-13`, `D-15`, and `D-16`.

## Boundary Notes

- Prepare/backend-selection trace carries structured manifest data.
- Benchmark text is human-readable evidence; benchmark JSON is the structured report contract.
- Run trace references only `gpuLoweredRegionId` plus runtime outcome attributes.
- Native Metal/CUDA ABI is unchanged.
- Public `Tensor` API remains logical and unchanged.

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.

## Next Phase Readiness

Plan 15-04 can close docs, validation evidence, public API guardrails, and source hygiene for Phase 15.

## Self-Check: PASSED

---
*Phase: 15-gpu-region-internal-lowered-dag-contract*
*Completed: 2026-05-01*
