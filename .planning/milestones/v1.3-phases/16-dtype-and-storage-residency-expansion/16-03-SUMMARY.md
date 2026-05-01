---
phase: 16-dtype-and-storage-residency-expansion
plan: "03"
status: complete
requirements-completed: [GPUSTORAGE-02, GPUSTORAGE-03]
completed: 2026-05-01
---

# Phase 16 Plan 03: DType Residency Trace And Report Evidence Summary

DType residency trace and report evidence is now visible from lowered GPU region manifests through benchmark coverage reports.

## DType residency trace and report evidence

- Added `dtypeResidency.*` lowered-region backend extension entries for supported and unsupported dtype roles.
- Added `UNSUPPORTED_DTYPE` lowered-region rejections for rejected dtype residency roles with stable `backend=... role=... dtype=...` details.
- Rendered dtype residency evidence in compact lowered-region text, benchmark text reports, and benchmark JSON reports.
- Added `dtypeResidencyReasons` coverage aggregation without fabricating evidence when trace manifests have none.

## Verification

| Command | Result |
|---------|--------|
| `rg -n "manifestRecordsDTypeResidencyAssumptions|prepareTraceRendersDTypeResidencyRejectionReasons|coverageSummaryCountsDTypeMaterializationReasons|benchmarkReportsRenderDTypeResidencyEvidence|dtypeResidency|dtype=BFLOAT16|dtype=INT32|dtype=BOOL|backend=GPU_METAL|backend=GPU_CUDA" src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java src/test/java/CompiledGraphTraceTest.java src/test/java/GpuCoverageSummaryTest.java src/test/java/BenchmarkSessionTest.java` | Passed |
| `rg -n "dtypeResidency|AcceleratorDTypeResidencyPolicy|UNSUPPORTED_DTYPE|renderCompact|GpuLoweredRegionRejection" src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifestRenderer.java` | Passed |
| `rg -n "dtypeResidencyReasons|dtypeResidencyEvidence|DType Residency Evidence|dtypeResidency" src/main/java/tuning/benchmark/report src/test/java/GpuCoverageSummaryTest.java src/test/java/BenchmarkSessionTest.java` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | Passed |

## Requirement Coverage

- `GPUSTORAGE-02`: Dtype-related Metal/CUDA rejections are rendered with stable `UNSUPPORTED_DTYPE`, backend, role, and dtype evidence.
- `GPUSTORAGE-03`: Coverage reports distinguish dtype residency evidence from real CPU materialization; hidden CPU materialization remains reportable.

## Deviations from Plan

None - plan executed exactly as written.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged

## Self-Check: PASSED
