---
phase: 40
plan: 40-03
status: complete
completed: 2026-05-02
requirements:
  - CUDAPARITY-03
key-files:
  - src/main/java/tuning/benchmark/report/CudaHotPathBlockerClass.java
  - src/main/java/tuning/benchmark/report/CudaHotPathBlockerPolicy.java
  - src/main/java/tuning/benchmark/report/GpuCoverageTriageReport.java
  - src/main/java/tuning/benchmark/report/TextGpuCoverageTriageReportRenderer.java
  - src/main/java/tuning/benchmark/report/JsonGpuCoverageTriageReportRenderer.java
  - src/test/java/GpuCoverageTriageReportTest.java
  - src/test/java/GpuHotPathCoverageTargetsTest.java
---

# Plan 40-03 Summary

## Completed

- Added CUDA hot-path blocker classifications: `V16_BLOCKER`, `ACCEPTED_CAPABILITY_GAP`, `FUTURE_SCOPE`, and `REQUIRES_NATIVE_EVIDENCE`.
- Added deterministic policy mapping for v1.6 CUDA blocker targets and BF16 accepted capability gaps.
- Added text and JSON triage report output under `cudaHotPathBlockers`.

## Verification

```bash
./gradlew test --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest
./gradlew test --tests backend.accelerator.lowering.GpuBackendParityReportTest
```

Both commands passed.
