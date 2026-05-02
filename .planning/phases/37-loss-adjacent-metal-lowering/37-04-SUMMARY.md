# Plan 37-04 Summary: Loss Coverage Report And Docs Closure

**Status:** Complete
**Completed:** 2026-05-02
**Requirement Coverage:** METALLOSS-01, METALLOSS-02, METALLOSS-03

## Completed

- Added `dense_loss_small` as a separate standard workload and hot-path target for dense loss coverage.
- Kept `cross_entropy_small` as the index-target loss blocker target instead of counting it as dense loss support.
- Registered Metal `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS` as native-executable target truth rows after Phase 37 parity and trace evidence.
- Added hard Metal coverage policy for `dense_loss_small`: native buffer binding required, lowered primitive evidence required, and CPU/tensor-array fallback rejected.
- Strengthened index-target gates so `cross_entropy_small` must expose `UNSUPPORTED_INDEX_SEMANTICS`/`CROSS_ENTROPY_LOSS_INDICES` blocker evidence.
- Updated workload catalog tests, coverage target tests, coverage gate tests, docs, and benchmark workload documentation.

## Verification

```bash
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest
./gradlew test --tests BenchmarkSuiteSessionTest
```

Both passed during implementation. Final Phase 37 closure gates are recorded in `37-VERIFICATION.md`.

## Residual Scope

- Index-target CE/NLL and index-target gradients remain CPU-owned with `UNSUPPORTED_INDEX_SEMANTICS`.
- CUDA dense loss remains unsupported until CUDA-specific lowering/native evidence exists.
- Dense loss support remains scoped to the Phase 37 `FLOAT32`, dense-layout, rank 1..4, public mean-reduced scalar contract.
