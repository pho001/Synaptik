# Summary 30-03: BF16 Parity, Tolerance, And Coverage Gates

**Status:** Complete
**Completed:** 2026-05-02

## What Changed

- Added BF16-specific hot-path coverage targets for MLP, LayerNorm, RMSNorm, and reduction-chain workloads.
- Added hard Metal BF16 coverage policies that require native buffer binding and reject CPU materialization, CPU fallback, and tensor-array fallback.
- Added BF16 dtype residency evidence to suite-level coverage summaries and target gate payloads.
- Added BF16 target gate validation that fails when supported BF16 targets do not expose `dtype=BFLOAT16` residency evidence.
- Added native Metal BF16 parity tests for exact raw buffer roundtrip, RELU, MATMUL, SUM reduction, LayerNorm, and softmax with explicit tolerance constants.

## Tolerance Policy

- BF16 upload/readback storage roundtrip is exact raw BF16 bit equality.
- BF16 MATMUL and reduction parity use `0.5f` tolerance for BF16-rounded numeric output.
- BF16 normalization and softmax parity use `0.025f` tolerance.

## Verification

```bash
./gradlew classes
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest
./gradlew metalTest
```

All commands passed.

## Notes

- Existing local profile artifacts under `profiles/platform/.../tuning/` were left unstaged.
- BF16 target gates are Metal-only hard native gates; CUDA keeps capability-gated behavior and does not inherit Metal BF16 assumptions.
