# Summary 25-04: SDPA Coverage Gate And Documentation Closure

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Updated transformer hot-path coverage expectations for Phase 25 outcomes:
  - Metal `transformer_block_hot_path` now requires native buffer evidence, at least one lowered primitive, zero tensor-array fallback, zero CPU fallback, and zero CPU materialization.
  - CUDA `transformer_block_hot_path` expects visible `CAPABILITY_MISSING` / `SCALED_DOT_PRODUCT_ATTENTION` / `target=transformer_block_hot_path` evidence instead of native support.
- Added regression coverage proving the supported SDPA policy fails hidden tensor-array fallback, CPU fallback, and lost native buffer binding.
- Kept report rendering on existing coverage fields: selected region length, lowered primitive count, fallback/materialization counters, reason codes, native evidence, and target coverage truth.
- Updated docs to describe the Metal supported SDPA path and CUDA capability-gated fallback expectations.
- Checked profile hygiene; local profile artifacts remain dirty and unstaged, and are not Phase 25 evidence.

## Verification

Passed:

```bash
./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests StandardWorkloadsTest
./gradlew metalTest
git diff --check
git status --short profiles/platform
```

Not run:

```bash
./gradlew cudaTest
```

`nvcc` is not available in this environment, so native CUDA compilation remains hardware/toolchain-gated.

## Deviations from Plan

- No new benchmark report schema field was required. Existing coverage summary and renderer fields already expose the required SDPA evidence through region length, lowered primitive count, fallback/materialization counters, native evidence, and target coverage truth.
