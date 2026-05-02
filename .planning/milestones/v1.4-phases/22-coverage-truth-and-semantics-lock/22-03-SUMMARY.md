# Summary 22-03: Representative Gate Baselines

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Added deterministic `reduction_chain_small` and `bool_compare_where_small` standard workloads.
- Expanded the GPU hot-path target registry from the old v1.3 four-target set to v1.4 coverage closure targets:
  `reduction_chain_small`, `transformer_block_hot_path`, `mlp_classifier_small`, `conv2d_resnet_3x3`,
  `max_pool2d_small`, `layer_norm_small`, `rms_norm_small`, `cross_entropy_small`, and
  `bool_compare_where_small`.
- Added target expectations for reductions, SDPA/attention, normalization, loss/indexing, conv/pool, and BOOL compare output paths.
- Updated standard workload and target registry tests.

## Verification

Passed:

```bash
./gradlew test --tests StandardWorkloadsTest --tests GpuHotPathCoverageTargetsTest --tests GpuTargetSemanticsContractTest --tests GpuCoverageSummaryTest
```

## Handoff

Phase 23 can now start reduction execution work with:

- a truth layer that prevents premature native-support claims,
- reduction semantics contracts for axis/keepDims/output shape,
- representative reduction workload coverage baseline.
