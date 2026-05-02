# 28-01 Summary: Target Coverage Policy Audit

## Completed

- Audited `GpuHotPathCoverageTargets.expectationsForBackend(...)` against `GpuTargetCoverageTruth` and Phase 23-27 outcomes.
- Promoted `reduction_chain_small` from permissive partial-blocker policy to a hard native/lowered policy for Metal and CUDA legal cases.
- Confirmed hard native policies for supported v1.4 targets:
  - reductions,
  - `layer_norm_small`,
  - `rms_norm_small`,
  - Metal `transformer_block_hot_path`,
  - `mlp_classifier_small`.
- Confirmed visible-blocker policies for unsupported or capability-gated targets:
  - CUDA `transformer_block_hot_path`,
  - `conv2d_resnet_3x3`,
  - `max_pool2d_small`,
  - `cross_entropy_small`,
  - `bool_compare_where_small`.
- Added tests proving supported targets require native buffer evidence and unsupported targets keep expected visible reasons.

## Verification

- `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageSummaryTest`

## Outcome

`GPUCLOSE-01` and `GPUCLOSE-02` now have target policies aligned to real v1.4 execution truth instead of permissive generic coverage.
