# Phase 28 Context: Coverage Regression Closure

## Goal

Prove v1.4 reduced CPU exits on representative workloads and harden regression gates against hidden fallback.

## Requirements

- `GPUCLOSE-01`: Representative transformer, normalization, loss/indexing, and conv/pool workloads have hard coverage thresholds for CPU materializations, fallback counts, region length, and backend execution path.
- `GPUCLOSE-02`: Regression gates fail when a supported target family regresses from native/buffer execution to tensor-array bridge or CPU fallback without an explicit capability reason.
- `GPUCLOSE-03`: Final benchmark reports show before/after fallback reduction for v1.4 target families and keep local calibration/profile artifacts out of committed state unless intentionally promoted.

## Current Findings

- `GpuHotPathCoverageTargets` already defines nine v1.4 target workloads:
  - `reduction_chain_small`
  - `transformer_block_hot_path`
  - `mlp_classifier_small`
  - `conv2d_resnet_3x3`
  - `max_pool2d_small`
  - `layer_norm_small`
  - `rms_norm_small`
  - `cross_entropy_small`
  - `bool_compare_where_small`
- Existing `GpuCoverageRegressionGate` already fails lost GPU coverage, hidden tensor-array fallback, unexpected CPU fallback/materialization, lost native buffer binding, shorter region length, lost lowered primitive count, lost multi-op region count, and lost fused subpattern evidence.
- Existing reports expose `targetCoverageGates`, `nativeEvidence`, `capabilitySkipped`, `targetCoverageTruth`, selected region metrics, fallback counts, CPU materializations, device handoffs, and native/tensor-array path counters.
- `GpuTargetCoverageTruth` separates matrix support from native executable proof. This is the right source of truth for deciding whether a workload should require hard native evidence or only visible fallback/rejection reasons.
- Phase 27 deliberately left conv/pool and native BOOL-producing compute unsupported with stable rejection evidence. Phase 28 must not convert those into native claims.
- Local profile artifacts under `profiles/platform/...` remain dirty from local benchmark/autotune work and must not become Phase 28 evidence unless intentionally promoted.

## Phase Direction

Phase 28 is a closure and hardening phase, not a new operation-family implementation phase.

The correct implementation should:

- tighten policies for families v1.4 actually made native/lowered executable, especially reductions, normalization, Metal SDPA, and fused/multi-op MLP-like paths;
- require visible reason evidence for families intentionally left unsupported or capability-gated, including conv/pool, BOOL-producing compare outputs, index/loss blockers, and CUDA SDPA;
- render a final v1.4 coverage report/evidence table that can be audited without reading local benchmark scratch files;
- keep benchmark/calibration artifacts out of committed state unless they are explicitly promoted to fixtures.

## Canonical References

- `.planning/ROADMAP.md` - Phase 28 scope and success criteria.
- `.planning/REQUIREMENTS.md` - `GPUCLOSE-01/02/03`.
- `.planning/phases/22-coverage-truth-and-semantics-lock/22-03-SUMMARY.md` - representative target registry baseline.
- `.planning/phases/23-forward-reductions-native-execution/23-03-SUMMARY.md` - reduction native execution closure.
- `.planning/phases/24-normalization-gpu-lowering/24-VERIFICATION.md` - normalization native/lowered execution closure.
- `.planning/phases/25-forward-sdpa-semantic-enablement/25-VERIFICATION.md` - Metal SDPA native evidence and CUDA capability fallback.
- `.planning/phases/26-loss-adjacent-and-indexing-gpu-coverage/26-VERIFICATION.md` - loss/indexing support-or-rejection evidence.
- `.planning/phases/27-conv-pool-and-bool-compare-outputs/27-VERIFICATION.md` - conv/pool and BOOL output support-or-rejection evidence.
- `docs/gpu-lowering-coverage.md` - current public lowering coverage contract.
- `docs/compute-flow.md` - trace/report and device residency explanations.

## Deferred Ideas

- Broader native conv/pool execution and native BOOL-producing GPU compute remain future operation coverage work.
- Real CUDA hardware performance proof remains a CUDA-equipped lane concern because `nvcc` is unavailable locally.
- Vendor library routing remains deferred to `GPULIB-*` requirements.

---

*Phase: 28-coverage-regression-closure*
*Context gathered: 2026-05-02*
