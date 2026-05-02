---
phase: 43
type: research
status: complete
requirements:
  - CUDATRAIN-01
  - CUDATRAIN-02
  - CUDATRAIN-03
---

# Phase 43 Research: CUDA Training And Index Semantics

## Phase Goal

Close CUDA training/backward and scatter/index-gradient evidence gaps with explicit blockers where native CUDA support is not proven.

## Current Evidence

- Phase 38 established that training/backward support is per backward operation and not inherited from forward support.
- `GpuTargetCoverageTruth` already lists backward-adjacent target rows for Metal and CUDA.
- CUDA currently has matrix-supported rows for selected backward-adjacent primitives, but native executable truth is conservative and must not be promoted without execution evidence.
- Phase 41 validates CUDA forward `GATHER` / `TAKE_ALONG_AXIS` dtype, layout, rank, shape, and static bounds before final `CAPABILITY_MISSING`.
- Metal has `MetalIndexWriteSemantics` for `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD`; CUDA does not yet have the same operation-specific index-write/gradient semantic rejection layer.
- Training coverage targets already exist: `training_transformer_block_hot_path`, `training_dense_loss_small`, `training_reduction_chain_small`, `training_layer_norm_small`, and `training_cross_entropy_small`.

## Planning Direction

1. Add a CUDA backward/training truth contract that separates:
   - native-executable CUDA rows with evidence,
   - matrix-supported-only rows,
   - capability-missing rows,
   - unsupported semantic rows.
2. Add CUDA index-write/gradient semantic validation before generic matrix rejection:
   - dense `FLOAT32` values/output,
   - dense static `INT32` indices,
   - rank/axis/shape legality,
   - static bounds proof,
   - final duplicate-index blocker.
3. Harden training hot-path report targets for CUDA:
   - visible blockers for unsupported training rows,
   - no hidden internal CPU materialization counted as support,
   - gradient publication remains a separate, allowed boundary only for supported training rows.
4. Do not promote CUDA training support without native execution, parity tests, and trace/report evidence.

## Verification Targets

- CUDA region lowerer tests for `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` dtype/layout/bounds/shape/duplicate blockers.
- Coverage truth tests proving CUDA backward rows are not inferred from forward support.
- Coverage target tests proving CUDA training hot paths are visible blockers until native evidence exists.
- Docs updates in `docs/cuda-backend.md` and `docs/gpu-lowering-coverage.md`.

