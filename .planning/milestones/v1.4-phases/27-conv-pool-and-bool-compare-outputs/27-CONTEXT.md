# Phase 27 Context: Conv Pool And Bool Compare Outputs

## Goal

Add GPU coverage for conv/pool targets and device-resident BOOL compare outputs feeding `WHERE` or mask consumers.

## Requirements

- `GPUCONVBOOL-01`: Conv/pool workloads receive GPU coverage through supported backend primitives/lowering or explicit capability-gated rejection for unsupported layouts and ranks.
- `GPUCONVBOOL-02`: BOOL-producing compare operations can produce device-resident BOOL outputs for supported Metal/CUDA paths and feed `WHERE`/mask consumers without CPU materialization.
- `GPUCONVBOOL-03`: Conv/pool and BOOL compare coverage reports expose selected region length, lowered primitive count, CPU exits, and backend execution path.

## Current Findings

- The public `Tensor` API already exposes compare ops (`GT`, `GE`, `LT`, `LE`, `EQ`, `NE`), logical BOOL ops (`LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`), BOOL reductions (`REDUCE_ALL`, `REDUCE_ANY`), `CONV2D`, `MAX_POOL2D`, and `AVG_POOL2D`.
- CPU execution covers conv/pool, compare, logical BOOL, and BOOL reductions.
- The shared GPU coverage matrix currently lists only a partial Phase 27 surface (`CONV2D`, `MAX_POOL2D`, `GT`, `EQ`).
- `GpuTargetSemanticsContract` already lists forward conv/pool and compare contracts, but not all conv/pool backward/GEMM variants, logical BOOL ops, or BOOL reductions.
- `AcceleratorDagNodeType` has no compare, logical BOOL, BOOL reduction, conv, or pool primitive ABI codes.
- `AcceleratorSubgraphLowerer` cannot lower compare, logical BOOL, BOOL reduction, conv, or pool nodes to the accelerator DAG.
- Metal currently treats native compute/output dtype as `FLOAT32`; BOOL is accepted only as an external predicate input role for `WHERE`.
- CUDA native dense buffer execution remains `FLOAT32` only and rejects non-`FLOAT32` output dtype for native compute.

## Phase Direction

Phase 27 should first make the full target surface explicit and testable. Native BOOL compare output and native conv/pool execution must only become `SUPPORTED` after the DAG ABI, lowering, backend legality, native execution, trace/report, and parity evidence all exist.

Until then, the correct behavior is stable capability/dtype rejection with coverage/report visibility, not silent fallback or an incomplete matrix.
