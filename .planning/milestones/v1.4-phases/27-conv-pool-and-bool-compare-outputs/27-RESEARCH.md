# Phase 27 Research

## Architecture Notes

- Public tensors remain logical; GPU residency belongs in compile/prepare/execute runtime state.
- Backend-neutral contracts should be updated before backend-specific Metal/CUDA support.
- Coverage matrix rows are the source of truth consumed by Metal and CUDA planner diagnostics.
- A matrix `SUPPORTED` row means the operation has real lowering and backend execution for the legal scoped case, not just documentation.

## Conv/Pool Surface

Target operations:

- `CONV2D`
- `CONV2D_GEMM`
- `CONV2D_BACKWARD_INPUT`
- `CONV2D_BACKWARD_WEIGHT`
- `CONV2D_BACKWARD_INPUT_GEMM`
- `CONV2D_BACKWARD_WEIGHT_GEMM`
- `MAX_POOL2D`
- `MAX_POOL2D_BACKWARD_INPUT`
- `AVG_POOL2D`
- `AVG_POOL2D_BACKWARD_INPUT`

CPU supports these paths, including lowering and workspace behavior. GPU does not yet expose native DAG primitives or vendor-library routing for these operations. The safe current contract is explicit `CAPABILITY_MISSING` rows plus semantic contracts for rank/layout/padding/dilation/group/count-include-pad/tie behavior.

## BOOL Compare Surface

Target operations:

- `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`
- `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`
- `REDUCE_ALL`, `REDUCE_ANY`

The current Metal/CUDA execution path can represent `BOOL` storage and Metal can bind external BOOL predicate inputs for `WHERE`, but native compute/output support for BOOL-producing nodes is not implemented. The safe current contract is `UNSUPPORTED_DTYPE` with an explicit note that external predicate residency is separate from native BOOL output compute.

## Native Work Needed Before Marking Supported

- Add backend-neutral DAG node types for compare/logical/bool-reduction and/or conv/pool primitives.
- Extend `AcceleratorSubgraphLowerer` with shape, broadcast, axis, and semantic parameter lowering.
- Extend Metal/CUDA legality adapters so dtype/layout/rank capability gates run before planner admission.
- Extend native Metal/CUDA shims for BOOL output buffers and conv/pool primitives, or route through verified backend libraries.
- Add CPU parity tests for compare-to-where masks and representative conv/pool workloads.
- Add coverage/report gates proving selected region length, lowered primitive count, backend path, and CPU exits.
