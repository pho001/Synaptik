# Tensor (src/main/java/Tensor)

## Purpose

The Tensor module provides the runtime tensor node abstraction used by graph execution, autodiff, and backend dispatch.

It combines:

- value container for runtime execution
- metadata (`shape`, `strides`, `label`, `requiresGrad`)
- graph links (`prevTensors`, producing operation)
- gradient references and backward-graph hooks
- compiled execution caches (resolved backend/kernel)

## Main Components

- Core runtime tensor:
  - [src/main/java/Tensor/Tensor.java](../Tensor/Tensor.java)
- API reference:
  - [src/main/java/Tensor/API.md](../Tensor/API.md)
- Metadata:
  - [src/main/java/Tensor/TensorMetadata.java](../Tensor/TensorMetadata.java)
- Operation helpers:
  - [src/main/java/Tensor/TensorOps.java](../Tensor/TensorOps.java)
  - [src/main/java/Tensor/TensorBinaryOps.java](../Tensor/TensorBinaryOps.java)
  - [src/main/java/Tensor/TensorUnaryOps.java](../Tensor/TensorUnaryOps.java)
  - [src/main/java/Tensor/TensorReduceOps.java](../Tensor/TensorReduceOps.java)
  - [src/main/java/Tensor/TensorLayoutOps.java](../Tensor/TensorLayoutOps.java)
  - [src/main/java/Tensor/TensorNaryOps.java](../Tensor/TensorNaryOps.java)
- Layout remap utility:
  - [src/main/java/Tensor/TensorRemap.java](../Tensor/TensorRemap.java)
- Storage/type abstraction (currently auxiliary):
  - [src/main/java/Tensor/TensorStorage.java](../Tensor/TensorStorage.java)
  - [src/main/java/Tensor/DataType.java](../Tensor/DataType.java)
  - [src/main/java/Tensor/Float16Storage.java](../Tensor/Float16Storage.java)
  - [src/main/java/Tensor/Float32Storage.java](../Tensor/Float32Storage.java)
  - [src/main/java/Tensor/Float64Storage.java](../Tensor/Float64Storage.java)

## Runtime Data Model

`Tensor` stores:

- operation (`Operation`) producing the node, or `null` for leaf/constant nodes
- input links (`prevTensors`)
- tensor values for execution
- gradient tensor reference (`gradient`)
- metadata via `TensorMetadata`
- cached compiled execution data:
  - resolved backend (`resolvedBackend`)
  - resolved CPU kernel (`resolvedCpuKernel`)
  - resolved CPU execution plan (`resolvedCpuExecutionPlan`)
  - resolved broadcast plan (`resolvedBroadcastPlan`)
  - resolved CPU config epoch (`resolvedCpuConfigEpoch`)
  - compiled graph handle (`compiledGraph`)

`TensorMetadata` handles:

- shape normalization
- stride computation and validation
- contiguous check
- flat index <-> spatial index mapping
- `label` and `requiresGrad` flags

## Execution Flow

1. Tensor expression graph is built through operations (`add`, `mul`, `pow`, `log`, etc.).
2. `Tensor.compute(optimizer)` builds `CompiledGraph` once and executes it.
3. `Tensor.compute()` uses profile-driven recommended optimizer from `OptimizerFactory`.
4. `CompiledGraph` executes forward (and backward section in training mode) via backend dispatch.

Unary transcendental APIs now include both exact and approximate variants:

- exact: `exp()`, `tanh()`
- approximate: `fastExp()`, `fastTanh()`

Related files:

- [src/main/java/Graph/CompiledGraph.java](../Graph/CompiledGraph.java)
- [src/main/java/Graph/optimizer/OptimizerFactory.java](../Graph/optimizer/OptimizerFactory.java)
- [src/main/java/Backend/ComputeEngine.java](../Backend/ComputeEngine.java)
- [src/main/java/Tensor/TensorUnaryOps.java](../Tensor/TensorUnaryOps.java)
- Numerics diagnostics harness: [src/main/java/Numerics/README.md](../Numerics/README.md)

## Gradient and Backward

- During graph compilation, backward nodes are built from forward nodes (`buildBackwardGraph()` path).
- Root gradient is seeded with `onesLike(root)`.
- `CompiledGraph` executes backward section after forward section in training mode.
- `Tensor.backward()` delegates to compiled graph backward execution.

Helpers:

- `Tensor.onesLike(...)`
- `Tensor.zerosLike(...)`

## Backend Selection

Tensor backend decision is:

1. `forcedBackend` if explicitly set on tensor.
2. operation preferred backend if provided.
3. fallback to `CPU`.

During compile, resolved backend and CPU kernel are cached per node to reduce runtime dispatch overhead.

## Non-Contiguous Layout Handling

`TensorRemap` provides layout remapping between tensors with identical shape and different strides.

Current remap behavior:

- compile-time remap plan support (`TensorRemap.RemapPlan`) reused in backend hot path
- typed fast paths for contiguous/same-stride copies (`System.arraycopy` for F64/F32/F16)
- odometer/range-walker offset traversal (avoids per-element `logicalToOffset` recomputation)
- supports sequential and parallel remap paths via `CpuThreadPool`

This avoids stride-indexing out-of-bounds issues on non-contiguous tensors and is used by backend materialization paths.

## Notes on DTypes and Storage

`TensorStorage` and `DataType` abstractions exist and provide `FLOAT16/FLOAT32/FLOAT64` storage wrappers.

Current execution path still operates on tensor runtime values used by existing kernels. Storage abstraction can be incrementally wired into kernels and tensor runtime as dtype support expands.
