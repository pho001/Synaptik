# Tensor (src/Tensor)

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
  - [src/Tensor/Tensor.java](src/Tensor/Tensor.java)
- Metadata:
  - [src/Tensor/TensorMetadata.java](src/Tensor/TensorMetadata.java)
- Operation helpers:
  - [src/Tensor/TensorOps.java](src/Tensor/TensorOps.java)
  - [src/Tensor/TensorBinaryOps.java](src/Tensor/TensorBinaryOps.java)
  - [src/Tensor/TensorUnaryOps.java](src/Tensor/TensorUnaryOps.java)
  - [src/Tensor/TensorReduceOps.java](src/Tensor/TensorReduceOps.java)
  - [src/Tensor/TensorLayoutOps.java](src/Tensor/TensorLayoutOps.java)
  - [src/Tensor/TensorNaryOps.java](src/Tensor/TensorNaryOps.java)
- Storage/type abstraction (currently auxiliary):
  - [src/Tensor/TensorStorage.java](src/Tensor/TensorStorage.java)
  - [src/Tensor/DataType.java](src/Tensor/DataType.java)
  - [src/Tensor/Float16Storage.java](src/Tensor/Float16Storage.java)
  - [src/Tensor/Float32Storage.java](src/Tensor/Float32Storage.java)
  - [src/Tensor/Float64Storage.java](src/Tensor/Float64Storage.java)

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

Related files:

- [src/Graph/CompiledGraph.java](src/Graph/CompiledGraph.java)
- [src/Graph/optimizer/OptimizerFactory.java](src/Graph/optimizer/OptimizerFactory.java)
- [src/Backend/ComputeEngine.java](src/Backend/ComputeEngine.java)

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

## Notes on DTypes and Storage

`TensorStorage` and `DataType` abstractions exist and provide `FLOAT16/FLOAT32/FLOAT64` storage wrappers.

Current execution path still operates on tensor runtime values used by existing kernels. Storage abstraction can be incrementally wired into kernels and tensor runtime as dtype support expands.
