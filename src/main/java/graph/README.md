# Graph (src/main/java/graph)

## Purpose

The `graph` package turns a tensor expression DAG into explicit execution artifacts:

- `CompiledGraph`: compile-time artifact
- `PreparedExecution`: runtime-bound artifact

This layer owns:

- forward/backward graph assembly
- optimizer application
- per-node prepared execution metadata
- fused-kernel preparation

## Main Components

- Compile artifact:
  - [src/main/java/graph/CompiledGraph.java](../graph/CompiledGraph.java)
- Runtime artifact:
  - [src/main/java/graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java)
  - [src/main/java/graph/execution/PreparedNodeExecution.java](../graph/execution/PreparedNodeExecution.java)
  - [src/main/java/graph/execution/CompiledNodeExecutionMetadata.java](../graph/execution/CompiledNodeExecutionMetadata.java)
- Fused codegen:
  - [src/main/java/graph/codegen/FusedExpressionPlan.java](../graph/codegen/FusedExpressionPlan.java)
  - [src/main/java/graph/codegen/CompiledFusedKernelFactory.java](../graph/codegen/CompiledFusedKernelFactory.java)
  - [src/main/java/graph/codegen/FusedKernelGeneratorRouter.java](../graph/codegen/FusedKernelGeneratorRouter.java)
  - [src/main/java/graph/codegen/FusedOperationGenerator.java](../graph/codegen/FusedOperationGenerator.java)
  - [src/main/java/graph/codegen/HFusedOperationGenerator.java](../graph/codegen/HFusedOperationGenerator.java)
- Optimizer module:
  - [src/main/java/graph/optimizer/README.md](../graph/optimizer/README.md)

## Compile Pipeline

`CompiledGraph` performs the following high-level steps:

1. create a forward execution anchor via `rootTensor.forwardOutput()`
2. topologically sort the forward graph
3. if the graph has differentiable leaf inputs:
   - seed root gradient
   - build backward graph nodes
   - collect backward targets
   - create a temporary super-root to unify forward and backward sinks
4. run the configured optimizer over the assembled graph
5. record the boundary between forward and backward execution sections

The core public entry points are:

- `CompiledGraph.compile(Tensor root, OptimizerConfig optimizerConfig)`
- `CompiledGraph.compile(Tensor root, GraphOptimizer optimizer)` (legacy/bridge form)

## Runtime Preparation

`CompiledGraph.prepare(...)` converts the compiled graph into a runtime-bound `PreparedExecution`.

Preparation builds:

- ordered forward steps
- ordered backward steps
- per-node backend metadata
- pre-resolved CPU execution plans
- prepared fused runtime kernels for `OpType.FUSED`

This preparation is runtime-specific because it depends on:

- `RuntimeConfig`
- backend planner thresholds
- approximation policy
- BLAS policy

## Execution Modes

The execution engine now uses engine-oriented names:

- `ExecutionMode.FORWARD`
- `ExecutionMode.FORWARD_BACKWARD`

Semantics:

- `FORWARD`
  - execute only prepared forward steps
- `FORWARD_BACKWARD`
  - execute prepared forward steps
  - synchronize output back to the root tensor
  - seed root gradient
  - execute prepared backward steps

Backward capability is exposed via:

- `CompiledGraph.supportsBackward()`
- `PreparedExecution.supportsBackward()`

## Execution Pipeline

`PreparedExecution.execute(mode)`:

1. creates `ExecutionContext`
2. iterates prepared forward steps in topological order
3. syncs root tensor data from the optimized result node
4. if mode is `FORWARD_BACKWARD`, zeroes gradients, seeds the root gradient, and runs backward steps

Actual kernel dispatch then goes through:

- [src/main/java/backend/ComputeEngine.java](../backend/ComputeEngine.java)

## Fused Codegen Path

When `FuseElementWiseRule` collapses a cluster:

1. the optimizer creates a `FusedOperation` descriptor
2. `FusedOperationFactory` converts the cluster to `FusedExpressionPlan`
3. `CompiledGraph.prepare(...)` asks `CompiledFusedKernelFactory` for a prepared runtime kernel
4. `FusedKernelGeneratorRouter` dispatches by dtype
5. `FusedOperationGenerator` emits `FLOAT32/FLOAT64` fused kernels
6. `HFusedOperationGenerator` emits `FLOAT16` fused kernels
7. `CpuFusedKernel` executes the compiled fused kernel stored in prepared node metadata

Important architectural point:

- fused bytecode is generated from plan IR
- `FusedOperation` is only a descriptor
- runtime compiled executable is not stored back on the operation descriptor

## Backward Graph Notes

Backward nodes are built from gradients attached to forward nodes and marked with:

- `tensor.setBackward(true)`

This marker lets optimizer rules preserve phase boundaries between forward and backward sections.

## Canonicalization Notes

Algebraic rewriting includes canonical sigmoid recognition in forward-only inference-style graphs:

- `1 / (1 + exp(-x)) -> sigmoid(x)`
- also recognizes the `mulScalar(-1)` form

For graphs that require gradients, this rewrite is intentionally skipped to preserve the current backward-graph construction semantics.

## Related Modules

- Tensor front-end:
  - [src/main/java/tensor/README.md](../tensor/README.md)
- Backend execution:
  - [src/main/java/backend/README.md](../backend/README.md)
- Optimizer:
  - [src/main/java/graph/optimizer/README.md](../graph/optimizer/README.md)
- Numerics tooling:
  - [src/main/java/numerics/README.md](../numerics/README.md)
