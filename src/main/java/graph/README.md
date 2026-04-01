# Graph (src/main/java/graph)

## Purpose

The Graph module compiles tensor expression DAGs into executable plans, runs forward/backward execution, and integrates optimizer + fused code generation.

## Main Components

- Compiled execution plan:
  - [src/main/java/graph/CompiledGraph.java](../graph/CompiledGraph.java)
- Legacy ASM sample generator:
  - [src/main/java/graph/ByteCodeGenerator.java](../graph/ByteCodeGenerator.java)
- Fused operation codegen:
  - [src/main/java/graph/codegen/FusedOperationGenerator.java](../graph/codegen/FusedOperationGenerator.java) (F32/F64)
  - [src/main/java/graph/codegen/HFusedOperationGenerator.java](../graph/codegen/HFusedOperationGenerator.java) (F16)
  - [src/main/java/graph/codegen/FusedKernelGeneratorRouter.java](../graph/codegen/FusedKernelGeneratorRouter.java)
  - [src/main/java/graph/codegen/CompiledFusedKernelFactory.java](../graph/codegen/CompiledFusedKernelFactory.java)
  - [src/main/java/graph/codegen/FusedExpressionPlan.java](../graph/codegen/FusedExpressionPlan.java)
- Optimizer module:
  - [src/main/java/graph/optimizer/README.md](../graph/optimizer/README.md)

## Compile Pipeline

`CompiledGraph.compile()` performs:

1. Build forward graph from `rootTensor.forwardOutput().topologicalSort()`.
2. If graph has trainable leaves:
  - seed root gradient
  - build backward graph nodes
  - mark backward nodes
  - create super-root to unify forward + backward sinks
3. Run optimizer over unified graph.
4. Cache the forward boundary used later by `prepare(...)`.

`CompiledGraph.prepare(RuntimeConfig)` then builds runtime-specific `PreparedExecution`:

- forward steps
- backward steps
- per-node execution metadata (`CompiledNodeExecutionMetadata`)

## Execution Pipeline

`PreparedExecution.execute(...)`:

1. Execute forward section in topological order.
2. Sync root tensor data from optimized/fused result node.
3. If mode is `FORWARD_BACKWARD`, execute prepared backward section.

Dispatch uses:

- [src/main/java/backend/ComputeEngine.java](../backend/ComputeEngine.java)

with prepared metadata per node for low-overhead execution.

Debug switches for pre-resolve behavior:

- `-Dcg.cpu.disableResolveExecutionHints=true` disables compile-time resolve pass in `CompiledGraph`.
- `-Dcg.cpu.disablePreResolvedExecutionPlan=true` disables cached CPU execution-plan reuse in `CPUBackend`.

Approximation-aware fused path:

- fused generated ops for `exp`/`tanh` receive explicit fused execution options
- explicit fused ops `fastExp` / `fastTanh` are also supported

## Training vs Inference Modes

- Inference mode:
  - forward execution only
  - backward section skipped
- Training mode:
  - forward section + backward section

Primary explicit API:

- `PreparedExecution.execute(FORWARD)`
- `PreparedExecution.execute(FORWARD_BACKWARD)`

## Backward Graph Handling

Backward nodes are collected from gradients attached to forward nodes, traversed in dependency-safe order, and marked via `tensor.setBackward(true)` so optimizer/rules can preserve phase boundaries.

`backward()` API remains available and executes backward-only traversal from compiled backward start when present.

## Fused Codegen Path

When optimizer fuses element-wise clusters:

1. Fusion rule produces `FusedOperation` node(s).
2. `FusedOperationFactory` converts the cluster into a `FusedExpressionPlan`-backed descriptor.
3. `CompiledGraph.prepare(...)` asks `CompiledFusedKernelFactory` for a compiled fused runtime executable.
4. `FusedKernelGeneratorRouter` selects dtype-specific codegen.
5. `FusedOperationGenerator` generates the `FLOAT32/FLOAT64` fused executable from plan IR.
6. `HFusedOperationGenerator` generates the `FLOAT16` fused executable from the same plan IR.
7. `CpuFusedKernel` executes the prepared fused kernel from node metadata.

Related files:

- [src/main/java/graph/optimizer/rules/FuseElementWiseRule.java](../graph/optimizer/rules/FuseElementWiseRule.java)
- [src/main/java/operations/FusedOperation.java](../operations/FusedOperation.java)
- [src/main/java/operations/FusedOperationFactory.java](../operations/FusedOperationFactory.java)
- [src/main/java/graph/codegen/FusedExpressionPlan.java](../graph/codegen/FusedExpressionPlan.java)
- [src/main/java/graph/codegen/CompiledFusedKernelFactory.java](../graph/codegen/CompiledFusedKernelFactory.java)
- [src/main/java/graph/codegen/FusedKernelGeneratorRouter.java](../graph/codegen/FusedKernelGeneratorRouter.java)
- [src/main/java/backend/kernels/cpu/CpuFusedKernel.java](../backend/kernels/cpu/CpuFusedKernel.java)

## Canonicalization Notes

Algebraic rewriting includes canonical sigmoid recognition in inference graphs:

- `1 / (1 + exp(-x)) -> sigmoid(x)`
- also recognizes `exp(x * -1)` form (`mulScalar(-1)`)

In training graphs (`requiresGrad=true` path), this rewrite is intentionally skipped to keep backward graph construction semantics unchanged.

## Notes

- `ByteCodeGenerator` is a minimal ASM sample/utility and not the primary fused execution path.
- Optimizer configuration and profile-driven runtime knobs are handled via `OptimizerFactory` and benchmark profile files.
- Numerics stability diagnostics and post-check tooling are documented in [src/main/java/numerics/README.md](../numerics/README.md).
