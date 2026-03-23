# Graph (src/Graph)

## Purpose

The Graph module compiles tensor expression DAGs into executable plans, runs forward/backward execution, and integrates optimizer + fused code generation.

## Main Components

- Compiled execution plan:
  - [src/Graph/CompiledGraph.java](src/Graph/CompiledGraph.java)
- Legacy ASM sample generator:
  - [src/Graph/ByteCodeGenerator.java](src/Graph/ByteCodeGenerator.java)
- Fused operation codegen:
  - [src/Graph/codegen/FusedOperationGenerator.java](src/Graph/codegen/FusedOperationGenerator.java) (F32/F64)
  - [src/Graph/codegen/HFusedOperationGenerator.java](src/Graph/codegen/HFusedOperationGenerator.java) (F16)
  - [src/Graph/codegen/FusedOperationGeneratorRouter.java](src/Graph/codegen/FusedOperationGeneratorRouter.java)
- Optimizer module:
  - [src/Graph/optimizer/README.md](src/Graph/optimizer/README.md)

## Compile Pipeline

`CompiledGraph.compile()` performs:

1. Build forward graph from `rootTensor.forwardOutput().topologicalSort()`.
2. If graph has trainable leaves:
  - seed root gradient
  - build backward graph nodes
  - mark backward nodes
  - create super-root to unify forward + backward sinks
3. Run optimizer over unified graph.
4. Pre-resolve per-node execution metadata:
  - resolved backend
  - resolved CPU kernel
5. Cache forward boundary index for two-phase execution (forward first, backward section after).

## Execution Pipeline

`CompiledGraph.execute()`:

1. Execute forward section in topological order.
2. Sync root tensor data from optimized/fused result node.
3. If training mode is on, execute remaining backward section.

Dispatch uses:

- [src/Backend/ComputeEngine.java](src/Backend/ComputeEngine.java)

with pre-resolved backend per node for low-overhead execution.

## Training vs Inference Modes

- Inference mode:
  - forward execution only
  - backward section skipped
- Training mode:
  - forward section + backward section

Switches:

- `setTrainingModeOff()` for inference behavior
- `setTrainingModeOn()` for training behavior

## Backward Graph Handling

Backward nodes are collected from gradients attached to forward nodes, traversed in dependency-safe order, and marked via `tensor.setBackward(true)` so optimizer/rules can preserve phase boundaries.

`backward()` API remains available and executes backward-only traversal from compiled backward start when present.

## Fused Codegen Path

When optimizer fuses element-wise clusters:

1. Fusion rule produces `FusedOperation` node(s).
2. `FusedOperationGeneratorRouter` selects dtype-specific codegen.
3. `FusedOperationGenerator` generates fused `apply(...)` bytecode for `FLOAT32/FLOAT64`.
4. `HFusedOperationGenerator` handles `FLOAT16`.
5. CPU fused kernel executes fused op in runtime path.

Related files:

- [src/Graph/optimizer/rules/FuseElementWiseRule.java](src/Graph/optimizer/rules/FuseElementWiseRule.java)
- [src/Operations/FusedOperation.java](src/Operations/FusedOperation.java)
- [src/Backend/kernels/cpu/CpuFusedKernel.java](src/Backend/kernels/cpu/CpuFusedKernel.java)

## Canonicalization Notes

Algebraic rewriting includes canonical sigmoid recognition in inference graphs:

- `1 / (1 + exp(-x)) -> sigmoid(x)`
- also recognizes `exp(x * -1)` form (`mulScalar(-1)`)

In training graphs (`requiresGrad=true` path), this rewrite is intentionally skipped to keep backward graph construction semantics unchanged.

## Notes

- `ByteCodeGenerator` is a minimal ASM sample/utility and not the primary fused execution path.
- Optimizer configuration and profile-driven runtime knobs are handled via `OptimizerFactory` and benchmark profile files.
