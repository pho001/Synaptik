# Graph (src/main/java/Graph)

## Purpose

The Graph module compiles tensor expression DAGs into executable plans, runs forward/backward execution, and integrates optimizer + fused code generation.

## Main Components

- Compiled execution plan:
  - [src/main/java/Graph/CompiledGraph.java](../Graph/CompiledGraph.java)
- Legacy ASM sample generator:
  - [src/main/java/Graph/ByteCodeGenerator.java](../Graph/ByteCodeGenerator.java)
- Fused operation codegen:
  - [src/main/java/Graph/codegen/FusedOperationGenerator.java](../Graph/codegen/FusedOperationGenerator.java) (F32/F64)
  - [src/main/java/Graph/codegen/HFusedOperationGenerator.java](../Graph/codegen/HFusedOperationGenerator.java) (F16)
  - [src/main/java/Graph/codegen/FusedOperationGeneratorRouter.java](../Graph/codegen/FusedOperationGeneratorRouter.java)
- Optimizer module:
  - [src/main/java/Graph/optimizer/README.md](../Graph/optimizer/README.md)

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
  - resolved CPU execution plan (dtype/remap/dispatch/broadcast hints)
  - CPU config epoch for staleness checks
5. Cache forward boundary index for two-phase execution (forward first, backward section after).

## Execution Pipeline

`CompiledGraph.execute()`:

1. Execute forward section in topological order.
2. Sync root tensor data from optimized/fused result node.
3. If training mode is on, execute remaining backward section.

During execution, graph runtime also sets training context in `ComputeEngine`, so backend approximation policy (`ApproxMode.TRAINING_ONLY`) can be applied consistently.

Dispatch uses:

- [src/main/java/Backend/ComputeEngine.java](../Backend/ComputeEngine.java)

with pre-resolved backend per node for low-overhead execution.

Debug switches for pre-resolve behavior:

- `-Dcg.cpu.disableResolveExecutionHints=true` disables compile-time resolve pass in `CompiledGraph`.
- `-Dcg.cpu.disablePreResolvedExecutionPlan=true` disables cached CPU execution-plan reuse in `CPUBackend`.

Approximation-aware fused path:

- fused generated ops for `exp`/`tanh` route through fused scalar/vector helpers
- helpers honor current `ComputeEngine` approximation mode
- explicit fused ops `fastExp` / `fastTanh` are also supported

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

- [src/main/java/Graph/optimizer/rules/FuseElementWiseRule.java](../Graph/optimizer/rules/FuseElementWiseRule.java)
- [src/main/java/Operations/FusedOperation.java](../Operations/FusedOperation.java)
- [src/main/java/Backend/kernels/cpu/CpuFusedKernel.java](../Backend/kernels/cpu/CpuFusedKernel.java)

## Canonicalization Notes

Algebraic rewriting includes canonical sigmoid recognition in inference graphs:

- `1 / (1 + exp(-x)) -> sigmoid(x)`
- also recognizes `exp(x * -1)` form (`mulScalar(-1)`)

In training graphs (`requiresGrad=true` path), this rewrite is intentionally skipped to keep backward graph construction semantics unchanged.

## Notes

- `ByteCodeGenerator` is a minimal ASM sample/utility and not the primary fused execution path.
- Optimizer configuration and profile-driven runtime knobs are handled via `OptimizerFactory` and benchmark profile files.
- Numerics stability diagnostics and post-check tooling are documented in [src/main/java/Numerics/README.md](../Numerics/README.md).
