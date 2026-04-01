# Optimizer (Graph/optimizer)

## Goal

- Transform the topologically sorted Tensor graph before execution.
- Reduce node count, remove redundant expressions, and improve locality/throughput.
- Preserve correctness for both forward and backward graph parts.

## Main Components

- Orchestrator:
  - [src/main/java/graph/optimizer/GraphOptimizer.java](../../graph/optimizer/GraphOptimizer.java)
  - [src/main/java/graph/optimizer/OptimizationRule.java](../../graph/optimizer/OptimizationRule.java)
  - [src/main/java/graph/optimizer/OptimizerFactory.java](../../graph/optimizer/OptimizerFactory.java)
- Rules:
  - [src/main/java/graph/optimizer/rules/AlgebraicRewritingRule.java](../../graph/optimizer/rules/AlgebraicRewritingRule.java)
  - [src/main/java/graph/optimizer/rules/CommonSubexpressionEliminationRule.java](../../graph/optimizer/rules/CommonSubexpressionEliminationRule.java)
  - [src/main/java/graph/optimizer/rules/FuseElementWiseRule.java](../../graph/optimizer/rules/FuseElementWiseRule.java)
  - [src/main/java/graph/optimizer/rules/MemoryOptimizerRule.java](../../graph/optimizer/rules/MemoryOptimizerRule.java)
- Fused codegen:
  - [src/main/java/operations/FusedOperation.java](../../operations/FusedOperation.java)
  - [src/main/java/operations/FusedOperationFactory.java](../../operations/FusedOperationFactory.java)
  - [src/main/java/graph/codegen/FusedExpressionPlan.java](../../graph/codegen/FusedExpressionPlan.java)
  - [src/main/java/graph/codegen/CompiledFusedKernelFactory.java](../../graph/codegen/CompiledFusedKernelFactory.java)
  - [src/main/java/graph/codegen/FusedKernelGeneratorRouter.java](../../graph/codegen/FusedKernelGeneratorRouter.java)
  - [src/main/java/graph/codegen/FusedOperationGenerator.java](../../graph/codegen/FusedOperationGenerator.java) (F32/F64)
  - [src/main/java/graph/codegen/HFusedOperationGenerator.java](../../graph/codegen/HFusedOperationGenerator.java) (F16)

## Data Flow

1. `Tensor.compile(...)` creates `CompiledGraph`.
2. `CompiledGraph.compile()` builds `finalGraph` and calls `optimizer.optimize(...)`.
3. Rules run sequentially over a topologically sorted node list.
4. `CompiledGraph.prepare(RuntimeConfig)` builds runtime-specific `PreparedExecution` steps and per-node metadata.
5. `PreparedExecution.execute(...)` runs the optimized order for inference or training.

Files:
- [src/main/java/tensor/Tensor.java](../../tensor/Tensor.java)
- [src/main/java/graph/CompiledGraph.java](../../graph/CompiledGraph.java)

## Rule API Contract

Rule interface:
- [src/main/java/graph/optimizer/OptimizationRule.java](../../graph/optimizer/OptimizationRule.java)

A rule:
- accepts `List<Tensor>` (topologically sorted graph),
- returns a new `List<Tensor>` representing the transformed graph.

Rules must:
- preserve dependencies and execution order,
- preserve gradient flow,
- return a consistent topological order.

## Rule Summary

- `AlgebraicRewritingRule`
  - local algebraic expression simplifications.
  - includes inference-only canonical sigmoid rewrite:
    - `inv(add(1, exp(neg(x)))) -> sigmoid(x)`
    - `inv(add(1, exp(mulScalar(x, -1)))) -> sigmoid(x)`
- `CommonSubexpressionEliminationRule`
  - merges equivalent subexpressions (with optional strict safety).
- `FuseElementWiseRule`
  - groups element-wise nodes into fused clusters using a cost model.
  - respects phase boundaries (forward/backward), materialization points, and shared-expensive policy.
- `MemoryOptimizerRule`
  - rewrites focused on improved memory behavior/reuse.

## Fused Operations

`FuseElementWiseRule` can replace a cluster with a single `FusedOperation` node.

- Fused node uses `OpType.FUSED`.
- `FusedOperation` is a descriptor backed by `FusedExpressionPlan`.
- Runtime execution goes through `CpuFusedKernel`, which executes a prepared `CompiledFusedKernel` from node metadata.
- Fused bytecode is generated from plan IR, not from live `Tensor` graph nodes.

Files:
- [src/main/java/graph/optimizer/rules/FuseElementWiseRule.java](../../graph/optimizer/rules/FuseElementWiseRule.java)
- [src/main/java/backend/kernels/cpu/CpuFusedKernel.java](../../backend/kernels/cpu/CpuFusedKernel.java)
- [src/main/java/graph/codegen/FusedKernelGeneratorRouter.java](../../graph/codegen/FusedKernelGeneratorRouter.java)
- [src/main/java/graph/codegen/CompiledFusedKernelFactory.java](../../graph/codegen/CompiledFusedKernelFactory.java)
- [src/main/java/graph/codegen/FusedExpressionPlan.java](../../graph/codegen/FusedExpressionPlan.java)
- [src/main/java/graph/codegen/FusedOperationGenerator.java](../../graph/codegen/FusedOperationGenerator.java)
- [src/main/java/graph/codegen/HFusedOperationGenerator.java](../../graph/codegen/HFusedOperationGenerator.java)

## Benchmark/Autotune Integration

Optimizer stage order and tuning knobs are controlled by the benchmark framework:
- [src/main/java/benchmark/OptimizerBenchmarkFramework.java](../../benchmark/OptimizerBenchmarkFramework.java)
- [src/main/java/benchmark/OptimizerCandidateFactory.java](../../benchmark/OptimizerCandidateFactory.java)
- [src/main/java/benchmark/TuningKnobs.java](../../benchmark/TuningKnobs.java)

Autotune is two-phase:
- Phase 1: coarse candidate screening.
- Phase 2: refined re-measurement of finalists.

Winning profiles are persisted:
- `config/optimizer-profile.json` (runtime training profile),
- `config/optimizer-hw-profiles.tsv` (runtime HW-bucket profiles),
- `build/optimizer-autotune/best-profile-training.json`,
- `build/optimizer-autotune/best-profile-inference.json`.
- `build/optimizer-autotune/candidate-history.tsv` (context-aware unsafe candidate cache).
- `build/numerics/autotune-postcheck-<dtype>-<timestamp>.tsv` (numerics post-check report for checked finalists).

Runtime profile selection priority is:

1. HW-bucket profile (`optimizer-hw-profiles.tsv`).
2. Architecture preset (`os.arch`, includes ARM/aarch64 and x86_64/amd64).
3. Best-profile overrides (`best-profile-*.json`).
4. Defaults.

Autotune now records context-specific unsafe candidates (`MISMATCH_*` + `NUMERICS_POSTCHECK_UNSAFE`) and skips them on subsequent runs with matching context.

CPU dispatch-related tuned knobs include:

- threshold knobs (`vectorMinSize`, `parallelMinSize`, `chunksPerWorker`, `minChunkSize`, `contiguousMaterializeThreshold`)
- low-cost scheduler threshold (`lowCostNsPerElementThreshold`)
- vector policies by op group (`vectorPolicyCheap`, `vectorPolicyTranscendental`, `vectorPolicyReduction`)

## Adding a New Rule

1. Add a class in `src/main/java/graph/optimizer/rules/` implementing `OptimizationRule`.
2. Implement the transformation `List<Tensor> -> List<Tensor>`.
3. Register the rule in `OptimizerFactory` or directly when assembling `GraphOptimizer`.
4. Validate correctness:
- numerical equivalence against baseline,
- gradient preservation,
- regression tests for edge cases.

## Build/Runtime Notes

- The project uses ASM (`org.ow2.asm`).
- CPU vector path uses `jdk.incubator.vector`.
- When running benchmark locally without Gradle, ASM must be on classpath.
