# Optimizer (src/main/java/graph/optimizer)

## Goal

The optimizer transforms a compiled tensor graph before runtime preparation.

Main objectives:

- reduce redundant work
- simplify expressions
- fuse profitable element-wise regions
- preserve correctness across forward and backward graph sections

## Main Components

- Orchestration:
  - [src/main/java/graph/optimizer/GraphOptimizer.java](../../graph/optimizer/GraphOptimizer.java)
  - [src/main/java/graph/optimizer/OptimizationRule.java](../../graph/optimizer/OptimizationRule.java)
  - [src/main/java/graph/optimizer/OptimizerFactory.java](../../graph/optimizer/OptimizerFactory.java)
  - [src/main/java/graph/optimizer/OptimizerProfiles.java](../../graph/optimizer/OptimizerProfiles.java)
- Config objects:
  - [src/main/java/config/optimizer/OptimizerConfig.java](../../config/optimizer/OptimizerConfig.java)
  - [src/main/java/config/optimizer/OptimizerStage.java](../../config/optimizer/OptimizerStage.java)
  - [src/main/java/config/optimizer/CseConfig.java](../../config/optimizer/CseConfig.java)
  - [src/main/java/config/optimizer/FuseConfig.java](../../config/optimizer/FuseConfig.java)
- Rules:
  - [src/main/java/graph/optimizer/rules/AlgebraicRewritingRule.java](../../graph/optimizer/rules/AlgebraicRewritingRule.java)
  - [src/main/java/graph/optimizer/rules/CommonSubexpressionEliminationRule.java](../../graph/optimizer/rules/CommonSubexpressionEliminationRule.java)
  - [src/main/java/graph/optimizer/rules/FuseElementWiseRule.java](../../graph/optimizer/rules/FuseElementWiseRule.java)
  - [src/main/java/graph/optimizer/rules/MemoryOptimizerRule.java](../../graph/optimizer/rules/MemoryOptimizerRule.java)
- Fusion support:
  - [src/main/java/graph/optimizer/fusion/FusedCostModel.java](../../graph/optimizer/fusion/FusedCostModel.java)
  - [src/main/java/graph/optimizer/fusion/FusedExternalInputCollector.java](../../graph/optimizer/fusion/FusedExternalInputCollector.java)
  - [src/main/java/graph/optimizer/fusion/FusedPrecisionResolver.java](../../graph/optimizer/fusion/FusedPrecisionResolver.java)
  - [src/main/java/graph/optimizer/fusion/FusedSignatureBuilder.java](../../graph/optimizer/fusion/FusedSignatureBuilder.java)

## Placement in the Execution Flow

Preferred flow today is:

1. construct tensor expression graph
2. call `CompiledGraph.compile(root, optimizerConfig)`
3. `CompiledGraph` builds forward/backward graph structure as needed
4. optimizer runs over the assembled topologically sorted graph
5. `CompiledGraph.prepare(runtimeConfig)` converts the optimized graph into runtime steps

The optimizer does not own runtime execution.
It only transforms the graph before preparation.

## Rule Contract

Each rule:

- accepts `List<Tensor>` representing a topologically sorted graph
- returns a transformed `List<Tensor>`

Rules must preserve:

- dependency ordering
- graph reachability
- backward-flow correctness
- phase boundaries between forward and backward sections

## Rule Summary

### `AlgebraicRewritingRule`

- local algebraic simplifications
- canonical sigmoid rewrite in forward-only graphs:
  - `inv(add(1, exp(neg(x)))) -> sigmoid(x)`
  - `inv(add(1, exp(mulScalar(x, -1)))) -> sigmoid(x)`

### `CommonSubexpressionEliminationRule`

- merges equivalent subexpressions
- can use stricter or more aggressive safety configuration through `CseConfig`

### `FuseElementWiseRule`

- groups profitable element-wise regions into fused clusters
- respects forward/backward phase boundaries
- uses explicit fuse policy from `FuseConfig`
- cooperates with fusion support helpers for:
  - external inputs
  - precision resolution
  - cluster signature building
  - cost decisions

### `MemoryOptimizerRule`

- graph rewrites aimed at better memory behavior and reuse patterns
- current planner is liveness-aware and slot-assignment-based
- planner policy is modeled explicitly through `MemoryPlannerPolicy`
- planner exposes internal liveness/slot metrics through `MemoryPlanSummary`
- current debug dumps can explain:
  - summary metrics
  - slot assignment
  - role per tensor
  - storage owner
  - birth / last-read interval
  - saved-forward report
- current summary metrics include:
  - reusable interval count
  - slot count
  - reuse count
  - peak live bytes
  - peak reusable bytes
  - peak saved-forward bytes
  - peak gradient-target bytes
  - forward/backward peak live bytes
  - saved-forward hold statistics

Relevant files:

- [src/main/java/graph/optimizer/rules/MemoryOptimizerRule.java](../../graph/optimizer/rules/MemoryOptimizerRule.java)
- [src/main/java/graph/optimizer/memory/MemoryPlanner.java](../../graph/optimizer/memory/MemoryPlanner.java)
- [src/main/java/graph/optimizer/memory/MemoryPlan.java](../../graph/optimizer/memory/MemoryPlan.java)
- [src/main/java/graph/optimizer/memory/MemoryPlanSummary.java](../../graph/optimizer/memory/MemoryPlanSummary.java)
- [src/main/java/graph/optimizer/memory/MemoryPlannerPolicy.java](../../graph/optimizer/memory/MemoryPlannerPolicy.java)

## Fused Operations

The fused optimizer path now follows a descriptor + prepared-runtime split.

Current model:

- optimizer replaces a cluster with a `FusedOperation` descriptor node
- `FusedOperationFactory` builds `FusedExpressionPlan`
- `CompiledGraph.prepare(...)` compiles a runtime fused executable through `CompiledFusedKernelFactory`
- prepared fused executable is stored in `CompiledNodeExecutionMetadata`
- `CpuFusedKernel` executes that prepared executable

This means:

- `FusedOperation` is not itself the compiled kernel
- live `Tensor` graph nodes are not passed directly to generated runtime code
- generated code consumes plan IR and prepared runtime bindings

## Config and Defaults

Primary compile-time config is:

- `OptimizerConfig`

Default presets:

- `OptimizerConfig.noOptimization()`
- `OptimizerConfig.trainingDefaults()`
- `OptimizerConfig.inferenceDefaults()`

`OptimizerFactory` converts these config objects into concrete `GraphOptimizer` instances.

For public graph compilation, `OptimizerConfig` is the intended API surface.
`GraphOptimizer` is a lower-level pipeline object used internally by optimizer/benchmark tooling, not the preferred public compile contract.

## Benchmark / Autotune Integration

The benchmark layer still owns persisted winning profiles and tuning history.

Important persisted files:

- `config/optimizer-profile.json`
- `config/optimizer-profile-f32.json`
- `config/optimizer-profile-f64.json`
- `config/optimizer-hw-profiles.tsv`
- `config/optimizer-hw-profiles-f32.tsv`
- `config/optimizer-hw-profiles-f64.tsv`
- `build/optimizer-autotune/best-profile-training.json`
- `build/optimizer-autotune/best-profile-inference.json`
- `build/optimizer-autotune/candidate-history.tsv`

That benchmark/autotune layer is still transitional. The architectural direction is documented in the local TODO/planning notes, but the current persisted profile chain above is what the codebase still uses today.

## Adding a New Rule

1. Add a new rule class under `src/main/java/graph/optimizer/rules/`.
2. Implement `OptimizationRule`.
3. Register it in `OptimizerFactory` or inject it manually into `GraphOptimizer`.
4. Validate:
   - numerical equivalence
   - gradient preservation
   - regressions for broadcasting, dtype handling, and fused boundaries

## Build / Runtime Notes

- Fused codegen uses ASM.
- CPU vector execution uses `jdk.incubator.vector`.
- Optimizer output is consumed by `CompiledGraph`, not directly by backend kernels.
