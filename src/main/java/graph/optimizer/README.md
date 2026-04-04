# Optimizer (src/main/java/graph/optimizer)

## Contents

- [Goal](#goal)
- [Main Components](#main-components)
- [Placement in the Execution Flow](#placement-in-the-execution-flow)
- [Rule Contract](#rule-contract)
- [Rule Summary](#rule-summary)
  - [`RewriteRule` Stage Family](#rewriterule-stage-family)
  - [`CommonSubexpressionEliminationRule`](#commonsubexpressioneliminationrule)
  - [`FuseElementWiseRule`](#fuseelementwiserule)
  - [`MemoryOptimizerRule`](#memoryoptimizerrule)
- [Fused Operations](#fused-operations)
- [Config and Defaults](#config-and-defaults)
- [Benchmark / Autotune Integration](#benchmark--autotune-integration)
- [Adding a New Rule](#adding-a-new-rule)
- [Build / Runtime Notes](#build--runtime-notes)

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
  - [src/main/java/graph/optimizer/OptimizerGraphSupport.java](../../graph/optimizer/OptimizerGraphSupport.java)
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

Operation-surface strategy is defined in:

- [src/main/java/operations/README.md](../../operations/README.md)

In particular, optimizer lowering should respect the difference between:

- compositional API surface
- canonical graph primitives
- semantic-preserving lowering opportunities

## Rule Contract

Each rule:

- accepts `List<Tensor>` representing a topologically sorted graph
- returns a transformed `List<Tensor>`

Rules must preserve:

- dependency ordering
- graph reachability
- backward-flow correctness
- phase boundaries between forward and backward sections
- semantic equivalence when lowering composed forms into specialized primitives

Shared graph rewrite mechanics such as input replacement and topological-closure rebuild now live in:

- [src/main/java/graph/optimizer/OptimizerGraphSupport.java](../../graph/optimizer/OptimizerGraphSupport.java)

## Rule Summary

### `RewriteRule` Stage Family

`OptimizerStage.AR` no longer maps to a class in `graph.optimizer.rules`.
It maps directly to the rewrite family in:

- [src/main/java/graph/optimizer/rewrite/RewriteRule.java](../../graph/optimizer/rewrite/RewriteRule.java)

That class is a composite stage wrapper over explicit rewrite delegates.

Current delegate order:

1. `AlgebraicRewrite`
2. `LinearLoweringRewrite`
3. `Conv2dLoweringRewrite`

This is the intended architectural boundary:

- `graph.optimizer.rules`
  - top-level optimizer stages such as CSE / FUSE / MEM
- `graph.optimizer.rewrite`
  - the internal rewrite family executed by `OptimizerStage.AR`

Why this split is useful:

- AR is one optimizer stage from the outside
- internally it can still host several rewrite passes with their own semantics

- local algebraic simplifications
- dispatches by `Operation.OpType`, not by concrete class names
- currently targets the numeric unary/binary subset where the rewrites are semantically valid
- current focus is classic algebraic cleanup such as:
  - identity elimination
  - `pow(x, 2) -> x * x`
  - inverse / negation / scalar canonicalization
  - local constant folding for scalar forms
- it does **not** currently lower compare/select surface forms such as:
  - canonical sigmoid decompositions into `sigmoid`
  - `where`-based relu/clamp patterns into specialized primitives

That boundary is intentional in the current code:

- algebraic rewrite stays in the numeric algebra layer
- piecewise/select lowering remains a separate future rewrite concern if reintroduced

### `CommonSubexpressionEliminationRule`

- merges equivalent subexpressions
- can use stricter or more aggressive safety configuration through `CseConfig`
- uses structural signatures instead of stringly class-name signatures
- signatures are built from:
  - `Operation.OpType`
  - forward/backward phase
  - strict-safety metadata (`requiresGrad`, backend, output shape)
  - structurally resolved input signatures
  - explicit operation parameters for parametric ops
- current parameter-aware coverage includes:
  - `pow`
  - `mulScalar`
  - `sum(dim, keepDims)`
  - `reduceMin(dim, keepDims)`
  - `reduceMax(dim, keepDims)`
  - `minGrad` / `maxGrad`
  - `reduceMinGrad` / `reduceMaxGrad`
  - layout/shape ops such as `reshape`, `permute`, `expand`, `expandDims`, `squeeze`
- `noop` and fused nodes remain CSE boundaries and are intentionally not merged

This matters because shape/layout transforms and reduction shape policy are not interchangeable computations.
For example, `sum(axis, false)` and `sum(axis, true)` must not collapse to the same node, and different `permute(...)` layouts must not merge just because they share the same input.

### `FuseElementWiseRule`

- groups profitable fused-compute regions into fused clusters
- respects forward/backward phase boundaries
- uses explicit fuse policy from `FuseConfig`
- cooperates with fusion support helpers for:
  - external input collection
  - access-chain validation
  - precision resolution
  - cluster signature building
  - cost decisions

Detailed behavior:

1. identify candidate roots that are retained graph nodes
2. walk backward only through fused-compute-compatible parents
3. stop at barriers
4. collect external inputs for the cluster boundary
5. resolve absorbable view/layout chains into fused access metadata
6. replace the cluster root with a `FusedOperation` descriptor
7. rebuild topological closure so dead absorbed nodes disappear from the optimized execution list

#### What counts as fused compute

Current fused compute algebra intentionally includes only:

- unary numeric ops
- binary numeric ops
- compare ops
- logical ops
- `where`

#### What does not count as fused compute

These are **not** fused compute nodes:

- `select`
- `reshape`
- `expand`
- `permute`
- `expandDims`
- `squeeze`

They are treated as:

- absorbable access/layout transforms on external inputs

#### Hard barriers

Current hard barriers include:

- indexing ops
  - `gather`
  - `takeAlongAxis`
  - `scatterAdd`
- reductions
- `matmul`
- losses
- special grad kernels

#### Example: accepted cluster

```java
Tensor out = Tensor.where(a.greaterThan(b), x, y).relu().mul(x);
```

Fused interpretation:

```java
// Step 1: bool intermediate
// cond = a > b
//
// Step 2: numeric select
// v = cond ? x : y
//
// Step 3: more numeric compute
// out = relu(v) * x
```

This is a valid fused compute cluster because:

- all nodes live over the same logical output space
- bool values are only intermediates inside the fused compute algebra
- there is no reduction or indexing barrier

#### Example: access chain absorbed into fused input metadata

```java
Tensor out = base.select(0, 1).permute(0).relu().exp();
```

Fusion rule behavior:

- `relu` and `exp` are fused compute nodes
- `select(...)` and `permute(...)` are not fused compute nodes
- the root fused node receives the backing tensor of `base`
- access metadata describes how to read the logical input view

#### Example: barrier

```java
Tensor out = base.gather(indices, 1).relu().exp();
```

Fusion rule behavior:

- `gather(...)` is a barrier
- only the arithmetic nodes above it may fuse
- `gather(...)` remains an ordinary graph node

### `MemoryOptimizerRule`

- graph rewrites aimed at better memory behavior and reuse patterns
- current planner is liveness-aware and slot-assignment-based
- planner policy is modeled explicitly through `MemoryPlannerPolicy`
- public compile-time config for the MEM stage is carried by `OptimizerConfig.memory()`
- `MemoryPlannerPolicy` is the runtime planner-facing projection of that config
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
  - reusable fresh allocation count
  - reuse hit-rate
  - allocated slot bytes
  - peak live bytes
  - peak reusable bytes
  - peak saved-forward bytes
  - peak gradient-target bytes
  - forward/backward peak live bytes
  - saved-forward hold statistics

#### Memory policy shape

The compile-time MEM-stage policy is:

```java
public record MemoryConfig(
        boolean separateForwardBackwardPools,
        boolean allowCrossPhaseReuse,
        boolean allowLargerBufferReuse,
        int minReusableBufferSize
) {}
```

Meaning of the fields:

- `separateForwardBackwardPools`
  - keeps forward-temp and backward-temp reuse pools separate
  - conservative default because it makes liveness reasoning and explain output cleaner
- `allowCrossPhaseReuse`
  - allows one reuse space across forward/backward when pools are not separated
  - currently a deliberate opt-in knob, not the default policy
- `allowLargerBufferReuse`
  - permits reusing a slot with `slot.size >= tensor.size`
  - improves reuse opportunities at the cost of potentially holding larger buffers
- `minReusableBufferSize`
  - excludes tiny temporaries from slot planning
  - useful when planning overhead or slot churn would outweigh any real gain

Important boundary:

- this is a **compile-time optimizer policy**
- it belongs to `OptimizerConfig`
- it is not a runtime execution knob like BLAS or approximation mode

#### Example: explicit MEM-stage config

```java
OptimizerConfig config = OptimizerConfig.trainingDefaults()
        .withMemory(new MemoryConfig(
                true,   // separateForwardBackwardPools
                false,  // allowCrossPhaseReuse
                true,   // allowLargerBufferReuse
                16      // minReusableBufferSize
        ));
```

This means:

- the graph still uses separate forward/backward reuse pools
- cross-phase reuse stays disabled
- larger reusable buffers may be reused for smaller temporaries
- very small temporaries below `16` elements are skipped by slot planning

#### MemoryPlanSummary as benchmark/autotune surface

`MemoryPlanSummary` is not only a debug artifact.

It is intended to be the stable machine-readable planner report for:

- benchmark comparisons
- future planner autotune
- policy evaluation without local probe scripts

Today it exposes both:

- human-readable explain output through `MemoryPlan.explain()`
- machine-readable metrics through `MemoryPlanSummary.toMetricMap()`

Relevant files:

- [src/main/java/graph/optimizer/rules/MemoryOptimizerRule.java](../../graph/optimizer/rules/MemoryOptimizerRule.java)
- [src/main/java/graph/optimizer/memory/MemoryPlanner.java](../../graph/optimizer/memory/MemoryPlanner.java)
- [src/main/java/graph/optimizer/memory/MemoryPlan.java](../../graph/optimizer/memory/MemoryPlan.java)
- [src/main/java/graph/optimizer/memory/MemoryPlanSummary.java](../../graph/optimizer/memory/MemoryPlanSummary.java)
- [src/main/java/graph/optimizer/memory/MemoryPlannerPolicy.java](../../graph/optimizer/memory/MemoryPlannerPolicy.java)
- [src/main/java/config/optimizer/MemoryConfig.java](../../config/optimizer/MemoryConfig.java)

## Fused Operations

The fused optimizer path now follows a descriptor + prepared-runtime split.

Current model:

- optimizer replaces a cluster with a `FusedOperation` descriptor node
- `FusedOperationFactory` builds `FusedExpressionPlan`
- `FusedAccessResolver` validates absorbable access chains and resolves backing runtime inputs
- `CompiledGraph.prepare(...)` compiles a runtime fused executable through `CompiledFusedKernelFactory`
- prepared fused executable is stored in `CompiledNodeExecutionMetadata`
- `CpuFusedKernel` executes that prepared executable

This means:

- `FusedOperation` is not itself the compiled kernel
- live `Tensor` graph nodes are not passed directly to generated runtime code
- generated code consumes:
  - typed fused node IR
  - typed external input access metadata
  - prepared runtime bindings resolved to backing tensors

Important current fused IR pieces:

- [src/main/java/graph/codegen/FusedNodePlan.java](../../graph/codegen/FusedNodePlan.java)
  - one fused compute node
  - explicit `opType`
  - explicit `outputType`
  - typed node attributes
- [src/main/java/graph/codegen/FusedExternalInputPlan.java](../../graph/codegen/FusedExternalInputPlan.java)
  - one external runtime input contract
  - backing tensor index
  - access shape/strides
  - `storageOffset`
  - access kind

Example:

```java
Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
Tensor view = base.select(0, 1);
Tensor out = view.relu().exp();
```

The fused descriptor path does **not** treat `select(...)` as a fused compute node.
Instead it creates:

- one backing runtime input for `base`
- one external access descriptor carrying:
  - logical output shape `[3]`
  - `storageOffset = 3`
  - effective strides `[1]`
- two fused compute nodes:
  - `RELU`
  - `EXP`

This is the main architectural difference between:

- compute algebra
- access algebra

## Config and Defaults

Primary compile-time config is:

- `OptimizerConfig`

Default presets:

- `OptimizerConfig.noOptimization()`
- `OptimizerConfig.trainingDefaults()`
- `OptimizerConfig.inferenceDefaults()`

`OptimizerFactory` converts these config objects into concrete `GraphOptimizer` instances.

`GraphOptimizer` itself is now just a rule pipeline object:

- it owns an ordered rule list
- it does not own graph state
- `OptimizerFactory` is the single standard place that maps `OptimizerStage` to concrete rule instances

For public graph compilation, `OptimizerConfig` is the intended API surface.
`GraphOptimizer` is a lower-level pipeline object used internally by optimizer/tuning tooling, not the preferred public compile contract.

Current compile-time config families inside `OptimizerConfig` are:

- `rewrite`
  - rewrite-family policy
  - currently carries `conv2d` lowering policy
- `cse`
  - common-subexpression-elimination policy
- `fuse`
  - fused compute cluster policy
- `memory`
  - MEM-stage planner policy

Example:

```java
OptimizerConfig config = new OptimizerConfig(
        List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.FUSE, OptimizerStage.MEM),
        RewriteConfig.defaults(),
        CseConfig.aggressiveDefaults(),
        FuseConfig.inferenceDefaults(),
        MemoryConfig.defaults()
);
```

Current rewrite-policy family includes:

- `Conv2dLoweringConfig`
  - `OFF`
  - `ALWAYS`
  - `HEURISTIC`

Important boundary:

- `conv2d` lowering to `CONV2D_GEMM` is a compile-time rewrite decision
- Java vs OpenBLAS inside the GEMM path remains a runtime/backend decision
- those two policies must stay separate

## Benchmark / Autotune Integration

Benchmark/autotune orchestration now lives in:

- [src/main/java/tuning/README.md](../../tuning/README.md)

Important architectural boundary:

- optimizer owns graph transformation policy
- `tuning` owns workload generation, measurement, validation, search, reporting, and persistence
- persisted execution profiles are serialized through:
  - [src/main/java/config/profile/ExecutionProfileIO.java](../../config/profile/ExecutionProfileIO.java)

That means:

- optimizer rules must be configurable through `OptimizerConfig`
- tuning must not invent a second hidden optimizer configuration model

## Adding a New Rule

1. Add a new rule class under `src/main/java/graph/optimizer/rules/`.
2. Implement `OptimizationRule`.
3. Register it in `OptimizerFactory`.
4. Reuse `OptimizerGraphSupport` if the rule rewrites graph edges or needs topological-closure rebuilding.
5. Validate:
   - numerical equivalence
   - gradient preservation
   - regressions for broadcasting, dtype handling, and fused boundaries

## Build / Runtime Notes

- Fused codegen uses ASM.
- CPU vector execution uses `jdk.incubator.vector`.
- Optimizer output is consumed by `CompiledGraph`, not directly by backend kernels.
