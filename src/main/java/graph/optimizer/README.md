# Optimizer

The optimizer is a purely graph-level layer. It transforms a topologically sorted graph before runtime preparation. It does not execute kernels and it does not decide hot-path dispatch at run time.

Its contract is simple:

- input: `List<Tensor>` in topological order
- output: a semantically equivalent `List<Tensor>` still in topological order

## Reading Guide

This document is for you if you are dealing with:

- when to add a new operation descriptor and when to add only a rewrite
- what the `AR` stage family looks like today
- which patterns are lowered into specialized primitives
- where the boundaries of CSE / FUSE / MEM are
- how to preserve forward/backward correctness

Related documentation:

- graph lifecycle: [../README.md](../README.md)
- operation descriptors: [../../operations/README.md](../../operations/README.md)
- backend families: [../../backend/README.md](../../backend/README.md)
- tuning/autotune: [../../tuning/README.md](../../tuning/README.md)

## Main Components

- orchestration
  - [GraphOptimizer.java](../../graph/optimizer/GraphOptimizer.java)
  - [OptimizationRule.java](../../graph/optimizer/OptimizationRule.java)
  - [OptimizerFactory.java](../../graph/optimizer/OptimizerFactory.java)
- shared graph rewrite support
  - [OptimizerGraphSupport.java](../../graph/optimizer/OptimizerGraphSupport.java)
- top-level stages
  - [rewrite/RewriteRule.java](../../graph/optimizer/rewrite/RewriteRule.java)
  - [rules/CommonSubexpressionEliminationRule.java](../../graph/optimizer/rules/CommonSubexpressionEliminationRule.java)
  - [rules/FuseElementWiseRule.java](../../graph/optimizer/rules/FuseElementWiseRule.java)
  - [rules/MemoryOptimizerRule.java](../../graph/optimizer/rules/MemoryOptimizerRule.java)
- fusion support
  - [fusion/FusedCostModel.java](../../graph/optimizer/fusion/FusedCostModel.java)
  - [fusion/FusedExternalInputCollector.java](../../graph/optimizer/fusion/FusedExternalInputCollector.java)
  - [fusion/FusedPrecisionResolver.java](../../graph/optimizer/fusion/FusedPrecisionResolver.java)
  - [fusion/FusedSignatureBuilder.java](../../graph/optimizer/fusion/FusedSignatureBuilder.java)

## Stage Model

The public optimizer stage order currently uses:

- `AR`
- `CSE`
- `FUSE`
- `MEM`

Mapping to implementations is centralized in [OptimizerFactory.java](../../graph/optimizer/OptimizerFactory.java).

Default preset reality:

- `OptimizerConfig.noOptimization()`
  - no stages
- `OptimizerConfig.trainingDefaults()`
  - `AR -> CSE -> MEM`
- `OptimizerConfig.inferenceDefaults()`
  - `AR -> CSE -> FUSE -> MEM`

This is important:

- the training default does not enable `FUSE` today
- the inference default does

## Core Design Rule

The optimizer must not become a "second runtime layer". What must remain a runtime decision:

- scalar/vector/parallel dispatch
- BLAS vs Java path for a concrete prepared matmul recipe
- chunk sizing
- approximation policy

What does belong to the optimizer:

- algebraic cleanup
- lowering into specialized graph primitives
- structural CSE
- fusion cluster formation
- memory planning

## Rule Contract

Every rule must preserve:

- dependency ordering
- reachability from sinks
- forward/backward phase boundaries
- dtype and shape semantics
- gradient correctness

A rule may:

- replace a node with another node
- rewrite an input edge
- rebuild topological closure from retained sinks

The main helper for this is [OptimizerGraphSupport.java](../../graph/optimizer/OptimizerGraphSupport.java).

## `AR`: Rewrite Family

`AR` is not a single small algebraic pass. It is a composite rewrite stage.

The current delegate order in [rewrite/RewriteRule.java](../../graph/optimizer/rewrite/RewriteRule.java) is:

1. optional `PiecewiseLoweringRewrite`
2. `AlgebraicRewrite`
3. `LinearLoweringRewrite`
4. `LossLoweringRewrite`
5. `ReductionLoweringRewrite`
6. `AttentionLoweringRewrite`
7. `AttentionBackwardLoweringRewrite`
8. optional `Conv2dLoweringRewrite`

That order is intentional:

- canonicalization/import cleanup runs before the other specializations
- algebraic cleanup simplifies the local graph shape first
- structural lowering into specialized primitives runs after that
- `conv2d` lowering remains explicitly policy-controlled

## `PiecewiseLoweringRewrite`

This pass is intentionally opt-in today. It primarily serves as a repair/canonicalization layer for:

- imported graphs
- manually decomposed patterns

Internal `Tensor` builders are not expected to rely on it for normal forward graphs.

It currently recognizes:

- canonical sigmoid
  - `1 / (1 + exp(-x)) -> sigmoid(x)`
- relu-like `where`
  - `where(x > 0, x, 0) -> relu(x)`
- clamp-like `where`
  - `where(x < t, t, x) -> clampMin(t)`
  - `where(x > t, t, x) -> clampMax(t)`

Config:

- [PiecewiseLoweringConfig.java](../../config/optimizer/PiecewiseLoweringConfig.java)

Default:

- everything disabled

This is important to state explicitly:

- if `Tensor.relu()` already creates a `relu` primitive, the rewrite does nothing
- its role is canonicalization/import cleanup, not normal forward construction

## `AlgebraicRewrite`

This is where local numeric simplification belongs. It is intentionally narrower than "any semantic lowering".

Typical examples:

- identity elimination
- scalar canonicalization
- local constant folding where safe
- rewrites such as `pow(x, 2) -> x * x`

What does not belong here today:

- attention pattern recognition
- softmax backward lowering
- cross-entropy lowering
- view/access rewrites

## `LinearLoweringRewrite`

Recognizes the pattern:

- `matmul(input, weight) + bias`

and replaces it with:

- `LINEAR(input, weight, bias)`

Conditions are purely shape/semantics based:

- `weight` must match a linear layer
- `bias` must be a 1D bias vector
- output shape must match the input batch prefix plus `outFeatures`

Why this matters:

- the backend receives an explicit structured primitive
- bias epilogue and packed weights can live inside one family
- runtime no longer has to rediscover the pattern

## `LossLoweringRewrite`

This is one of the most important rewrite families today because it replaces actually used loss patterns with specialized primitives.

It currently lowers:

- forward cross-entropy-from-indices patterns into `CROSS_ENTROPY_LOSS_INDICES`
- backward patterns into `CROSS_ENTROPY_LOSS_INDICES_GRAD`

The approximate forward shape it recognizes is:

- `neg(gather(logSoftmax(logits), targetIndices))`
- optionally followed by `sum()` or `mean()`

The backward shape recognizes the decomposed softmax/scatter-based gradient pattern and replaces it with a specialized gradient primitive.

This is exactly the right layer for this:

- it is a graph semantics problem
- not a backend runtime heuristic
- the backend can then expose a much cleaner specialized kernel family

## `ReductionLoweringRewrite`

This pass lowers backward patterns for structured reduction families.

Currently:

- softmax backward pattern -> `SOFTMAX_GRAD`
- log-softmax backward pattern -> `LOG_SOFTMAX_GRAD`

What gets recognized is the backward graph shape, not the forward API call.

This is important:

- the public tensor surface can still build gradients out of tensor ops
- the optimizer can later replace them with a specialized primitive

That preserves:

- a clean public API
- a fast backend

## `AttentionLoweringRewrite`

Recognizes the forward scaled dot-product attention pattern:

- `scores = q.matmul(k^T)`
- optional scaling through `mulScalar`
- optional masking through `where(mask, scores, fill)`
- `softmax(scores)`
- `softmax(scores).matmul(v)`

If the pattern matches, it rewrites it to:

- `SCALED_DOT_PRODUCT_ATTENTION`

The mask fill scalar is validated by dtype. The rewrite is not a generic "try to guess attention at any cost" pass. It is a fairly narrow and controlled pattern detector.

## `AttentionBackwardLoweringRewrite`

This pass looks into the backward section of the graph and replaces decomposed backward patterns with a specialized primitive:

- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`

It can recognize gradient paths for:

- query
- key
- value

It uses an index over forward attention nodes so that it does not pair a backward pattern with the wrong forward primitive.

That is important:

- this is not a local one-node rewrite
- it is structured backward lowering built around knowledge of the forward primitive

## `Conv2dLoweringRewrite`

`conv2d` lowering is not always enabled. It is controlled by explicit policy:

- `OFF`
- `HEURISTIC`
- `ALWAYS`

The rewrite turns:

- `CONV2D`

into:

- `CONV2D_GEMM`

when policy allows it.

Important boundary:

- compile-time rewrite decides whether the graph carries direct or GEMM-lowered conv primitives
- runtime still decides the concrete backend compute details inside the chosen family

## `CSE`: Common Subexpression Elimination

`CommonSubexpressionEliminationRule` does not use naive string comparison. It works with structural signatures.

The signature takes into account:

- `Operation.OpType`
- forward/backward phase
- input signatures
- explicit operation parameters
- safety metadata

This matters for examples such as:

- `sum(axis, keepDims=false)` vs `sum(axis, keepDims=true)`
- different `permute(...)`
- `pow` with different exponents
- scalar-parameter ops

`noop` and fused nodes intentionally remain CSE boundaries.

## `FUSE`: Elementwise Fusion

`FuseElementWiseRule` creates fused clusters only where the model makes sense:

- one output-space loop
- local per-element compute

The fused compute algebra currently includes:

- unary numeric ops
- binary numeric ops
- compare ops
- logical ops
- `where`

It does not include:

- indexing
- reductions
- matmul
- structured losses
- special gradient kernels

View/access ops are not treated as compute nodes. They can instead be absorbed as external input access metadata.

## `MEM`: Memory Planning

`MemoryOptimizerRule` is a compile-time planner, not a runtime allocator.

Its job is to:

- analyze liveness
- assign reusable slots
- reduce peak memory footprint
- return explain/summary data

Policy flows through:

- [MemoryConfig.java](../../config/optimizer/MemoryConfig.java)
- [MemoryPlannerPolicy.java](../../graph/optimizer/memory/MemoryPlannerPolicy.java)

Typical knobs:

- separate forward/backward pools
- cross-phase reuse
- larger-buffer reuse
- minimum reusable buffer size

## Example: Forward Lowering

```java
Tensor logits = x.linear(w, b);
Tensor loss = logits.crossEntropyLossIndices(targets, 1);
```

In the ideal runtime graph after `AR`, you may already have:

- `LINEAR`
- `CROSS_ENTROPY_LOSS_INDICES`

instead of the decomposed combination:

- `MATMUL`
- `ADD`
- `LOG_SOFTMAX`
- `GATHER`
- `NEG`
- `MEAN`

## Example: Backward Lowering

The public autograd builder may compose backward through regular tensor operations. After `AR`, it may be rewritten into:

- `SOFTMAX_GRAD`
- `LOG_SOFTMAX_GRAD`
- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`
- `CROSS_ENTROPY_LOSS_INDICES_GRAD`

This is a key design pattern in the project:

- forward/backward formulas can be built purely out of `Tensor` operations
- the optimizer can later replace them with structured primitives

## Config Surface

The primary public optimizer config is:

- [OptimizerConfig.java](../../config/optimizer/OptimizerConfig.java)

It contains:

- `stageOrder`
- `rewrite`
- `cse`
- `fuse`
- `memory`

An important consequence:

- tuning must not invent a second hidden optimizer config model
- anything that belongs to graph policy must be expressible through `OptimizerConfig`

## Adding A New Rewrite

Correct process:

1. decide whether a new rewrite should exist at all
   - is this really not just a new operation descriptor?
   - is it maybe a runtime/backend knob instead?
2. if it is a rewrite:
   - place it under `graph.optimizer.rewrite` if it belongs to the `AR` family
   - or under `graph.optimizer.rules` if it is a standalone top-level stage
3. implement `OptimizationRule`
4. use `OptimizerGraphSupport` for edge rewrites and closure rebuild
5. register it in `RewriteRule` or `OptimizerFactory`
6. add tests for:
   - forward correctness
   - backward correctness
   - dtype coverage
   - broadcast/layout invariants

## When Not To Add A Rewrite

Do not add a rewrite when:

- the public `Tensor` builder should create the correct primitive directly
- it is only a backend-specific dispatch decision
- it is a tuning knob rather than a graph transformation
- the pattern is benchmark-only synthetic shape with no real graph meaning

## Common Mistakes

- mixing graph policy with runtime policy
- lowering a pattern that should already be a canonical primitive in the public API
- ignoring the backward section and rewriting only the forward form
- rewriting based on node label instead of `Operation.OpType` and parameters
- relying on rewrite as a repair step for internal builder inconsistency

## Related Modules

- graph lifecycle: [../README.md](../README.md)
- operations: [../../operations/README.md](../../operations/README.md)
- backend: [../../backend/README.md](../../backend/README.md)
- tuning: [../../tuning/README.md](../../tuning/README.md)
