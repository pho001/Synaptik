# Optimizer

The optimizer is the graph-level transformation layer that runs after the graph has already been built and topologically sorted, but before runtime preparation and kernel dispatch.

Its contract is intentionally narrow:

- input: `List<Tensor>` in topological order
- output: a semantically equivalent `List<Tensor>` still in topological order

The optimizer does not execute kernels. It also does not decide runtime dispatch details such as vector width, parallel chunk sizing, BLAS routing, or approximation mode. Those remain runtime concerns.

## Reading Guide

This package is the right place to look when you need to answer questions such as:

- when to add a new primitive operation vs when to add only a rewrite
- which graph patterns are currently recognized and lowered
- what `AR`, `CSE`, `FUSE`, and `MEM` really do today
- how forward/backward boundaries are preserved
- how graph-level fusion differs from runtime backend dispatch

Related documentation:

- graph lifecycle: [../README.md](../README.md)
- operation descriptors: [../../operations/README.md](../../operations/README.md)
- backend families: [../../backend/README.md](../../backend/README.md)
- tuning and autotune: [../../tuning/README.md](../../tuning/README.md)

## Main Components

- orchestration
  - [GraphOptimizer.java](./GraphOptimizer.java)
  - [OptimizationRule.java](./OptimizationRule.java)
  - [OptimizerFactory.java](./OptimizerFactory.java)
  - [OptimizerProfiles.java](./OptimizerProfiles.java)
- shared graph rewrite support
  - [OptimizerGraphSupport.java](./OptimizerGraphSupport.java)
- public stage implementations
  - [rewrite/RewriteRule.java](./rewrite/RewriteRule.java)
  - [rules/CommonSubexpressionEliminationRule.java](./rules/CommonSubexpressionEliminationRule.java)
  - [rules/FuseElementWiseRule.java](./rules/FuseElementWiseRule.java)
  - [rules/MemoryOptimizerRule.java](./rules/MemoryOptimizerRule.java)

## Public Stage Model

The public optimizer stage order currently uses four stage names:

1. `AR`
2. `CSE`
3. `FUSE`
4. `MEM`

`OptimizerFactory` maps them as follows:

- `AR` -> [rewrite/RewriteRule.java](./rewrite/RewriteRule.java)
- `CSE` -> [rules/CommonSubexpressionEliminationRule.java](./rules/CommonSubexpressionEliminationRule.java)
- `FUSE` -> [rules/FuseElementWiseRule.java](./rules/FuseElementWiseRule.java)
- `MEM` -> [rules/MemoryOptimizerRule.java](./rules/MemoryOptimizerRule.java)

Default presets:

- `OptimizerConfig.noOptimization()`
  - no stages
- `OptimizerConfig.trainingDefaults()`
  - `AR -> CSE -> MEM`
- `OptimizerConfig.inferenceDefaults()`
  - `AR -> CSE -> FUSE -> MEM`

This matters because:

- training defaults currently do not enable graph fusion
- inference defaults do enable graph fusion
- `CSE` also uses different safety defaults between training and inference

## Rule Execution Model

At the top level, [GraphOptimizer.java](./GraphOptimizer.java) is intentionally simple:

1. take a sorted graph
2. apply each configured rule in order
3. require that each rule returns a non-null graph

The interesting mechanics live inside the individual rules and in [OptimizerGraphSupport.java](./OptimizerGraphSupport.java):

- `rewriteInputs(...)`
  - rewires already-replaced inputs before the current tensor is processed
- `resolveReplacement(...)`
  - follows chained replacements until the final tensor is found
- `consumerFreeSinks(...)`
  - finds current sinks in the graph
- `rebuildTopologicalClosure(...)`
  - rebuilds a clean execution list after a rule removed or replaced nodes

The abstract rewrite family base class [rewrite/AbstractRewriteRule.java](./rewrite/AbstractRewriteRule.java) uses the same general pattern:

1. remember the original sinks
2. walk tensors in topological order
3. rewrite current inputs through the replacement map
4. optionally replace the current tensor
5. propagate backward flags and gradient references
6. rebuild the closure from the resolved sinks

That design is important because optimizer rules are allowed to mutate graph structure, but the final graph still has to remain reachable, topologically valid, and semantically equivalent.

## Stage Documentation

Detailed stage documentation lives in separate files:

- [AR.md](./AR.md)
  - composite rewrite family: algebraic cleanup, canonicalization, and lowering into structured primitives
- [CSE.md](./CSE.md)
  - structural common subexpression elimination
- [FUSE.md](./FUSE.md)
  - graph-level elementwise cluster formation and fused op creation
- [MEM.md](./MEM.md)
  - memory lifetime analysis, slot assignment, and buffer reuse

## Design Boundary

The optimizer should not become a hidden second runtime. The current design boundary is:

What belongs in the optimizer:

- algebraic cleanup
- pattern canonicalization
- lowering from decomposed graphs into structured primitives
- structural deduplication
- fusion cluster formation
- memory planning

What does not belong in the optimizer:

- scalar vs vector dispatch
- Java vs BLAS selection for a prepared recipe
- thread count decisions
- vector width decisions
- approximation mode selection
- hot-path runtime tuning

## When To Add A Rewrite

Add a rewrite when all of the following are true:

- the pattern is recognizable structurally in the graph
- the replacement is semantically cleaner than the decomposed form
- backend specialization benefits from seeing the structured primitive directly
- the transformation is not a runtime hardware decision

Typical examples:

- `matmul + bias -> linear`
- decomposed cross-entropy-from-indices -> `crossEntropyLossIndices`
- decomposed softmax backward -> `softmaxGrad`
- decomposed attention forward/backward -> attention primitives

## When To Add A New Primitive

Add a new operation descriptor when the replacement represents a meaningful semantic unit that backends should be able to target directly.

That is usually the case when at least one of these is true:

- the backend can provide a materially better specialized implementation
- the primitive has stable semantics that appear repeatedly in real models
- keeping the computation decomposed would force runtime rediscovery of the same structure again and again

## Correctness Requirements

Every optimizer rule must preserve:

- reachability from the real sinks
- topological validity
- dtype semantics
- shape semantics
- backward/gradient correctness
- forward/backward phase boundaries

Whenever a rewrite seems attractive but weakens those guarantees, it does not belong here in its current form.

## End-To-End Example

Consider this decomposed forward fragment:

```text
t1 = matmul(x, w)
t2 = add(t1, b)
t3 = add(t2, zeros_like(t2))
t4 = add(t2, zeros_like(t2))
t5 = relu(t4)
t6 = mulScalar(t5, 0.5)
```

With a typical inference-style order:

```text
AR -> CSE -> FUSE -> MEM
```

the stages conceptually do this:

1. `AR`
   - `add(matmul(x, w), b)` becomes `linear(x, w, b)`
   - trivial `+ zeros_like(...)` noise collapses
2. `CSE`
   - repeated structurally identical subexpressions are shared
3. `FUSE`
   - the remaining elementwise tail may become one fused node
4. `MEM`
   - short-lived temporaries may reuse storage slots

That is why stage order matters.
Each stage is intentionally small, but the overall graph shape changes step by step.

## Typical Debugging Questions

### "Why was this decomposed graph not lowered?"

Inspect `AR`.
Common causes:

- the pattern is not in the canonical form the matcher expects
- shape/rank/axis preconditions are not satisfied
- the relevant rewrite family is disabled

### "Why do I still have duplicated work?"

Inspect `CSE`.
Common causes:

- the expressions are only mathematically equivalent, not structurally identical
- leaf identity intentionally differs
- strict safety keeps them separate

### "Why did this chain not become one fused node?"

Inspect `FUSE`.
Common causes:

- a non-elementwise consumer introduced a boundary
- there is a cross-phase edge
- the cluster score did not reach threshold
- a shared expensive node was intentionally preserved

### "Why is this still allocating?"

Inspect `MEM`.
Common causes:

- dtype or graph shape is outside reuse support
- the tensor is a true output, saved-forward value, or gradient target
- lifetime overlap prevents slot reuse

## Practical Inspection Order

When debugging optimizer behavior, the fastest order is usually:

1. confirm the stage order on the `ExecutionProfile`
2. inspect `AR`
3. inspect `CSE`
4. inspect `FUSE`
5. inspect `MEM`

That order matters because memory planning cannot rescue a graph that was never canonicalized, deduplicated, or fused the way you expected.
