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
