# Optimizer Package

The optimizer is the graph-level transformation layer between graph construction and runtime preparation.

Input:

- a topologically ordered compile-time graph snapshot

Output:

- a semantically equivalent topologically ordered graph snapshot

The optimizer does not execute kernels.
It also does not decide runtime dispatch knobs such as vector width, worker count, BLAS thresholds, or approximation mode.

## Public Stage Model

The user-facing stage names are currently:

1. `AR`
2. `CSE`
3. `FUSE`
4. `MEM`

`OptimizerFactory` maps them to:

- `AR` -> rewrite family
- `CSE` -> common subexpression elimination
- `FUSE` -> graph-level elementwise fusion
- `MEM` -> memory planning and reuse

Default presets:

- `OptimizerConfig.noOptimization()`
  - no stages
- `OptimizerConfig.trainingDefaults()`
  - `AR -> CSE -> MEM`
- `OptimizerConfig.inferenceDefaults()`
  - `AR -> CSE -> FUSE -> MEM`

So today:

- training defaults do not enable graph fusion
- inference defaults do enable graph fusion

## What The Optimizer Owns

The optimizer owns graph structure.

Typical examples:

- replace `x + 0` with `x`
- lower `matmul + bias` into `LINEAR`
- replace `softmax` gradient patterns with `SOFTMAX_GRAD`
- fuse a chain like `relu(add(mul(x, y), b))` into one `FUSED` node
- plan memory reuse after final graph shape is known

The optimizer does not own:

- vector widths
- worker counts
- matmul tile thresholds
- BLAS provider thread behavior
- approximation policy

Those belong to runtime config, backend planners, and the tuning layer.

## Current Execution Model

`GraphOptimizer` now supports an iterative fixpoint prefix followed by terminal stages.

Current factory behavior:

- iterative prefix:
  - everything except `MEM`
- terminal suffix:
  - `MEM`

That means a stage order such as:

```text
AR -> CSE -> FUSE -> MEM
```

is executed as:

1. run `AR -> CSE -> FUSE`
2. fingerprint the graph
3. if the graph changed, run `AR -> CSE -> FUSE` again
4. stop at structural fixpoint or the configured round cap
5. run `MEM` exactly once

This is important because the system no longer relies on repeated semantic recompile as an accidental extra optimization strategy.
The fixpoint behavior is explicit, bounded, and compile-local.

## Snapshot Boundary

Optimization runs on a compile-time graph snapshot rather than directly on the live semantic graph.

That gives the project two benefits at once:

- the optimizer still sees the whole joint forward/backward graph
- the semantic graph does not accumulate optimizer-mutated state across repeated compiles

This is one of the most important architectural boundaries in the current compiler design.

## Shared Rewrite Mechanics

Several optimizer rules rely on the same support utilities in [OptimizerGraphSupport.java](./OptimizerGraphSupport.java).

Important helpers:

- `rewriteInputs(...)`
  - rewires already replaced inputs before processing the current node
- `resolveReplacement(...)`
  - follows replacement chains to the final node
- `observableRoots(...)`
  - finds graph roots that should stay observable
- `rebuildTopologicalClosureFromRoots(...)`
  - rebuilds a clean, reachable topological list after a pass changed the graph

Those helpers are what make local rewrites safe in a larger mutable graph snapshot.

## Stage Responsibilities At A Glance

### `AR`

Composite rewrite/lowering stage.

Owns:

- algebraic simplification
- piecewise canonicalization
- structural lowerings into primitives such as:
  - `LINEAR`
  - `SOFTMAX_GRAD`
  - `LOG_SOFTMAX_GRAD`
  - `CROSS_ENTROPY_LOSS_INDICES`
  - `CROSS_ENTROPY_LOSS_INDICES_GRAD`
  - `SCALED_DOT_PRODUCT_ATTENTION`
  - `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`
  - optional conv2d GEMM lowerings

See [AR.md](./AR.md).

### `CSE`

Structural common subexpression elimination.

Owns:

- detect structurally identical tensors
- keep one representative
- redirect the others

See [CSE.md](./CSE.md).

### `FUSE`

Graph-level elementwise fusion.

Owns:

- choose safe and worthwhile elementwise clusters
- replace them with one `FUSED` primitive

See [FUSE.md](./FUSE.md).

### `MEM`

Memory planning and buffer reuse.

Owns:

- view aliasing at runtime
- temporary slot reuse
- reusable interval planning

See [MEM.md](./MEM.md).

## Example: Why Fixpoint Matters

Consider this schematic graph:

```text
z = exp(log(x)).add(0)
```

One `AR` round can simplify:

- `exp(log(x)) -> x`
- `x + 0 -> x`

But in more complex graphs, a first rewrite may expose new CSE or fusion opportunities only after the first round completes.
The fixpoint prefix makes those opportunities deterministic without requiring repeated public `compile()` calls.

## What To Change When

Use this rule of thumb:

- new algebraic identity or pattern lowering:
  - `AR`
- duplicate elimination:
  - `CSE`
- elementwise cluster profitability:
  - `FUSE`
- allocation/reuse policy:
  - `MEM`

If a change depends on runtime sizes, thresholds, or hardware policy, it probably belongs outside the optimizer.
