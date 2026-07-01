# Optimizer Package

The optimizer is the graph-level transformation layer between graph construction and runtime preparation.

Input:

- a topologically ordered compile-time graph snapshot

Output:

- a semantically equivalent topologically ordered graph snapshot

The optimizer does not execute kernels.
It also does not decide prepared-execution dispatch knobs such as vector width, worker count, BLAS thresholds, or approximation mode.

## Graph Optimization Model

`OptimizerFactory` now owns only backend-neutral graph optimization:

- algebraic/light canonical rewrites
- constant folding
- common subexpression elimination
- dead-code elimination
- optional backend-neutral lowering

Backend ownership planning, partition optimization, and memory planning are compile-flow responsibilities owned by
`CompileConfig`, `BackendPlanningConfig`, `PartitionExecutionConfig`, and `MemoryPlanningConfig`.

The contiguous simplification block `AR -> CF -> CSE -> DCE` is executed by `SimplificationFixpointRule`, not as four one-shot
passes. The loop stops when the graph fingerprint is stable, when the max iteration count is reached, or when the next
iteration does not improve the structural graph score.

## What The Optimizer Owns

The optimizer owns graph structure.

Typical examples:

- replace `x + 0` with `x`
- fold small constant-only expressions
- remove nodes unreachable from forward output or gradient publication roots
- lower backend-neutral compound operations when optional lowering is enabled

The optimizer does not own:

- backend ownership/partition planning
- partition fusion/execution-unit planning
- memory reuse planning
- vector widths
- worker counts
- matmul tile thresholds
- BLAS provider thread behavior
- approximation policy

Those belong to compile backend planning, partition optimization, memory planning, runtime profiles, backend preparers, and the tuning layer.

## Current Execution Model

`GraphOptimizer` is an ordered pipeline, but simplification is a nested fixpoint stage.

The graph optimizer pipeline is:

```text
SIMPLIFICATION_FIXPOINT(AR -> CF -> CSE -> DCE) -> optional LOWER
```

is executed as:

1. repeat `AR`, `CF`, `CSE`, `DCE` until stable/improvement stops/max iterations
2. run `LOWER`

Heavy executable/decomposition lowering is not part of `AR`. Backend-neutral operation lowering runs in `LOWER`;
backend-specific executable lowering still happens later, where the target backend and partition contract are known.

Backend planning, partition optimization, and memory planning run later in the compile flow. They are not `GraphOptimizer`
rules.

## Snapshot Boundary

Optimization runs on a compile-time graph snapshot rather than directly on the live semantic graph.

That gives the project two benefits at once:

- the optimizer still sees the whole joint forward/backward graph
- the semantic graph does not accumulate optimizer-mutated state across repeated compiles

This is one of the most important architectural boundaries in the current compiler design.

## Shared Rewrite Mechanics

Several optimizer rules operate through the mutable optimizer working graph model in [OptimizerGraph.java](./OptimizerGraph.java).

Important operations:

- `rewriteInputs(...)`
  - rewires already replaced inputs before processing the current node
- `resolveReplacement(...)`
  - follows replacement chains to the final node
- `observableRoots(...)`
  - finds graph roots that should stay observable
- `rebuildTopologicalClosureFromRoots(...)`
  - rebuilds a clean, reachable topological list after a pass changed the graph

Those operations make local rewrites safe in a larger mutable graph snapshot.

## Stage Responsibilities At A Glance

### `AR`

Algebraic and light canonical rewrite stage.

Owns:

- algebraic simplification
- piecewise canonicalization
- cheap local canonical forms that do not need backend ownership context

Does not own:

- `matmul + bias -> LINEAR`
- loss/attention/reduction decomposition
- conv2d DAG lowering
- backend-specific executable primitive selection

See [AR.md](./AR.md).

### `CF`

Deterministic constant folding for small pure constant subgraphs.

Owns:

- scalar known-constant elementwise evaluation
- safe boolean predicate folding
- preserving dtype semantics while replacing folded operation nodes with leaf constants

### `CSE`

Structural common subexpression elimination.

Owns:

- detect structurally identical tensors
- keep one representative
- redirect the others

See [CSE.md](./CSE.md).

### `DCE`

Dead-code elimination.

Owns:

- keeping forward output reachable
- keeping gradient publication roots reachable
- removing disconnected compile-only intermediates

### `LOWER`

Backend-neutral operation lowering.

Owns:

- `matmul + bias -> LINEAR`
- loss/reduction/attention specialized operation surfaces
- optional conv2d lowering to canonical window/matmul primitives according to rewrite config and its `Conv2dDagLoweringProfile` thresholds

Does not own:

- selecting CUDA/Metal/CPU executable primitives
- partition-internal backend fusion
- device buffer/layout planning

### Partition optimization

Compile phase outside `GraphOptimizer`.

Owns:

- choose safe and worthwhile execution units inside already owned partitions
- group compatible elementwise chains
- preserve unit-kernel boundaries for reductions, matmul, and other barriers

See [FUSE.md](./FUSE.md).

### Memory planning

Compile phase outside `GraphOptimizer`.

Owns:

- view aliasing at runtime
- temporary slot reuse
- reusable interval planning
- partition handoff planning

See [MEM.md](./MEM.md).

## Example: Stage Interaction

Consider this schematic graph:

```text
z = exp(log(x)).add(0)
```

`AR` can simplify:

- `exp(log(x)) -> x`
- `x + 0 -> x`

After that, later stages can act on the already simplified graph:

- `CF` can fold constant-only subgraphs exposed by `AR`
- `CSE` can collapse duplicates that remain after rewriting
- `DCE` can remove nodes made unreachable by replacements
- `LOWER` can create backend-neutral specialized operation surfaces
- backend planning can create CPU or accelerator ownership partitions
- partition optimization can group surviving elementwise chains inside owned partitions
- memory planning can plan reuse on the final graph shape

The simplification stages are replayed by `SimplificationFixpointRule` until stable or no longer improving. `LOWER` runs after simplification
when enabled. Backend planning, partition optimization, and memory planning are later compile phases.

## What To Change When

Use this rule of thumb:

- new algebraic identity or light canonical pattern:
  - `AR`
- deterministic small constant evaluation:
  - `CF`
- duplicate elimination:
  - `CSE`
- unreachable graph simplification:
  - `DCE`
- backend-neutral op surface lowering:
  - `LOWER`
- backend ownership policy:
  - `BackendPlanningConfig` and `planning.backend.BackendPlanningService`
- elementwise cluster profitability inside partitions:
  - `PartitionExecutionConfig` and `planning.partition.execution`
- allocation/reuse policy:
  - `MemoryPlanningConfig` and `planning.memory`

If a change depends on runtime sizes, thresholds, or hardware policy, it probably belongs outside the optimizer.
