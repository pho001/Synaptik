# Region Optimization / FUSE

`FUSE` is historical shorthand for region optimization. In the current architecture it is a compile-flow phase outside the backend-neutral `GraphOptimizer`.

Its responsibility is:

- choose safe and worthwhile elementwise subgraphs
- publish optimized region units for backend lowering
- leave fused operation descriptors, code generation, and machine-level execution strategy to backend preparation and runtime

It is not a code generator by itself.

## Entry Points

- optimizer:
  - [../compile/planning/region/DefaultRegionOptimizer.java](../compile/planning/region/DefaultRegionOptimizer.java)
- config:
  - [../../config/optimizer/FuseConfig.java](../../config/optimizer/FuseConfig.java)
- fused dispatch planning:
  - [../../backend/cpu/fused/plan/FusedDispatchPlanner.java](../../backend/cpu/fused/plan/FusedDispatchPlanner.java)

## What Counts As Fusable

The CPU elementwise fusion frontier is driven by operation metadata:

```text
op.isFusable()
```

So the stage does not maintain its own separate hardcoded list of all allowed ops.

The rule also refuses to fuse:

- nodes with `null` operation
- nodes already marked as `FUSED`

## High-Level Algorithm

The current implementation consumes backend planning `Partition` artifacts and follows three steps:

1. snapshot the current graph as immutable compiled nodes
2. optimize each partition into one or more execution units
3. publish `OptimizedRegion` artifacts for lowering and prepare

The optimizer does not mutate `Tensor.operation` or rewrite graph inputs directly. Compile session calls it only for accepted partitions.

## Consumer Maps

Region optimization builds two consumer views:

- all-consumer view
  - used for liveness and materialization logic across the whole combined graph
- same-phase consumer view
  - used for local fusion decisions inside one phase

This matters because a node may look locally fuseable in forward, but still need materialization because backward also consumes it.

## Materialization Points

A tensor becomes a materialization point if any of the following is true:

- it has no operation
- it is not an elementwise fusion candidate
- it has no consumers in the whole graph, so it is a sink
- it has a same-phase consumer that is not elementwise
- it has a cross-phase consumer
- it is already fused
- it is a shared expensive node and config says to preserve such nodes

Shared expensive currently means:

- total consumer count > 1
- op exists
- concrete operation metadata marks the operation as non-cheap, currently through the legacy optimizer
  compatibility path

Materialization points are the places where the fused cluster growth stops.

## Cluster Construction

Fusion starts only from retained elementwise materialization points.

For one such root, the rule walks backward and swallows predecessors only if:

- predecessor is not another materialization point
- predecessor has an operation
- predecessor is fusable
- predecessor is not already fused
- predecessor lives in the same phase

Clusters with size `<= 1` are immediately discarded.

## External Inputs

After a cluster is built:

- all parent inputs of all cluster nodes are collected
- cluster-internal nodes are removed from that input set
- the remaining tensors become runtime inputs of the fused primitive

So the cluster becomes:

- one fused expression DAG
- one fused output root
- a list of external runtime inputs

## Acceptance Rules

A cluster is accepted only if:

- size is at least 2
- size does not exceed `maxClusterNodes`
- it is not rejected by the special broadcast-heavy affine heuristic
- its final fusion score is at least `scoreThreshold`

## Current Score Model

The score is built from:

- benefit:
  - more nodes
  - more internal edges
  - bonus for non-cheap nodes
- cost:
  - more external inputs
  - shared expensive nodes
  - access penalty estimated from fused input access kinds

In simplified form:

```text
score =
  (nodes - 1)
  + internalEdgeBonus * internalEdges
  + nonCheapBonus * nonCheapNodes
  - externalInputPenalty * externalInputs
  - sharedExpensivePenalty * sharedExpensive
  - accessPenalty
```

## Cheap vs Non-Cheap Dispatch Families

The cost model classifies fused plans into dispatch families used later by runtime/backends.

Current public families:

- `CHEAP_CONTIGUOUS`
- `CHEAP_STRIDED`
- `NON_CHEAP_CONTIGUOUS`
- `NON_CHEAP_STRIDED`

A plan is considered cheap only if:

- all external inputs are numeric
- outputs are numeric
- all fused ops are from the cheap numeric set

Cheap numeric examples today include:

- `ADD`
- `SUB`
- `MUL`
- `MIN`
- `MAX`
- `NEG`
- `MUL_SCALAR`
- `RELU`
- `CLAMP_MIN`
- `CLAMP_MAX`
- `ABS`

Non-cheap examples include:

- `DIV`
- `INV`
- `SQRT`
- `EXP`
- `FAST_EXP`
- `LOG`
- `TANH`
- `FAST_TANH`
- `SIGMOID`
- `POW`
- compare ops
- logical ops
- `WHERE`

## Broadcast-Heavy Rejection

The special rejection helper exists because some small affine-style broadcast-heavy expressions are structurally legal to fuse but not profitable in the current backend design.

The rejection currently targets plans that:

- have at least two `BROADCAST_STRIDED` inputs
- are composed only from a narrow affine-ish op subset
- contain at least one `DIV` or `SQRT`

## Worked Example

Consider:

```text
y = relu(add(mul(x, w), b))
```

with:

- `x` shape `[4, 8]`
- `w` shape `[4, 8]`
- `b` shape `[8]`

Possible fused cluster:

```text
mul(x, w) -> add(..., b) -> relu(...)
```

External inputs:

- `x`
- `w`
- `b`

Cluster output:

- `y`

If the score passes threshold and no materialization boundary blocks growth, this becomes a fused elementwise execution unit in an optimized region.

## What Happens On Success

On success:

1. the original graph remains the semantic source of truth
2. the optimized region records a fused elementwise execution unit
3. lowering records the unit's ordered node ids and external input ids
4. CPU prepare builds backend-owned fused plan descriptors from the lowered unit

The cluster internals remain visible in compile artifacts. Prepared execution can run them as one backend fused step without relying on graph-level `FUSED` mutation.

## What FUSE Does Not Decide

`FUSE` does not decide:

- which ASM vector width to use
- whether the fused node runs scalar or vector in the prepared execution plan
- how many workers to use
- exact machine code shape

Those belong to backend preparation driven by the selected runtime profile.
In the current architecture they are fixed during `prepare(...)`, not rediscovered inside `execute(...)`.

So the correct mental model is:

- `FUSE` decides optimized region shape
- `prepare(...)` plus backend planning decide execution shape
- `execute(...)` runs the prepared recipe
