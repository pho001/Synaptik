# FUSE Stage

`FUSE` is the graph-level elementwise fusion stage.

Its responsibility is:

- choose safe and worthwhile elementwise subgraphs
- replace them with one `FUSED` primitive
- leave the final machine-level execution strategy to backend preparation and runtime

It is not a code generator by itself.

## Entry Points

- rule:
  - [rules/FuseElementWiseRule.java](./rules/FuseElementWiseRule.java)
- config:
  - [../../config/optimizer/FuseConfig.java](../../config/optimizer/FuseConfig.java)
- cost model:
  - [fusion/FusedCostModel.java](./fusion/FusedCostModel.java)

## What Counts As Fusable

The fusion frontier is driven by operation metadata:

```text
op.opType().isFusable()
```

So the stage does not maintain its own separate hardcoded list of all allowed ops.

The rule also refuses to fuse:

- nodes with `null` operation
- nodes already marked as `FUSED`

## High-Level Algorithm

The current implementation follows four steps:

1. build consumer maps
2. determine materialization points
3. grow candidate clusters backward from retained roots
4. score and accept/reject those clusters

## Consumer Maps

The pass builds two consumer views:

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
- `op.isCheap() == false`

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

If the score passes threshold and no materialization boundary blocks growth, this becomes one `FUSED` node.

## What Happens On Success

On success:

1. the original cluster root tensor is kept
2. its operation is replaced with a fused operation descriptor
3. its runtime inputs become the cluster external inputs
4. internal swallowed nodes disappear from the main execution graph

The cluster internals still exist inside the fused operation plan, but they are no longer separate main-graph execution nodes.

## What FUSE Does Not Decide

`FUSE` does not decide:

- which ASM vector width to use
- whether the fused node runs scalar or vector at runtime
- how many workers to use
- exact machine code shape

Those belong to backend preparation and runtime tuning.

So the correct mental model is:

- `FUSE` decides graph shape
- runtime/backend decide execution shape
