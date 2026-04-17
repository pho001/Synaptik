# FUSE Stage

`FUSE` is the graph-level elementwise fusion stage.

Its purpose is not to decide the final machine code shape. Its purpose is to:

- identify elementwise graph regions that are safe and worthwhile to fuse
- replace those regions with a single fused graph primitive
- leave the low-level execution strategy to the runtime/backend

## Entry Points

- main rule: [rules/FuseElementWiseRule.java](./rules/FuseElementWiseRule.java)
- config: [../../config/optimizer/FuseConfig.java](../../config/optimizer/FuseConfig.java)
- cost model: [fusion/FusedCostModel.java](./fusion/FusedCostModel.java)

Defaults:

- training defaults are conservative
- inference defaults are more permissive

Current presets:

- training
  - `maxClusterNodes = 64`
  - `scoreThreshold = 0.55`
  - `internalEdgeBonus = 0.30`
  - `externalInputPenalty = 0.20`
  - `sharedExpensivePenalty = 1.00`
  - `nonCheapBonus = 0.35`
  - `preserveSharedExpensiveNodes = true`
- inference
  - `maxClusterNodes = 96`
  - `scoreThreshold = 0.00`
  - `internalEdgeBonus = 0.50`
  - `externalInputPenalty = 0.10`
  - `sharedExpensivePenalty = 0.50`
  - `nonCheapBonus = 0.35`
  - `preserveSharedExpensiveNodes = false`

## What Counts As Fusable

The rule does not hardcode a list of all fusable operations. It delegates that to the operation descriptor:

```text
op.opType().isFusable()
```

So the current fusion frontier is defined by the operation type metadata, not by `FuseElementWiseRule` itself.

The rule also explicitly refuses to fuse:

- null ops
- nodes already marked as `FUSED`

## High-Level Algorithm

The pass has four main steps:

1. build consumer maps
2. mark materialization points
3. build candidate clusters backward from those retained roots
4. replace accepted clusters with fused operations and rebuild the graph

## Step 1: Consumer Maps

The rule builds two different consumer views:

- `allConsumersMap` / `allConsumerCounts`
  - across the whole combined graph, including forward and backward
- `samePhaseConsumersMap`
  - only consumers that live in the same phase as the producer

This split matters because fusion is phase-local, but materialization and liveness need to respect the whole graph.

## Step 2: Materialization Points

The most important concept in this stage is the materialization point.

A tensor becomes a materialization point if any of the following is true:

- it is already fused
- it has no operation
- it is not an elementwise fusion candidate
- it has no consumers in the whole graph, so it is a sink
- it has a same-phase consumer that is not elementwise
- it has a cross-phase consumer
- it is a shared expensive node and `preserveSharedExpensiveNodes` is enabled

In code terms, a "shared expensive node" means:

- total consumer count is greater than 1, and
- the op exists, and
- `op.isCheap() == false`

This is the core recomputation boundary logic.

### Why This Matters

If a tensor is not materialized, it may be swallowed into a fused cluster and recomputed inside the fused kernel.

If it is materialized, it remains as an explicit graph node and becomes a boundary for cluster growth.

## Step 3: Cluster Construction

The rule only starts cluster building from retained elementwise materialization points.

For each such root:

1. create a queue starting from the root
2. walk backward through inputs
3. swallow predecessors only if all of these hold:
   - the predecessor is not itself a materialization point
   - it has an operation
   - it is fusable
   - it is not already fused
   - it is in the same phase as the current node

This produces a backward-grown cluster rooted at the original retained output tensor.

Clusters of size `<= 1` are discarded immediately.

## Step 4: External Inputs

Once a cluster exists, the rule collects all inputs to all cluster nodes and removes the nodes that are internal to the cluster.

The remaining tensors become the fused operation's runtime inputs.

Conceptually:

```text
cluster nodes      = internal expression DAG
external inputs    = leaves entering that DAG from the outside
cluster root       = tensor that survives and becomes the fused op output
```

## Acceptance Rules

A candidate cluster is accepted only if all of the following hold:

- cluster size is at least 2
- cluster size does not exceed `maxClusterNodes`
- `FusedCostModel.rejectBroadcastHeavySmallAffinePlan(...)` does not reject the preview plan
- the final score is at least `scoreThreshold`

### Score Formula

The current score model is:

```text
benefit =
    (nodes - 1)
  + internalEdgeBonus * internalEdges
  + nonCheapBonus * nonCheapNodes

cost =
    externalInputPenalty * externalInputs.size()
  + sharedExpensivePenalty * sharedExpensive
  + estimateFusionAccessPenalty(previewPlan)

score = benefit - cost
```

Where:

- `internalEdges`
  - number of parent links that stay inside the cluster
- `nonCheapNodes`
  - number of nodes whose op reports `isCheap() == false`
- `sharedExpensive`
  - number of non-cheap cluster nodes with more than one consumer in the whole graph

## Access Penalty Model

The access penalty comes from [fusion/FusedCostModel.java](./fusion/FusedCostModel.java).

Current per-input access penalties:

- `DIRECT_CONTIGUOUS` -> `0.00`
- `OFFSET_CONTIGUOUS` -> `0.10`
- `DIRECT_STRIDED` -> `0.35`
- `OFFSET_STRIDED` -> `0.55`
- `BROADCAST_STRIDED` -> `0.75`

This is a graph-level heuristic. It does not try to be a cycle-accurate hardware model.

## Broadcast-Heavy Small-Affine Rejection

The special rejection helper `rejectBroadcastHeavySmallAffinePlan(...)` currently rejects a narrow family of plans that look like small broadcast-heavy normalization-style formulas.

Current rejection conditions:

- the fused plan has at least 2 `BROADCAST_STRIDED` inputs
- every node is in this limited set:
  - `ADD`
  - `SUB`
  - `MUL`
  - `DIV`
  - `SQRT`
  - `RELU`
  - `CLAMP_MIN`
  - `CLAMP_MAX`
  - `ABS`
  - `NOOP`
- at least one node is `DIV` or `SQRT`

This heuristic exists to avoid fusing certain small affine-normalization style expressions that are structurally legal but not profitable in the current backend setup.

## Cheap vs Non-Cheap Families

The cost model also classifies fused plans into dispatch families.

Current dispatch families:

- `CHEAP_CONTIGUOUS`
- `CHEAP_STRIDED`
- `NON_CHEAP_CONTIGUOUS`
- `NON_CHEAP_STRIDED`

A plan counts as cheap numeric only if:

- all external inputs are numeric, not `BOOL`
- all fused outputs are numeric, not `BOOL`
- every fused node is in this set:
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
  - `NOOP`

Ops such as these are currently treated as non-cheap:

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
- comparisons
- logical ops
- `WHERE`

## What Happens On Success

If a cluster is accepted:

1. the root tensor is mutated in place
2. `FusedOperationFactory.create(...)` builds a fused operation and runtime input list
3. the root tensor keeps its identity, but now points at the fused op
4. swallowed internal tensors are dropped from the outer execution list

That in-place mutation is important. External references to the root tensor do not need to be rewritten to a different object.

## What The Final Graph Keeps

After fusion, the final retained graph keeps:

- all materialization points
- fused roots
- anything from rejected clusters

Then `OptimizerGraphSupport.rebuildTopologicalClosure(...)` reconstructs the final execution list from those retained nodes.

## Examples

### Example 1: simple elementwise chain

```text
t1 = add(x, y)
t2 = relu(t1)
t3 = mulScalar(t2, 0.5)
```

If all three nodes are same-phase and no boundary condition applies, the rule can fuse them into one fused op rooted at `t3`.

### Example 2: non-elementwise consumer creates a boundary

```text
t1 = add(x, y)
t2 = relu(t1)
t3 = matmul(t2, w)
```

`t2` has a same-phase non-elementwise consumer (`matmul`), so it becomes a materialization point and the elementwise chain cannot be swallowed across that boundary.

### Example 3: cross-phase boundary

If a forward tensor is consumed by a backward tensor, that forward tensor becomes a materialization point even if the local formula is elementwise.

## Important Limits

- `FUSE` does not decide vector width or parallelism.
- `FUSE` does not cross forward/backward boundaries.
- `FUSE` does not fuse arbitrary non-elementwise regions.
- the current profitability model is heuristic, not hardware-calibrated

## Practical Meaning

Use this stage when you want the runtime to see fewer, larger elementwise expressions without encoding those fused patterns manually in the public API.

If a graph looks like a long chain of fusable elementwise ops, `FUSE` is the stage that decides whether that chain becomes one fused node or stays decomposed.
