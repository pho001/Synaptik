# Memory Planning / MEM

`MEM` is historical shorthand for the compile-time memory planning and reuse phase. It is outside the backend-neutral `GraphOptimizer`.

It runs after structural graph shape is already finalized.
Its job is to reduce allocation pressure by:

- aliasing view-like tensors to storage owners
- reusing temporary buffers where lifetimes do not overlap
- keeping required forward/backward values alive when needed

## Entry Points

- planner:
  - [../compile/planning/memory/MemoryPlanner.java](../compile/planning/memory/MemoryPlanner.java)
- input:
  - [../compile/planning/memory/MemoryPlanningInput.java](../compile/planning/memory/MemoryPlanningInput.java)
- plan model:
  - [../compile/planning/memory/MemoryPlan.java](../compile/planning/memory/MemoryPlan.java)
  - [../compile/planning/memory/MemoryPlanSummary.java](../compile/planning/memory/MemoryPlanSummary.java)
- policy:
  - [../compile/planning/memory/MemoryPlannerPolicy.java](../compile/planning/memory/MemoryPlannerPolicy.java)

## High-Level Split

The implementation is split into:

1. planning
   - pure analysis in `MemoryPlanner.plan(...)`
2. compile artifact publication
   - `CompileSession` stores the finalized `MemoryPlan` in `CompileArtifacts`

That split is useful because runtime binding can consume compile-time memory artifacts instead of rebuilding memory policy.

## Step 1: Find The Forward Boundary

The planner needs to distinguish forward and backward lifetime regions.

Current rule:

- scan the graph from the end
- find the last `NOOP` labeled `Tensor.SYSTEM_FORWARD_OUTPUT_LABEL`
- use its index as the forward boundary
- if none is found, treat the whole graph as one phase

This matters because some forward values must stay alive across the forward/backward boundary.

## Step 2: Resolve Storage Owners

Not every tensor owns its own storage.
Some tensors are views or aliases.

Current alias-at-runtime rules:

- always alias input 0 for:
  - `NOOP`
  - `EXPAND`
  - `SELECT`
  - `PERMUTE`
  - `EXPAND_DIMS`
  - `SQUEEZE`
- `RESHAPE` aliases input 0 only if the input is contiguous

Everything else owns storage by default.

So the memory planner works in terms of storage owners, not merely tensor identities.

## Step 3: Track Consumers And Last Reads

For each storage owner the planner computes:

- consumer count
- last read index
- whether the owner is needed again from backward

If a storage owner has no consumers, its `lastReadIndex` is treated as `Integer.MAX_VALUE`.
That intentionally keeps true outputs alive to the end.

## Step 4: Assign Memory Roles

Each tensor receives one role.

Current roles:

- `LEAF`
- `FORWARD_TEMP`
- `SAVED_FORWARD`
- `GRADIENT_TARGET`
- `BACKWARD_TEMP`
- `VIEW_ALIAS`

Interpretation:

- `LEAF`
  - original input/parameter style tensors
- `FORWARD_TEMP`
  - forward-only temporary
- `SAVED_FORWARD`
  - forward value that must survive into backward
- `GRADIENT_TARGET`
  - tensor used as a gradient publication target
- `BACKWARD_TEMP`
  - backward-only temporary
- `VIEW_ALIAS`
  - does not own runtime storage

## Step 5: Build Reusable Intervals

Only some owners become reusable intervals.

Current eligibility:

- tensor is its own storage owner
- role is one of:
  - `FORWARD_TEMP`
  - `BACKWARD_TEMP`
  - `SAVED_FORWARD`
- flat size is at least `policy.minReusableBufferSize()`

This is conservative by design.

## Step 6: Assign Slots

Reusable intervals are sorted by:

1. `birthIndex`
2. `lastReadIndex`

Then slot assignment proceeds greedily:

1. release expired active slots
2. choose the smallest compatible free slot
3. if none exists, allocate a new slot id
4. assign the chosen slot to the interval owner

### Compatibility rules

A free slot is compatible only if:

- dtype matches
- forward/backward pool policy allows phase sharing
- size policy allows equal-size or larger-buffer reuse

Intervals with role `SAVED_FORWARD` are treated as `"shared"` for phase compatibility.

## Step 7: Apply The Plan At Runtime

`RuntimeMemoryBinder` consumes the plan during prepare/runtime binding.

### View aliases

If a tensor is `VIEW_ALIAS`, runtime storage is aliased from the resolved storage owner.

### Slotted owners

If an owner has a slot:

- the slot buffer is allocated lazily
- the whole slot buffer is zero-filled
- the tensor is assigned that reused buffer

### Non-slotted owners

If no slot is assigned:

- the tensor receives a fresh dense buffer of its flat size

## Why Reused Buffers Are Zero-Filled

The current implementation explicitly zero-fills reused slot buffers before assignment.

This is conservative but safe:

- it avoids stale-data bleed between logically unrelated tensors
- it keeps semantics straightforward during debugging

It is not necessarily the globally cheapest possible policy, but it is the current correct one.

## Summary Metrics

`MemoryPlanSummary` exposes metrics such as:

- interval count
- slot count
- reuse count
- reuse hit rate
- allocated slot bytes
- peak total live bytes
- peak reusable bytes
- peak saved-forward bytes
- peak gradient-target bytes
- peak forward live bytes
- peak backward live bytes
- saved-forward count
- gradient-target count
- average saved-forward hold distance

Those metrics are the right place to look when evaluating a memory-reuse change.

## Worked Example

Suppose a forward graph has these simplified temporaries:

```text
t1 = a.add(b)
t2 = t1.relu()
t3 = t2.mul(c)
```

If:

- `t1` is dead after `t2`
- `t2` is dead after `t3`
- sizes and dtype match

then `MEM` may place `t1` and `t2` into the same reusable slot because their live intervals do not overlap.

Conceptually:

- slot 0:
  - first holds `t1`
  - later reuses the same buffer for `t2`

That reduces allocation count without changing graph semantics.

## What MEM Does Not Do

`MEM` does not:

- change graph formulas
- decide graph optimization ordering
- decide prepared-execution vector widths
- choose BLAS thresholds

It only plans and applies storage reuse over the already optimized graph.
