# MEM Stage

`MEM` is the graph-level memory planning and buffer reuse stage.

Its purpose is to analyze tensor lifetimes in the already optimized graph and reduce allocation pressure by:

- aliasing view-like tensors to their storage owners
- assigning reusable temporaries to shared storage slots
- allocating fresh buffers only when reuse is not allowed or not possible

## Entry Points

- rule: [rules/MemoryOptimizerRule.java](./rules/MemoryOptimizerRule.java)
- planner: [memory/MemoryPlanner.java](./memory/MemoryPlanner.java)
- plan model: [memory/MemoryPlan.java](./memory/MemoryPlan.java)
- policy: [memory/MemoryPlannerPolicy.java](./memory/MemoryPlannerPolicy.java)

## Activation And Scope

The stage is globally gated by the system property:

```text
cg.optimizer.enableMemoryReuse
```

If that property is false, the stage is a no-op.

The stage also bails out early when:

- the graph is null or empty
- the graph contains mixed dtypes
- the graph dtype is `BFLOAT16`
- the graph dtype is `INT32`
- the graph dtype is `BOOL`

So today the rule only applies to uniform:

- `FLOAT64`
- `FLOAT32`

graphs.

## High-Level Split

The implementation is split into two parts:

1. `MemoryPlanner.plan(...)`
   - pure planning
   - computes lifetimes, reusable intervals, slot assignment, and summary metrics
2. `MemoryOptimizerRule.apply(...)`
   - materializes the plan onto actual tensor runtime storage

That split is useful because the same plan can be inspected and debugged independently of execution.

## Step 1: Forward Boundary Detection

The planner first determines where forward ends and backward begins.

Current rule:

- scan from the end of the graph
- find the last `NOOP` whose label equals `Tensor.SYSTEM_FORWARD_OUTPUT_LABEL`
- use its index as the forward boundary
- if no such tensor exists, treat the entire graph as one phase

This boundary is later used for:

- deciding whether a forward tensor must be saved for backward
- separating forward and backward slot pools when policy requires it

## Step 2: Storage Owner Resolution

Not every tensor owns storage. Some tensors are views or aliases.

The planner resolves a `storageOwner` for every tensor.

Current aliasing rules:

- always alias input 0 at runtime for:
  - `NOOP`
  - `EXPAND`
  - `SELECT`
  - `PERMUTE`
  - `EXPAND_DIMS`
  - `SQUEEZE`
- `RESHAPE` aliases input 0 only if the input is contiguous

Everything else owns its own storage by default.

This is why the plan is owner-based rather than tensor-based.

## Step 3: Consumer Tracking

The planner computes, for each storage owner:

- how many times it is consumed
- the index of its last read
- whether a forward owner is used again from the backward side

If an owner has no consumers at all, its `lastReadIndex` is set to `Integer.MAX_VALUE`.

That is deliberate. It means consumer-free results are treated as live until the end, which prevents incorrect reuse of true outputs.

## Step 4: Role Assignment

Every tensor receives a `NodeLifetime`:

- `birthIndex`
- `lastReadIndex`
- `role`
- `storageOwner`

Current roles are:

- `LEAF`
- `FORWARD_TEMP`
- `SAVED_FORWARD`
- `GRADIENT_TARGET`
- `BACKWARD_TEMP`
- `VIEW_ALIAS`

Role assignment rules:

- alias views -> `VIEW_ALIAS`
- tensors referenced as some other tensor's `.getGradient()` -> `GRADIENT_TARGET`
- forward owners that are consumed from backward -> `SAVED_FORWARD`
- backward tensors -> `BACKWARD_TEMP`
- other computed forward tensors -> `FORWARD_TEMP`
- tensors with no op -> `LEAF`

## Step 5: Reusable Interval Selection

Not every owner becomes reusable.

An owner is eligible for a `ReusableInterval` only if:

- it is its own storage owner
- its role is one of:
  - `FORWARD_TEMP`
  - `BACKWARD_TEMP`
  - `SAVED_FORWARD`
- its flat size is at least `policy.minReusableBufferSize()`

This is intentionally conservative:

- leafs are not pooled
- alias views do not get their own slots
- gradient targets are not pooled here

## Step 6: Slot Assignment

Reusable intervals are sorted by:

1. `birthIndex`
2. `lastReadIndex`

Then the planner walks them in order.

For each interval:

1. release all active slots whose `lastReadIndex < currentBirth`
2. choose the best compatible free slot
3. if no compatible slot exists, create a new slot
4. record slot ownership for the interval's storage owner

### Slot Compatibility Rules

A free slot is compatible only if:

- dtype matches
- if `separateForwardBackwardPools == true`, phase must match
- if `allowLargerBufferReuse == false`, slot size must equal interval size
- if `allowLargerBufferReuse == true`, slot size must be at least interval size

Among compatible slots, the planner chooses the smallest fitting one.

### Phase Model

Current phase labels are:

- `forward`
- `backward`
- `shared`

Intervals with role `SAVED_FORWARD` are treated as `shared`.

Important current implementation detail:

- `MemoryPlannerPolicy` contains `allowCrossPhaseReuse`
- but the slot chooser does not branch on that flag directly
- effective cross-phase behavior is currently controlled by `separateForwardBackwardPools`

So today the practical behavior is:

- if pools are separated, forward and backward slots do not mix except through `shared`
- if pools are not separated, cross-phase reuse is effectively allowed as long as other constraints match

That is the current code behavior and should be kept in mind when tuning policy.

## Step 7: Summary Metrics

The planner computes a `MemoryPlanSummary` with metrics such as:

- reusable interval count
- slot count
- reuse count
- reuse hit rate
- allocated slot bytes
- peak live bytes
- peak reusable bytes
- peak saved-forward bytes
- peak gradient-target bytes
- peak forward live bytes
- peak backward live bytes
- saved-forward count
- gradient-target count
- average saved-forward hold distance

These are useful when validating whether a memory optimization change actually improved the plan.

## Applying The Plan

`MemoryOptimizerRule` applies the plan separately for:

- `FLOAT64`
- `FLOAT32`

The logic is the same, only the backing array type changes.

### For view aliases

If `lifetime.role() == VIEW_ALIAS`, runtime storage is aliased via:

```text
tensor.aliasRuntimeFrom(lifetime.storageOwner())
```

### For slotted owners

If the tensor's owner has a slot:

1. fetch or lazily allocate the slot buffer
2. zero-fill the whole buffer
3. assign that buffer to the tensor

This happens through:

- `double[]` buffers for `FLOAT64`
- `float[]` buffers for `FLOAT32`

### For non-slotted owners

If no slot is assigned, the rule allocates a fresh dense array of the tensor's flat size.

## Why Zero-Fill Happens

The rule explicitly fills reused slot buffers with zeros before assignment.

This is conservative but correct:

- it avoids stale data when a kernel does not overwrite every position
- it makes alias/reuse bugs easier to reason about

The cost is extra memory bandwidth, so it is a correctness-first choice.

## Debugging Hooks

`MemoryOptimizerRule` exposes the most recent plan statically:

- `lastPlan()`
- `lastExplain()`
- `lastSummary()`

`MemoryPlan.explain()` prints:

- plan summary
- slot assignment by slot id
- per-node assignment
- saved forward values

This is the first thing to inspect when reuse behavior looks suspicious.

## Example

Consider a simple forward-only chain:

```text
t1 = add(x, y)
t2 = relu(t1)
t3 = mul(t2, z)
```

If:

- `t1` is no longer needed after `t2`
- `t2` is no longer needed after `t3`
- shapes and dtypes match

then `t1` and `t2` may share the same slot across time.

Conceptually:

```text
slot 0:
  [t1 live ........]
             [t2 live ........]
```

The tensors remain distinct graph nodes, but their runtime backing storage is reused.

## Important Limits

- the stage only works on uniform `FLOAT32` or `FLOAT64` graphs
- it does not currently reuse `BFLOAT16`, `INT32`, or `BOOL`
- it does not pool gradient targets
- reuse is shape-size based, not semantic
- zero-fill can trade some raw speed for safety

## Practical Meaning

`MEM` is the stage that turns a graph from "allocate a fresh dense array for almost everything" into "reuse storage where lifetime analysis proves it is safe".

If you are debugging unexpected peak memory or trying to understand why a temporary still allocates fresh storage, this is the stage to inspect first.

## Concrete Reuse Example

Consider:

```text
t1 = add(x, y)
t2 = relu(t1)
t3 = mul(t2, z)
t4 = exp(t3)
```

If all tensors share dtype/size and each one dies before the next becomes live, the plan may reuse one slot sequentially:

```text
slot 0 -> t1
slot 0 -> t2
slot 0 -> t3
slot 0 -> t4
```

The nodes stay distinct.
Only their runtime backing storage is reused.

## Saved-Forward Example

Now consider:

```text
f1 = linear(x, w, b)
f2 = relu(f1)
loss = sum(f2)
g1 = backward helper that still needs f2
```

`f2` may become `SAVED_FORWARD` instead of a short-lived forward temp.
That means the planner keeps it alive across the forward/backward boundary instead of reusing its storage too early.

## Practical Interpretation Of Summary Metrics

Useful heuristics when reading `MemoryPlanSummary`:

- high `reuseHitRate`
  - many eligible intervals successfully shared storage
- high `savedForwardCount`
  - backward is forcing many forward activations to stay live
- high `peakSavedForwardBytes`
  - saved activations dominate memory pressure
- many intervals but low reuse
  - sizes, phases, or lifetime overlap are blocking reuse

## Debug Checklist

If a tensor did not reuse storage and you expected it to:

1. confirm the graph is uniform `FLOAT32` or `FLOAT64`
2. check whether it is a true owner or only a view alias
3. check whether it became `SAVED_FORWARD` or `GRADIENT_TARGET`
4. check whether its size passed `minReusableBufferSize`
5. check whether another overlapping lifetime occupied the slot
6. check whether phase separation blocked reuse
