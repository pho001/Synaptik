# Influence backend ownership (planned workflow)

## Outcome

This guide explains how users will express backend intent and how compile-time planning will turn it into ownership. Configuration and planning APIs are not implemented, so exact options and defaults are not yet available.

## Mental model

```text
user intent + backend capabilities + graph estimates
  -> backend-neutral score
  -> owner identity for each node or segment
```

Intent is a preference or requirement, not a concrete kernel choice. A capability says that a backend can accept the work. Scoring compares valid candidates using compile-time estimates such as transfer and boundary costs.

## Scenario

Assume a region can run on CPU or Metal. Moving its input to Metal has an estimated cost of 20 units, Metal execution saves 50 units, and an extra ownership boundary costs 10 units. A simple interpreted comparison is a net Metal benefit of `50 - 20 - 10 = 20` units, so Metal may win. These numbers illustrate the factors only; no implemented scoring formula or unit is promised.

The selected plan records `owner = Metal`. MPSGraph versus a custom Metal kernel remains a prepare-time backend decision.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Runtime changes backend after a failure | Selection was deferred too late. | Resolve support during compile and preparation; runtime follows the prepared schedule. |
| CPU scalar and OpenBLAS appear as separate backends | Routes were confused with ownership. | Treat both as CPU-internal prepare choices. |
| Current device residency changes compile scoring | Mutable run state leaked into planning. | Use compile-time estimates and immutable profiles only. |

## Limitations

Exact intent types, scoring policy, availability behavior, and diagnostics remain to be specified by focused tasks. See [Partition scoring](../architecture/partition-scoring.md) and the [planning master plan](../planning/modules/planning/master-plan.md).
