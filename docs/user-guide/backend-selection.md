# Influence backend ownership (planned workflow)

## Outcome

This guide explains how users will express backend intent and how compile-time planning will turn
it into ownership. The current Java API can construct one hard backend eligibility target.
Configuration and planning APIs are not implemented, so a requirement cannot yet be attached to
a compile request and exact preference options and defaults are not available.

## Mental model

```text
optional hard requirement + backend capabilities + availability
  -> valid ownership candidates
user preference + valid candidates + graph estimates
  -> backend-neutral score
  -> owner identity for each node or segment
```

A hard requirement removes candidates that do not meet its target; it is not a preference or a
capability claim. Preference influences how planning compares the remaining candidates. A
capability says that a backend can accept particular work. None of these concepts chooses a
concrete kernel; that remains a backend prepare decision.

## Current hard-target values

These constructors are current and runnable with `modules:backend-contract`:

```java
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendDeviceIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.backend.contract.DeviceClassRequirement;

BackendId metal = new BackendId("metal");
BackendDeviceId metalZero = new BackendDeviceId(metal, "0");

BackendRequirement exactBackend = new BackendIdRequirement(metal);
BackendRequirement exactDevice = new BackendDeviceIdRequirement(metalZero);
BackendRequirement acceleratorClass =
        new DeviceClassRequirement(DeviceClass.ACCELERATOR);
```

`exactBackend` means that later eligible ownership must use a `BackendId` equal to `metal`.
`exactDevice` targets a `BackendDeviceId` equal to `metalZero` and therefore also its owning
backend. `acceleratorClass` permits any later eligible device in the `ACCELERATOR` class; it does
not prefer Metal over another accelerator backend. Each record retains and returns the exact
non-null reference supplied to it.

Constructing these values does not discover or register Metal, inspect an availability snapshot,
query capability, or select ownership. The family has no `AUTO`, `ANY`, or `NONE` value because
absence belongs to the later configuration field. It also has no preference, fallback,
combination, matcher, or score. Later planning will combine a supplied hard target with
availability and capability facts and fail rather than silently relax the target when no eligible
candidate remains; the failure type and message are not yet specified.

## Scenario

Assume a region can run on CPU or Metal. Moving its input to Metal has an estimated cost of 20 units, Metal execution saves 50 units, and an extra ownership boundary costs 10 units. A simple interpreted comparison is a net Metal benefit of `50 - 20 - 10 = 20` units, so Metal may win. These numbers illustrate the factors only; no implemented scoring formula or unit is promised.

The selected plan records `owner = Metal`. MPSGraph versus a custom Metal kernel remains a prepare-time backend decision.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Runtime changes backend after a failure | Selection was deferred too late. | Resolve support during compile and preparation; runtime follows the prepared schedule. |
| CPU scalar and OpenBLAS appear as separate backends | Routes were confused with ownership. | Treat both as CPU-internal prepare choices. |
| Current device residency changes compile scoring | Mutable run state leaked into planning. | Use compile-time estimates and immutable profiles only. |
| An accelerator-class requirement is treated as a Metal preference | Hard eligibility was confused with ranking. | Keep the class as a candidate filter and express preference in the later config-owned intent contract. |

## Limitations

Exact configuration placement, preference types, scoring policy, requirement evaluation, and
no-match diagnostics remain to be specified by focused tasks. See [Public API
status](../api/public-api.md), [Partition scoring](../architecture/partition-scoring.md), and the
[planning master plan](../planning/modules/planning/master-plan.md).
