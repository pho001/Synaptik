# Influence backend ownership (planned workflow)

## Outcome

This guide explains how users will express backend intent and how compile-time planning will turn
it into ownership. The current Java API can construct one hard backend eligibility target and
place it in `BackendIntent`, or construct an unconstrained intent with no hard target. The later
planning module also has a current immutable operation-capability query and explicitly supplied
provider interface. Current `PartitionScoringConfig` can separately record an optional soft
`DeviceClass` preference. Planning has one internal per-query hard-eligibility evaluator, but no
user-callable entry point. The later `CompileConfig` aggregate, reusable/public capability matrix,
public planning orchestration, preference interpretation, score calculation, and ownership planner
are not implemented, so these values cannot yet be attached to a compile request.

## Mental model

```text
optional hard requirement + current provider capability answers + availability
  -> current internal provider-ordered eligible BackendId values
  -> planned public ownership candidates
optional coarse class preference + valid candidates + graph estimates
  -> backend-neutral score
  -> owner identity for each node or segment
```

A hard requirement removes candidates that do not meet its target; it is not a preference or a
capability claim. Preference influences how planning compares the remaining candidates. A
capability says that a backend can accept particular work. None of these concepts chooses a
concrete kernel; that remains a backend prepare decision.

## Current hard-target and intent values

These constructors are current and runnable with `modules:backend-contract` and `modules:config`:

```java
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendDeviceIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.backend.contract.DeviceClassRequirement;
import io.github.pho001.synaptik.config.compile.BackendIntent;

BackendId metal = new BackendId("metal");
BackendDeviceId metalZero = new BackendDeviceId(metal, "0");

BackendRequirement exactBackend = new BackendIdRequirement(metal);
BackendRequirement exactDevice = new BackendDeviceIdRequirement(metalZero);
BackendRequirement acceleratorClass =
        new DeviceClassRequirement(DeviceClass.ACCELERATOR);

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireMetal = BackendIntent.requiring(exactBackend);
```

`exactBackend` means that later eligible ownership must use a `BackendId` equal to `metal`.
`exactDevice` targets a `BackendDeviceId` equal to `metalZero` and therefore also its owning
backend. `acceleratorClass` permits any later eligible device in the `ACCELERATOR` class; it does
not prefer Metal over another accelerator backend. Each record retains and returns the exact
non-null reference supplied to it.

Constructing these values does not discover or register Metal, inspect an availability snapshot,
query capability, or select ownership. The family has no `AUTO`, `ANY`, or `NONE` value because
absence belongs to `BackendIntent.hardRequirement()`. `unconstrained` therefore records only an
empty hard target; it does not promise a default backend, automatic discovery, or fallback.
`requireMetal` retains the exact `exactBackend` reference inside its optional. The requirement
family and intent have no preference, fallback, combination, matcher, or score. Later planning
will combine a supplied hard target with
availability and capability facts and fail rather than silently relax the target when no eligible
candidate remains; the failure type and message are not yet specified.

## Current soft-preference value

Use `PartitionScoringConfig` to record either no explicit coarse class preference or one soft CPU
or accelerator preference:

```java
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;

PartitionScoringConfig neutralRanking = PartitionScoringConfig.neutral();
PartitionScoringConfig preferAccelerator =
        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR);
```

`neutralRanking.preferredDeviceClass()` is empty. That absence does not select CPU, accelerator,
automatic discovery, fallback, equal scores, or a successful owner. `preferAccelerator` retains
the exact `DeviceClass.ACCELERATOR` reference. Later planning may use it only after hard
eligibility, so it does not make an eligible CPU candidate invalid, weaken `requireMetal`, or
guarantee that an accelerator candidate is selected.

Construction records metadata only. It does not inspect the hard requirement, availability, or
capability; enumerate candidates; calculate or compare scores; select a backend or device; or
prepare or execute work. The later aggregate will decide which scoring configuration is its
default; `neutral()` does not decide that policy today.

## Current capability boundary

`OperationCapabilityQuery` currently snapshots one backend-independent `Operation` plus its
ordered input and output `TensorDescriptor` references. An explicitly supplied
`BackendCapabilityProvider` names one stable `BackendId` and returns a deterministic boolean
answer for that immutable occurrence. This is a library integration boundary rather than a user
selection setting: constructing a query does not attach `BackendIntent`, discover a backend,
inspect `BackendAvailabilitySnapshot`, evaluate a requirement, or select ownership.

No production provider implementation or public planning consumer exists yet. Internal planning
can now validate one complete provider/snapshot set and combine non-empty availability, an exact
hard requirement, and backend-level support into an immutable provider-ordered `BackendId` list
for one query. A no-match list is empty and a hard requirement is never relaxed. Exact-device and
device-class matching proves only reported availability; it does not establish device-level
support or choose a device.

The internal result and evaluator are package-private, so users cannot invoke them or complete
backend selection with the current API. A provider's `true` means only that its named backend can
semantically own the occurrence. A `false` carries no diagnostic reason and does not by itself say
whether the backend is registered or available. The runnable inputs above are useful for preparing
configuration, while compile aggregation, public planning orchestration, scoring, ownership,
preparation, runtime, and execution remain planned workflows.

## Scenario

Assume a region can run on CPU or Metal. Moving its input to Metal has an estimated cost of 20 units, Metal execution saves 50 units, and an extra ownership boundary costs 10 units. A simple interpreted comparison is a net Metal benefit of `50 - 20 - 10 = 20` units, so Metal may win. These numbers illustrate the factors only; no implemented scoring formula or unit is promised.

The selected plan records `owner = Metal`. MPSGraph versus a custom Metal kernel remains a prepare-time backend decision.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Runtime changes backend after a failure | Selection was deferred too late. | Resolve support during compile and preparation; runtime follows the prepared schedule. |
| CPU scalar and OpenBLAS appear as separate backends | Routes were confused with ownership. | Treat both as CPU-internal prepare choices. |
| Current device residency changes compile scoring | Mutable run state leaked into planning. | Use compile-time estimates and immutable profiles only. |
| An accelerator-class requirement is treated as a Metal preference | Hard eligibility was confused with ranking. | Keep the class as a candidate filter; later scoring configuration owns preference. |
| A preferred accelerator is treated as a required accelerator | A soft ranking input was confused with hard eligibility. | Use `DeviceClassRequirement` for a hard target; a scoring preference never filters candidates. |
| `BackendIntent.unconstrained()` is treated as guaranteed automatic fallback | Absence of a hard target was confused with selection behavior. | Treat it only as no hard eligibility constraint; later planning may still have no valid candidate. |
| A capability-provider `false` is treated as proof that the backend is unavailable | Semantic support was confused with supplied availability. | Evaluate capability and availability as separate facts in later planning. |

## Limitations

`BackendIntent` placement and hard-target optionality, `PartitionScoringConfig`, the operation
query/provider contracts, and one internal per-query hard-eligibility step are current. Provider
implementations, device-level capability or selection, reusable/public capability matrices,
public planning orchestration, preference interpretation, score calculation, profiles, compile
aggregation, ownership, compiler integration, preparation, runtime, execution, and no-match
diagnostics remain to be specified by focused tasks. See [Public API
status](../api/public-api.md), [Partition scoring](../architecture/partition-scoring.md), and the
[planning master plan](../planning/modules/planning/master-plan.md).
