# Influence backend ownership (planned workflow)

## Outcome

This guide explains how users will express backend intent and how compile-time planning will turn
it into ownership. The current Java API can construct one hard backend eligibility target and
place it in `BackendIntent`, or construct an unconstrained intent with no hard target. The later
planning module also has a current immutable operation-capability query and explicitly supplied
provider interface. Current `PartitionScoringConfig` can separately record an optional soft
`DeviceClass` preference. Planning has one internal per-query hard-eligibility evaluator, but no
user-callable entry point. It now also has an internal cost-free selector that applies the
preference to the hard-eligible identities and returns one backend owner. The later `CompileConfig`
aggregate, reusable/public capability matrix, public planning orchestration or owner selector,
cost scoring, and owner-map assembly are not implemented, so users still cannot attach these
values to a compile request. The public immutable `PlannedPartition` recipe and internal maximal
same-owner generator are current, but the generator is not user-callable. Public immutable
`LogicalMemoryRequirement` and `LogicalMemoryPlan` recipes plus internal derivation from the graph
and ordered partitions are also current; that derivation is likewise not user-callable.

## Mental model

```text
optional hard requirement + current provider capability answers + availability
  -> current internal provider-ordered eligible BackendId values
optional coarse class preference + that complete candidate list + associated snapshots
  -> current internal first preferred match, otherwise first eligible
  -> one BackendId owner
later orchestration assembles one owner for every graph NodeId
  -> current internal consecutive same-owner grouping
  -> current immutable PlannedPartition recipes
  -> current internal producer/consumer/output derivation
  -> current immutable LogicalMemoryPlan
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
family and intent have no preference, fallback, combination, matcher, or score. Current internal
planning combines a supplied hard target with availability and capability facts. Its baseline
selector fails rather than silently relaxing the target when no eligible candidate remains; that
internal failure is not yet a public compile failure contract.

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

`neutralRanking.preferredDeviceClass()` is empty. The current internal baseline then uses the first
hard-eligible backend in provider order; the config value itself still selects no CPU,
accelerator, discovery behavior, or successful owner. `preferAccelerator` retains the exact
`DeviceClass.ACCELERATOR` reference. Internal planning uses it only after hard eligibility: the
first eligible backend with a reported accelerator-class device wins, or the first eligible
backend wins when none matches. It does not make an eligible CPU candidate invalid, weaken
`requireMetal`, or guarantee that an accelerator candidate exists.

Construction records metadata only. It does not inspect the hard requirement, availability, or
capability; enumerate candidates; calculate or compare scores; select a backend or device; or
prepare or execute work. The package-private selector, not the config value, performs the current
baseline comparison. The later aggregate will decide which scoring configuration is its default;
`neutral()` does not decide that policy today.

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

The internal result, evaluator, and selector are package-private, so users cannot invoke them or
complete backend selection with the current API. The selector consumes the provider-ordered list
directly, validates equal-ID snapshot associations, allows extra unique snapshots, and returns the
first preferred-class match or the first eligible backend. Provider order resolves ties. An empty
matching snapshot is a preference nonmatch, and an empty eligible list fails before snapshot
elements are read with the internal message
`no hard-eligible backend is available for ownership selection`. It returns the exact eligibility
identity reference and never selects a device.

A provider's `true` still means only that its named backend can semantically own the occurrence.
A `false` carries no diagnostic reason and does not by itself say whether the backend is registered
or available. The runnable inputs above are useful for preparing configuration, while compile
aggregation, public planning orchestration, owner-map assembly, cost scoring, preparation,
runtime, and execution remain planned workflows.

## Current partition recipe and internal grouping

`PlannedPartition` is a current public immutable record containing one `BackendId owner` and one
non-empty ordered `List<NodeId> nodeIds`. Users can construct and inspect that data type directly,
but doing so does not validate graph membership or run planning. Its constructor snapshots list
membership, rejects null or duplicate node identities, and retains the exact owner and node-ID
references.

Planning's current generator is package-private. It requires a complete owner map for an existing
`CompiledGraphModel`, validates it before producing output, and groups maximal runs over the
stored topological node sequence. For graph order `[n0, n1, n2, n3]` with owner values
`[cpu, cpu, metal, cpu]`, the conceptual result is:

```text
owner cpu:   [n0, n1]
owner metal: [n2]
owner cpu:   [n3]
```

The last CPU node does not join the first CPU partition because the Metal-owned node separates
them. Graph connectivity does not replace this sequence rule: two consecutive independent nodes
with equal owners do join. A phase change, fan-out, merge, repeated input, graph output, or
multi-output producer does not split an otherwise equal-owner run. Graph inputs and outputs are
values, so a zero-node pass-through graph produces no partition.

This is current internal planning behavior, not a runnable user workflow. No public API currently
builds the complete owner map or calls the generator. The recipe also contains no boundary value,
transfer, materialization, memory, selected device, route, kernel, executable, or runtime state.

## Current logical-memory recipe and internal derivation

After partitioning, current package-private planning can validate an ordered complete partition
list against `CompiledGraphModel` and derive one `LogicalMemoryRequirement` per graph value. Each
generated requirement retains the value's exact `ValueId` and `TensorDescriptor`, an optional
producing partition, distinct consuming partitions in partition order, and whether the graph
declares the value as an output. `LogicalMemoryPlan` preserves those requirements in graph-value
order.

For a value produced in CPU partition `p0`, consumed in Metal partition `p1`, and declared as a
graph output, the logical facts are:

```text
producerPartition = p0
consumerPartitions = [p1]
graphOutput = true
```

This means the value is a partition output, a partition input of `p1`, a cross-owner boundary,
and a graph-output preservation obligation. It does not select a CPU-to-Metal transfer, physical
buffer, publication target, Metal representation, or execution step. A value produced and
consumed only inside one partition, with no graph-output obligation, remains partition-internal.
A graph input has no producing partition; a zero-node pass-through graph has no partitions but
still has a logical requirement for its declared input/output value.

The records are current public DTOs, but direct construction does not validate graph-relative
facts. The internal derivation performs complete membership, coverage, graph-order, and
adjacent-owner checks first. It accepts no `ForwardPublicationBinding` or
`GradientPublicationBinding` and calculates no element or byte count, lifetime, slot, allocation,
transfer, device, route, or kernel. No public API currently connects capability, owner selection,
partitioning, and logical-memory derivation end to end.

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
| `BackendIntent.unconstrained()` is treated as guaranteed automatic fallback | Absence of a hard target was confused with selection behavior. | Treat it only as no hard eligibility constraint; internal planning may still have no valid candidate. |
| A capability-provider `false` is treated as proof that the backend is unavailable | Semantic support was confused with supplied availability. | Evaluate capability and availability as separate facts in planning. |

## Limitations

`BackendIntent` placement and hard-target optionality, `PartitionScoringConfig`, the operation
query/provider contracts, internal per-query hard eligibility, and internal baseline owner
selection are current. The public partition recipe and internal consecutive same-owner generator
are also current, as are the public logical-memory recipes and their internal derivation. Provider
implementations, device-level capability or selection, reusable/public capability matrices,
public planning orchestration or owner selection, cost scoring, profiles, compile aggregation,
owner-map assembly, compiler integration, physical memory, preparation, runtime, execution, and
public no-match diagnostics remain to be specified by focused tasks. See [Public API
status](../api/public-api.md), [Partition scoring](../architecture/partition-scoring.md), and the
[planning master plan](../planning/modules/planning/master-plan.md).
