# Task 0007: Representation Creation and [Residency](../../../../glossary.md#residency) Foundation

> **Editorial historical reading key.**
>
> Read this completed task as the historical creation/residency layer, not today's full execution
> contract. Package-private `RunStateCreation` receives dense caller representations and invokes
> backend-supplied creators; Runtime controls ordering, identity checks, validity bookkeeping and
> cleanup transfer, while concrete backends implement allocation and physical release. Borrowed inputs
> remain caller-owned; successfully created resources pass to the complete `RunState` only after
> construction succeeds. Residency means a representation is bound, whereas validity records whether
> that copy contains the logical value; setting a bit performs no physical work.
>
> The [current creation
> contract](../../../../reference/runtime/creation.md#current-prepared-representation-creation) also
> includes initialized buffers. The preserved two-component record requirement is historical: today's
> [prepared owner](../../../../reference/runtime/prepared-execution.md#current-prepared-execution) has
> explicit persistent-resource ownership, closure and run leases. Recorded test evidence remains
> evidence of this task's original scope.

## Status

Complete

## Goal

Add the smallest [Runtime](../../../../glossary.md#term-runtime)-owned prepared description and cold per-[run](../../../../glossary.md#term-run) setup needed to turn current
prepared memory geometry into one complete [`RunState`](../../../../glossary.md#run-state--runstate), while keeping concrete physical classes
and creation mechanics inside concrete [backends](../../../../glossary.md#backend).

```text
immutable prepared description
  -> cold caller-input binding and backend creation
  -> mutable per-run validity in RunState
```

The prepared description distinguishes caller-supplied [buffers](../../../../glossary.md#term-buffer) from backend-created buffers and
workspace callbacks. Cold setup validates every caller input before invoking a [callback](../../../../glossary.md#term-callback), creates
run-owned representations deterministically, and transfers cleanup only after `RunState`
construction succeeds. Caller inputs begin valid. Created buffers begin invalid. Workspaces are
run-owned backend-local scratch and never represent a coherent logical value.

Preserve the completed aggregate exactly:

```java
public record PreparedExecution(
        PreparedMemoryPlan memoryPlan,
        PreparedSchedule schedule) {}
```

The new [prepared representation plan](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) remains reachable through the existing schedule component.

## [Scope](../../../../glossary.md#term-scope)

- Add one [immutable](../../../../glossary.md#term-immutable) public [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) in `runtime.resource` with nested typed
  caller-input, created-buffer, buffer-creator, and workspace-creator contracts.
- Add one `PreparedSchedule.RepresentationCreationStep` that retains the plan. It is optional for
  compatibility, but when present must be the sole creation step and the first occurrence.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Add one package-private `RunStateCreation` cold [operation](../../../../glossary.md#operation) in `runtime.run`.
- Accept caller inputs in deterministic dense encounter order, always as [`BORROWED`](../../../../glossary.md#ownership-borrowing-and-lease).
- Invoke buffer creators in buffer/representation order, then workspace creators in workspace
  order; every callback result is `RUN_OWNED`.
- Validate all caller references and [identity](../../../../glossary.md#term-identity-meanings) uniqueness before invoking any [callback](../../../../glossary.md#term-callback).
- On partial failure, close successfully created resources once in reverse creation order, never
  close borrowed resources, preserve the original unchecked failure, and suppress cleanup
  failures in encounter order.
- Add one array-backed validity bit for every [`RunState`](../../../../glossary.md#run-state--runstate) buffer representation and explicit
  constant-time query/mutation methods.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Initialize `BORROWED` [buffers](../../../../glossary.md#term-buffer) valid and `RUN_OWNED` buffers invalid, including through the
  existing public `RunState` constructor.
- Define binding presence as residency for this foundation: the exact physical object exists for
  the run until closure. Validity separately records whether it contains the logical slot value.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency))</span>
- Keep workspaces structurally resident, [run](../../../../glossary.md#term-run)-owned, and outside logical validity.
- Preserve the existing cold typed [binding](../../../../glossary.md#term-binding) and direct-reference [hot path](../../../../glossary.md#term-hot-path).
- Add focused exact-surface, validation, creation, rollback, validity, schedule, and isolation
  tests and finalize affected Javadocs/documentation in a separate clean documentation context.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Synchronize this task, the Runtime master plan, and the roadmap without rewriting completed
  history.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- transfer routes or callbacks, copies, [materialization](../../../../glossary.md#materialization), or transfer schedule steps
- [kernels](../../../../glossary.md#kernel), schedule execution, a runner, or automatic validity changes around execution
- hidden coherence, implicit write-back, dirty-state policy, or automatic [synchronization](../../../../glossary.md#term-synchronization)
- creation, eviction, replacement, or removal after [`RunState`](../../../../glossary.md#run-state--runstate) construction
- [lazy](../../../../glossary.md#term-lazy-evaluation) allocation, [pooling](../../../../glossary.md#term-pooling), caching, reuse, [aliasing](../../../../glossary.md#term-aliasing), or [lifetime](../../../../glossary.md#term-lifetime)/interference analysis
- physical access, addresses, native handles, backend/device inspection, or concrete physical
  representation classes in Runtime<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device))</span>
- broad allocators, factories, managers, registries, services, service locators, discovery, or a
  public orchestration facade<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- maps, reflection, string [dispatch](../../../../glossary.md#term-dispatch), raw `Object`, unchecked generics, synchronization, or public
  concrete-backend switches
- [lowering](../../../../glossary.md#lowering), route/[kernel](../../../../glossary.md#kernel) selection, fallback, tuning/[cache](../../../../glossary.md#term-cache) behavior, tracing, or profiling
- caller-input names, graph/value identities, Prepare source associations, publication,
  ownership transfer/lease, <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lease glossary](../../../../glossary.md#term-lease))</span>[`RunResult`](../../../../glossary.md#run-result--runresult), or Config 0007
- changes to [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan),
  `BufferRepresentationBinding`, `RunResourceOwnership`, or the two representation interfaces
- Prepare, Engine, concrete backend, Gradle, dependency, architecture, ADR, architecture-test,
  conformance-test, or integration-test implementation<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- detailed [Runtime](../../../../glossary.md#term-runtime) 0008–0011 or [Prepare](../../../../glossary.md#prepare) 0003 specifications

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): core invariants; Runtime, Prepare, and
  concrete-backend ownership; run lifecycle; service-locator and dependency rules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend ownership glossary](../../../../glossary.md#backend-ownership); [run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [ADR 0006: No Runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Runtime master plan](../master-plan.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)
- [Runtime 0005](0005-prepared-schedule-contract.md)
- [Runtime 0006](0006-prepared-execution-aggregate.md)
- [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Prepared recipes remain immutable, reusable, and thread-safe. Every active run receives one
  distinct mutable <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [mutable glossary](../../../../glossary.md#term-mutable))</span>[`RunState`](../../../../glossary.md#run-state--runstate) and distinct [run](../../../../glossary.md#term-run)-owned [callback](../../../../glossary.md#term-callback) results.
- Runtime owns prepared representation coordinates, per-run orchestration, ownership, logical
  validity/residency, rollback, cleanup, and run isolation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [residency glossary](../../../../glossary.md#residency); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Concrete backend modules implement creator callbacks and physical representation classes. They
  own allocation, release, transfer, and access mechanics. Runtime invokes but never interprets
  them.<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [release glossary](../../../../glossary.md#term-release))</span>
- Caller inputs are borrowed and initially valid. Created buffers are run-owned and initially
  invalid. All workspaces are run-owned scratch without logical validity.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- A buffer may have multiple resident representations and any subset may be valid. Zero valid
  copies is permitted before an internal buffer is first produced.
- Validity mutation changes a fact only. It performs no copy, kernel call, storage access,
  ownership transition, or implicit invalidation.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- Runtime 0008 and later runner/execution work must query and mutate validity explicitly around
  successful prepared actions; they must not infer coherence from ownership or concrete class.
- Creation is cold. `PreparedExecutable.bind` and `BoundInvocation.execute` remain unchanged and
  gain no creation, validity lookup, map access, or repeated cast.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- The representation plan enters [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) only through its existing exact schedule.
  `PreparedExecution(memoryPlan, schedule)` remains the entire aggregate [shape](../../../../glossary.md#shape).
- Runtime remains independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backends. Backend Contract remains closed. Config 0007 is not a dependency.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- If implementation needs a new module edge, an incompatible completed signature change, or an
  architecture-contract update, stop and report the exact decision instead of editing.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>

## Package impact

Existing packages used or changed:

- `runtime.memory` — consume [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) and dense encounter order without modification.
- `runtime.resource` — add the [immutable](../../../../glossary.md#term-immutable) representation description beside [nominal](../../../../glossary.md#term-nominal) physical roles.
- `runtime.run` — add package-private cold setup and explicit per-[run](../../../../glossary.md#term-run) validity.
- `runtime.schedule` — add the sole currently justified non-executable setup variant.

No package is added, moved, or removed.

Type placement:

- `runtime.resource.PreparedRepresentationPlan` — Runtime owns reusable backend-neutral
  coordinates and callback roles; concrete backends supply implementations.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [callback glossary](../../../../glossary.md#term-callback))</span>
- Its nested preparation variants and creators — meaningful only inside that plan, so they do not
  become five more top-level types.
- `runtime.run.RunStateCreation` — package-private cold operation for the future Runtime runner,
  not a public lifecycle facade.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `PreparedSchedule.RepresentationCreationStep` — keeps creation reachable through the unchanged
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) aggregate.

Tests mirror `runtime.run` and `runtime.schedule`. No root [facade](../../../../glossary.md#term-facade) or generic helper package is
introduced.

## Exact public and package-private [API](../../../../glossary.md#term-application-programming-interface-api) declarations

Add exactly:

```java
package io.github.pho001.synaptik.runtime.resource;

public record PreparedRepresentationPlan(
        PreparedMemoryPlan memoryPlan,
        List<List<PreparedRepresentationPlan.BufferPreparation>> bufferPreparations,
        List<PreparedRepresentationPlan.WorkspaceCreator> workspaceCreators) {

    public sealed interface BufferPreparation permits CallerInput, CreatedBuffer {}

    public record CallerInput() implements BufferPreparation {}

    public record CreatedBuffer(BufferCreator creator) implements BufferPreparation {}

    @FunctionalInterface
    public interface BufferCreator {
        BufferRepresentation create();
    }

    @FunctionalInterface
    public interface WorkspaceCreator {
        WorkspaceRepresentation create();
    }
}
```

Extend the existing schedule family exactly:

```java
public sealed interface Step permits ExecutionStep, RepresentationCreationStep {
    PreparedMemoryPlan memoryPlan();
}

public record RepresentationCreationStep(
        PreparedRepresentationPlan representationPlan) implements Step {
    @Override
    public PreparedMemoryPlan memoryPlan();
}
```

Add exactly these [`RunState`](../../../../glossary.md#run-state--runstate) members:

```java
public boolean isBufferRepresentationValid(
        int bufferIndex,
        int representationIndex);

public void setBufferRepresentationValid(
        int bufferIndex,
        int representationIndex,
        boolean valid);
```

Add exactly this package-private [operation](../../../../glossary.md#operation):

```java
final class RunStateCreation {
    static RunState create(
            PreparedRepresentationPlan representationPlan,
            List<BufferRepresentation> callerInputs);
}
```

`RunState` adds one private final `boolean[][] bufferValidity` matching `bufferBindings`. No other
public, protected, or package-private member or top-level production type is added. Its existing
constructor and all completed [signatures](../../../../glossary.md#term-signature) remain unchanged.

## Prepared representation-plan semantics

[`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) retains the exact non-null memory plan, snapshots both [buffer](../../../../glossary.md#term-buffer)-list
levels and the workspace list, and retains exact [immutable](../../../../glossary.md#term-immutable) preparation/[callback](../../../../glossary.md#term-callback) references.
Callback implementations must be immutable and [thread-safe](../../../../glossary.md#term-thread-safe) because one plan may create concurrent
runs.

[Validation](../../../../glossary.md#term-validation) order:

1. require `memoryPlan`, `bufferPreparations`, then `workspaceCreators` non-null;
2. require outer buffer count equal `memoryPlan.buffers().size()`;
3. require workspace count equal `memoryPlan.workspaces().size()`;
4. scan buffer positions: require each inner list non-null and non-empty, then each preparation
   non-null in representation order;
5. [snapshot](../../../../glossary.md#term-snapshot) each inner list and then the outer list; and
6. [scan](../../../../glossary.md#term-cumulative-scan) workspace creators for null in order, then snapshot.

Exact failures:

- `NullPointerException("memoryPlan")`
- `NullPointerException("bufferPreparations")`
- `NullPointerException("workspaceCreators")`
- `IllegalArgumentException("bufferPreparations size must equal prepared buffer count N")`
- `IllegalArgumentException("workspaceCreators size must equal prepared workspace count N")`
- `NullPointerException("bufferPreparations[i]")`
- `IllegalArgumentException("bufferPreparations[i] must not be empty")`
- `NullPointerException("bufferPreparations[i][j]")`
- `NullPointerException("workspaceCreators[i]")`

`CreatedBuffer` validates its component with `NullPointerException("creator")`. `CallerInput` has
no component or global [identity](../../../../glossary.md#term-identity-meanings); every [occurrence](../../../../glossary.md#term-occurrence) is one dense input position. Creator methods
declare no checked exception. Each successful callback must return one non-null fresh [run](../../../../glossary.md#term-run)-owned
representation not reused by another position or concurrent run. Cold creation validates the
non-null and within-run identity portions.

Plan construction invokes no callback, allocates no physical storage, creates no [`RunState`](../../../../glossary.md#run-state--runstate), and
changes no validity.

## Schedule semantics

`RepresentationCreationStep` validates `representationPlan` with
`NullPointerException("representationPlan")`, retains it exactly, and returns exactly
`representationPlan.memoryPlan()`.

Existing top-level and same-plan [validation](../../../../glossary.md#term-validation) order remains unchanged. After a non-null step reports
the exact schedule plan, apply this additional rule:

- a `RepresentationCreationStep` at any index other than zero fails with
  `IllegalArgumentException("steps[i] representation creation must be the first schedule occurrence")`.

There can therefore be zero or one creation step, only at index zero. Existing empty and
executable-only schedules remain valid. Repeated execution occurrences remain valid. [Prepare](../../../../glossary.md#prepare) 0003
may later require one creation prefix for a runnable final result; task 0007 does not invalidate
foundation schedules or implement that validator.

## Cold creation, [ownership](../../../../glossary.md#term-ownership-meanings), and cleanup

`RunStateCreation.create` accepts caller inputs as one flat list in the encounter order of all
`CallerInput` occurrences: [buffer](../../../../glossary.md#term-buffer) position first, then representation position. It performs no
schedule traversal, executable [binding](../../../../glossary.md#term-binding), transfer, or execution.

Exact order:

1. require `representationPlan`, then `callerInputs` non-null;
2. count caller-input occurrences without invoking callbacks;
3. require exact caller-input count;
4. [scan](../../../../glossary.md#term-cumulative-scan) all caller inputs for null and repeated exact [identity](../../../../glossary.md#term-identity-meanings);
5. traverse buffer preparations, retaining caller inputs as [`BORROWED`](../../../../glossary.md#ownership-borrowing-and-lease) or invoking creators and
   retaining results as `RUN_OWNED`;
6. invoke workspace creators in order and retain every result as [run](../../../../glossary.md#term-run)-owned;
7. construct one [`RunState`](../../../../glossary.md#run-state--runstate) with the exact plan and completed structures; and
8. return it, transferring cleanup of created results only after construction succeeds.

Exact pre-creation failures:

- `NullPointerException("representationPlan")`
- `NullPointerException("callerInputs")`
- `IllegalArgumentException("callerInputs size must equal caller-input preparation count N")`
- `NullPointerException("callerInputs[i]")`
- `IllegalArgumentException("representation is already bound to this run")`

Null [callback](../../../../glossary.md#term-callback) results fail with:

- `NullPointerException("bufferPreparations[i][j] creator result")`; or
- `NullPointerException("workspaceCreators[i] result")`.

A callback result duplicating a caller input or earlier result fails with the existing exact
`IllegalArgumentException("representation is already bound to this run")`.

Any `RuntimeException` or `Error` after callback invocation starts triggers cleanup of all
successfully created results in reverse creation order. The original failure is rethrown unchanged;
cleanup failures are suppressed in cleanup encounter order. Borrowed inputs are never closed. A
non-null duplicate callback result is not closed as a second owned object because it [aliases](../../../../glossary.md#aliasing-and-arena) a
borrowed or already-owned physical object.

Successful creation order is created buffers in dense buffer/representation order, then
workspaces. Existing `RunState.close()` already closes workspaces in reverse and then run-owned
buffers in reverse, so successful cleanup is also reverse creation order and skips borrowed
inputs.

Use arrays and direct iteration only: no map, reflection, string [dispatch](../../../../glossary.md#term-dispatch), [lookup](../../../../glossary.md#term-lookup) service,
[synchronization](../../../../glossary.md#term-synchronization), [backend](../../../../glossary.md#backend) selection, or physical access.

## Validity, [residency](../../../../glossary.md#residency), and transitions

For one open state:

- every bound [buffer](../../../../glossary.md#term-buffer)/workspace representation is resident until closure;
- a [buffer validity](../../../../glossary.md#buffer-validity) bit says whether that resident copy currently contains the logical slot value;
- [`BORROWED`](../../../../glossary.md#ownership-borrowing-and-lease) starts valid and `RUN_OWNED` starts invalid;
- multiple copies may be valid after a later copy;
- zero copies may be valid before an internal value is first produced; and
- workspaces have no validity because scratch is not a coherent logical-value copy.

The new methods first reject closure with `IllegalStateException("run state is closed")`, then
validate `bufferIndex`, then `representationIndex`. They retain the existing exact
`IndexOutOfBoundsException` messages `bufferIndex out of range: X` and
`representationIndex out of range: X`.

`isBufferRepresentationValid` returns the bit in [constant](../../../../glossary.md#term-constant) time. `setBufferRepresentationValid`
stores exactly the supplied [boolean](../../../../glossary.md#data-type--datatype) in constant time. It does not inspect storage, copy data, call
a [backend](../../../../glossary.md#backend), change [ownership](../../../../glossary.md#term-ownership-meanings), or change another bit. Repeating a value is valid.

[Runtime](../../../../glossary.md#term-runtime) 0008 can later require a valid source, perform an explicit backend-owned transfer, and
mark the destination valid only after success. Runner/execution work can later mark written output
copies valid and explicitly invalidate stale copies after successful computation. This task
implements none of those actions or policies.

[`RunState`](../../../../glossary.md#run-state--runstate) stays single-orchestrator-thread state. Add no [synchronization](../../../../glossary.md#term-synchronization), volatile/[atomic](../../../../glossary.md#resource-transaction-and-rollback) state,
or permission to [race](../../../../glossary.md#term-race) validity, [binding](../../../../glossary.md#term-binding), execution, or closure. Separate runs have independent
validity arrays.

## Affected files

Expected [Runtime](../../../../glossary.md#term-runtime) production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java` — add.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/package-info.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunStateCreation.java` — add package-private.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/package-info.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/PreparedSchedule.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/package-info.java`.

Expected Runtime tests:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateCreationTest.java` — add; also lock the exact nested plan/[callback](../../../../glossary.md#term-callback) surface.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateTest.java`.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/schedule/PreparedScheduleTest.java`.

Expected explanatory documentation:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation status only; no rule change.
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a contradiction is found: `AGENTS.md`, `ARCHITECTURE.md`, ADRs 0006/0010/0011,
focused architecture documents, documentation rules/profiles, planning guide, completed Runtime
0001–0006, [Prepare](../../../../glossary.md#prepare) master/tasks 0001–0002 and Draft 0003, Config master and Draft 0007, [Backend](../../../../glossary.md#backend)
Contract master/current contracts, current Runtime and relevant Prepare/Config/Backend Contract
source/tests/generated Javadocs, [Compile](../../../../glossary.md#compile)/[Tensor](../../../../glossary.md#tensor)/Training APIs, user and other backend guides,
Runtime/root [Gradle](../../../../glossary.md#term-gradle)/settings, and architecture/[conformance](../../../../glossary.md#term-conformance)/integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 18 paths: 7 [Runtime](../../../../glossary.md#term-runtime) production/[Javadoc](../../../../glossary.md#term-javadoc), 3 Runtime test, 5 explanatory documentation, and
3 [planning](../../../../glossary.md#term-planning) paths.

This is the smallest credible [atomic](../../../../glossary.md#resource-transaction-and-rollback) ceiling: description, schedule reachability, safe cold
creation/[rollback](../../../../glossary.md#term-rollback), per-[run](../../../../glossary.md#term-run) validity, focused tests, and public/[backend](../../../../glossary.md#backend) documentation are one
[lifecycle](../../../../glossary.md#lifecycle) contract. Splitting them would leave task 0007 unable either to create a complete state
safely or expose the facts required by Runtime 0008 and the runner.

No Java/test outside Runtime, [Gradle](../../../../glossary.md#term-gradle), [architecture contract](../../../../glossary.md#architecture-contract), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), architecture-test,
backend-[conformance](../../../../glossary.md#term-conformance), or integration path may change. Stop if another type, package, [module](../../../../glossary.md#term-module) edge,
[owner](../../../../glossary.md#term-owner), or path is needed. Do not create a later task specification.

## Acceptance criteria

- Exact planned public/package-private surfaces exist in only their named packages, with no extra
  facade, allocator/factory/manager/registry/service, ID, or helper type.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry))</span>
- Prepared descriptions and callbacks are immutable reusable references; mutable physical objects
  exist only as caller inputs or callback results inside a run.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [mutable glossary](../../../../glossary.md#term-mutable); [callback glossary](../../../../glossary.md#term-callback); [run glossary](../../../../glossary.md#term-run))</span>
- Concrete backends can implement callbacks and nominal physical representations without Runtime
  importing or inspecting them.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [nominal glossary](../../../../glossary.md#term-nominal); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Caller inputs are [`BORROWED`](../../../../glossary.md#ownership-borrowing-and-lease)/valid; created [buffers](../../../../glossary.md#term-buffer) are `RUN_OWNED`/invalid; all workspaces are
  run-owned and have no validity state.
- A buffer supports one or more resident representations and an independent bit per copy.
- Cold setup validates all caller inputs before creation and implements exact deterministic
  all-or-cleaned rollback, reverse order, suppression, and borrowed exclusion.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Validity query/mutation is explicit, constant-time, dense-indexed, closed-state guarded, and
  free of storage action or implicit coherence.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Runtime 0008 and runner work can use these facts without changing this contract, while this task
  implements no route, copy, materialization, kernel, execution transition, or schedule runner.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [kernel glossary](../../../../glossary.md#kernel))</span>
- The representation plan is reachable through [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule); [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) remains
  exactly memory plan plus schedule.
- Existing typed cold binding and direct hot execution remain unchanged and free of creation,
  validity, map, reflection, strings, lookup, or repeated casts.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Tests lock exact shape, validation/messages/order, immutable snapshots, callback counts,
  identity, rollback/suppression, initial/mutated validity, workspace exclusion, isolated runs,
  schedule compatibility, and forbidden mechanisms.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Every affected contract has complete Javadoc for inputs, returns, nullability, ownership,
  identity, lifecycle, thread safety, side effects, failures, and exclusions.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Runtime/Public API, focused boundary status, backend guide, glossary, task, master plan, and
  roadmap consistently distinguish current creation/validity from later transfers, publication,
  runner, Engine, and concrete backend behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [publication glossary](../../../../glossary.md#term-publication); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- A separate clean documentation pass finalizes Javadocs, examples, glossary, links, terminology,
  generated pages, planning evidence, and no-change conclusions without repeating successful Java
  tests absent a recorded reason.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Architecture, dependencies, Gradle, Backend Contract, Config, Prepare Java, concrete backends,
  architecture tests, conformance, and integration remain unchanged with reasoned conclusions.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Exact scope, task ordering/status, later-spec absence, links, anchors, terminology, fences,
  newlines, whitespace, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.run.RunStateCreationTest \
  --tests io.github.pho001.synaptik.runtime.run.RunStateTest \
  --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest
```

Final affected [module](../../../../glossary.md#term-module) after Java stabilizes:

```bash
./gradlew :modules:runtime:test
```

Documentation pass:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md \
  docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/writing-a-backend.md \
  docs/glossary.md \
  docs/planning/modules/runtime/tasks/0007-representation-creation-and-residency-foundation.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

If the temporary Markdown validator is absent/incompatible, use an equivalent temporary validator
outside the repository for local [targets](../../../../glossary.md#term-target)/anchors, unique effective anchors, balanced backtick and
tilde fences, final newlines, and trailing whitespace.

Also verify:

- exact record/nested/generic/[sealed](../../../../glossary.md#term-sealed)/functional-interface and schedule surfaces;
- exact package-private [operation](../../../../glossary.md#operation) and absence of public setup;
- every specified validation, message, order, identity, call count, cleanup, suppression, initial
  state, mutation, closure, and isolation rule;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- no [callback](../../../../glossary.md#term-callback) before complete caller validation;
- unchanged [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) source/compiled surface and unchanged [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable),
  [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), and their hot-path [bytecode](../../../../glossary.md#term-bytecode);
- no forbidden map/reflection/raw/string/registry/synchronization/concrete-backend/upstream/Config/
  Prepare/Engine/transfer/kernel/runner/publication mechanism or import;<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [synchronization glossary](../../../../glossary.md#term-synchronization); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [kernel glossary](../../../../glossary.md#kernel); [publication glossary](../../../../glossary.md#term-publication))</span>
- unchanged [Runtime](../../../../glossary.md#term-runtime)/root [Gradle](../../../../glossary.md#term-gradle), dependencies, and Java 26 configuration;
- exact 18 paths and no Java/test outside Runtime;
- Runtime 0001–0006 and Prepare 0001–0002 remain Complete; Runtime 0007 synchronizes through
  implementation and becomes Complete only after all gates; Runtime 0008–0011, Prepare 0003, and
  Config 0007 remain Draft without detailed specs; and
- final documentation/terminology/status/whitespace gates.

Repository-wide and architecture tests are deferred to the Runtime prepared-execution [checkpoint](../../../../glossary.md#term-checkpoint)
or [CI](../../../../glossary.md#term-continuous-integration-ci) because this changes one module without a [dependency](../../../../glossary.md#term-dependency), build, architecture, concrete-[backend](../../../../glossary.md#backend),
or end-to-end behavior change. [Conformance](../../../../glossary.md#term-conformance)/integration tests are not applicable yet.

The documentation context reuses successful Runtime test evidence unless it changes executable
behavior or records a concrete risk.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0001–0006 — Complete.
- Prepare 0001–0002 — Complete; they establish geometry and dense resource coordinates, but
  Runtime imports none of their types.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0011 — Accepted and sufficient for this [ownership](../../../../glossary.md#term-ownership-meanings)/creation/validity split.
- Existing Runtime dependencies and Java 26 build contract — unchanged.

Config 0007, [Prepare](../../../../glossary.md#prepare) 0003, Runtime 0008+, concrete [backends](../../../../glossary.md#backend), [Engine](../../../../glossary.md#term-engine), transfers, [publication](../../../../glossary.md#term-publication),
tracing, and tuning are not dependencies.

## Follow-up tasks

- Runtime 0008 remains Draft for explicit transfer/materialization steps using these coordinates
  and validity mutations.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- [Runtime](../../../../glossary.md#term-runtime) 0009 remains Draft for [publication](../../../../glossary.md#term-publication)/result associations and [ownership](../../../../glossary.md#term-ownership-meanings) transitions.
- Runtime 0010 remains Draft for cold schedule consumption, caller input handoff, binding,
  execution, and explicit validity transitions.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [binding glossary](../../../../glossary.md#term-binding))</span>
- Runtime 0011 remains Draft for contract closure.
- Prepare 0003 remains Draft for public preparation and validation that a runnable schedule has
  exactly one compatible creation prefix.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [validation glossary](../../../../glossary.md#term-validation))</span>
- Concrete backend tasks later implement creator callbacks, physical representations, transfers,
  and kernels.<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create any follow-up specification here.

## Javadocs and documentation impact

- Fully document the representation plan and nested contracts: prepared/run distinction, dense
  order, snapshots, callback obligations, ownership, reuse, validation, and exclusions.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Document package-private `RunStateCreation`: caller order, callback order, transfer point,
  rollback, original/suppressed failures, and non-public role.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Update [`RunState`](../../../../glossary.md#run-state--runstate)/package Javadocs for structural residency, validity, initial state, mutation,
  workspace exclusion, closure, and unchanged single-thread/cold-binding rules.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-normative))</span><span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency))</span>
- Update schedule Javadocs for first-only setup, same-plan [identity](../../../../glossary.md#term-identity-meanings), reachability through
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), compatibility, and no runner behavior.
- Runtime API needs the current mental model and a focused example with borrowed valid input,
  created invalid buffer, run-owned workspace, explicit validity change, and cleanup. Label any
  future-runner use conceptual.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Public [API](../../../../glossary.md#term-application-programming-interface-api) lists the new current contracts without claiming a public [run](../../../../glossary.md#term-run)/[Prepare](../../../../glossary.md#prepare) [facade](../../../../glossary.md#term-facade).
- Focused architecture page changes implementation status/mechanics only, not rules.
- Backend guide shows immutable typed callbacks and fresh physical results without registry,
  Runtime lookup, or hot allocation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Glossary adds only reusable prepared-representation-plan and buffer-validity terms and updates
  status; do not duplicate algorithms.
- Generate and inspect [Runtime](../../../../glossary.md#term-runtime) Javadocs for the new/nested pages, `RunState`, [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule),
  and affected package summaries.

Required no-change conclusions:

- [Architecture contract](../../../../glossary.md#architecture-contract), [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) pages, and ADRs already assign these [owners](../../../../glossary.md#term-owner).
- [Compile](../../../../glossary.md#compile), [Tensor](../../../../glossary.md#tensor), and Training APIs do not expose physical representations/validity/runner.
- Prepare Java/Javadocs/history stay unchanged because Draft Prepare 0003 is the later consumer.
- Config/Config 0007 stay unchanged because creation/validity needs no invocation/[publication](../../../../glossary.md#term-publication) option.
- [Backend](../../../../glossary.md#backend) Contract stays closed because these are Runtime/concrete-backend roles, not identity DTOs.
- Other guides stay unchanged absent a concrete contradiction; no [Engine](../../../../glossary.md#term-engine) workflow/backend exists.
- Gradle/architecture tests stay unchanged because no edge/rule changes; conformance/integration
  remain inapplicable.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture impact

Expected impact: None.

The architecture and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0011 already require explicit prepared creation, [backend](../../../../glossary.md#backend)-owned physical
mechanics, [Runtime](../../../../glossary.md#term-runtime)-owned per-[run](../../../../glossary.md#term-run) orchestration/validity/[residency](../../../../glossary.md#residency)/cleanup, and isolated runs. The
new schedule variant is a compatible representation of setup work and keeps the two-component
[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) intact.

If creators must move to [Prepare](../../../../glossary.md#prepare)/Backend Contract, Runtime must inspect a [concrete backend](../../../../glossary.md#concrete-backend)/[device](../../../../glossary.md#term-device),
`PreparedExecution` needs another component, a completed [signature](../../../../glossary.md#term-signature) must change incompatibly, or a
[module](../../../../glossary.md#term-module) edge/architecture rule must change, stop without editing and report the decision.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, ADRs 0006/0010/0011, the focused
Runtime/Prepare/backend architecture documents, documentation rules and General/API-Javadoc/
Architecture/Backend-Guide/Planning/Example profiles, Runtime and Prepare master plans, Runtime
tasks 0001–0007, Prepare tasks 0001–0002, current Runtime and relevant Prepare/Backend Contract/
Config source/tests/generated Javadocs, Runtime/Public APIs, backend guide, glossary, and builds.

Implement docs/planning/modules/runtime/tasks/0007-representation-creation-and-residency-foundation.md
exactly within its 18-path ceiling. Preserve PreparedExecution as exactly memoryPlan + schedule
and all completed contracts except the specified compatible additions. Do not implement transfer/
copy/materialization, kernels, execution/runner, publication/result, Config 0007, Prepare/Engine,
concrete backends, discovery/registries/services, pooling, tracing/tuning, dependency/Gradle/
architecture changes, or later specs. Stop on any architecture, package, API, validation, or scope
conflict.

Run focused and final Runtime validation plus exact surface/mechanism/scope/status checks. Hand the
actual diff and evidence to a separate clean documentation context. That pass must follow
documentation-rules.md, independently finalize Javadocs/docs/examples/glossary/planning evidence,
inspect generated Runtime Javadocs, record no-change conclusions, and not repeat successful Java
tests without executable changes or a concrete risk.

Mark Complete only after every gate. Return both context IDs, exact paths, commands/results/counts,
issues/follow-up, and repository status. Do not commit or push.
```

## Local decisions

- Use one [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan), not callbacks on physical interfaces or executables, to
  keep reusable descriptions immutable and physical objects per-run.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run))</span>
- Nest preparations/creators because their vocabulary is plan-local.
- Use zero-component `CallerInput` occurrences; dense encounter order replaces a new ID/map.
- Put one creation step at the schedule prefix so [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) remains unchanged.
- Keep creation package-private; the later runner is the public consumer.
- Initialize validity from existing ownership: borrowed means caller input/current value;
  run-owned means newly created storage without a value.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Use one boolean per representation and one setter, permitting zero/multiple valid copies without
  automatic coherence.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Treat binding as residency because every prepared representation exists before cold binding and
  remains until close; another always-true flag would duplicate arrays.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency))</span>
- Validate callers before creators and compare [identity](../../../../glossary.md#term-identity-meanings) with direct iteration, not a map.
- Do not close a duplicate callback result as a second object because it aliases an already
  borrowed/owned resource.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>

## Known limitations

- No public input names or [graph](../../../../glossary.md#graph)/value IDs; later runner uses dense caller-input order.
- All representations are created before binding and remain resident until close; no lazy
  creation, eviction, or dynamic residency.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [residency glossary](../../../../glossary.md#residency))</span>
- Validity is an orchestration fact; Runtime cannot prove bytes were copied/computed before a bit
  is set.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The setter permits zero/multiple valid copies and has no action-specific transition policy.
- Callback contracts cannot prove concrete immutability, thread safety, freshness, geometry, or
  physical cleanup behavior; backend tests must.<span data-glossary-navigation="true"> ([Callback glossary](../../../../glossary.md#term-callback))</span>
- Empty/executable-only schedules remain constructible; Prepare 0003 later validates runnable
  completeness.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- No transfer, [publication](../../../../glossary.md#term-publication)/result transition, runner, production [backend](../../../../glossary.md#backend), or [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle).

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root` ran the focused command:

  ```bash
  ./gradlew :modules:runtime:test \
    --tests io.github.pho001.synaptik.runtime.run.RunStateCreationTest \
    --tests io.github.pho001.synaptik.runtime.run.RunStateTest \
    --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest
  ```

  Its final stabilized result was `BUILD SUCCESSFUL`; JUnit XML reported three suites and 37
  tests (<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>`RunStateCreationTest` 10, `RunStateTest` 15, and `PreparedScheduleTest` 12), with zero
  failures, errors, or skips.
- The implementation context ran exactly one final affected-module command after executable Java
  stabilized: <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`./gradlew :modules:runtime:test`. It passed with `BUILD SUCCESSFUL`; JUnit XML
  reported 11 suites and 94 tests with zero failures, errors, or skips. Gradle reported nine
  actionable tasks, one executed and eight up-to-date.
- Documentation-focused context `/root/runtime_0007_docs` applied the General, API/Javadoc,
  Architecture, Backend Guide, Planning, and Example profiles. It independently read the
  architecture contract, ADRs 0006/0010/0011, focused lifecycle/module/dependency/boundary
  explanations, planning guide/roadmap, Runtime and Prepare master/task history, final Runtime
  source/tests, adjacent Runtime/Prepare/Backend Contract/Config contracts, APIs, backend guide,
  glossary, builds, and the actual dirty diff.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning); [architecture contract glossary](../../../../glossary.md#architecture-contract); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency); [Prepare glossary](../../../../glossary.md#prepare))</span>
- The documentation pass changed Javadocs, package documentation, explanatory documentation,
  glossary entries, and planning evidence only; it changed no executable behavior or tests. It
  therefore reused the successful 37-test and 94-test implementation evidence and did not repeat
  either Java test command.
- Final [Runtime](../../../../glossary.md#term-runtime) [Javadoc](../../../../glossary.md#term-javadoc) generation passed:

  ```bash
  ./gradlew :modules:runtime:javadoc
  ```

  [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL`; five tasks were actionable, two executed and three
  up-to-date. Generated pages were inspected for [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) and all five nested
  contracts, [`RunState`](../../../../glossary.md#run-state--runstate), [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), `RepresentationCreationStep`, and the resource/run/
  schedule package summaries. They expose dense ordering, immutable reuse, callback freshness,
  ownership, nullability, initial validity, structural residency, constant-time mutation,
  first-only creation, lifecycle, rollback, suppression, and deliberate boundaries.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [immutable glossary](../../../../glossary.md#term-immutable); [callback glossary](../../../../glossary.md#term-callback); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [residency glossary](../../../../glossary.md#residency); [constant glossary](../../../../glossary.md#term-constant); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- The focused current Runtime [API](../../../../glossary.md#term-application-programming-interface-api) example was compiled and executed with Java 26:

  ```bash
  javac --release 26 -cp modules/runtime/build/classes/java/main \
    -d /tmp/runtime-0007-api-example /tmp/Runtime0007ApiExample.java
  java -cp modules/runtime/build/classes/java/main:/tmp/runtime-0007-api-example \
    Runtime0007ApiExample
  ```

  Both commands passed with no output. The example checks borrowed-valid and created-invalid
  initial state, one explicit validity mutation, workspace exclusion, exact plan retention,
  borrowed cleanup exclusion, run-owned cleanup, and idempotence. The future runner is labeled
  conceptual; the example does not claim a copy, kernel, or automatic transition.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [kernel glossary](../../../../glossary.md#kernel))</span>
- The exact eight-file Markdown command passed and reported `validated 8 Markdown files`:

  ```bash
  python3 /tmp/validate_synaptik_markdown.py \
    docs/api/runtime-api.md \
    docs/api/public-api.md \
    docs/architecture/runtime-prepare-backend-boundary.md \
    docs/backend-guide/writing-a-backend.md \
    docs/glossary.md \
    docs/planning/modules/runtime/tasks/0007-representation-creation-and-residency-foundation.md \
    docs/planning/modules/runtime/master-plan.md \
    docs/planning/roadmap.md
  ```

  It checked local link targets and heading anchors, unique effective anchors, balanced backtick
  and tilde fences, final newlines, and trailing whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `javap -p` confirmed the exact `PreparedRepresentationPlan` record and nested sealed/record/
  functional-interface surface; package-private <span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>`RunStateCreation.create`; `RunState`'s sole
  `boolean[][]` plus exactly two public validity methods; the sealed schedule creation variant;
  and unchanged two-component [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution). [Bytecode](../../../../glossary.md#term-bytecode) inspection confirmed
  `BoundInvocation.execute` remains one state-open [guard](../../../../glossary.md#term-guard) plus `executeBound`, while
  `PreparedExecutable.bind` retains its existing compatibility and direct-[binding](../../../../glossary.md#term-binding) flow.
- Source and compiled-mechanism scans found no map, reflection, `ServiceLoader`, synchronization,
  atomic/concurrent state, raw <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>`Object`, unchecked suppression, upstream module, concrete backend,
  registry/service, transfer, kernel, publication, or runner mechanism in affected production.
  Source/tests confirm exact plan validation and snapshot order, complete caller prevalidation,
  buffer-then-workspace creation, identity uniqueness, borrowed-valid/run-owned-invalid state,
  independent constant-time mutation, original-failure preservation, reverse rollback with
  suppressed cleanup failures, successful reverse cleanup, run isolation, and first-only
  creation-step compatibility.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [registry glossary](../../../../glossary.md#term-registry); [publication glossary](../../../../glossary.md#term-publication); [snapshot glossary](../../../../glossary.md#term-snapshot); [buffer glossary](../../../../glossary.md#term-buffer); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `git diff --exit-code` checks confirmed no change to `ARCHITECTURE.md`, ADRs 0006/0010/0011,
  lifecycle/module/dependency pages, Runtime/root builds, Prepare/Backend Contract/Config Java,
  concrete backends, Engine, or architecture/conformance/integration tests. Root configuration
  remains Java toolchain and release 26; Runtime adds no override or dependency.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release))</span>
- The final task inventory contains exactly 18 authorized paths: seven Runtime production/
  Javadoc paths, three Runtime test paths, five explanatory documents, and three planning paths.
  No Java/test path outside Runtime, Gradle path, architecture contract/ADR/test, backend-
  conformance path, integration path, or later task specification changed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Status/specification checks confirm Runtime 0001–0007 and Prepare 0001–0002 are Complete;
  Runtime 0008–0011, Prepare 0003, and Config 0007 remain Draft; detailed specifications exist
  only through Runtime 0007 and Prepare 0002, with none for those later rows.
- Final `git diff --check` passed with no output.
- No-change conclusions:
  - `ARCHITECTURE.md`, ADRs 0006/0010/0011, and lifecycle/module/dependency explanations already
    assign immutable prepared work, per-run orchestration/validity, backend-owned physical
    mechanics, explicit composition, and module direction. Only the focused boundary page needed
    implementation-status detail; no rule or decision changed.
  - Compile, Tensor, and Training APIs remain unchanged because none exposes Runtime physical
    representations, validity, or a runner.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor))</span>
  - Prepare Java/Javadocs/history remain unchanged because Runtime consumes established dense
    geometry and Draft Prepare 0003 is the later public construction/validation consumer.
  - Config and Draft Config 0007 remain unchanged because creation and validity consume no run or
    publication option. Backend Contract remains closed because creators and physical roles belong
    to Runtime/concrete backends rather than the identity DTO leaf.<span data-glossary-navigation="true"> ([DTO glossary](../../../../glossary.md#data-transfer-object--dto); [leaf glossary](../../../../glossary.md#term-leaf))</span>
  - Other guides and Engine remain unchanged because no public lifecycle or production backend
    exists; the backend contributor guide alone needed the current typed-callback pattern.
  - Gradle/dependencies/Java 26, architecture tests, backend conformance, integration tests, and
    concrete backends remain unchanged because no edge, architecture rule, backend implementation,
    or end-to-end behavior changed. Later specifications remain absent by progressive planning.

## Implementation notes

- Added only the exact immutable representation plan and nested preparation/creator contracts,
  package-private cold creation operation, two validity methods and backing array, and one sealed
  schedule variant specified by the task.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [sealed glossary](../../../../glossary.md#term-sealed))</span>
- Cold creation prevalidates every borrowed caller input before callbacks, creates buffers then
  workspaces, and gives a complete state cleanup ownership only after construction succeeds.
  Partial failure preserves the original unchecked failure and rolls back prior created results in
  reverse order without closing borrowed or duplicate identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [buffers glossary](../../../../glossary.md#term-buffer); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Binding presence is structural residency. Borrowed buffers initialize valid, created run-owned
  buffers initialize invalid, workspaces have no validity, and the setter changes exactly one bit
  without physical or coherence behavior.<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency); [run glossary](../../../../glossary.md#term-run))</span>
- The documentation pass finalized all seven production/package Javadocs, the focused current API
  example, Runtime/Public API status, focused architecture implementation status, typed backend
  callback guidance, the two reusable glossary terms, and synchronized planning evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [callback glossary](../../../../glossary.md#term-callback); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: implemented and documented immutable prepared representation origins,
  package-private all-or-cleaned cold setup, structural residency, independent per-copy validity,
  and first-only schedule reachability while preserving the existing aggregate and hot path.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [residency glossary](../../../../glossary.md#residency); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Files changed or created: exactly the authorized 18 paths—seven Runtime production/Javadoc,
  three Runtime test, five explanatory documentation, and three planning paths.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the implementation context's successful focused 37-test and final
  94-test Runtime evidence; Runtime Javadoc/generated pages, Java 26 API example, eight-file
  Markdown, exact surface/order/failure/rollback/validity/hot-path/import/mechanism/build/Java-26/
  scope/status/later-specification, final-newline/fence/whitespace, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [rollback glossary](../../../../glossary.md#term-rollback); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` gates
  passed.
- Documentation-agent review: completed in clean context `/root/runtime_0007_docs` without
  executable behavior/test changes or repeated Java tests; implementation context was `/root`.
- Documentation impact: Runtime/Public APIs, focused architecture status, backend guide, task,
  Runtime master plan, and roadmap now distinguish current creation/residency/validity from later
  transfer, execution, publication, public Prepare, Engine, and runner work.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [publication glossary](../../../../glossary.md#term-publication); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Javadoc review: all seven affected production/package paths were independently finalized and
  generated pages inspected for every new/nested/current contract and package summary.
- Glossary impact: added only reusable prepared-representation-plan and buffer-validity terms and
  synchronized current residency, schedule, and run-state status.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run))</span>
- Unresolved issues: None.
- Follow-up required: None for this task. Runtime 0008–0011, Prepare 0003, and Config 0007 remain
  separate Draft planning work without detailed specifications.

Status: Complete
