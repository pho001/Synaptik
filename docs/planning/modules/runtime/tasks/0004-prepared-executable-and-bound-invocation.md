# Task 0004: [Prepared Executable](../../../../glossary.md#prepared-executable--preparedexecutable) and [Bound Invocation](../../../../glossary.md#bound-invocation--boundinvocation)

## Status

Complete

## Goal

Implement the smallest [Runtime](../../../../glossary.md#term-runtime)-owned cold-[binding](../../../../glossary.md#term-binding) boundary authorized by
[ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>. One [immutable](../../../../glossary.md#term-immutable),
reusable [backend](../../../../glossary.md#backend)-owned [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) recipe selects exact dense positions from one open
[`RunState`](../../../../glossary.md#run-state--runstate), performs checked backend compatibility before execution, and returns one per-[run](../../../../glossary.md#term-run)
backend-owned [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) that retains direct typed resource references.

The exact public surface is:

```java
package io.github.pho001.synaptik.runtime.execution;

public abstract class PreparedExecutable {
    protected PreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            List<PreparedExecutable.BufferSelection> bufferSelections,
            List<PreparedExecutable.WorkspaceSelection> workspaceSelections);

    public final PreparedMemoryPlan memoryPlan();
    public final BoundInvocation bind(RunState runState);

    protected abstract boolean acceptsBufferRepresentation(
            int selectionIndex,
            BufferRepresentation representation);

    protected abstract boolean acceptsWorkspaceRepresentation(
            int selectionIndex,
            WorkspaceRepresentation representation);

    protected abstract BoundInvocation bindCompatible(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations);

    public record BufferSelection(int bufferIndex, int representationIndex) {}
    public record WorkspaceSelection(int workspaceIndex) {}
}

public abstract class BoundInvocation {
    protected BoundInvocation(RunState runState);
    public final void execute();
    protected abstract void executeBound();
}
```

The [concrete backend](../../../../glossary.md#concrete-backend) subclasses both abstract classes. Its `PreparedExecutable` implementation is
an immutable recipe created only after route choice and shared slot assignment. Its
`BoundInvocation` implementation stores checked concrete [buffer](../../../../glossary.md#term-buffer)/workspace references directly in
typed fields. Runtime supplies plan, selection, run-association, and closed-state [validation](../../../../glossary.md#term-validation); it
never knows a concrete backend representation class.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
immutable backend PreparedExecutable
  + exact prepared-plan identity
  + ordered dense buffer/workspace selections
  + one open matching RunState
       |
       v  cold: array lookup + explicit backend instanceof checks
backend-owned BoundInvocation
  + exact RunState association
  + direct concrete typed references
       |
       v  hot: one closed-state guard + direct backend call
execute prepared region
```

The complete [heterogeneous](../../../../glossary.md#term-heterogeneous) Java type relation cannot be expressed in one shared generic
[signature](../../../../glossary.md#term-signature). The abstract hooks confine that check to a deliberate [backend](../../../../glossary.md#backend) implementation point.
The hot call has no [graph](../../../../glossary.md#graph), [planning](../../../../glossary.md#term-planning), routing, [lookup](../../../../glossary.md#term-lookup), selection, or cast input.

`PreparedUnit` is deliberately not introduced. No current schedule, [partition](../../../../glossary.md#partition)-finalization, or
input/output-role consumer establishes a distinct [invariant](../../../../glossary.md#term-invariant) for it. Adding it now would only wrap
one executable or duplicate its selections. [Prepare](../../../../glossary.md#prepare) 0002 or [Runtime](../../../../glossary.md#term-runtime) 0005 must justify and define
that association when [prepared partition](../../../../glossary.md#prepared-partition--preparedpartition) or schedule construction is current.

## [Scope](../../../../glossary.md#term-scope)

- Add the exact two top-level public abstract classes and two nested public selection records
  shown above in `io.github.pho001.synaptik.runtime.execution`.
- Make [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) retain the exact non-null [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) and private immutable
  array snapshots of ordered buffer and workspace selections.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Interpret `bufferIndex` and `workspaceIndex` as dense zero-based positions in
  `PreparedMemoryPlan.buffers()` and `PreparedMemoryPlan.workspaces()` encounter order, never as
  slot numeric components.
- Interpret `representationIndex` as the dense zero-based position in the selected run state's
  ordered bindings for that prepared buffer position.<span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Permit empty selection lists and repeated selections. Repetition is required for repeated
  operand roles and does not duplicate ownership in <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>[`RunState`](../../../../glossary.md#run-state--runstate).
- Validate prepared-plan-relative selection bounds at executable construction and run-dynamic
  representation bounds at cold binding.
- Require exact `PreparedMemoryPlan` [reference identity](../../../../glossary.md#term-reference-identity) between the recipe and `RunState`; equal
  geometry from another plan is not the same association.
- Resolve selected [nominal](../../../../glossary.md#term-nominal) representations into fresh cold-path arrays in selection order.
- Let [concrete backend](../../../../glossary.md#concrete-backend) hooks perform explicit `instanceof` compatibility checks without
  reflection, raw `Object`, unchecked generic access, or a public concrete-[backend](../../../../glossary.md#backend) switch.
- Standardize incompatibility diagnostics in the final [Runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding) method.
- Require `bindCompatible` to create a backend-owned [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) that retains the exact
  supplied `RunState` and direct concrete typed references, not the nominal selection arrays.
- Check the exact [run](../../../../glossary.md#term-run)-state association of the returned invocation.
- [Guard](../../../../glossary.md#term-guard) `BoundInvocation.execute()` with one minimal open-state check before delegating to the
  backend's `executeBound()` implementation.
- Add focused tests for exact surface, validation/failure order, immutable snapshots, selection
  order, exact reference retention, checked compatibility, lifecycle, direct typed binding,
  concurrency contract, and hot-path exclusions.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Finalize all affected Javadocs and explanatory documentation in the required separate clean
  documentation-focused context.

## Out of [scope](../../../../glossary.md#term-scope)

- `PreparedUnit`, [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), Prepare assignment/finalization, or source-to-slot
  associations
- input/output/result role [metadata](../../../../glossary.md#term-metadata) or schedule-step [identity](../../../../glossary.md#term-identity-meanings)
- [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), transfer, [materialization](../../../../glossary.md#materialization), [residency](../../../../glossary.md#residency)/validity mutation, [publication](../../../../glossary.md#term-publication),
  [`RunResult`](../../../../glossary.md#run-result--runresult), [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), or a runner
- representation creation, physical allocation, storage access mechanics, transfer mechanics, or
  ownership changes<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- auxiliary [binding](../../../../glossary.md#term-binding) resources, native-handle acquisition, or an `AutoCloseable` invocation
  lifecycle<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [immutable](../../../../glossary.md#term-immutable) persistent prepared-resource [ownership](../../../../glossary.md#term-ownership) or closing [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable)
- [concrete backend](../../../../glossary.md#concrete-backend) production classes, [kernels](../../../../glossary.md#kernel), routes, native bridges, or [backend](../../../../glossary.md#backend) configuration
- [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Model](../../../../glossary.md#term-model), [Engine](../../../../glossary.md#term-engine), [`Operation`](../../../../glossary.md#operation), `CompiledNode`, partition, trace,
  backend identity, or device parameters on <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [trace glossary](../../../../glossary.md#trace); [backend identity glossary](../../../../glossary.md#backend-identity--backendid); [parameters glossary](../../../../glossary.md#term-parameter))</span>`bind` or `execute`
- allocation, transfer, residency, route/config search, lowering, kernel selection, backend
  discovery, fallback, service lookup, tuning, profiling, or trace emission<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- maps, registries, service locators, string [dispatch](../../../../glossary.md#term-dispatch), reflection in production, raw `Object`,
  unchecked casts/generics, or public switches over concrete backend types
- [concurrency](../../../../glossary.md#term-concurrency) support for one [`RunState`](../../../../glossary.md#run-state--runstate) or one [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation)
- [pooling](../../../../glossary.md#term-pooling), [aliasing](../../../../glossary.md#term-aliasing), hidden coherence/write-back, distributed sharding, or multi-[device](../../../../glossary.md#term-device) scheduling
- [Runtime](../../../../glossary.md#term-runtime) 0005+, Prepare 0002, or any other detailed task specification
- Gradle, dependency, architecture-contract, ADR, architecture-test, backend-conformance, or
  integration-test changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [Runtime](../../../../glossary.md#term-runtime) [service locator](../../../../glossary.md#term-service-locator)
  - [Prepare](../../../../glossary.md#prepare) and [run](../../../../glossary.md#term-run) lifecycles
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership); [binding glossary](../../../../glossary.md#term-binding))</span>
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0006: No Runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Runtime master plan](../master-plan.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)

## Architecture constraints

- [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) is immutable reusable prepared state and may bind concurrently to distinct
  open [`RunState`](../../../../glossary.md#run-state--runstate) instances. Concrete subclasses must also be [immutable](../../../../glossary.md#term-immutable) and [thread-safe](../../../../glossary.md#term-thread-safe).
- Every [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) belongs to exactly one `RunState` and one logical run. It is not
  thread-safe; callers must not execute it concurrently or race execution with state closure.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [race glossary](../../../../glossary.md#term-race))</span>
- A [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation) does not own or close its `RunState`, buffer representations, workspace
  representations, or immutable prepared resources.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- Runtime owns common plan/selection/lifecycle validation. Concrete backends own representation
  implementations, checked type compatibility, typed direct-reference fields, and region work.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle); [validation glossary](../../../../glossary.md#term-validation); [backends glossary](../../../../glossary.md#backend))</span>
- Dynamic compatibility checks occur only during `bind`. `execute()` performs only the minimal
  run-open guard followed by the backend call.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>
- Runtime remains independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backends. Backend Contract remains closed.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- No implementation choice, backend lookup, route choice, allocation, transfer, or publication
  may be deferred into <span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [publication glossary](../../../../glossary.md#term-publication))</span>`execute()`.
- If exact compatibility requires a shared registry, raw/untyped carrier, reflective class token,
  unchecked generic API, or another module edge, stop and report an architecture conflict.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [carrier glossary](../../../../glossary.md#term-storage-carrier); [token glossary](../../../../glossary.md#term-token); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.runtime.memory` — consume the exact [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) reference and
  its dense entry counts without modification.
- `io.github.pho001.synaptik.runtime.resource` — consume nominal buffer/workspace representation
  roles without adding physical access.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- `io.github.pho001.synaptik.runtime.run` — consume [`RunState`](../../../../glossary.md#run-state--runstate); update only its Javadoc and package
  status wording for the now-current binding lifecycle.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [binding glossary](../../../../glossary.md#term-binding); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Package added:

- `io.github.pho001.synaptik.runtime.execution` — public prepared executable and per-run bound
  invocation contracts.<span data-glossary-navigation="true"> ([prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [run glossary](../../../../glossary.md#term-run); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>

Type placement:

- `runtime.execution.PreparedExecutable` — Runtime owns the backend-neutral reusable recipe and
  final cold-binding template; concrete backend modules subclass it.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
- `PreparedExecutable.BufferSelection` — nested because the dense buffer/representation pair is
  meaningful only to executable binding.
- `PreparedExecutable.WorkspaceSelection` — nested for the corresponding workspace position.
- `runtime.execution.BoundInvocation` — Runtime owns the common per-run lifecycle guard while a
  concrete backend subclass owns typed fields and execution.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>

Tests mirror `runtime.execution`. No root [facade](../../../../glossary.md#term-facade) or other Java package is added.

## Affected files

Expected [Runtime](../../../../glossary.md#term-runtime) production paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutable.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/BoundInvocation.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java` — [Javadoc](../../../../glossary.md#term-javadoc) only
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/package-info.java` —
  current-status wording only

Expected Runtime test paths:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutableTest.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/execution/BoundInvocationTest.java`

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status clarification
  only; no architecture rule change
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs 0006/0010/0011,
[lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/overview docs, Runtime 0001–0003, [Prepare](../../../../glossary.md#prepare) 0001/master plan, all current
Runtime/Prepare source/tests/generated Javadocs/build files, Config/[Trace](../../../../glossary.md#trace)/[Engine](../../../../glossary.md#term-engine)/[Backend](../../../../glossary.md#backend) Contract
and concrete-backend plans, backend [partition](../../../../glossary.md#partition)-preparer guide, and architecture/[conformance](../../../../glossary.md#term-conformance)/
integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 15 paths:

- 5 [Runtime](../../../../glossary.md#term-runtime) production/[Javadoc](../../../../glossary.md#term-javadoc) paths;
- 2 Runtime test paths;
- 5 explanatory documentation paths; and
- 3 Runtime/global [planning](../../../../glossary.md#term-planning) paths.

No Java/test path outside Runtime, [Gradle](../../../../glossary.md#term-gradle) path, [architecture contract](../../../../glossary.md#architecture-contract), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), architecture-test,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration path may change. Stop if another public type, package, [module](../../../../glossary.md#term-module)
edge, executable behavior [owner](../../../../glossary.md#term-owner), or path is required. Do not create a later task specification.

## [Validation](../../../../glossary.md#term-validation), ordering, and failure rules

Selection constructors validate components in declaration order:

- negative `BufferSelection.bufferIndex`:
  `IllegalArgumentException("bufferIndex must be non-negative")`;
- negative `BufferSelection.representationIndex`:
  `IllegalArgumentException("representationIndex must be non-negative")`; and
- negative `WorkspaceSelection.workspaceIndex`:
  `IllegalArgumentException("workspaceIndex must be non-negative")`.

[`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) construction validates in this exact order:

1. require `memoryPlan`, then `bufferSelections`, then `workspaceSelections` non-null;
2. [scan](../../../../glossary.md#term-cumulative-scan) [buffer](../../../../glossary.md#term-buffer) selections in supplied order, rejecting the first null entry and then the first
   `bufferIndex` outside `memoryPlan.buffers()`;
3. [snapshot](../../../../glossary.md#term-snapshot) buffer selections into a private array;
4. scan workspace selections in supplied order, rejecting the first null entry and then the first
   `workspaceIndex` outside `memoryPlan.workspaces()`; and
5. snapshot workspace selections into a private array.

Exact construction failures are:

- `NullPointerException("memoryPlan")`, `NullPointerException("bufferSelections")`, or
  `NullPointerException("workspaceSelections")`;
- `NullPointerException("bufferSelections[i]")` or
  `NullPointerException("workspaceSelections[i]")`;
- `IllegalArgumentException("bufferSelections[i].bufferIndex out of prepared-plan range: X")`;
  and
- `IllegalArgumentException("workspaceSelections[i].workspaceIndex out of prepared-plan range: X")`.

The executable retains the exact plan and selection objects but not caller list containers.
Selection order is [argument](../../../../glossary.md#term-argument) order for both compatibility hooks and `bindCompatible`. Empty lists
and repeated selections are valid. Construction allocates only private [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) arrays and performs no
physical resource [operation](../../../../glossary.md#operation) or identifier allocation/consumption.

`bind` validates in this exact order:

1. require `runState` non-null;
2. reject a closed state with `IllegalStateException("run state is closed")`;
3. require `runState.memoryPlan() == memoryPlan`, otherwise throw
   `IllegalArgumentException("run state memory plan does not match prepared executable memory plan")`;
4. resolve buffer selections in order; reject the first representation index outside the selected
   run-state buffer with
   `IllegalArgumentException("bufferSelections[i].representationIndex out of run-state range: X")`;
5. invoke `acceptsBufferRepresentation(i, representation)` exactly once for each resolved buffer;
   a false result fails with
   `IllegalArgumentException("bufferSelections[i] is incompatible with prepared executable")`;
6. resolve workspaces in order and invoke
   `acceptsWorkspaceRepresentation(i, representation)` exactly once; a false result fails with
   `IllegalArgumentException("workspaceSelections[i] is incompatible with prepared executable")`;
7. invoke `bindCompatible` exactly once with the exact run state and fresh nominal arrays in the
   original selection order;<span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate))</span>
8. reject a null result with `NullPointerException("boundInvocation")`; and
9. require the returned invocation to retain that exact [run](../../../../glossary.md#term-run)-state reference, otherwise throw
   `IllegalArgumentException("bound invocation does not belong to supplied run state")`.

The [backend](../../../../glossary.md#backend) compatibility hooks must use explicit checked type tests such as `instanceof`. A
normal incompatibility returns false so [Runtime](../../../../glossary.md#term-runtime) emits the exact indexed diagnostic. The backend
must not use reflection, class-name strings, raw `Object`, unchecked casts/generics, a [registry](../../../../glossary.md#term-registry),
or a shared/concrete-backend switch. `bindCompatible` may use ordinary checked casts after the
successful compatibility pass, but it must create an invocation with concrete typed fields and
must not retain the [nominal](../../../../glossary.md#term-nominal) arrays as its hot-path access mechanism.

[Binding](../../../../glossary.md#term-binding) may allocate ordinary JVM arrays and the [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation) object. This task deliberately
forbids acquisition of an independently closeable/native auxiliary binding resource. Therefore a
failed bind changes no [ownership](../../../../glossary.md#term-ownership-meanings) and needs no cleanup protocol. A future task must add an explicit
`AutoCloseable` [lifecycle](../../../../glossary.md#lifecycle) and partial-failure cleanup before permitting such resources.

[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) construction requires `runState` non-null, checks it is open, and retains the
exact reference. Exact failures are `NullPointerException("runState")` and
`IllegalStateException("run state is closed")`.

`execute()` first checks the retained state and fails with
`IllegalStateException("run state is closed")` before any backend call. Otherwise it invokes
`executeBound()` exactly once and returns normally, or propagates the backend's exact
`RuntimeException` or `Error` without fallback, wrapping, cleanup, or retry. Sequential calls while
the state remains open are permitted; no one-shot/idempotence promise is made. Concurrent calls
or a [race](../../../../glossary.md#term-race) with `RunState.close()` are unsupported because both objects belong to one
single-orchestrator-thread run.

## Hot-path and [lifecycle](../../../../glossary.md#lifecycle) contract

- [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) and its selection snapshots are immutable. Concrete subclasses must retain
  only immutable prepared state and may be bound concurrently to different run states.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run))</span>
- [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) strongly retains its exact [`RunState`](../../../../glossary.md#run-state--runstate); retaining the Java object does not keep
  its representations logically open after `RunState.close()`.
- The invocation never owns or closes the state or any representation.
- Each successful [backend](../../../../glossary.md#backend) subclass stores direct concrete typed references during cold [binding](../../../../glossary.md#term-binding).
- `execute()` accepts no argument and performs no array/list/map/slot lookup, representation
  selection, compatibility test/cast, graph inspection, backend discovery, lowering, kernel or
  route selection, configuration search, allocation, transfer, residency decision, publication,
  tuning, profiling selection, or tracing emission.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [lookup glossary](../../../../glossary.md#term-lookup); [graph glossary](../../../../glossary.md#graph); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication))</span>
- The only shared hot-path work is one `RunState.isClosed()` [guard](../../../../glossary.md#term-guard) and one direct virtual call to
  `executeBound()`.

## Acceptance criteria

- The exact two-top-level/two-nested public surface exists only in `runtime.execution`; no
  `PreparedUnit` or other execution type is added.
- Constructors, final template methods, protected hooks, nested records, modifiers, component
  types, and declared exceptions match this task exactly.
- Prepared selections are immutable private-array snapshots in deterministic supplied order,
  retain exact selection references, and use dense positions rather than slot numeric values.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Construction and [binding](../../../../glossary.md#term-binding) follow the exact [validation](../../../../glossary.md#term-validation) order and messages above.
- Binding requires exact plan identity, rejects closed state, validates run-dynamic representation
  indices, checks compatibility once per selected representation, and validates exact invocation
  association before returning.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [run glossary](../../../../glossary.md#term-run))</span>
- Focused fake [backend](../../../../glossary.md#backend) tests prove explicit checked `instanceof` compatibility and direct concrete
  buffer/workspace fields in the bound subclass. No concrete backend production code is added.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Bound execution checks state closure before backend work, performs no resource lookup/cast, and
  propagates backend unchecked failures unchanged.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Tests prove one prepared recipe binds concurrently to distinct states while each invocation and
  run-owned resource remains isolated; the test does not concurrently execute one invocation.
- Production uses no map, reflection, string [dispatch](../../../../glossary.md#term-dispatch), raw `Object`, unchecked generic access,
  registry, service locator, public concrete-backend switch, graph/compiler/planning/Prepare/Model/
  Engine type, concrete backend import, or identifier side effect.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- No allocation/access mechanics, validity/residency, transfer, schedule, runner, publication,
  result, persistent-resource lifecycle, tracing emission, tuning, backend, Prepare, Gradle,
  dependency, or architecture behavior is added.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Every public/protected contract has complete Javadoc for inputs, result, failure, ownership,
  reference retention, immutability, lifecycle, thread safety, selection order, hot-path behavior,
  and unsupported behavior.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- A separate clean documentation-focused agent finalizes affected Java/package Javadocs, Runtime
  and Public API status, focused boundary status, backend guide, glossary, planning evidence, and
  reasoned no-change conclusions in the same overall change.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Runtime 0001–0003 and Prepare 0001 remain Complete; Runtime 0004 becomes Complete only after all
  gates; Runtime 0005+ and Prepare 0002 remain Draft without detailed specifications.
- Exact 15-path [scope](../../../../glossary.md#term-scope), Markdown/link/anchor/fence/newline/whitespace, and `git diff --check` gates
  pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused validation:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutableTest \
  --tests io.github.pho001.synaptik.runtime.execution.BoundInvocationTest

./gradlew :modules:runtime:test
```

[Run](../../../../glossary.md#term-run) the focused command while developing and one final [module](../../../../glossary.md#term-module) command after executable Java
stabilizes.

Documentation-focused pass after final Javadocs and documentation:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md \
  docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/writing-a-backend.md \
  docs/glossary.md \
  docs/planning/modules/runtime/tasks/0004-prepared-executable-and-bound-invocation.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. Validate local link [targets](../../../../glossary.md#term-target) and heading anchors, unique effective anchors, balanced
backtick and tilde fences, final newlines, and trailing whitespace.

Required source/[scope](../../../../glossary.md#term-scope)/status checks:

- exact public/protected/nested surface and package placement;
- exact validation order/messages, array snapshots, selection order, reference retention, plan
  identity, compatibility call counts, invocation association, and lifecycle behavior;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- fake [backend](../../../../glossary.md#backend) bound objects have direct concrete typed fields and execute without resource access;
- production mechanism/import scan for no maps, reflection, raw/unchecked access, strings used for
  dispatch, registries/service locators, concrete backends, or forbidden graph/lifecycle types;<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph))</span>
- unchanged [Runtime](../../../../glossary.md#term-runtime)/root [Gradle](../../../../glossary.md#term-gradle) files and Java 26 root [toolchain](../../../../glossary.md#term-toolchain)/[release](../../../../glossary.md#term-release) configuration;
- exact 15-path ceiling and no Java/test outside Runtime;
- Runtime 0001–0004 and [Prepare](../../../../glossary.md#prepare) 0001 status [synchronization](../../../../glossary.md#term-synchronization) after completion;
- Runtime 0005–0008 and Prepare 0002 remain Draft, exactly Runtime 0001–0004 have specs, exactly
  Prepare 0001 has a spec, and no later spec exists; and
- final newlines, trailing whitespace, and `git diff --check`.

Repository-wide tests and architecture tests are deferred to the Runtime prepared-contract
capability [checkpoint](../../../../glossary.md#term-checkpoint) or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one module without a module
edge, [dependency](../../../../glossary.md#term-dependency) rule, shared build contract, [concrete backend](../../../../glossary.md#concrete-backend), or end-to-end behavior.
Backend-[conformance](../../../../glossary.md#term-conformance) and integration tests are not applicable yet.

The documentation context reuses successful Runtime Java-test evidence unless it changes
executable behavior or records a concrete reason to repeat it.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0001–0003 — Complete.
- [Prepare](../../../../glossary.md#prepare) 0001 — Complete but not imported by Runtime.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010 staged preparation — Accepted.
- ADR 0011 per-[run](../../../../glossary.md#term-run) [ownership](../../../../glossary.md#term-ownership-meanings) and checked cold [binding](../../../../glossary.md#term-binding) — Accepted and sufficient for this task.
- Existing Runtime [dependency](../../../../glossary.md#term-dependency) and Java 26 build contracts — unchanged.

Prepare 0002, Runtime 0005+, concrete [backends](../../../../glossary.md#backend), schedules, [publication](../../../../glossary.md#term-publication)/results, transfer,
[residency](../../../../glossary.md#residency), allocation, and [Engine](../../../../glossary.md#term-engine) are not dependencies of this bounded contract task.

## Follow-up tasks

- Prepare 0002 may use this executable contract when it defines shared assignment associations and
  backend finalization against assigned slots. It remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Runtime](../../../../glossary.md#term-runtime) 0005 must decide whether a distinct `PreparedUnit` is justified by the actual prepared
  partition/schedule consumer and then define ordered schedule work. It must not be created here.<span data-glossary-navigation="true"> ([prepared partition glossary](../../../../glossary.md#prepared-partition--preparedpartition))</span>
- Runtime 0006–0008 remain Draft for prepared aggregate/lifecycle, runner/dynamic execution, and
  closure audit.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Javadocs and documentation impact

- New execution-package Javadocs must explain the cold-versus-hot boundary, ordered dense
  selections, exact plan/run association, checked compatibility, direct typed references,
  lifecycle, threading, failures, and all deliberate exclusions.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [`RunState`](../../../../glossary.md#run-state--runstate) and `runtime.run` package Javadocs need current wording for binding and post-close
  invocation rejection; executable behavior in <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`RunState` does not change.
- [Runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api) must make [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), and their fake-backend example
  current while preserving schedules, allocation, residency, transfer, publication/results, and
  runner as planned.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication))</span>
- Public API status must list the new Runtime execution contracts without claiming a runnable
  public lifecycle.
- The focused architecture page receives implementation-status changes and a concrete current
  cold-binding explanation only. It must not alter the architecture rule.
- The backend guide must show explicit checked compatibility and direct concrete fields without
  suggesting a current concrete backend or Prepare finalizer.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [Prepare glossary](../../../../glossary.md#prepare))</span>
- The glossary makes `PreparedExecutable` current, adds `BoundInvocation`, and updates `RunState`
  current limits. No ordinary programming term is added.
- Lifecycle, module-boundary, dependency, overview, ADRs, other API/user guides, backend partition
  analysis, Config/Trace/Engine/backend plans, architecture tests, conformance, integration, and
  Gradle remain review-only unless the implementation reveals a concrete contradiction.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [partition glossary](../../../../glossary.md#partition); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Architecture impact

Expected impact: None.

[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0011 and the [architecture contract](../../../../glossary.md#architecture-contract) already authorize the exact checked cold-[binding](../../../../glossary.md#term-binding) boundary,
[backend](../../../../glossary.md#backend)-owned typed invocation objects, direct references, one-[run](../../../../glossary.md#term-run) association, and hot-path
exclusions. This task introduces no [module](../../../../glossary.md#term-module) edge or rule. Stop if implementation requires a shared
resource [registry](../../../../glossary.md#term-registry), generic catch-all abstraction, unchecked public [API](../../../../glossary.md#term-application-programming-interface-api), concrete-backend knowledge
inside [Runtime](../../../../glossary.md#term-runtime), execution after run closure, or auxiliary-resource [ownership](../../../../glossary.md#term-ownership) without an explicit
[lifecycle](../../../../glossary.md#lifecycle).

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, ADRs 0006/0010/0011, the focused
Runtime/Prepare/backend lifecycle, module, dependency, and boundary docs, documentation rules and
General/API-Javadoc/Architecture/Backend-Guide/Planning profiles, the Runtime and Prepare master
plans, Runtime 0001–0003, Prepare 0001, current Runtime/Prepare source/tests/build/generated
Javadocs, Runtime/Public APIs, backend guide, glossary, and
docs/planning/modules/runtime/tasks/0004-prepared-executable-and-bound-invocation.md.

Implement Runtime 0004 exactly within its two-top-level/two-nested surface and 15-path ceiling.
Add only immutable prepared executable selections, final checked cold binding, the per-run
backend-owned bound-invocation lifecycle guard, focused fake-backend tests/Javadocs, and the
specified documentation/status updates. Do not add PreparedUnit, Prepare finalization, allocation,
auxiliary binding resources, residency, transfer, schedule/runner, publication/result, concrete
backend, tracing/tuning, dependency/Gradle/architecture changes, or later task specs. Stop on any
architecture, package, API, validation, or scope conflict.

Run the focused tests, one final Runtime module test, and all source/scope/status checks. Then hand
the actual diff and exact Java evidence to a separate documentation-focused clean context. That
pass must follow documentation-rules.md, independently finalize affected Javadocs/docs/glossary/
planning evidence, and not repeat Java tests unless executable behavior changes or a concrete risk
is recorded. Mark Complete only after every implementation and documentation gate passes. Return
both context IDs, exact paths, commands/results/counts, unresolved issues, follow-up, and the
repository completion status format.
```

## Local decisions

- Use abstract Runtime template classes rather than interfaces so common plan identity,
  selection, invocation association, and closed-state rules are final and cannot drift across
  backends.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend))</span>
- Use two nested immutable selection records and private arrays. This gives prepared finalization
  explicit dense coordinates without maps, boxing-dependent lookup, slot-value-sized arrays, or a
  redundant <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lookup glossary](../../../../glossary.md#term-lookup))</span>`PreparedUnit`.
- Allow repeated selections because one prepared region may consume the same physical
  representation in more than one logical argument role. <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>[`RunState`](../../../../glossary.md#run-state--runstate) still owns each exact
  representation only once.
- Keep compatibility hooks boolean and backend-owned. Runtime owns exact failure diagnostics while
  concrete code uses <span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>`instanceof` and constructs a typed invocation after all checks pass.
- Require exact [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) object [identity](../../../../glossary.md#term-reference-identity). `RunState` and the executable must refer to
  the same prepared recipe; structural equality is insufficient across independently constructed
  plans.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Retain a minimal execute-time state guard. Cold validation alone cannot prevent a previously
  bound invocation from running after its state closes, and one boolean check adds no graph/map/
  resource lookup.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [validation glossary](../../../../glossary.md#term-validation); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [graph glossary](../../../../glossary.md#graph))</span>
- Permit sequential calls while open but prohibit concurrent execution/races. Later schedule work
  owns normal call cardinality.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Defer auxiliary binding resources completely. The bound invocation owns no closeable state, so
  no <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`AutoCloseable` wrapper or partial-failure cleanup protocol is justified now.

## Known limitations

- No current shared contract labels buffer selections as input, output, or in-place roles.
  Prepare 0002/Runtime 0005 must introduce only the associations their concrete consumers need.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) does not describe representation counts or kinds. The immutable executable
  records a non-negative representation index, and binding validates it against each actual
  matching <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [binding glossary](../../../../glossary.md#term-binding))</span>[`RunState`](../../../../glossary.md#run-state--runstate) before [backend](../../../../glossary.md#backend) compatibility.
- Abstract classes document but cannot mechanically prove that every backend subclass is
  immutable, uses `instanceof` correctly, or stores only direct concrete fields. Focused fake
  backend tests lock the intended extension pattern; future concrete backends require their own
  unit and conformance tests.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- No schedule, allocation, validity/residency, transfer, publication, result, runner, persistent-
  resource cleanup, or concrete backend behavior exists.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Repository-wide and architecture-test validation remains deferred to the Runtime prepared-
  contract checkpoint or continuous integration because no dependency or architecture rule
  changes.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/implement_runtime_0004` ran the focused command:

  ```bash
  ./gradlew :modules:runtime:test \
    --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutableTest \
    --tests io.github.pho001.synaptik.runtime.execution.BoundInvocationTest
  ```

  It passed two suites and 18 tests (`PreparedExecutableTest` 13 and
  `BoundInvocationTest` 5) with zero failures, errors, and skips.
- The same implementation context ran the single final affected-module command after executable
  Java stabilized:<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

  ```bash
  ./gradlew :modules:runtime:test
  ```

  It passed eight suites and 63 tests with zero failures, errors, and skips. Documentation context
  `/root` changed Javadocs and documentation but no executable Java, so it reused both successful
  results and did not repeat either Java test command.
- Documentation context `/root` applied the General, API/Javadoc, Architecture, Backend Guide,
  Planning, and Example profiles. It independently reviewed the architecture contract,
  documentation rules, Planning Guide and roadmap, Runtime/Prepare master plans, ADRs
  0006/0010/0011, Runtime tasks 0001–0004, Prepare 0001, final Runtime production/tests, affected
  predecessor Javadocs, Runtime/Public APIs, focused boundary explanation, backend guide,
  glossary, build/toolchain files, and the exact diff.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning); [architecture contract glossary](../../../../glossary.md#architecture-contract); [Prepare glossary](../../../../glossary.md#prepare); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- Final [Runtime](../../../../glossary.md#term-runtime) [Javadoc](../../../../glossary.md#term-javadoc) generation passed:

  ```bash
  ./gradlew :modules:runtime:javadoc
  ```

  [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL`; five tasks were actionable, two executed and three
  up-to-date. The generated execution, run-state, and package pages expose the exact plan identity,
  dense indices, repeated-selection, checked compatibility, direct typed-field, run association,
  post-close guard, ownership, lifecycle, thread-safety, failure, and hot-path contracts.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings); [guard glossary](../../../../glossary.md#term-guard); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The changed Runtime [API](../../../../glossary.md#term-application-programming-interface-api) example compiled and ran with Java 26:

  ```bash
  javac --release 26 -cp modules/runtime/build/classes/java/main \
    -d /tmp/runtime-0004-api-example /tmp/Runtime0004ApiExample.java
  java -cp modules/runtime/build/classes/java/main:/tmp/runtime-0004-api-example \
    Runtime0004ApiExample
  ```

  Both commands passed with no output. The example checks direct typed execution exactly once and
  the exact post-close failure. The backend-guide skeleton also compiled with
  `javac --release 26` against the Runtime main classes.
- Targeted Markdown validation passed for all eight changed documentation/planning files. It
  checked local targets and heading anchors, unique effective anchors, balanced backtick and tilde
  fences, final newlines, and trailing whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- Final source/API checks confirmed exactly two top-level public abstract classes and two nested
  public records in `runtime.execution`, the planned constructors/modifiers/hooks, exact messages
  and validation order, private array snapshots, selection order, exact reference retention,
  exact plan identity, one compatibility call per selected representation, invocation
  association, and the post-close lifecycle guard. <span data-glossary-navigation="true"> ([close lifecycle glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>`javap -p` confirmed the compiled surface;
  `javap -p -c BoundInvocation` confirmed the hot method performs one `RunState.isClosed()` call
  followed by `executeBound()`.
- Production imports are limited to current Runtime contracts plus `java.util.List` and
  `java.util.Objects`. Source and bytecode checks found no map, reflection, string dispatch, raw
  or unchecked access, registry/service locator, concrete backend, graph/compiler/planning/
  Prepare/Model/Engine type, or forbidden hot-path operation.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [concrete backend glossary](../../../../glossary.md#concrete-backend); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine); [operation glossary](../../../../glossary.md#operation))</span>
- Root/settings/Runtime Gradle files are unchanged. Root configuration alone retains Java 26
  toolchain and release settings, and Runtime dependencies remain Config, Backend Contract, and
  Trace only.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [Trace glossary](../../../../glossary.md#trace))</span>
- The final scope audit found exactly the 15 authorized paths: five Runtime production/Javadoc
  paths, two Runtime test paths, five explanatory documentation paths, and three planning paths.
  No other Java/test, Gradle, architecture/ADR/test, backend-conformance, integration, module, or
  later-specification path changed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Status/specification checks confirmed Runtime 0001–0004 and Prepare 0001 are Complete; Runtime
  0005–0008 and Prepare 0002 remain Draft; exactly Runtime 0001–0004 and Prepare 0001 have detailed
  specifications; and no later specification exists.
- `git diff --check` passed after final documentation and status [synchronization](../../../../glossary.md#term-synchronization) with no output.
- No-change conclusions:
  - `ARCHITECTURE.md` and ADRs 0006/0010/0011 already authorize the exact cold-binding boundary,
    backend-owned typed invocation, resource ownership, and service-locator prohibition; no
    architecture decision or dependency rule changed.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
  - Runtime 0001–0003 memory/resource contracts and Javadocs remain accurate except for the
    required [`RunState`](../../../../glossary.md#run-state--runstate) and `runtime.run` current-[binding](../../../../glossary.md#term-binding) wording finalized here.
  - Prepare 0001 and its partition-preparer guide remain analysis-only and unchanged; Prepare
    0002 still owns assignment/finalization and has no detailed specification.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
  - Compile, Tensor, Config, Trace, Engine, user-guide, lifecycle, overview, module-boundary, and
    dependency documents need no change because this task adds none of their public workflows,
    semantics, configuration, tracing, composition, or architecture rules.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor))</span>
  - Architecture tests need no update because module edges and tested dependency rules are
    unchanged. Backend-conformance and integration tests remain inapplicable because there is no
    production concrete backend or end-to-end runner.
  - Root and Runtime Gradle files need no change because the existing module and Java 26 contracts
    compile the new surface without another dependency or override.

## Implementation notes

- Added only [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), and execution-package documentation plus
  the two focused fake-backend suites in the planned package.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- The executable snapshots ordered dense selections, retains exact plan/selection references,
  requires exact plan identity during binding, performs buffer-then-workspace checked
  compatibility, and validates the returned invocation's exact state association.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- The bound invocation strongly retains but does not own its state, rejects execution after
  closure, and delegates directly to backend work without resource lookup or rebinding.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Finalized [Runtime](../../../../glossary.md#term-runtime) 0004 execution Javadocs, affected [`RunState`](../../../../glossary.md#run-state--runstate)/package wording, Runtime/Public
  APIs, focused boundary status, backend integration guidance, glossary terminology, and planning
  evidence. No executable Java changed during the documentation pass.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: implemented and documented the exact immutable prepared-executable recipe,
  ordered dense cold-binding selections, checked backend compatibility hooks, and per-run direct-
  reference bound invocation lifecycle guard.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [run glossary](../../../../glossary.md#term-run); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [guard glossary](../../../../glossary.md#term-guard))</span>
- Files changed or created: exactly the 15 authorized paths—five Runtime production/Javadoc
  paths, two Runtime tests, five explanatory documentation paths, and three planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the implementation context's successful focused 18-test and final
  63-test Runtime evidence; Runtime Javadoc, two Java 26 example compilations and one execution,
  generated/API/bytecode/import/mechanism/build/toolchain, eight-file Markdown, exact scope/status/
  later-specification, final-newline/fence/whitespace, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [bytecode glossary](../../../../glossary.md#term-bytecode); [toolchain glossary](../../../../glossary.md#term-toolchain); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` gates passed.
- Documentation-agent review: completed in clean documentation context `/root` without executable
  Java changes or duplicate Java-test execution.
- Documentation impact: Runtime/Public APIs, focused architecture status, backend guide, glossary,
  task, master plan, and roadmap now distinguish current cold binding from later Prepare
  finalization, allocation, residency, transfer, scheduling, publication, and runner work.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: finalized the three new execution production/package paths and the affected
  [`RunState`](../../../../glossary.md#run-state--runstate)/`runtime.run` [lifecycle](../../../../glossary.md#lifecycle) wording. [Runtime](../../../../glossary.md#term-runtime) 0001–0003 Javadocs otherwise remain accurate.
- Glossary impact: made [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) current, added [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), and synchronized
  physical-representation and `RunState` cold-[binding](../../../../glossary.md#term-binding) boundaries.
- Unresolved issues: None.
- Follow-up required: None for this task. Prepare 0002 and Runtime 0005 remain Draft pending a
  separate frontier-planning step.

Status: Complete
