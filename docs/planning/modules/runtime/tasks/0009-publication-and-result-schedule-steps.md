# Task 0009: Publication and Result Schedule Steps

## Status

Complete

## Goal

Add the smallest Runtime-owned publication and result-lifetime contracts that let an immutable
prepared schedule name an already-created buffer representation as one stable ordered result.

The capability consists of:

- one immutable `PreparedPublication` recipe using only an exact `PreparedMemoryPlan` reference
  and dense run-state buffer/representation coordinates;
- one per-run `BoundPublication` that retains the exact selected representation directly after
  cold binding and records whether its one publication occurrence completed;
- one `PreparedSchedule.PublicationStep` in a dense ordered publication suffix; and
- one closeable `RunResult` that leases the complete exact `RunState` after every ordered bound
  publication has completed.

Publication performs no physical work. It requires the exact run state to be open and the
selected resident representation to be valid at the moment of publication, then records the
occurrence as published. It performs no representation discovery, route choice, transfer,
materialization, backend callback, allocation, conversion, copy, or fallback. When a result needs
another representation, an explicit Runtime 0008 `BufferTransferStep` must precede publication.

## Rationale and mental model

```text
Compiler publication roles and graph values
  -> Prepare translates them before Runtime
  -> Runtime PreparedPublication(resultIndex, bufferIndex, representationIndex)

immutable prepared schedule
  -> creation / execution / transfer occurrences
  -> dense publication suffix: result 0, result 1, ...

one open exact-plan RunState
  -> cold bind each PreparedPublication once
  -> BoundPublication retains the exact physical representation directly
  -> publish only if that selected copy is valid
  -> RunResult leases the complete RunState
  -> RunResult.close() closes resources still owned by that state
```

The prepared recipe says *which existing copy* and *which ordered result position*. The bound
object is the per-run publication state. `RunResult` is the completed lifetime owner, but it adds
no public value-reading or export surface. Keeping the whole `RunState` leased preserves the
implemented ownership matrix, duplicate-identity protection, reverse cleanup, borrowed-input
rules, and one-state-per-complete-run invariant without adding mutable individual-resource
ownership transfer to `BufferRepresentationBinding`.

The Runtime result order is a flat dense order. Prepare later projects the compiler's ordered
forward bindings followed by ordered gradient bindings into that order. Runtime never sees or
imports `ForwardPublicationBinding`, `GradientPublicationBinding`, `PublicationPlan`, `TensorId`,
`ValueId`, graph facts, Planning types, or Prepare types.

## Exact API

Add these three public final classes in `io.github.pho001.synaptik.runtime.run`:

```java
public final class PreparedPublication {
    public PreparedPublication(
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int representationIndex,
            int resultIndex);

    public PreparedMemoryPlan memoryPlan();

    public int bufferIndex();

    public int representationIndex();

    public int resultIndex();

    public BoundPublication bind(RunState runState);
}

public final class BoundPublication {
    public void publish();

    public boolean isPublished();
}

public final class RunResult implements AutoCloseable {
    public RunResult(
            RunState runState,
            List<BoundPublication> publications);

    public int resultCount();

    public boolean isClosed();

    @Override
    public void close();
}
```

`BoundPublication` has no public or protected constructor. `PreparedPublication.bind` is its only
construction path. It has package-private final accessors used only by `RunResult` to validate
and snapshot the completed association:

```java
final RunState runState();

final PreparedPublication publication();

final BufferRepresentation representation();
```

Extend the existing schedule family exactly:

```java
public sealed interface Step
        permits ExecutionStep,
                RepresentationCreationStep,
                BufferTransferStep,
                PublicationStep {
    PreparedMemoryPlan memoryPlan();
}

public record PublicationStep(
        PreparedPublication publication) implements Step {
    @Override
    public PreparedMemoryPlan memoryPlan();
}

public int publicationCount();
```

`publicationCount()` returns the number of `PublicationStep` occurrences. It introduces no new
record component or stored mutable cache. `PreparedSchedule` remains the existing two-component
record. `PreparedExecution` remains exactly its existing two components:

```java
public record PreparedExecution(
        PreparedMemoryPlan memoryPlan,
        PreparedSchedule schedule) {}
```

No class adds a builder, factory, overload, interface, nested result value, generic payload,
public representation accessor, public `RunState` accessor, public result accessor, result ID
type, closeable publication binding, custom equality/hash/text, or serialization contract.

## Scope

- Add the exact API above and no broader public surface.
- Use dense `PreparedMemoryPlan.buffers()` position, dense representation position in the matching
  `RunState`, and dense zero-based result position as the complete Runtime publication identity.
- Make `PreparedPublication` immutable, reusable, and thread-safe.
- Cold-bind one prepared publication against one exact matching open `RunState` and resolve the
  physical representation exactly once.
- Retain that exact physical representation directly in `BoundPublication` so publication itself
  performs no representation lookup.
- Require the selected representation to be valid at the exact publication moment.
- Add a publication-only suffix to `PreparedSchedule`; suffix occurrences must have result
  indices `0..N-1` in encounter order.
- Permit zero results through an empty publication suffix and `RunResult(state, List.of())`.
- Permit distinct ordered result positions to alias the same exact buffer/representation
  coordinate and physical representation.
- Reject repeated publication of one `BoundPublication`; a result position has one occurrence.
- Construct `RunResult` only from a complete dense ordered list of successfully published bound
  occurrences for one exact open state.
- Transfer cleanup responsibility for the complete state to `RunResult` only after its constructor
  validates the complete publication list.
- Preserve borrowed-input ownership and existing `RunState.close()` cleanup order and failure
  behavior.
- Add focused API, validation, order, identity, aliasing, lifecycle, partial-failure, threading,
  bytecode, and forbidden-mechanism tests.
- Finalize affected Javadocs and explanatory documentation in a separate clean documentation
  context after implementation tests stabilize.
- Preserve completed Runtime 0001–0008, Prepare 0001–0002, Compiler 0001–0006, and every existing
  public contract not named here.

## Out of scope

- output value conversion, reading, access, mapping, download, export, host-storage exposure, or
  Tensor construction
- Tensor publication, Tensor or graph identity in Runtime, forward/gradient role classification,
  or public Engine result facade
- `ForwardPublicationBinding`, `GradientPublicationBinding`, `PublicationPlan`, `TensorId`,
  `ValueId`, `CompiledGraphModel`, graph output facts, Planning contracts, or Prepare contracts in
  Runtime
- Config `RunOptions`, `PublicationPolicy`, defaults, filtering, requested-result selection, or
  any other run/publication policy
- a runner, schedule traversal/consumption, caller-input handoff, run-state creation, executable
  binding/execution, transfer binding/execution, or failure orchestration across schedule steps
- executable-output validity or invalidation, write-set contracts, coherence, dirty bits, or
  hidden mutation tracking
- representation discovery, source choice, transfer route search, fallback, multi-hop transfer,
  implicit materialization, lazy creation, allocation, replacement, eviction, pooling, copying,
  conversion, or backend work during publication
- storage allocation/access implementation, concrete backend implementation, native bridge,
  kernel selection, backend discovery, lowering, or prepare orchestration
- individual representation ownership transfer, mutation of `RunResourceOwnership`, detachable
  resources, reference counting, shared result leases, or closing only published outputs
- public result values, labels, names, typed result roles, forward/gradient grouping, or result
  lookup by identity
- tracing, profiling, Config, tuning, measurement, cache behavior, retry, fallback, cancellation,
  barriers, branches, parallel scheduling, or synchronization
- changes to `PreparedExecution`, `PreparedExecutable`, `BoundInvocation`,
  `PreparedBufferTransfer`, `BoundBufferTransfer`, `PreparedRepresentationPlan`,
  `RunStateCreation`, memory geometry, or existing representation validity semantics
- Compiler, Model, Planning, Prepare, Config, Trace, Backend Contract, Engine, concrete-backend,
  Gradle, dependency, architecture-contract, ADR, architecture-test, backend-conformance, or
  integration-test behavior
- detailed Runtime 0010–0011 or Prepare 0003 specifications

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): Runtime ownership, one complete-run state,
  explicit publication, result lease/transfer, hot-path, and dependency rules.
- [ADR 0006: No Runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Public API](../../../../api/public-api.md)
- [Backend guide](../../../../backend-guide/writing-a-backend.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)
- [Runtime 0005](0005-prepared-schedule-contract.md)
- [Runtime 0006](0006-prepared-execution-aggregate.md)
- [Runtime 0007](0007-representation-creation-and-residency-foundation.md)
- [Runtime 0008](0008-prepared-buffer-transfer-and-materialization-schedule.md)

## Architecture constraints

- Runtime publication identities are Runtime-owned dense prepared-plan/run-state coordinates.
- Prepared recipes remain immutable and reusable. Every active logical invocation still has
  exactly one mutable `RunState` for the complete heterogeneous schedule.
- `PreparedExecution` remains exactly memory plan plus schedule. Publication is reachable through
  the schedule and result lifetime begins only after per-run completion.
- Publication names one already-created representation exactly. Structural residency already
  exists; logical validity must be true at publication time.
- If the selected representation is invalid, publication fails deterministically. It performs no
  automatic transfer, source selection, materialization, allocation, conversion, or fallback.
- `PreparedBufferTransfer` remains the only explicit representation-copy/materialization work. A
  required transfer occurrence precedes the publication suffix.
- `BoundPublication` retains a direct representation reference established during cold binding.
  Its `publish()` hot method performs only state-open, one-shot, and dense validity checks plus one
  boolean state transition. It performs no physical-resource lookup or backend call.
- Publication does not change `RunState` validity or `BufferRepresentationBinding` ownership.
- A successfully constructed `RunResult` leases the complete state. Closing it delegates to that
  state, which skips borrowed inputs and releases every still-run-owned representation exactly as
  already implemented.
- Construction failure transfers no cleanup responsibility. Before a result exists, the later
  runner remains responsible for closing the state after any failure.
- One bound publication and one result are not thread-safe. They must not race publication,
  result construction, validity mutation, execution, transfer, or state/result closure.
- Separate runs may bind the same immutable prepared publication to distinct states. Their bound
  publication flags, results, validity, and run-owned resources remain isolated.
- Runtime remains independent of Model, Planning, Compiler, Prepare, Engine, and concrete
  backends. Existing Runtime Gradle dependencies remain unchanged and none is needed by the new
  types.
- If implementation needs graph identity, a new dependency, physical work, individual ownership
  mutation, a third `PreparedExecution` component, or output access, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.runtime.memory` — supplies the exact plan and dense buffer geometry.
- `io.github.pho001.synaptik.runtime.resource` — supplies the nominal selected physical buffer
  representation without modification.
- `io.github.pho001.synaptik.runtime.execution` — remains the owner of prepared executable and
  transfer actions and the unchanged two-component `PreparedExecution`.

Packages changed:

- `io.github.pho001.synaptik.runtime.run` — owns prepared-to-run publication binding, one
  occurrence's per-run publication state, and the completed result lease over the whole run.
- `io.github.pho001.synaptik.runtime.schedule` — owns the ordered publication suffix and its one
  new sealed step variant.

Type placement:

- `io.github.pho001.synaptik.runtime.run.PreparedPublication` — immutable Runtime-only association
  from exact prepared/run coordinates to one stable result position.
- `io.github.pho001.synaptik.runtime.run.BoundPublication` — one cold-bound per-run direct-
  reference occurrence and its one-shot publication state.
- `io.github.pho001.synaptik.runtime.run.RunResult` — completed closeable lease over the exact
  whole-run state and private ordered published-representation snapshot.
- `io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.PublicationStep` — one ordered
  occurrence in the required publication suffix.

Tests mirror `runtime.run` and `runtime.schedule`. No package is added, moved, or removed.

## Prepared publication construction contract

`PreparedPublication` stores exactly four private final fields matching its constructor. It is not
a record because its cold bind operation and identity-based plan association are the contract,
while structural value equality is neither required nor exposed.

Construction validates in this exact order:

1. require `memoryPlan` non-null;
2. require `bufferIndex` non-negative;
3. require `bufferIndex < memoryPlan.buffers().size()`;
4. require `representationIndex` non-negative; and
5. require `resultIndex` non-negative.

Exact failures are:

- `NullPointerException("memoryPlan")`;
- `IllegalArgumentException("bufferIndex must be non-negative")`;
- `IllegalArgumentException("bufferIndex out of prepared-plan range: X")`;
- `IllegalArgumentException("representationIndex must be non-negative")`; and
- `IllegalArgumentException("resultIndex must be non-negative")`.

The four accessors return the exact plan reference and exact primitive values. Construction does
not inspect a run, resolve a representation, query validity, invoke a callback, acquire a
resource, or mutate ownership.

Representation-position bounds remain cold per-run validation because `PreparedMemoryPlan`
describes buffer geometry, not the number of physical representations created for a run.

## Cold binding contract

`PreparedPublication.bind` validates and acts in this exact order:

1. require `runState` non-null;
2. require the state open;
3. require `runState.memoryPlan() == memoryPlan()`;
4. require `representationIndex` to exist at `bufferIndex`;
5. resolve the exact `BufferRepresentationBinding` once;
6. retain its exact `BufferRepresentation` directly in a new `BoundPublication`; and
7. return that non-null bound object.

Exact failures are:

- `NullPointerException("runState")`;
- `IllegalStateException("run state is closed")`;
- `IllegalArgumentException("run state memory plan does not match prepared publication memory plan")`; and
- `IllegalArgumentException("representationIndex out of run-state range: X")`.

The range failure is deliberately a cold recipe-association failure rather than the public
`RunState` accessor's `IndexOutOfBoundsException`. Binding checks the count before resolution so
the exact publication diagnostic is stable. `bufferIndex` cannot be out of the matching state
because constructor validation proved it against the exact same plan.

Binding allocates only the ordinary bound object. It performs no compatibility test: publication
does not interpret or access backend-specific storage. It changes no validity or ownership,
invokes no physical work, and acquires no closeable resource.

## Bound publication and publication-moment contract

`BoundPublication` retains exactly the supplied exact `RunState`, exact `PreparedPublication`,
and exact selected `BufferRepresentation`, plus one private boolean initially `false`. Its
constructor is package-private and called only after `PreparedPublication.bind` has completed the
shared validation above.

`publish()` validates and acts in this exact order on every call:

1. require the retained state open;
2. reject an already-published occurrence;
3. query validity for the prepared buffer/representation coordinate exactly once;
4. reject an invalid selected representation; and
5. set only this bound occurrence's `published` flag to `true`.

Exact failures are:

- `IllegalStateException("run state is closed")`;
- `IllegalStateException("publication is already complete")`; and
- `IllegalStateException("published buffer representation is invalid")`.

`isPublished()` returns only the local flag and remains available after state closure. It performs
no state access. A failed call leaves the flag false and changes no Runtime validity or ownership.
A successful call changes only that flag. The selected direct representation is retained for
later result construction but is neither accessed nor returned.

Sequential repeat calls fail rather than silently succeeding. This makes an accidentally repeated
schedule occurrence observable and prevents one occurrence from masquerading as two result
positions. Separate bound publications for distinct result positions may intentionally retain the
same exact representation.

`publish()` is the hot schedule action but invokes no backend hook. Bytecode must contain no
representation lookup, map, reflection, registry/service lookup, string dispatch, class test,
cast, boxing, allocation, graph inspection, transfer, copy, conversion, tracing, retry, fallback,
cleanup, or ownership mutation.

## Schedule construction and ordering contract

`PublicationStep` retains one exact non-null `PreparedPublication` and derives
`memoryPlan()` directly from it. Null fails with `NullPointerException("publication")`.
Construction invokes no bind or publication action.

`PreparedSchedule` preserves its existing validation order for top-level references, each step's
null/plan association, and the first-only representation-creation rule. After those checks for
each encountered step, it enforces publication structure:

1. before the first publication, creation, executable, and transfer steps retain current rules;
2. the first publication must have `resultIndex() == 0`;
3. each later publication must have the next dense result index; and
4. after the first publication, every remaining occurrence must also be a publication.

Exact added failures are:

- `IllegalArgumentException("steps[X] publication resultIndex must equal publication encounter index Y")`; and
- `IllegalArgumentException("steps[X] non-publication occurrence follows publication suffix")`.

The existing null, exact-plan, and creation-position diagnostics win before these added checks for
the same step. List snapshotting still occurs only after the complete scan succeeds.

`publicationCount()` counts the publication suffix and returns `0` for no publication. It performs
no allocation and is safe on the immutable schedule. The suffix makes results final schedule
deliveries without requiring executable-output invalidation semantics in this task. Empty,
creation-only, executable-only, transfer-only, and mixed pre-publication schedules remain valid.
Existing executable and transfer repetition remains valid.

## Multiple results, duplicates, and aliasing

- Result order is exactly publication encounter order and exactly dense `resultIndex` order.
- The result count is the number of publication occurrences; zero is valid.
- One result position occurs exactly once. Gaps, reordering, and duplicate positions are rejected
  by schedule construction.
- Two or more distinct result positions may use the same buffer and representation coordinates.
  Each binds to a distinct `BoundPublication`, and the final private result snapshot retains the
  same exact physical representation at each corresponding position.
- Aliasing is identity retention only. Runtime does not infer physical aliasing between different
  representation objects or compare representation equality.
- Compiler forward/gradient grouping, repeated gradient `ValueId` values, and Tensor identities do
  not enter Runtime. Prepare later chooses coordinates and preserves the compiler-established
  role order by assigning consecutive Runtime result indices.

## Run result construction contract

`RunResult` stores exactly one private final exact `RunState` and one private final
`BufferRepresentation[]` ordered snapshot. It exposes only result count and lifecycle state; it
does not expose the array, a list, a representation, the state, bytes, storage, or a Tensor.

Construction validates in this exact order:

1. require `runState` non-null;
2. require `publications` non-null;
3. require the state open;
4. for each list position in encounter order, require the bound publication non-null;
5. require it to belong to the exact supplied state;
6. require its prepared `resultIndex()` to equal the list position; and
7. require it to be published.

Exact failures are:

- `NullPointerException("runState")`;
- `NullPointerException("publications")`;
- `IllegalStateException("run state is closed")`;
- `NullPointerException("publications[X]")`;
- `IllegalArgumentException("publications[X] does not belong to supplied run state")`;
- `IllegalArgumentException("publications[X] result index does not match encounter order")`; and
- `IllegalStateException("publications[X] is not published")`.

Only after the complete scan succeeds does cleanup responsibility for the entire state pass from
the run/runner to the result. The supplied list container is not retained. The result snapshots
the exact direct representation from every bound publication into a private array in encounter
order. Empty publications produce a zero-length array and successfully lease the open state.

Constructor failure transfers nothing, closes nothing, and mutates nothing. The caller that
still owns the run must close the state. Successful construction creates no individual resource
ownership transition: `BufferRepresentationBinding` values and `RunResourceOwnership` remain
unchanged.

`resultCount()` returns the private array length and remains available after closure.
`isClosed()` returns the exact retained state's lifecycle status. `close()` delegates to the
idempotent `RunState.close()` contract; the first call marks the state closed before cleanup,
skips borrowed buffers, and preserves its established reverse-order failure behavior. Later calls
are inert through that existing state contract. The result does not retry, wrap, replace, or
independently suppress cleanup failure.

After successful construction, callers must close only through the result and must not separately
close or mutate the leased state. The new API intentionally does not expose that state, though a
caller that constructed it already holds a reference and is responsible for obeying this lifetime
rule.

## Ownership, lifetime, and cleanup contract

- Before `RunResult` construction succeeds, the run/runner owns cleanup of the open state.
- `PreparedPublication` and `BoundPublication` own no representation or state and close nothing.
- Publication transfers no individual representation ownership.
- After successful `RunResult` construction, the result exclusively leases the complete state
  cleanup lifecycle.
- Run-owned buffers and every workspace remain state-owned and close in existing deterministic
  reverse order when the result closes.
- Borrowed buffer representations remain caller-owned and are never closed by the result/state.
  If one is published, its original caller lifetime and synchronization obligation continues for
  the entire result lease; publication does not silently promote it to run-owned storage.
- Duplicate result positions that retain one exact representation do not duplicate cleanup. The
  state still contains that physical identity once and closes it according to its one binding.
- `PreparedExecution` owns no per-run or result resource and remains close-free.
- There is no detachable output, individual lease, reference count, shared-result close protocol,
  or result finalizer.

## Failure, partial publication, and rollback contract

Prepared construction and cold binding acquire no physical resource. Their failures require no
rollback.

If one `publish()` fails, earlier bound publications may retain `isPublished() == true`, while the
failed and later occurrences remain false. No result exists, no cleanup ownership transferred,
and no representation validity or ownership changed. This task adds no rollback method because
the flags are private per-run occurrence state with no external resource effect. The later runner
must discard those bound objects and close the still-run-owned `RunState`; it must not construct a
partial `RunResult`.

If `RunResult` construction fails, it retains no snapshot and transfers no cleanup responsibility.
The caller still closes the exact state. If result construction succeeds and a later close fails,
the result/state remains closed and the exact existing cleanup failure propagates.

Publication never catches, wraps, retries, falls back, transfers, allocates, copies, or invokes a
backend, so it introduces no backend failure channel and no physical rollback claim.

## Thread-safety and side effects

- `PreparedPublication`, `PreparedSchedule`, and `PreparedExecution` are immutable and thread-safe.
- One prepared publication may bind concurrently to distinct open matching states.
- `BoundPublication`, `RunState`, and `RunResult` are not thread-safe.
- A caller must not race publication with itself, validity changes, bound execution/transfer,
  state closure, result construction, or result closure.
- Cold binding allocates one ordinary bound object and reads one state binding.
- Successful publication mutates one local boolean only.
- Successful result construction allocates one ordinary result plus one private reference array
  and transfers the semantic cleanup lease for the state.
- Result close has exactly the existing `RunState.close()` physical cleanup side effects.
- No other method performs backend, storage, validity, ownership, configuration, tracing, graph,
  transfer, or publication-policy work.

## Concrete lifecycle example

This is a current Java 26 contract example for the implemented Runtime 0009 API. It calls the
publication contracts directly because schedule traversal remains Runtime 0010 work.

### Goal and initial state

Publish two ordered results from one open state. Result zero uses buffer 1 representation 0.
Result one intentionally aliases that same exact representation. A preceding Runtime 0008
transfer has already made the selected representation valid.

```java
PreparedPublication first = new PreparedPublication(plan, 1, 0, 0);
PreparedPublication alias = new PreparedPublication(plan, 1, 0, 1);

PreparedSchedule schedule = new PreparedSchedule(
        plan,
        List.of(
                creationStep,
                executionStep,
                transferStep,
                new PreparedSchedule.PublicationStep(first),
                new PreparedSchedule.PublicationStep(alias)));

BoundPublication boundFirst = first.bind(runState);
BoundPublication boundAlias = alias.bind(runState);
boundFirst.publish();
boundAlias.publish();

try (RunResult result =
        new RunResult(runState, List.of(boundFirst, boundAlias))) {
    assert result.resultCount() == 2;
}
```

### Result and interpretation

The schedule fixes result order as `0, 1` and requires publication to be its suffix. Both bound
objects retain the same exact already-created representation, but the result count remains two.
Neither publication copies or reads storage. The successful result constructor leases the whole
state; closing the result closes each state-owned resource once, while any borrowed input remains
caller-owned.

If the selected representation were invalid, `publish()` would fail with
`published buffer representation is invalid`. Runtime would not search for another valid copy or
materialize one. The runner would discard both bound publications, close the state, and return no
partial result.

## Affected files

Expected production and test paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/PreparedPublication.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/BoundPublication.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunResult.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/PreparedSchedule.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/package-info.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/PreparedPublicationTest.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/BoundPublicationTest.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunResultTest.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/schedule/PreparedScheduleTest.java`

Expected documentation and planning paths in the implementation change:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`
- `docs/planning/modules/runtime/tasks/0009-publication-and-result-schedule-steps.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless a concrete contradiction is found:

- `ARCHITECTURE.md`
- `docs/architecture/current-architecture-plan.md`
- `docs/architecture/lifecycle.md`
- `docs/architecture/module-boundaries.md`
- `docs/architecture/dependency-rules.md`
- `docs/design/decisions/0006-no-runtime-service-locator.md`
- `docs/design/decisions/0010-staged-backend-preparation.md`
- `docs/design/decisions/0011-per-run-runtime-resource-ownership.md`
- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/api/training-api.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/engine/master-plan.md`
- current Compiler publication source/tests and generated Runtime Javadocs
- Runtime/Prepare/Config/Engine build files and Java 26 root configuration
- architecture, backend-conformance, and integration tests

If a review-only path needs a substantive change beyond the focused implementation-status page,
stop and report why the planned boundary is insufficient rather than silently expanding scope.

## Maximum scope

The implementation task may create or modify at most the exact 18 paths listed under expected
production, test, documentation, and planning paths.

No Java or test path outside `modules/runtime` may change. No additional production type, test
file, documentation file, build file, or later task specification may be added. If implementation
needs another path or cannot remain within this ceiling, stop and propose a separately reviewed
follow-up or architecture decision.

This planning-only task itself changes exactly three planning Markdown paths: this specification,
the Runtime master plan, and the repository roadmap.

## Acceptance criteria

- The exact three-class public `runtime.run` surface and sole `PublicationStep` addition exist
  with no extra constructor, overload, accessor, factory, builder, result value, or identity type.
- `PreparedExecution` remains bytecode-exactly its existing two-component record surface.
- Runtime production imports no Compiler, Model, Planning, Prepare, Engine, or concrete-backend
  type, including every forbidden publication and graph identity named in this task.
- Prepared publication construction and binding follow the exact order, exception types, and
  messages specified above.
- Cold binding requires one exact matching open state, validates the representation position, and
  retains the exact selected physical representation directly.
- Publication requires that exact state open and selected copy valid at publication time, then
  changes only one local one-shot flag.
- Invalid, closed, and repeated publication fail exactly without hidden fallback or any validity,
  ownership, backend, transfer, allocation, conversion, or copy side effect.
- Schedule publication occurrences form a dense `0..N-1` suffix. Empty results are valid;
  reordered, gapped, duplicated, or non-suffix publications fail exactly.
- Distinct result positions may intentionally alias one exact representation; no equality-based
  deduplication or duplicate cleanup is added.
- `RunResult` accepts only a complete dense ordered same-state list of published occurrences,
  privately snapshots exact representations, and exposes only count/closed/close behavior.
- Successful result construction leases the complete `RunState`; constructor failure transfers
  nothing; result close preserves existing borrowed/run-owned cleanup and failure semantics.
- Partial publication produces no partial result and requires no flag rollback or individual
  resource rollback; later runner cleanup remains possible through the unchanged state.
- No public output value conversion/access/export, Tensor publication, Config policy, trace,
  concrete backend, Prepare orchestration, or runner behavior is added.
- Hot publication bytecode has no physical representation lookup, backend call, map, reflection,
  registry/service, string dispatch, boxing, allocation, graph/route/config work, transfer, copy,
  conversion, cleanup, or fallback.
- Existing Runtime tests and completed contracts remain green and unchanged except the one
  schedule test extended for the new sealed variant and suffix rules.
- Javadocs describe exact inputs, results, nullability, identity, ownership, lifecycle, threading,
  side effects, failures, aliases, empty results, and exclusions.
- The documentation pass distinguishes Compiler logical publication roles from Runtime prepared
  publication coordinates and keeps current versus planned boundaries explicit.
- A separate clean documentation-focused agent finalizes affected Javadocs, documentation,
  glossary impact, generated-page inspection, planning evidence, and no-change conclusions
  without repeating successful Java tests absent executable change or a recorded concrete risk.
- Exactly the authorized 18 implementation paths change; links, anchors, terminology, fences,
  final newlines, whitespace, status, and `git diff --check` pass.
- Runtime 0001–0008 and Prepare 0001–0002 remain Complete. Runtime 0009 becomes Complete only
  after implementation and documentation gates. Runtime 0010–0011 and Prepare 0003 remain Draft
  without detailed specifications.

## Tests / validation

Implementation development:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.run.PreparedPublicationTest \
  --tests io.github.pho001.synaptik.runtime.run.BoundPublicationTest \
  --tests io.github.pho001.synaptik.runtime.run.RunResultTest \
  --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest
```

Final affected module after executable Java stabilizes:

```bash
./gradlew :modules:runtime:test
```

Documentation-focused pass after final Javadocs/documentation, without repeating successful Java
tests unless executable Java changes or a concrete risk is recorded:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md \
  docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/writing-a-backend.md \
  docs/glossary.md \
  docs/planning/modules/runtime/tasks/0009-publication-and-result-schedule-steps.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

If the temporary validator is absent or incompatible, create an equivalent validator outside the
repository. It must check local file targets and heading anchors, unique effective anchors,
balanced backtick and tilde fences, final newlines, and trailing whitespace.

Also verify:

- exact source, compiled, reflection, constructor-visibility, record-component, and sealed-family
  surfaces;
- exact validation order/messages, exact plan/state/reference identity, direct representation
  retention, one-shot flag behavior, and no state validity/ownership mutation;
- zero, one, multiple, aliased, reordered, gapped, repeated, partial, closed, invalid, and cleanup-
  failure cases;
- immutable prepared objects and isolated bound/result state across separate runs;
- unchanged compiled/source `PreparedExecution`, `PreparedExecutable`, `BoundInvocation`,
  `PreparedBufferTransfer`, `BoundBufferTransfer`, `PreparedRepresentationPlan`, `RunState`, and
  `RunStateCreation` surfaces;
- hot `BoundPublication.publish()` bytecode contains only open/one-shot/validity/flag behavior and
  no physical-resource resolution or forbidden mechanism;
- no forbidden upstream/concrete-backend imports or map/reflection/raw/unchecked/string/boxing/
  registry/service/synchronization/discovery/route/config/tuning/tracing/Tensor/value-access
  mechanism;
- unchanged Runtime/root Gradle, dependencies, architecture contract/ADRs/tests, Java 26
  configuration, Prepare/Compiler/Model/Config/Trace/Backend Contract/Engine/concrete-backend Java,
  conformance, and integration;
- exact 18-path implementation ceiling and no Java/test path outside Runtime;
- synchronized Runtime 0009 status and absence of Runtime 0010–0011 and Prepare 0003 detailed
  specifications; and
- final documentation/status/scope/whitespace gates.

Repository-wide and architecture validation is deferred to Runtime 0011, continuous integration,
or the prepared-execution checkpoint because implementation changes one module without a module
edge, build, architecture, concrete-backend, or end-to-end runner change. Backend conformance and
integration tests remain inapplicable until a concrete backend and runner consume the contracts.

The initial planning-only pass ran no Java, Gradle, Javadoc, architecture, conformance, or
integration tests. Implementation and documentation evidence is recorded below.

## Mandatory clean documentation pass

After implementation and the final Runtime module test, hand the exact diff and test evidence to
a distinct clean documentation-focused agent/thread. It must apply the General, API/Javadoc,
Architecture, Backend Guide, Planning, and Example profiles as appropriate and inspect actual
source/tests rather than trusting the implementation summary.

The pass must:

- finalize all three new class Javadocs, both affected package summaries, and schedule Javadocs;
- explain compiler logical publication roles versus Runtime prepared coordinates at first use;
- document the publication suffix, exact validity moment, no hidden fallback, aliasing, empty
  results, partial failure, whole-state result lease, borrowed lifetime, and close behavior;
- keep result storage private and avoid inventing any value access, Tensor conversion, Config
  policy, Engine facade, or runner API;
- update Runtime/Public APIs, focused implementation status, backend-author guidance, and glossary
  only for the implemented reusable terms and current behavior;
- inspect generated Runtime Javadoc pages for the three new types, `PublicationStep`,
  `publicationCount()`, and both affected package summaries;
- validate links, anchors, terminology, examples, fences, newlines, whitespace, exact scope, and
  planning status; and
- record exact files reviewed/changed, reused Java evidence, commands/results, limitations,
  unresolved issues, and reasoned no-change conclusions.

Required no-change conclusions unless a concrete contradiction is found:

- `ARCHITECTURE.md` and ADRs already assign publication/result lifetime to Runtime and physical
  cleanup mechanics to representations; no architecture decision or dependency rule changes.
- Lifecycle, module-boundary, and dependency pages remain accurate as architecture explanations;
  only the focused Runtime/Prepare/Backend implementation-status page needs synchronization.
- Compile API and compiler publication Javadocs remain accurate because Compiler identities and
  stable logical role ordering are unchanged and never imported by Runtime.
- Tensor and Training APIs remain unchanged because this task publishes nominal Runtime
  representations, not Tensor values, gradients, optimizers, or public value access.
- Prepare Java/Javadocs and Prepare 0003 remain unchanged because translation/orchestration and
  complete runnable-result validation are not implemented here.
- Config and its Draft run/publication options remain unchanged because Runtime 0009 adds no
  policy or request filtering.
- `PreparedExecution` and all completed execution/transfer/creation/run-state Javadocs remain
  accurate except the two package summaries and schedule text explicitly listed.
- Trace, Backend Contract, Engine, concrete backends, other guides, build/dependency files,
  architecture tests, conformance, and integration remain unchanged because no producer,
  implementation, module edge, or runnable lifecycle is added.

## Dependencies

- Runtime 0001–0008 — Complete and preserved.
- Runtime 0003 and ADR 0011 supply the one-state ownership and cleanup model.
- Runtime 0005 supplies the exact-plan sealed schedule family.
- Runtime 0006 fixes `PreparedExecution` as memory plan plus schedule.
- Runtime 0007 supplies already-created representation coordinates and explicit per-copy validity.
- Runtime 0008 supplies the only explicit transfer/materialization action required before
  publication when the selected result representation is not already valid.
- Compiler 0006 supplies stable ordered logical forward/gradient publication roles for later
  Prepare translation, read-only; it is not a Runtime dependency.
- Existing Runtime dependencies and Java 26 build boundaries remain unchanged.

Prepare 0003, Runtime 0010+, Config run/publication options, Trace run payloads, Engine, concrete
backends, output access, and tuning are not dependencies of this bounded contract.

## Follow-up tasks

- Runtime 0010 remains Draft for cold creation, binding of executable/transfer/publication
  occurrences, executable validity transitions, ordered schedule traversal, failure cleanup,
  caller-input handoff, and final `RunResult` construction. It must consume the direct bound
  objects without lookup in the hot path.
- Runtime 0011 remains Draft for contract closure and the prepared-execution checkpoint.
- Prepare 0003 remains Draft for public preparation orchestration, translation of compiler
  publication roles to Runtime coordinates, and complete runnable-result validation.
- Config 0007 remains Draft for declarative run/publication policy after a stable consumer
  requires it.
- Engine later owns the public Tensor/value-facing result access and lifecycle facade.
- Concrete backend tasks later implement physical representations/executables/transfers; Runtime
  publication itself requires no backend implementation.

Do not create any follow-up detailed specification in this task.

## Decisions and rejected alternatives

- **Use dense Runtime coordinates and a dense result index.** This is the smallest identity model
  already supported by prepared plans and run state. Importing Compiler binding types, graph IDs,
  Tensor IDs, Planning facts, or Prepare assignments would violate the Runtime boundary.
- **Use one immutable prepared recipe and one per-run bound occurrence.** This matches cold
  binding, retains the physical representation directly, and keeps lookup out of publication.
  Publishing directly from the schedule recipe would require a hot `RunState` lookup.
- **Require a publication suffix.** This fixes stable final-result order and prevents later
  scheduled work after a representation has been published without inventing executable-output
  invalidation or snapshot copying. Arbitrary interleaving is rejected.
- **Make result indices dense and encounter-ordered.** A map or ID registry adds lookup and allows
  gaps that no current consumer needs. List position is the stable identity.
- **Allow duplicate coordinates across distinct result positions.** Compiler gradient roles may
  share one logical value. Runtime preserves requested multiplicity and order without creating
  identity copies or deduplicating physical references.
- **Reject a repeated bound publication.** One schedule occurrence publishes once. Silent
  idempotence could conceal duplicated traversal and cannot create the next result position.
- **Require validity at publication time with no fallback.** Discovering another valid copy or
  inserting an implicit copy would repeat Prepare work and violate explicit transfer semantics.
- **Lease the complete `RunState` to `RunResult`.** The current ownership flag is immutable and
  cleanup already spans all buffers/workspaces. Transferring individual resources would require
  mutable ownership removal, workspace-lifetime decisions, duplicate/alias accounting, partial-
  transfer rollback, and changes to completed APIs. Whole-state leasing is coherent and smaller.
- **Keep published representations private.** A public representation, byte, host-storage, Tensor,
  conversion, or export accessor would establish an output access API that belongs to later
  Engine/result design.
- **Permit empty results.** Empty compiler/publication requests and foundation schedules already
  admit empty boundaries. A zero-result lease still closes run-owned resources deterministically.
- **Do not add `PreparedResultPlan` or a third `PreparedExecution` component.** The schedule already
  owns ordered occurrences and can validate/count the publication suffix. Another aggregate
  would duplicate association and contradict the completed minimal root without necessity.
- **Do not mutate `RunState` with a result-owner flag.** The semantic lease transfers only after
  complete result validation. The runner owns the state before that point, and the result owns its
  close call afterward; no current public state mutation is required.
- **Do not add publication rollback.** A bound publication changes only its private boolean. On
  partial failure, discarding bound objects and closing the still-run-owned state is exact and
  leaves no physical publication side effect to undo.

## Known limitations

- `RunResult` intentionally exposes no output value or representation access. Engine/result API
  work must define that separately without retroactively importing graph identities into Runtime.
- The whole-state lease may retain internal buffers and workspaces longer than an individual-
  output transfer design. This is the accepted minimal tradeoff that preserves current cleanup
  invariants; finer lifetime/reclamation needs a later proved contract.
- A published borrowed representation remains caller-owned. The result cannot extend its physical
  lifetime beyond the caller's existing obligation.
- Runtime cannot verify that valid physical bytes match logical semantics. Executable/transfer and
  concrete-backend tests own physical correctness.
- Publication suffix validation proves structural order only. Complete creation, execution,
  transfer, output-validity, and publication reachability belongs to Prepare 0003/Runtime 0010.
- No executable-output invalidation contract exists yet. The suffix prevents post-publication
  work but Runtime 0010 must define explicit validity transitions before a runnable lifecycle.
- Partial publication leaves earlier bound objects locally marked until discarded; no result or
  ownership transfer exists and no physical rollback is needed.
- No public runner, Engine, concrete backend, Config run policy, tracing, output access,
  conformance, or end-to-end execution consumes the contract yet.
- Repository-wide, architecture, conformance, and integration validation remain deferred for the
  reasons in the validation section.

## Architecture impact

Expected impact: None.

The architecture already assigns prepared publication, output ownership lease/transfer, and
failure cleanup to Runtime. Leasing the complete `RunState` is directly compatible with ADR 0011
and preserves every completed component and module edge. No architecture contract, ADR,
dependency, build, or architecture-test change is required.

If implementation requires individual resource detachment, graph/Compiler identity in Runtime,
another module edge, physical publication work, output access, or a third `PreparedExecution`
component, stop and report the exact architectural decision rather than inventing a workaround.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the focused Runtime architecture
and ADR references, documentation rules/profiles, Runtime master plan, Runtime tasks 0003–0009,
Compiler publication contracts read-only, and the directly relevant current Runtime
source/tests/Javadocs and public/backend documentation.

Implement docs/planning/modules/runtime/tasks/0009-publication-and-result-schedule-steps.md exactly
within its 18-path ceiling. Preserve every completed contract and stop on any architecture,
package, API, validation, ownership, or scope conflict. Keep PreparedExecution exactly its
existing two components. Do not add output access/conversion/export, Tensor publication,
Compiler/Model/Planning/Prepare identities or imports, Config policy, tracing, concrete backend,
runner/traversal, executable-output invalidation, implicit transfer/materialization, allocation,
copy, route search, dependency/Gradle/architecture change, or a later task specification.

Run the focused tests, one final Runtime module test, and all exact surface/mechanism/scope/status
checks. Then hand the actual diff and exact Java evidence to a separate clean documentation-
focused context. That pass must follow documentation-rules.md, independently finalize affected
Javadocs, docs, examples, glossary impact, and planning evidence, and must not repeat successful
Java tests without an executable change or recorded concrete risk.

Mark Complete only after every implementation and documentation gate passes. Return both context
IDs, exact changed paths, commands/results/test counts, no-change conclusions, unresolved issues,
follow-up, and the repository completion status. Do not commit or push.
```

## Local decisions

- Kept `PreparedExecution` bytecode-exactly at its existing two record components; publication is
  reachable only through the schedule.
- Used dense Runtime coordinates and a dense result suffix, retaining Compiler/Model/Planning/
  Prepare identities outside Runtime.
- Leased the complete `RunState` to `RunResult` after full validation instead of mutating
  individual ownership bindings. This preserves borrowed ownership, alias handling, duplicate-
  identity protection, and existing cleanup order.
- Kept all result representations private. Runtime 0009 establishes lifetime only; public value
  access remains a later Engine/result contract.

## Validation evidence

- Implementation context: `019fbe60-a915-71c2-a138-43dacaa7a69f`.
- Focused implementation validation:
  `./gradlew :modules:runtime:test --tests io.github.pho001.synaptik.runtime.run.PreparedPublicationTest --tests io.github.pho001.synaptik.runtime.run.BoundPublicationTest --tests io.github.pho001.synaptik.runtime.run.RunResultTest --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest`
  passed 4 suites and 32 tests with no failures, errors, or skips.
- Final implementation validation: exactly one
  `./gradlew :modules:runtime:test` passed 16 suites and 130 tests with no failures, errors, or
  skips. Clean documentation context `019fbe69-07e8-7a20-b132-c3b70c663d4d` reused this evidence;
  it changed no executable Java behavior and did not rerun Java tests.
- Implementation checks passed for exact source/compiled/reflection surface, constructor
  visibility, record components and sealed permits, validation order/messages, exact identity and
  alias behavior, partial/empty lifecycle, `PreparedExecution` preservation, direct-field hot
  path, `BoundPublication.publish()` bytecode, forbidden imports/mechanisms, Java 26/build
  boundaries, ownership, scope, status, and whitespace.
- Documentation context applied the General, API/Javadoc, Architecture, Backend Guide, Planning,
  and Example profiles. It reviewed the architecture contract, focused Runtime/Prepare/Backend
  explanation and ADRs, Runtime 0005–0009 plans, final source/tests/Javadocs, Runtime/Public APIs,
  backend guide, glossary, master plan, roadmap, and Java 26 build boundary.
- `./gradlew :modules:runtime:javadoc` passed after the final Javadoc edit; generated pages for
  `PreparedPublication`, `BoundPublication`, `RunResult`, `PublicationStep`,
  `publicationCount()`, and both affected package summaries contained the intended contracts.
- `javac -cp modules/runtime/build/classes/java/main -d /tmp/runtime-publication-doc-example /tmp/RuntimePublicationDocExample.java` followed by
  `java -ea -cp modules/runtime/build/classes/java/main:/tmp/runtime-publication-doc-example RuntimePublicationDocExample`
  compiled and ran the focused Runtime publication example with assertions enabled.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md docs/api/public-api.md docs/architecture/runtime-prepare-backend-boundary.md docs/backend-guide/writing-a-backend.md docs/glossary.md docs/planning/modules/runtime/tasks/0009-publication-and-result-schedule-steps.md docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md`
  passed all 8 files for local targets, heading anchors, unique effective anchors, balanced
  backtick/tilde fences, final newlines, and trailing whitespace.
- Final manual/current-versus-planned review confirms Compiler logical publication roles remain
  distinct from Runtime dense coordinates; publication performs no physical work or fallback;
  aliases and empty results are preserved; partial publication transfers no lease; and successful
  construction leases the whole state while exposing no value, representation, or state.
- Exact 18-path scope, later-specification absence, synchronized status, and `git diff --check`
  passed after final planning updates.
- Repository-wide, architecture, backend-conformance, and integration suites were not run: this
  task changes one module with no dependency/build/architecture/concrete-backend/runner boundary,
  and the task defers those suites to Runtime 0011, continuous integration, or the prepared-
  execution checkpoint.

## Implementation notes

- Added final `PreparedPublication`, `BoundPublication`, and `RunResult` types in `runtime.run`.
- Extended the sealed schedule with `PublicationStep` and dense suffix validation while retaining
  the existing two-component schedule and prepared-execution roots.
- Added focused tests for exact API shape, construction/binding failures, validity and one-shot
  behavior, dense ordering, aliases, empty/partial results, cleanup, concurrency isolation, and
  hot-path exclusions.
- Finalized all affected Javadocs and the five authorized explanatory documentation paths. No
  executable behavior changed during the documentation pass.

## Completion summary

- Completed changes: Runtime-owned prepared publication coordinates, per-run direct publication,
  dense publication suffix, and whole-state result lease.
- Files changed or created: exactly the 18 authorized production, test, API, architecture,
  backend-guide, glossary, and planning paths listed in this task.
- Tests and validation: focused 4 suites/32 tests and final Runtime 16 suites/130 tests passed;
  exact API/bytecode/mechanism checks, Runtime Javadoc, generated-page inspection, eight-file
  Markdown validation, scope/status checks, and whitespace checks passed.
- Documentation-agent review: clean context `019fbe69-07e8-7a20-b132-c3b70c663d4d` completed the
  independent targeted pass and reused stable implementation-test evidence.
- Documentation impact: Runtime/Public API references, focused boundary status, backend guidance,
  glossary, master plan, roadmap, and the task record now describe current publication/result
  behavior and preserve planned runner/output-access boundaries.
- Javadoc review: all three new types, `PreparedSchedule` additions, and both changed package
  summaries document inputs, results, failures, identity, aliases, lifecycle, threading, side
  effects, and exclusions; generated output passed inspection.
- Glossary impact: added `PreparedPublication`, `BoundPublication`, and `RunResult` definitions and
  synchronized the prepared-schedule, prepared-representation, Runtime-status, and run-state
  entries.
- No-change conclusions: `ARCHITECTURE.md`, ADRs, general lifecycle/module/dependency pages,
  Compile/Tensor/Training APIs, Prepare/Config/Trace/Backend Contract/Engine/concrete-backend
  source and Javadocs, `PreparedExecution` and other completed Runtime APIs, Gradle/build files,
  architecture tests, conformance tests, and integration tests remain accurate because no
  architecture decision, module edge, producer contract, policy, physical implementation,
  public value access, or runnable lifecycle changed.
- Unresolved issues: None.
- Follow-up required: None for Runtime 0009. Runtime 0010–0011 and Prepare 0003 remain Draft
  without detailed specifications.

Status: Complete
