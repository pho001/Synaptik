# Task 0008: Prepared Buffer Transfer and Materialization Schedule

## Status

Complete

## Goal

Add the smallest Runtime-owned prepared and per-run bound contracts for copying one logical
buffer value between two distinct, already-created physical representation positions of the same
prepared buffer slot, and make that work expressible as one ordered prepared-schedule occurrence.

The public capability consists of:

- one immutable reusable `PreparedBufferTransfer` recipe supplied by a concrete backend;
- one per-run `BoundBufferTransfer` with direct source and destination physical references held by
  its concrete-backend subclass; and
- one `PreparedSchedule.BufferTransferStep` occurrence retaining the prepared recipe.

Materialization is not a second operation. When the destination is an equivalent already-created
representation of the same logical buffer, performing this buffer transfer materializes that
value there. The task adds no allocation, representation creation, second materialization type,
or second schedule variant.

For one bound action, destination validity controls the transition:

1. if the destination is already valid, return without requiring a valid source and without
   invoking backend work;
2. otherwise require the source to be valid immediately before physical work;
3. invoke the backend action exactly once; and
4. only after successful return, mark only the destination valid.

If backend work throws a `RuntimeException` or `Error`, every Runtime validity bit remains
unchanged. The source and every other valid copy remain valid after success; this task introduces
no invalidation or hidden coherence policy.

## Rationale and mental model

```text
one logical prepared buffer position
  -> source representation position (already created)
  -> destination representation position (already created)

immutable PreparedBufferTransfer
  -> cold bind against one exact open RunState
  -> checked concrete compatibility
  -> backend-owned BoundBufferTransfer with direct physical references

hot bound action
  -> destination valid? no-op
  -> source valid? otherwise fail
  -> one backend copy/materialization action
  -> mark only destination valid after success
```

Runtime owns the coordinates, validity transition, exact-plan/run association, and schedule
occurrence. A concrete backend owns the physical representation classes, compatibility checks,
direct fields, and copy mechanics. This preserves the established cold checked boundary and keeps
resource resolution out of the hot backend action.

The word *materialization* describes why a transfer may be required: an already-created
destination representation needs the same logical value in a form required by later prepared
work. It does not authorize another resource origin, allocation path, route search, or coherence
protocol.

## Exact API

Add these two public abstract classes in `io.github.pho001.synaptik.runtime.execution`:

```java
public abstract class PreparedBufferTransfer {
    protected PreparedBufferTransfer(
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex);

    public final PreparedMemoryPlan memoryPlan();

    public final int bufferIndex();

    public final int sourceRepresentationIndex();

    public final int destinationRepresentationIndex();

    public final BoundBufferTransfer bind(RunState runState);

    protected abstract boolean acceptsSourceBufferRepresentation(
            BufferRepresentation representation);

    protected abstract boolean acceptsDestinationBufferRepresentation(
            BufferRepresentation representation);

    protected abstract BoundBufferTransfer bindCompatible(
            RunState runState,
            BufferRepresentation sourceRepresentation,
            BufferRepresentation destinationRepresentation);
}

public abstract class BoundBufferTransfer {
    protected BoundBufferTransfer(
            RunState runState,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex);

    public final void execute();

    protected abstract void executeTransfer();
}
```

Extend the existing schedule family exactly:

```java
public sealed interface Step
        permits ExecutionStep, RepresentationCreationStep, BufferTransferStep {
    PreparedMemoryPlan memoryPlan();
}

public record BufferTransferStep(
        PreparedBufferTransfer transfer) implements Step {
    @Override
    public PreparedMemoryPlan memoryPlan();
}
```

`PreparedBufferTransfer` and `BoundBufferTransfer` add no interface, nested type, factory,
builder, overload, public mutator, public resource accessor, close method, result, or custom
object method. `BufferTransferStep` is the only new schedule variant. Existing
`PreparedExecution(PreparedMemoryPlan, PreparedSchedule)`, `PreparedExecutable`,
`BoundInvocation`, `PreparedRepresentationPlan`, `RunState`, and `RunStateCreation` declarations
remain unchanged.

`BoundBufferTransfer` has package-private final association accessors used only by
`PreparedBufferTransfer.bind` to validate the returned bound object:

```java
final RunState runState();

final int bufferIndex();

final int sourceRepresentationIndex();

final int destinationRepresentationIndex();
```

These methods are not public or protected backend API. No other package-private member or
top-level production type is added.

## Scope

- Add the exact prepared/bound transfer pair and schedule variant above.
- Address one prepared buffer by dense `PreparedMemoryPlan.buffers()` position and two distinct
  dense representation positions within that buffer's `RunState` binding list.
- Require both physical representations to exist before binding; this task creates neither one.
- Confine physical-type checks and representation resolution to `PreparedBufferTransfer.bind`.
- Require concrete backend subclasses of `BoundBufferTransfer` to retain direct concrete typed
  source and destination references established by the bind hook.
- Implement the exact destination-no-op, source-valid, one-action, success-only destination-valid
  transition.
- Preserve source and unrelated-copy validity on every successful action.
- Preserve all Runtime validity bits when the backend action throws.
- Propagate the exact unchecked backend failure without retry, fallback, wrapping, cleanup, or
  publication.
- Extend `PreparedSchedule` with the one exact-plan transfer occurrence while preserving its
  existing creation-prefix rule, empty schedules, executable-only schedules, repeated
  occurrences, and immutable encounter order.
- Add focused tests for API shape, validation, identity, binding order, compatibility hooks,
  direct references, no-op/action/failure transitions, schedule behavior, threading boundaries,
  and forbidden hot-path mechanisms.
- Finalize affected Javadocs and explanatory documentation in a distinct clean documentation
  context after implementation tests stabilize.
- Preserve completed Runtime 0001–0007 and Prepare 0001–0002 history.

## Out of scope

- a separate materialization recipe, bound action, schedule step, enum value, or policy
- representation creation, allocation, lazy creation, replacement, eviction, removal, pooling,
  reuse, aliasing, or lifetime/interference analysis
- transfer-route discovery, route search, graph search, shortest-path selection, fallback, or
  multi-hop transfer composition
- backend selection, lowering, kernel selection, concrete backend implementation, native bridge,
  device inspection, or physical storage/access implementation in Runtime
- a runner, schedule traversal/consumption, `RunState` creation, executable binding/execution,
  output invalidation after executable work, retries, barriers, branches, or parallel scheduling
- automatic invalidation of stale copies, dirty bits, preferred-copy policy, hidden write-back,
  implicit synchronization, or a general coherence protocol
- publication, ownership transfer/lease, `RunResult`, result cleanup, delivery target, or Config
- public Prepare orchestration, complete prepared-result validation, Engine composition, caller
  input handoff, or a public run lifecycle
- changes to `PreparedExecution`, `PreparedExecutable`, `BoundInvocation`,
  `PreparedRepresentationPlan`, `RunState`, `RunStateCreation`, memory geometry, or resource
  ownership contracts
- Compiler, Planning, Model, Prepare Java, Config, Trace, Backend Contract, Engine, concrete
  backend, Gradle, dependency, architecture-contract, ADR, architecture-test,
  backend-conformance, or integration-test behavior
- tracing, profiling, configuration/tuning interpretation, measurement, cache lookup/mutation, or
  model-autotuning
- maps, registries, services, service locators, raw `Object`, unchecked generic access,
  reflection, class-name tests, string dispatch, boxing, synchronization, or public concrete-
  backend switches
- detailed Runtime 0009–0011 or Prepare 0003 specifications

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): core invariants; Runtime and concrete-
  backend ownership; run lifecycle; hot-path and dependency rules.
- [ADR 0006: No Runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0004](0004-prepared-executable-and-bound-invocation.md)
- [Runtime 0005](0005-prepared-schedule-contract.md)
- [Runtime 0006](0006-prepared-execution-aggregate.md)
- [Runtime 0007](0007-representation-creation-and-residency-foundation.md)

## Architecture constraints

- Prepared transfer recipes are immutable, reusable, and thread-safe. Each bind produces a new
  per-run bound action associated with one exact open `RunState`.
- Every source and destination coordinate refers to distinct representation positions of one
  logical buffer position. A transfer never crosses buffer slots.
- Every representation is already structurally resident because Runtime 0007 creates and binds
  the complete state before cold binding. Transfer performs no allocation or resource lookup.
- Runtime owns explicit validity checks and mutation. Concrete backends own physical copy/access
  mechanics and retain direct typed physical references in the bound subclass.
- Java compatibility checks occur once at cold binding. The hot action performs no representation
  lookup, compatibility cast, map access, reflection, registry/service lookup, string dispatch,
  boxing, backend discovery, route selection, graph inspection, or allocation.
- The only hot shared-state access is constant-time dense boolean validity query/mutation through
  the exact retained `RunState`; the backend action uses direct concrete fields.
- A valid destination makes the action an explicit no-op even if the source is invalid. Backend
  work is not invoked and no validity bit changes.
- An invalid destination requires a valid source immediately before work. An invalid source fails
  before backend invocation and changes no validity.
- Successful work marks only the destination valid. Source and all other copies remain unchanged.
- Backend failure may have physically touched the destination, but Runtime validity remains
  unchanged and therefore continues to classify that destination as invalid.
- One `BoundBufferTransfer` is mutable-run-associated and not thread-safe. Callers must not race
  its action with itself, other validity transitions, execution, or state closure.
- Runtime remains independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backends. No existing module edge or architecture rule changes.
- If implementation needs another operation kind, resource origin, module edge, ownership rule,
  or completed API change, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.runtime.memory` — supplies the exact plan and dense buffer geometry.
- `io.github.pho001.synaptik.runtime.resource` — supplies nominal physical buffer roles without
  modification.
- `io.github.pho001.synaptik.runtime.run` — supplies exact per-run bindings and validity without
  modification.

Packages changed:

- `io.github.pho001.synaptik.runtime.execution` — owns reusable prepared work, cold binding, and
  direct-reference per-run actions; the prepared/bound transfer pair follows the existing
  `PreparedExecutable`/`BoundInvocation` boundary.
- `io.github.pho001.synaptik.runtime.schedule` — owns the closed semantic schedule family and adds
  the transfer occurrence.

Type placement:

- `io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer` — one immutable reusable
  backend-supplied transfer/materialization recipe.
- `io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer` — one backend-owned per-run
  direct-reference transfer action with Runtime validity orchestration.
- `io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.BufferTransferStep` — one ordered
  occurrence of that recipe in the existing schedule.

Tests mirror `runtime.execution` and `runtime.schedule`. No package is added, moved, or removed.

## Prepared transfer construction contract

`PreparedBufferTransfer` stores exactly four private final fields matching its constructor
parameters. Construction validates in this exact order:

1. require `memoryPlan` non-null;
2. require `bufferIndex` non-negative;
3. require `bufferIndex < memoryPlan.buffers().size()`;
4. require `sourceRepresentationIndex` non-negative;
5. require `destinationRepresentationIndex` non-negative; and
6. require the source and destination representation indices to be distinct.

Exact failures are:

- `NullPointerException("memoryPlan")`;
- `IllegalArgumentException("bufferIndex must be non-negative")`;
- `IllegalArgumentException("bufferIndex out of prepared-plan range: X")`;
- `IllegalArgumentException("sourceRepresentationIndex must be non-negative")`;
- `IllegalArgumentException("destinationRepresentationIndex must be non-negative")`; and
- `IllegalArgumentException("sourceRepresentationIndex and destinationRepresentationIndex must be distinct")`.

The four final accessors return the exact plan reference and exact primitive coordinates. The
constructor performs no state access, physical compatibility check, allocation, callback,
validity query, or schedule action. Representation-position bounds remain cold per-run checks
because `PreparedMemoryPlan` does not describe representation counts.

## Cold binding contract

`PreparedBufferTransfer.bind` validates and acts in this exact order:

1. require `runState` non-null;
2. require the state open;
3. require `runState.memoryPlan() == memoryPlan()`;
4. require the source representation index to exist at `bufferIndex`;
5. require the destination representation index to exist at `bufferIndex`;
6. resolve the exact source representation once;
7. call `acceptsSourceBufferRepresentation` exactly once and require `true`;
8. resolve the exact destination representation once;
9. call `acceptsDestinationBufferRepresentation` exactly once and require `true`;
10. call `bindCompatible` exactly once with the exact state and exact resolved nominal references;
11. require the returned bound action non-null; and
12. require it to retain the exact state and all four exact coordinates.

Exact shared failures are:

- `NullPointerException("runState")`;
- `IllegalStateException("run state is closed")`;
- `IllegalArgumentException("run state memory plan does not match prepared buffer transfer memory plan")`;
- `IllegalArgumentException("sourceRepresentationIndex out of run-state range: X")`;
- `IllegalArgumentException("destinationRepresentationIndex out of run-state range: X")`;
- `IllegalArgumentException("source buffer representation is incompatible with prepared buffer transfer")`;
- `IllegalArgumentException("destination buffer representation is incompatible with prepared buffer transfer")`;
- `NullPointerException("boundBufferTransfer")`;
- `IllegalArgumentException("bound buffer transfer does not belong to supplied run state")`; and
- `IllegalArgumentException("bound buffer transfer does not match prepared buffer transfer positions")`.

Range failure occurs before either compatibility hook. Source compatibility failure occurs before
destination resolution/compatibility. `bindCompatible` runs only after both checks succeed.
Binding may allocate only the ordinary bound object and concrete-backend immutable/direct-reference
fields. It changes no validity or ownership, invokes no physical transfer, and acquires no
auxiliary closeable resource.

The concrete prepared subclass must be immutable and thread-safe. Its compatibility hooks use
explicit checked type tests. Its `bindCompatible` implementation performs the justified checked
casts and constructs a concrete `BoundBufferTransfer` subclass retaining direct typed source and
destination fields. It must not retain a nominal array or perform later slot resolution.

## Bound transfer construction and identity

The protected `BoundBufferTransfer` constructor retains exactly one `RunState` and the three
primitive coordinates. It validates in this exact order:

1. require `runState` non-null;
2. require it open;
3. require `bufferIndex` to address the state;
4. require `sourceRepresentationIndex` to address that buffer;
5. require `destinationRepresentationIndex` to address that buffer; and
6. require the representation indices to be distinct.

It uses existing dense-index diagnostics where applicable:

- `NullPointerException("runState")`;
- `IllegalStateException("run state is closed")`;
- `IndexOutOfBoundsException("bufferIndex out of range: X")`;
- `IndexOutOfBoundsException("representationIndex out of range: X")`; and
- `IllegalArgumentException("sourceRepresentationIndex and destinationRepresentationIndex must be distinct")`.

For either representation range failure, the existing `representationIndex` diagnostic carries
the rejected source or destination value; source is checked first. Construction performs no
physical work or validity mutation. The package-private accessors return the retained association
for final bind validation.

## Hot action and validity transition

`BoundBufferTransfer.execute()` is final and follows this exact order on every call:

1. query destination validity exactly once; the existing `RunState` open guard first rejects a
   closed state with `IllegalStateException("run state is closed")`;
2. if the destination is valid, return immediately;
3. query source validity exactly once;
4. if the source is invalid, throw
   `IllegalStateException("source buffer representation is invalid")`;
5. invoke `executeTransfer()` exactly once;
6. if it returns normally, set only the destination validity bit to `true`; and
7. return.

No validity bit is written before `executeTransfer()`. A thrown `RuntimeException` or `Error` is
propagated unchanged, with no retry, fallback, wrapping, cleanup, or suppressed failure. The
source, destination, and every other Runtime validity bit therefore retain their pre-call values.
On normal return, source and unrelated bits remain unchanged and destination becomes valid.

Sequential calls are permitted while the state remains open. After the first success, a later
call observes the valid destination and is a no-op. The bound action is not thread-safe and must
not race with state closure, execution, another transfer, or direct validity mutation.

`executeTransfer()` receives no argument because the backend subclass already retains direct
concrete typed source and destination references. It returns `void`, owns no resource, and must
not mutate Runtime validity itself. It may perform only the backend transfer/materialization work
prepared for this pair. It must not rediscover a backend, route, or representation.

## Schedule contract

`BufferTransferStep` is a public nested final record with exactly one component,
`PreparedBufferTransfer transfer`.

Its compact constructor requires the component non-null with
`NullPointerException("transfer")`, retains the exact reference, and performs no binding or
physical work. `memoryPlan()` returns exactly `transfer.memoryPlan()` without caching or copying.

Add `BufferTransferStep` to the exact permitted subclass family and declared nested surface.
Existing `PreparedSchedule` validation remains in the same order:

1. top-level null checks;
2. each step's exact-plan association in encounter order;
3. the existing first-only `RepresentationCreationStep` rule; and
4. final `List.copyOf` snapshot.

Transfer occurrences have no additional construction-time ordering restriction. Empty,
creation-only, executable-only, transfer-only, and mixed schedules remain representable at this
foundation layer when their exact plans match. Repeated transfer step or recipe references are
valid and each list position is one explicit occurrence. This task does not claim such schedules
are runnable; later Prepare validation and the runner own complete lifecycle ordering.

## Identity, ownership, side effects, and threading

- Plan association always uses exact reference identity, never structural equality.
- Buffer and representation indices are dense positions, not numeric `BufferSlot` values,
  backend/device IDs, addresses, handles, or graph identities.
- Prepared recipes and schedule steps own only immutable Java state and no physical resource.
- Bound actions own no state or physical representation. They retain references whose lifecycle
  remains owned by the run and must not close them.
- Binding transfers no ownership and changes no validity.
- A valid-destination action has no physical or Runtime side effect.
- An invalid-source failure has no physical or Runtime side effect.
- A successful backend action may mutate only the prepared destination physical representation;
  Runtime then changes only its destination validity bit.
- On backend failure the destination's physical contents may be partial or unspecified, but its
  Runtime validity remains false and all other validity remains unchanged.
- Prepared transfer recipes may bind concurrently to distinct states. A bound transfer may be
  used sequentially only and must not race any mutation or closure of its state.

## Performance limits

- Prepared construction and access are constant-time.
- Cold binding performs direct dense array/index access, two compatibility hooks, and one bound
  construction; it uses no list snapshot, map, registry, reflection, string dispatch, boxing,
  service lookup, backend discovery, graph inspection, route search, or physical action.
- Hot execution performs one destination validity operation, optionally one source validity
  operation, at most one direct backend virtual call, and after success one destination validity
  operation. Each existing `RunState` validity operation retains its constant-time open and dense-
  index guards; this task adds no duplicate resource resolution or alternative validity API.
- Hot execution performs no representation lookup, compatibility check/cast, nominal-array walk,
  allocation, synchronization, identifier creation, route/configuration choice, tracing, or
  publication.
- Tests must inspect source and bytecode sufficiently to lock the direct-reference/no-forbidden-
  mechanism boundary that ordinary behavior assertions cannot prove.

## Affected files

Expected Runtime production/Javadoc paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedBufferTransfer.java` — add.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/BoundBufferTransfer.java` — add.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/package-info.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/PreparedSchedule.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/package-info.java`.

Expected Runtime test paths:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/execution/PreparedBufferTransferTest.java` — add.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/execution/BoundBufferTransferTest.java` — add.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/schedule/PreparedScheduleTest.java`.

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status/mechanics only;
  no architecture rule change.
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `AGENTS.md`, `ARCHITECTURE.md`, ADRs
0006/0010/0011, focused lifecycle/module/dependency documents, documentation rules/profiles,
planning guide, completed Runtime 0001–0007, Prepare 0001–0002 and Draft 0003, current Runtime and
relevant Prepare source/tests/generated Javadocs, Runtime/Public APIs, backend guide, glossary,
Compile/Tensor/Training APIs, Runtime/root builds, and architecture/conformance/integration tests.

## Maximum scope

At most 16 paths: five Runtime production/Javadoc paths, three Runtime test paths, five
explanatory documentation paths, and three planning paths.

No Java/test path outside Runtime, Gradle, architecture contract, ADR, architecture-test,
backend-conformance, or integration path may change. Stop if another type, package, module edge,
behavior owner, or path is required. Do not create a later task specification.

## Test requirements

`PreparedBufferTransferTest` must cover:

- exact public abstract surface, fields, constructor, final accessors/bind method, abstract hooks,
  absence of nested/interfaces/extra members, and package placement;
- constructor validation order, exact exception types/messages, plan identity, primitive
  retention, distinct-position rule, and no action at construction;
- bind null/closed/foreign-plan/range order and exact messages;
- exact source-then-destination resolution and compatibility hook count/order;
- stop-on-first-incompatibility and no `bindCompatible` call before complete compatibility;
- exact nominal references supplied to one bind hook;
- null, foreign-state, and mismatched-position bound results;
- immutable recipe reuse across distinct isolated run states; and
- forbidden imports, allocation/resource action, lookup, discovery, registry, reflection, string,
  boxing, generic-payload, and hot-path mechanisms.

`BoundBufferTransferTest` must cover:

- exact public abstract template, private final association fields, protected constructor, final
  execute method, one abstract backend hook, package-private association accessors, and absence of
  extra public/protected surface;
- constructor null/closed/index/distinct validation in exact order and no physical/validity work;
- destination-valid no-op without source requirement or backend call;
- destination-invalid/source-invalid failure before backend work with exact message;
- successful one-call action and change of only destination validity;
- repeated sequential call becoming a destination-valid no-op;
- exact propagation of `RuntimeException` and `Error`, one attempted backend call, and unchanged
  full validity matrix on failure;
- source and other copies remaining valid after success;
- rejection after state closure before any validity/backend work;
- direct concrete source/destination field use in the test backend subclass; and
- absence of resource lookup, compatibility cast, allocation, map, reflection, string dispatch,
  boxing, registry/service, discovery, retry, cleanup, publication, or invalidation in the hot
  action.

`PreparedScheduleTest` must cover:

- the exact updated declared/permitted nested family and `BufferTransferStep` record shape;
- null transfer failure, exact transfer/plan retention, and `memoryPlan()` identity;
- same-plan and foreign-plan validation with existing first-failure order/messages;
- transfer-only and mixed creation/execution/transfer schedules;
- preservation of the creation-first rule;
- immutable order, repeated transfer steps/recipes, and construction action absence; and
- unchanged existing schedule semantics and tests.

No existing `RunState`, `PreparedExecutable`, `BoundInvocation`, or `PreparedExecution` test is
modified. No architecture, conformance, or integration test is added because no module edge,
concrete backend, or end-to-end runner behavior changes.

## Acceptance criteria

- The exact prepared/bound transfer pair and sole new schedule variant exist only in their named
  packages with no second materialization type or step.
- Source and destination are distinct representation positions of one prepared buffer position,
  and both representations must already exist in the exact run state.
- Construction and binding follow every exact validation order, failure type/message, call count,
  and identity rule in this task.
- Cold binding supplies exact direct physical references to a backend-owned bound object and
  changes no resource ownership or validity.
- A valid destination is an explicit no-op even with an invalid source.
- An invalid destination requires a valid source immediately before exactly one backend call.
- Success marks only destination valid; source and other copies keep their validity.
- Backend failure is propagated unchanged and leaves the entire Runtime validity matrix unchanged.
- The hot path contains no physical-resource lookup, map, reflection, registry/service, string
  dispatch, boxing, backend rediscovery, graph/route/configuration work, allocation, retry,
  invalidation, cleanup, publication, or tracing.
- Materialization is explained only as the same operation producing an equivalent already-created
  destination representation.
- `PreparedExecution` remains exactly memory plan plus schedule; no runner or traversal is added.
- Existing Runtime 0001–0007 and Prepare 0001–0002 behavior/history remain unchanged.
- Javadocs fully describe inputs, returns, nullability, exact identity, ownership, lifecycle,
  threading, side effects, validity transitions, performance, failures, and exclusions.
- Runtime/Public APIs, focused boundary status, backend guide, glossary, task, master plan, and
  roadmap distinguish current transfer/materialization contracts from later publication, runner,
  output invalidation, Prepare orchestration, Engine, and concrete backend work.
- A separate clean documentation-focused pass finalizes affected Javadocs/docs/examples/glossary,
  generated-page and Markdown validation, planning evidence, and reasoned no-change conclusions
  without repeating successful Java tests absent an executable change or concrete risk.
- Exactly the authorized 16 paths change; Runtime 0009–0011 and Prepare 0003 remain Draft without
  detailed specifications; links, anchors, terminology, fences, final newlines, whitespace, and
  `git diff --check` pass.

## Tests / validation

Implementation development:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedBufferTransferTest \
  --tests io.github.pho001.synaptik.runtime.execution.BoundBufferTransferTest \
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
  docs/planning/modules/runtime/tasks/0008-prepared-buffer-transfer-and-materialization-schedule.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

If the temporary validator is absent or incompatible, create an equivalent validator outside the
repository. It must check local file targets and heading anchors, unique effective anchors,
balanced backtick and tilde fences, final newlines, and trailing whitespace.

Also verify:

- exact source, compiled, reflection, and permitted-family surfaces;
- exact validation order/messages, identity, hook/action counts, no-op/success/failure validity,
  and state-closure behavior;
- direct typed source/destination retention in the concrete test bound action;
- unchanged `PreparedExecution`, `PreparedExecutable`, `BoundInvocation`,
  `PreparedRepresentationPlan`, `RunState`, `RunStateCreation`, and their compiled surfaces;
- hot-method bytecode contains only the closed/validity/action/success transition and no resource
  resolution or forbidden mechanism;
- no forbidden upstream/concrete-backend imports or map/reflection/raw/unchecked/string/boxing/
  registry/service/synchronization/discovery/route/search/config/tuning/tracing/publication/
  runner mechanism;
- unchanged Runtime/root Gradle, dependencies, architecture/ADRs/tests, Java 26 configuration,
  Prepare/Config/Trace/Backend Contract/Engine/concrete backend Java, conformance, and integration;
- exact 16-path ceiling and no Java/test outside Runtime;
- Runtime 0001–0007 and Prepare 0001–0002 remain Complete; Runtime 0008 becomes Complete only
  after all gates; Runtime 0009–0011 and Prepare 0003 remain Draft without detailed specs; and
- final documentation/status/scope/whitespace gates.

Repository-wide and architecture validation is deferred to Runtime 0011, continuous integration,
or the prepared-execution checkpoint because this task changes one module without a dependency,
build, architecture, concrete-backend, or end-to-end runner change. Backend conformance and
integration tests remain inapplicable until a concrete backend and runner consume the contract.

## Documentation pass

After implementation and final Runtime module tests, hand the exact diff and test evidence to a
distinct clean documentation-focused agent/thread. It must apply the General, API/Javadoc,
Architecture, Backend Guide, Planning, and Example profiles as appropriate and inspect the actual
source/tests rather than trusting the handoff summary.

The pass must:

- finalize both new class Javadocs, both affected package summaries, and the schedule Javadocs;
- document materialization as the same already-created-representation transfer, never a second
  operation;
- add a current Runtime API example and backend-author pattern that use direct concrete fields,
  demonstrate destination no-op/success/failure validity, and label runner usage conceptual;
- update Public API and focused architecture implementation status without changing authority;
- update the glossary for the reusable `PreparedBufferTransfer`/`BoundBufferTransfer` distinction
  and materialization boundary only if those terms need reusable cross-document navigation;
- inspect generated Runtime Javadoc pages for both new classes, `BufferTransferStep`, and the
  affected package summaries;
- validate links, anchors, terminology, examples, fences, newlines, whitespace, exact scope, and
  planning status; and
- record exact files reviewed/changed, reused test evidence, commands/results, limitations,
  unresolved issues, and reasoned no-change conclusions.

Required no-change conclusions unless a concrete contradiction is found:

- `ARCHITECTURE.md`, ADRs, lifecycle/module/dependency pages already assign explicit transfer,
  Runtime validity, backend physical mechanics, cold binding, and module direction.
- `PreparedExecution`, `PreparedExecutable`, `BoundInvocation`, representation creation,
  `RunState`, and their unaffected Javadocs remain accurate because this task adds a parallel
  transfer action without changing those contracts.
- Prepare Java/Javadocs remain unchanged because public orchestration and complete runnable-
  schedule validation remain Draft Prepare 0003.
- Compile, Tensor, and Training APIs remain unchanged because transfer coordinates and physical
  validity are Runtime-only facts.
- Config, Trace, Backend Contract, Engine, concrete backends, other guides, Gradle/dependencies,
  architecture tests, conformance, and integration remain unchanged because no consumer,
  implementation, module edge, or end-to-end behavior is added.

## Dependencies

- Runtime 0001–0007 — Complete and preserved.
- Runtime 0007 supplies complete already-created representation coordinates and explicit
  independent per-copy validity.
- Runtime 0004 supplies the established prepared/cold-bound/direct-reference design pattern.
- Runtime 0005 supplies the exact-plan sealed schedule family.
- ADR 0011 supplies the immutable prepared/per-run/backend physical ownership split.
- Existing Runtime dependencies and Java 26 build contract remain unchanged.

Prepare 0003, Runtime 0009+, Config, Trace run payloads, concrete backends, Engine, publication,
runner behavior, and tuning are not dependencies of this bounded contract.

## Follow-up tasks

- Runtime 0009 remains Draft for Runtime-owned publication/result association, delivery, and
  ownership transition.
- Runtime 0010 remains Draft for cold schedule traversal/binding, caller-input handoff, bound
  action ordering, executable-output validity/invalidation, publication, and dynamic execution.
- Runtime 0011 remains Draft for contract closure and the prepared-execution checkpoint.
- Prepare 0003 remains Draft for public preparation orchestration and complete prepared-result
  validation.
- Concrete backend tasks later implement physical representation and transfer mechanics and add
  applicable conformance coverage.

Do not create any follow-up detailed specification in this task.

## Decisions

- Use one prepared/bound pair parallel to `PreparedExecutable`/`BoundInvocation`; this keeps
  heterogeneous type checks cold and physical references direct.
- Keep both representation coordinates within one buffer index. A cross-buffer copy is not
  materialization of one logical buffer and requires a separately justified contract.
- Require distinct source/destination positions. A same-position action has no transfer meaning
  and is represented by omitting the occurrence.
- Treat materialization as transfer to an equivalent already-created destination. Add no second
  API or schedule kind.
- Make destination validity the first transition check. An already-valid destination is a no-op
  even when the selected source is invalid.
- Write only destination validity and only after backend success. Preserve source and other copies
  to avoid inventing coherence or invalidation policy.
- Permit repeated and mixed schedule occurrences. Complete runnable-order validation belongs to
  later Prepare/runner work, not the immutable foundation schedule.
- Keep `PreparedExecution` unchanged because the schedule already makes the recipe reachable.
- Use primitive dense coordinates and arrays only; add no ID, map, registry, route model, or
  generic payload.

## Known limitations

- Runtime cannot prove that backend work copied correct bytes; concrete backend tests must verify
  physical semantics and failure behavior.
- A failed backend action may have partially modified destination storage, but the destination
  remains invalid and Runtime performs no rollback.
- No automatic source choice, route search, multi-hop copy, or fallback exists; the prepared
  recipe names exactly one source and destination pair.
- No execution-output invalidation or freshness policy exists. Runtime 0010 must define explicit
  executable transitions without changing this transfer contract silently.
- Schedules remain immutable recipes rather than validated runnable programs; Prepare 0003 and
  Runtime 0010 remain future owners.
- All representations remain resident until state closure; there is no lazy creation or eviction.
- No concrete backend, runner, publication/result, or Engine lifecycle consumes the contract yet.
- The task does not require a production backend, so direct-reference behavior is demonstrated
  and locked with local concrete test subclasses and source/bytecode inspection.
- Repository-wide, architecture, conformance, and integration validation remain deferred for the
  reasons in the validation section.

## Architecture impact

Expected impact: None.

The architecture already assigns explicit prepared transfer/materialization work and per-run
validity to Runtime, physical transfer/access mechanics to concrete backends, and heterogeneous
compatibility to cold binding. This task implements that contract without a new dependency,
ownership rule, architecture decision, or `PreparedExecution` component.

If implementation requires another module edge, a second materialization concept, route search,
allocation, hidden coherence, or a completed signature change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the focused Runtime architecture
and ADR references, documentation rules/profiles, Runtime master plan, Runtime tasks 0004–0008,
and the directly relevant current Runtime source/tests/Javadocs and public/backend documentation.

Implement docs/planning/modules/runtime/tasks/0008-prepared-buffer-transfer-and-materialization-schedule.md
exactly within its 16-path ceiling. Preserve every completed contract and stop on any architecture,
package, API, validation, or scope conflict. Do not add a second materialization kind, allocation,
route search, runner/traversal, executable output invalidation, publication/result, Prepare/Engine,
concrete backend, config/tuning/tracing/coherence policy, dependency/Gradle/architecture change, or
later task specification.

Run the focused tests, one final Runtime module test, and all exact surface/mechanism/scope/status
checks. Then hand the actual diff and exact Java evidence to a separate clean documentation-focused
context. That pass must follow documentation-rules.md, independently finalize affected Javadocs,
docs, examples, glossary impact, and planning evidence, and must not repeat successful Java tests
without an executable change or a recorded concrete risk.

Mark Complete only after every implementation and documentation gate passes. Return both context
IDs, exact changed paths, commands/results/test counts, no-change conclusions, unresolved issues,
follow-up, and the repository completion status. Do not commit or push.
```

## Local decisions

- The public class names and schedule variant are fixed by this task; implementation must not
  rename them or replace them with records/interfaces.
- `execute()` is the common final transition method and `executeTransfer()` is the sole backend
  hot hook, mirroring the established final-template pattern without conflating it with executable
  computation.
- Bound association coordinates are retained as primitive fields so final cold binding can reject
  a backend-created object for the wrong state or positions without reflection or a registry.
- Exact validation messages are part of the contract and tests because they locate cold binding
  failures before physical work.

## Validation evidence

- Implementation context: `019fbe31-7ba9-7b20-b4ca-c7a5ea9cf4d9`
  (`/root/runtime_0008_impl`). The final focused pass reported 31 tests: 8
  `PreparedBufferTransferTest`, 8 `BoundBufferTransferTest`, and 15
  `PreparedScheduleTest`, with zero failures, errors, or skips. The one final
  `./gradlew :modules:runtime:test` pass reported 13 suites and 113 tests with zero failures,
  errors, or skips. No executable Java changed after that pass.
- Documentation context: `/root/runtime_0008_docs`. It independently inspected the actual diff,
  relevant architecture and planning contracts, current source/tests, affected Javadocs, and the
  General, API/Javadoc, Architecture, Backend Guide, Planning, and Example documentation profiles.
  It reused the successful Java evidence and did not repeat Java tests.
- `./gradlew :modules:runtime:javadoc` completed successfully with five actionable tasks, one
  executed and four up-to-date. Generated pages for `PreparedBufferTransfer`,
  `BoundBufferTransfer`, `PreparedSchedule.BufferTransferStep`, and both affected package
  summaries were inspected for the exact transfer/materialization, identity, lifecycle,
  validity, failure, direct-reference, threading, and current-versus-planned boundaries.
- `/tmp/validate_synaptik_markdown.py` validated all eight affected Markdown files. The validator
  checked local targets and anchors, unique effective anchors, balanced backtick and tilde
  fences, final newlines, and trailing whitespace.
- `javap -p` confirmed the exact compiled prepared/bound class surfaces, package-private final
  bound-association accessors, unchanged `PreparedSchedule` record surface, the exact sealed
  `Step` family, and the sole `BufferTransferStep` addition. `javap -p -c` confirmed that hot
  `BoundBufferTransfer.execute()` performs only destination-valid no-op, source-valid rejection,
  one direct backend hook, and success-only destination validity.
- Source and test inspection confirmed exact construction/binding order and diagnostics, direct
  concrete `source` and `destination` fields in the test bound subclass, the required validity
  transitions and failure propagation, and absence of forbidden hot-path imports or mechanisms.
- `git diff --exit-code` confirmed no changes to the named unaffected Runtime contracts, Prepare,
  Compile, Tensor, Training, build files, architecture contract, ADRs, focused lifecycle/module/
  dependency documents, or architecture tests. Scope inspection confirmed exactly the authorized
  16 paths, no Java/test path outside Runtime, and no Runtime 0009-0011 or Prepare 0003 detailed
  specification.
- `git diff --check` and the final status/scope/planning-state gates passed after this evidence was
  recorded. Runtime 0001-0007 and Prepare 0001-0002 remain Complete; Runtime 0008 is Complete;
  Runtime 0009-0011 and Prepare 0003 remain Draft.

## Implementation notes

- Added the exact immutable `PreparedBufferTransfer` recipe and per-run `BoundBufferTransfer`
  template in Runtime execution. Cold binding resolves and checks the two already-created
  representations once and validates the exact returned state and coordinates.
- Added only `PreparedSchedule.BufferTransferStep` to the closed schedule family. Existing exact-
  plan validation, encounter order, immutability, and creation-prefix behavior remain intact.
- The final hot action makes an already-valid destination a no-op, rejects an invalid source,
  calls backend work once, and marks only the destination valid after success. Backend failure
  leaves Runtime validity unchanged and propagates unchanged.
- Finalized all affected class, method, record, and package Javadocs. Runtime/Public API,
  focused-boundary, backend-author, glossary, master-plan, and roadmap documentation now describe
  materialization as this same transfer to an equivalent already-created representation and keep
  runner, executable-output invalidation, publication/result, Prepare orchestration, Engine, and
  concrete backend implementation explicitly planned.
- No architecture contract, ADR, module edge, dependency, Gradle setting, Java version, or
  concrete-backend/end-to-end behavior changed. `PreparedExecution`, `PreparedExecutable`,
  `BoundInvocation`, representation creation, `RunState`, and their Javadocs remain accurate
  because this is a parallel action contract. Prepare Java/Javadocs remain unchanged because
  public orchestration and complete runnable-schedule validation remain Draft Prepare 0003.
  Compile, Tensor, and Training APIs remain unchanged because physical coordinates and validity
  are Runtime-only facts. Config, Trace, Backend Contract, Engine, other guides, architecture
  tests, backend conformance, and integration remain unchanged because no consumer, concrete
  implementation, module edge, or runnable lifecycle was added.

## Completion summary

- Completed changes: implemented and documented the reusable prepared transfer, per-run direct-
  reference bound action, success-only validity transition, and the one exact-plan schedule
  occurrence.
- Files changed or created: five Runtime production/Javadoc paths, three Runtime test paths, five
  explanatory documentation paths, and three planning paths, exactly matching the 16-path ceiling.
- Tests and validation: focused Runtime tests 31/31; final Runtime module tests 113/113; Runtime
  Javadoc successful; generated pages, compiled/reflection surface, hot bytecode, direct fields,
  forbidden mechanisms, Markdown, unchanged-area, scope, planning-state, and whitespace gates
  passed.
- Unresolved issues: none within task scope.
- Required follow-up: Runtime 0009-0011, Prepare 0003, concrete backend transfer mechanics, and
  end-to-end consumption remain their recorded Draft/future work and are not required for this
  bounded contract.

Status: Complete
