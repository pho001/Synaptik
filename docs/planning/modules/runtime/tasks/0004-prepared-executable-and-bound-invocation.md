# Task 0004: Prepared Executable and Bound Invocation

## Status

Ready

## Goal

Implement the smallest Runtime-owned cold-binding boundary authorized by
[ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md). One immutable,
reusable backend-owned `PreparedExecutable` recipe selects exact dense positions from one open
`RunState`, performs checked backend compatibility before execution, and returns one per-run
backend-owned `BoundInvocation` that retains direct typed resource references.

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

The concrete backend subclasses both abstract classes. Its `PreparedExecutable` implementation is
an immutable recipe created only after route choice and shared slot assignment. Its
`BoundInvocation` implementation stores checked concrete buffer/workspace references directly in
typed fields. Runtime supplies plan, selection, run-association, and closed-state validation; it
never knows a concrete backend representation class.

## Rationale and mental model

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

The complete heterogeneous Java type relation cannot be expressed in one shared generic
signature. The abstract hooks confine that check to a deliberate backend implementation point.
The hot call has no graph, planning, routing, lookup, selection, or cast input.

`PreparedUnit` is deliberately not introduced. No current schedule, partition-finalization, or
input/output-role consumer establishes a distinct invariant for it. Adding it now would only wrap
one executable or duplicate its selections. Prepare 0002 or Runtime 0005 must justify and define
that association when prepared partition or schedule construction is current.

## Scope

- Add the exact two top-level public abstract classes and two nested public selection records
  shown above in `io.github.pho001.synaptik.runtime.execution`.
- Make `PreparedExecutable` retain the exact non-null `PreparedMemoryPlan` and private immutable
  array snapshots of ordered buffer and workspace selections.
- Interpret `bufferIndex` and `workspaceIndex` as dense zero-based positions in
  `PreparedMemoryPlan.buffers()` and `PreparedMemoryPlan.workspaces()` encounter order, never as
  slot numeric components.
- Interpret `representationIndex` as the dense zero-based position in the selected run state's
  ordered bindings for that prepared buffer position.
- Permit empty selection lists and repeated selections. Repetition is required for repeated
  operand roles and does not duplicate ownership in `RunState`.
- Validate prepared-plan-relative selection bounds at executable construction and run-dynamic
  representation bounds at cold binding.
- Require exact `PreparedMemoryPlan` reference identity between the recipe and `RunState`; equal
  geometry from another plan is not the same association.
- Resolve selected nominal representations into fresh cold-path arrays in selection order.
- Let concrete backend hooks perform explicit `instanceof` compatibility checks without
  reflection, raw `Object`, unchecked generic access, or a public concrete-backend switch.
- Standardize incompatibility diagnostics in the final Runtime binding method.
- Require `bindCompatible` to create a backend-owned `BoundInvocation` that retains the exact
  supplied `RunState` and direct concrete typed references, not the nominal selection arrays.
- Check the exact run-state association of the returned invocation.
- Guard `BoundInvocation.execute()` with one minimal open-state check before delegating to the
  backend's `executeBound()` implementation.
- Add focused tests for exact surface, validation/failure order, immutable snapshots, selection
  order, exact reference retention, checked compatibility, lifecycle, direct typed binding,
  concurrency contract, and hot-path exclusions.
- Finalize all affected Javadocs and explanatory documentation in the required separate clean
  documentation-focused context.

## Out of scope

- `PreparedUnit`, `PreparedPartition`, Prepare assignment/finalization, or source-to-slot
  associations
- input/output/result role metadata or schedule-step identity
- `PreparedSchedule`, transfer, materialization, residency/validity mutation, publication,
  `RunResult`, `PreparedExecution`, or a runner
- representation creation, physical allocation, storage access mechanics, transfer mechanics, or
  ownership changes
- auxiliary binding resources, native-handle acquisition, or an `AutoCloseable` invocation
  lifecycle
- immutable persistent prepared-resource ownership or closing `PreparedExecutable`
- concrete backend production classes, kernels, routes, native bridges, or backend configuration
- graph, compiler, planning, Prepare, Model, Engine, `Operation`, `CompiledNode`, partition, trace,
  backend identity, or device parameters on `bind` or `execute`
- allocation, transfer, residency, route/config search, lowering, kernel selection, backend
  discovery, fallback, service lookup, tuning, profiling, or trace emission
- maps, registries, service locators, string dispatch, reflection in production, raw `Object`,
  unchecked casts/generics, or public switches over concrete backend types
- concurrency support for one `RunState` or one `BoundInvocation`
- pooling, aliasing, hidden coherence/write-back, distributed sharding, or multi-device scheduling
- Runtime 0005+, Prepare 0002, or any other detailed task specification
- Gradle, dependency, architecture-contract, ADR, architecture-test, backend-conformance, or
  integration-test changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - Concrete backend modules
  - Runtime service locator
  - Prepare and run lifecycles
  - Dependency rules
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0006: No Runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)

## Architecture constraints

- `PreparedExecutable` is immutable reusable prepared state and may bind concurrently to distinct
  open `RunState` instances. Concrete subclasses must also be immutable and thread-safe.
- Every `BoundInvocation` belongs to exactly one `RunState` and one logical run. It is not
  thread-safe; callers must not execute it concurrently or race execution with state closure.
- A bound invocation does not own or close its `RunState`, buffer representations, workspace
  representations, or immutable prepared resources.
- Runtime owns common plan/selection/lifecycle validation. Concrete backends own representation
  implementations, checked type compatibility, typed direct-reference fields, and region work.
- Dynamic compatibility checks occur only during `bind`. `execute()` performs only the minimal
  run-open guard followed by the backend call.
- Runtime remains independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backends. Backend Contract remains closed.
- No implementation choice, backend lookup, route choice, allocation, transfer, or publication
  may be deferred into `execute()`.
- If exact compatibility requires a shared registry, raw/untyped carrier, reflective class token,
  unchecked generic API, or another module edge, stop and report an architecture conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.runtime.memory` — consume the exact `PreparedMemoryPlan` reference and
  its dense entry counts without modification.
- `io.github.pho001.synaptik.runtime.resource` — consume nominal buffer/workspace representation
  roles without adding physical access.
- `io.github.pho001.synaptik.runtime.run` — consume `RunState`; update only its Javadoc and package
  status wording for the now-current binding lifecycle.

Package added:

- `io.github.pho001.synaptik.runtime.execution` — public prepared executable and per-run bound
  invocation contracts.

Type placement:

- `runtime.execution.PreparedExecutable` — Runtime owns the backend-neutral reusable recipe and
  final cold-binding template; concrete backend modules subclass it.
- `PreparedExecutable.BufferSelection` — nested because the dense buffer/representation pair is
  meaningful only to executable binding.
- `PreparedExecutable.WorkspaceSelection` — nested for the corresponding workspace position.
- `runtime.execution.BoundInvocation` — Runtime owns the common per-run lifecycle guard while a
  concrete backend subclass owns typed fields and execution.

Tests mirror `runtime.execution`. No root facade or other Java package is added.

## Affected files

Expected Runtime production paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutable.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/BoundInvocation.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java` — Javadoc only
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

Expected planning paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs 0006/0010/0011,
lifecycle/module/dependency/overview docs, Runtime 0001–0003, Prepare 0001/master plan, all current
Runtime/Prepare source/tests/generated Javadocs/build files, Config/Trace/Engine/Backend Contract
and concrete-backend plans, backend partition-preparer guide, and architecture/conformance/
integration tests.

## Maximum scope

At most 15 paths:

- 5 Runtime production/Javadoc paths;
- 2 Runtime test paths;
- 5 explanatory documentation paths; and
- 3 Runtime/global planning paths.

No Java/test path outside Runtime, Gradle path, architecture contract, ADR, architecture-test,
backend-conformance, or integration path may change. Stop if another public type, package, module
edge, executable behavior owner, or path is required. Do not create a later task specification.

## Validation, ordering, and failure rules

Selection constructors validate components in declaration order:

- negative `BufferSelection.bufferIndex`:
  `IllegalArgumentException("bufferIndex must be non-negative")`;
- negative `BufferSelection.representationIndex`:
  `IllegalArgumentException("representationIndex must be non-negative")`; and
- negative `WorkspaceSelection.workspaceIndex`:
  `IllegalArgumentException("workspaceIndex must be non-negative")`.

`PreparedExecutable` construction validates in this exact order:

1. require `memoryPlan`, then `bufferSelections`, then `workspaceSelections` non-null;
2. scan buffer selections in supplied order, rejecting the first null entry and then the first
   `bufferIndex` outside `memoryPlan.buffers()`;
3. snapshot buffer selections into a private array;
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
Selection order is argument order for both compatibility hooks and `bindCompatible`. Empty lists
and repeated selections are valid. Construction allocates only private JVM arrays and performs no
physical resource operation or identifier allocation/consumption.

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
   original selection order;
8. reject a null result with `NullPointerException("boundInvocation")`; and
9. require the returned invocation to retain that exact run-state reference, otherwise throw
   `IllegalArgumentException("bound invocation does not belong to supplied run state")`.

The backend compatibility hooks must use explicit checked type tests such as `instanceof`. A
normal incompatibility returns false so Runtime emits the exact indexed diagnostic. The backend
must not use reflection, class-name strings, raw `Object`, unchecked casts/generics, a registry,
or a shared/concrete-backend switch. `bindCompatible` may use ordinary checked casts after the
successful compatibility pass, but it must create an invocation with concrete typed fields and
must not retain the nominal arrays as its hot-path access mechanism.

Binding may allocate ordinary JVM arrays and the bound invocation object. This task deliberately
forbids acquisition of an independently closeable/native auxiliary binding resource. Therefore a
failed bind changes no ownership and needs no cleanup protocol. A future task must add an explicit
`AutoCloseable` lifecycle and partial-failure cleanup before permitting such resources.

`BoundInvocation` construction requires `runState` non-null, checks it is open, and retains the
exact reference. Exact failures are `NullPointerException("runState")` and
`IllegalStateException("run state is closed")`.

`execute()` first checks the retained state and fails with
`IllegalStateException("run state is closed")` before any backend call. Otherwise it invokes
`executeBound()` exactly once and returns normally, or propagates the backend's exact
`RuntimeException` or `Error` without fallback, wrapping, cleanup, or retry. Sequential calls while
the state remains open are permitted; no one-shot/idempotence promise is made. Concurrent calls
or a race with `RunState.close()` are unsupported because both objects belong to one
single-orchestrator-thread run.

## Hot-path and lifecycle contract

- `PreparedExecutable` and its selection snapshots are immutable. Concrete subclasses must retain
  only immutable prepared state and may be bound concurrently to different run states.
- `BoundInvocation` strongly retains its exact `RunState`; retaining the Java object does not keep
  its representations logically open after `RunState.close()`.
- The invocation never owns or closes the state or any representation.
- Each successful backend subclass stores direct concrete typed references during cold binding.
- `execute()` accepts no argument and performs no array/list/map/slot lookup, representation
  selection, compatibility test/cast, graph inspection, backend discovery, lowering, kernel or
  route selection, configuration search, allocation, transfer, residency decision, publication,
  tuning, profiling selection, or tracing emission.
- The only shared hot-path work is one `RunState.isClosed()` guard and one direct virtual call to
  `executeBound()`.

## Acceptance criteria

- The exact two-top-level/two-nested public surface exists only in `runtime.execution`; no
  `PreparedUnit` or other execution type is added.
- Constructors, final template methods, protected hooks, nested records, modifiers, component
  types, and declared exceptions match this task exactly.
- Prepared selections are immutable private-array snapshots in deterministic supplied order,
  retain exact selection references, and use dense positions rather than slot numeric values.
- Construction and binding follow the exact validation order and messages above.
- Binding requires exact plan identity, rejects closed state, validates run-dynamic representation
  indices, checks compatibility once per selected representation, and validates exact invocation
  association before returning.
- Focused fake backend tests prove explicit checked `instanceof` compatibility and direct concrete
  buffer/workspace fields in the bound subclass. No concrete backend production code is added.
- Bound execution checks state closure before backend work, performs no resource lookup/cast, and
  propagates backend unchecked failures unchanged.
- Tests prove one prepared recipe binds concurrently to distinct states while each invocation and
  run-owned resource remains isolated; the test does not concurrently execute one invocation.
- Production uses no map, reflection, string dispatch, raw `Object`, unchecked generic access,
  registry, service locator, public concrete-backend switch, graph/compiler/planning/Prepare/Model/
  Engine type, concrete backend import, or identifier side effect.
- No allocation/access mechanics, validity/residency, transfer, schedule, runner, publication,
  result, persistent-resource lifecycle, tracing emission, tuning, backend, Prepare, Gradle,
  dependency, or architecture behavior is added.
- Every public/protected contract has complete Javadoc for inputs, result, failure, ownership,
  reference retention, immutability, lifecycle, thread safety, selection order, hot-path behavior,
  and unsupported behavior.
- A separate clean documentation-focused agent finalizes affected Java/package Javadocs, Runtime
  and Public API status, focused boundary status, backend guide, glossary, planning evidence, and
  reasoned no-change conclusions in the same overall change.
- Runtime 0001–0003 and Prepare 0001 remain Complete; Runtime 0004 becomes Complete only after all
  gates; Runtime 0005+ and Prepare 0002 remain Draft without detailed specifications.
- Exact 15-path scope, Markdown/link/anchor/fence/newline/whitespace, and `git diff --check` gates
  pass.

## Tests / validation

Implementation-focused validation:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutableTest \
  --tests io.github.pho001.synaptik.runtime.execution.BoundInvocationTest

./gradlew :modules:runtime:test
```

Run the focused command while developing and one final module command after executable Java
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
repository. Validate local link targets and heading anchors, unique effective anchors, balanced
backtick and tilde fences, final newlines, and trailing whitespace.

Required source/scope/status checks:

- exact public/protected/nested surface and package placement;
- exact validation order/messages, array snapshots, selection order, reference retention, plan
  identity, compatibility call counts, invocation association, and lifecycle behavior;
- fake backend bound objects have direct concrete typed fields and execute without resource access;
- production mechanism/import scan for no maps, reflection, raw/unchecked access, strings used for
  dispatch, registries/service locators, concrete backends, or forbidden graph/lifecycle types;
- unchanged Runtime/root Gradle files and Java 26 root toolchain/release configuration;
- exact 15-path ceiling and no Java/test outside Runtime;
- Runtime 0001–0004 and Prepare 0001 status synchronization after completion;
- Runtime 0005–0008 and Prepare 0002 remain Draft, exactly Runtime 0001–0004 have specs, exactly
  Prepare 0001 has a spec, and no later spec exists; and
- final newlines, trailing whitespace, and `git diff --check`.

Repository-wide tests and architecture tests are deferred to the Runtime prepared-contract
capability checkpoint or continuous integration. This task changes one module without a module
edge, dependency rule, shared build contract, concrete backend, or end-to-end behavior.
Backend-conformance and integration tests are not applicable yet.

The documentation context reuses successful Runtime Java-test evidence unless it changes
executable behavior or records a concrete reason to repeat it.

## Dependencies

- Runtime 0001–0003 — Complete.
- Prepare 0001 — Complete but not imported by Runtime.
- ADR 0010 staged preparation — Accepted.
- ADR 0011 per-run ownership and checked cold binding — Accepted and sufficient for this task.
- Existing Runtime dependency and Java 26 build contracts — unchanged.

Prepare 0002, Runtime 0005+, concrete backends, schedules, publication/results, transfer,
residency, allocation, and Engine are not dependencies of this bounded contract task.

## Follow-up tasks

- Prepare 0002 may use this executable contract when it defines shared assignment associations and
  backend finalization against assigned slots. It remains Draft without a detailed specification.
- Runtime 0005 must decide whether a distinct `PreparedUnit` is justified by the actual prepared
  partition/schedule consumer and then define ordered schedule work. It must not be created here.
- Runtime 0006–0008 remain Draft for prepared aggregate/lifecycle, runner/dynamic execution, and
  closure audit.

## Javadocs and documentation impact

- New execution-package Javadocs must explain the cold-versus-hot boundary, ordered dense
  selections, exact plan/run association, checked compatibility, direct typed references,
  lifecycle, threading, failures, and all deliberate exclusions.
- `RunState` and `runtime.run` package Javadocs need current wording for binding and post-close
  invocation rejection; executable behavior in `RunState` does not change.
- Runtime API must make `PreparedExecutable`, `BoundInvocation`, and their fake-backend example
  current while preserving schedules, allocation, residency, transfer, publication/results, and
  runner as planned.
- Public API status must list the new Runtime execution contracts without claiming a runnable
  public lifecycle.
- The focused architecture page receives implementation-status changes and a concrete current
  cold-binding explanation only. It must not alter the architecture rule.
- The backend guide must show explicit checked compatibility and direct concrete fields without
  suggesting a current concrete backend or Prepare finalizer.
- The glossary makes `PreparedExecutable` current, adds `BoundInvocation`, and updates `RunState`
  current limits. No ordinary programming term is added.
- Lifecycle, module-boundary, dependency, overview, ADRs, other API/user guides, backend partition
  analysis, Config/Trace/Engine/backend plans, architecture tests, conformance, integration, and
  Gradle remain review-only unless the implementation reveals a concrete contradiction.

## Architecture impact

Expected impact: None.

ADR 0011 and the architecture contract already authorize the exact checked cold-binding boundary,
backend-owned typed invocation objects, direct references, one-run association, and hot-path
exclusions. This task introduces no module edge or rule. Stop if implementation requires a shared
resource registry, generic catch-all abstraction, unchecked public API, concrete-backend knowledge
inside Runtime, execution after run closure, or auxiliary-resource ownership without an explicit
lifecycle.

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
  backends.
- Use two nested immutable selection records and private arrays. This gives prepared finalization
  explicit dense coordinates without maps, boxing-dependent lookup, slot-value-sized arrays, or a
  redundant `PreparedUnit`.
- Allow repeated selections because one prepared region may consume the same physical
  representation in more than one logical argument role. `RunState` still owns each exact
  representation only once.
- Keep compatibility hooks boolean and backend-owned. Runtime owns exact failure diagnostics while
  concrete code uses `instanceof` and constructs a typed invocation after all checks pass.
- Require exact `PreparedMemoryPlan` object identity. `RunState` and the executable must refer to
  the same prepared recipe; structural equality is insufficient across independently constructed
  plans.
- Retain a minimal execute-time state guard. Cold validation alone cannot prevent a previously
  bound invocation from running after its state closes, and one boolean check adds no graph/map/
  resource lookup.
- Permit sequential calls while open but prohibit concurrent execution/races. Later schedule work
  owns normal call cardinality.
- Defer auxiliary binding resources completely. The bound invocation owns no closeable state, so
  no `AutoCloseable` wrapper or partial-failure cleanup protocol is justified now.

## Known limitations

- No current shared contract labels buffer selections as input, output, or in-place roles.
  Prepare 0002/Runtime 0005 must introduce only the associations their concrete consumers need.
- `PreparedMemoryPlan` does not describe representation counts or kinds. The immutable executable
  records a non-negative representation index, and binding validates it against each actual
  matching `RunState` before backend compatibility.
- Abstract classes document but cannot mechanically prove that every backend subclass is
  immutable, uses `instanceof` correctly, or stores only direct concrete fields. Focused fake
  backend tests lock the intended extension pattern; future concrete backends require their own
  unit and conformance tests.
- No schedule, allocation, validity/residency, transfer, publication, result, runner, persistent-
  resource cleanup, or concrete backend behavior exists.
- Repository-wide and architecture-test validation remains deferred to the Runtime prepared-
  contract checkpoint or continuous integration because no dependency or architecture rule
  changes.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
