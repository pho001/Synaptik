# Task 0007: Representation Creation and Residency Foundation

## Status

Complete

## Goal

Add the smallest Runtime-owned prepared description and cold per-run setup needed to turn current
prepared memory geometry into one complete `RunState`, while keeping concrete physical classes
and creation mechanics inside concrete backends.

```text
immutable prepared description
  -> cold caller-input binding and backend creation
  -> mutable per-run validity in RunState
```

The prepared description distinguishes caller-supplied buffers from backend-created buffers and
workspace callbacks. Cold setup validates every caller input before invoking a callback, creates
run-owned representations deterministically, and transfers cleanup only after `RunState`
construction succeeds. Caller inputs begin valid. Created buffers begin invalid. Workspaces are
run-owned backend-local scratch and never represent a coherent logical value.

Preserve the completed aggregate exactly:

```java
public record PreparedExecution(
        PreparedMemoryPlan memoryPlan,
        PreparedSchedule schedule) {}
```

The new prepared representation plan remains reachable through the existing schedule component.

## Scope

- Add one immutable public `PreparedRepresentationPlan` in `runtime.resource` with nested typed
  caller-input, created-buffer, buffer-creator, and workspace-creator contracts.
- Add one `PreparedSchedule.RepresentationCreationStep` that retains the plan. It is optional for
  compatibility, but when present must be the sole creation step and the first occurrence.
- Add one package-private `RunStateCreation` cold operation in `runtime.run`.
- Accept caller inputs in deterministic dense encounter order, always as `BORROWED`.
- Invoke buffer creators in buffer/representation order, then workspace creators in workspace
  order; every callback result is `RUN_OWNED`.
- Validate all caller references and identity uniqueness before invoking any callback.
- On partial failure, close successfully created resources once in reverse creation order, never
  close borrowed resources, preserve the original unchecked failure, and suppress cleanup
  failures in encounter order.
- Add one array-backed validity bit for every `RunState` buffer representation and explicit
  constant-time query/mutation methods.
- Initialize `BORROWED` buffers valid and `RUN_OWNED` buffers invalid, including through the
  existing public `RunState` constructor.
- Define binding presence as residency for this foundation: the exact physical object exists for
  the run until closure. Validity separately records whether it contains the logical slot value.
- Keep workspaces structurally resident, run-owned, and outside logical validity.
- Preserve the existing cold typed binding and direct-reference hot path.
- Add focused exact-surface, validation, creation, rollback, validity, schedule, and isolation
  tests and finalize affected Javadocs/documentation in a separate clean documentation context.
- Synchronize this task, the Runtime master plan, and the roadmap without rewriting completed
  history.

## Out of scope

- transfer routes or callbacks, copies, materialization, or transfer schedule steps
- kernels, schedule execution, a runner, or automatic validity changes around execution
- hidden coherence, implicit write-back, dirty-state policy, or automatic synchronization
- creation, eviction, replacement, or removal after `RunState` construction
- lazy allocation, pooling, caching, reuse, aliasing, or lifetime/interference analysis
- physical access, addresses, native handles, backend/device inspection, or concrete physical
  representation classes in Runtime
- broad allocators, factories, managers, registries, services, service locators, discovery, or a
  public orchestration facade
- maps, reflection, string dispatch, raw `Object`, unchecked generics, synchronization, or public
  concrete-backend switches
- lowering, route/kernel selection, fallback, tuning/cache behavior, tracing, or profiling
- caller-input names, graph/value identities, Prepare source associations, publication,
  ownership transfer/lease, `RunResult`, or Config 0007
- changes to `PreparedExecution`, `PreparedExecutable`, `BoundInvocation`, `PreparedMemoryPlan`,
  `BufferRepresentationBinding`, `RunResourceOwnership`, or the two representation interfaces
- Prepare, Engine, concrete backend, Gradle, dependency, architecture, ADR, architecture-test,
  conformance-test, or integration-test implementation
- detailed Runtime 0008–0011 or Prepare 0003 specifications

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): core invariants; Runtime, Prepare, and
  concrete-backend ownership; run lifecycle; service-locator and dependency rules.
- [ADR 0006: No Runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)
- [Runtime 0005](0005-prepared-schedule-contract.md)
- [Runtime 0006](0006-prepared-execution-aggregate.md)
- [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Prepared recipes remain immutable, reusable, and thread-safe. Every active run receives one
  distinct mutable `RunState` and distinct run-owned callback results.
- Runtime owns prepared representation coordinates, per-run orchestration, ownership, logical
  validity/residency, rollback, cleanup, and run isolation.
- Concrete backend modules implement creator callbacks and physical representation classes. They
  own allocation, release, transfer, and access mechanics. Runtime invokes but never interprets
  them.
- Caller inputs are borrowed and initially valid. Created buffers are run-owned and initially
  invalid. All workspaces are run-owned scratch without logical validity.
- A buffer may have multiple resident representations and any subset may be valid. Zero valid
  copies is permitted before an internal buffer is first produced.
- Validity mutation changes a fact only. It performs no copy, kernel call, storage access,
  ownership transition, or implicit invalidation.
- Runtime 0008 and later runner/execution work must query and mutate validity explicitly around
  successful prepared actions; they must not infer coherence from ownership or concrete class.
- Creation is cold. `PreparedExecutable.bind` and `BoundInvocation.execute` remain unchanged and
  gain no creation, validity lookup, map access, or repeated cast.
- The representation plan enters `PreparedExecution` only through its existing exact schedule.
  `PreparedExecution(memoryPlan, schedule)` remains the entire aggregate shape.
- Runtime remains independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backends. Backend Contract remains closed. Config 0007 is not a dependency.
- If implementation needs a new module edge, an incompatible completed signature change, or an
  architecture-contract update, stop and report the exact decision instead of editing.

## Package impact

Existing packages used or changed:

- `runtime.memory` — consume `PreparedMemoryPlan` and dense encounter order without modification.
- `runtime.resource` — add the immutable representation description beside nominal physical roles.
- `runtime.run` — add package-private cold setup and explicit per-run validity.
- `runtime.schedule` — add the sole currently justified non-executable setup variant.

No package is added, moved, or removed.

Type placement:

- `runtime.resource.PreparedRepresentationPlan` — Runtime owns reusable backend-neutral
  coordinates and callback roles; concrete backends supply implementations.
- Its nested preparation variants and creators — meaningful only inside that plan, so they do not
  become five more top-level types.
- `runtime.run.RunStateCreation` — package-private cold operation for the future Runtime runner,
  not a public lifecycle facade.
- `PreparedSchedule.RepresentationCreationStep` — keeps creation reachable through the unchanged
  `PreparedExecution` aggregate.

Tests mirror `runtime.run` and `runtime.schedule`. No root facade or generic helper package is
introduced.

## Exact public and package-private API declarations

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

Add exactly these `RunState` members:

```java
public boolean isBufferRepresentationValid(
        int bufferIndex,
        int representationIndex);

public void setBufferRepresentationValid(
        int bufferIndex,
        int representationIndex,
        boolean valid);
```

Add exactly this package-private operation:

```java
final class RunStateCreation {
    static RunState create(
            PreparedRepresentationPlan representationPlan,
            List<BufferRepresentation> callerInputs);
}
```

`RunState` adds one private final `boolean[][] bufferValidity` matching `bufferBindings`. No other
public, protected, or package-private member or top-level production type is added. Its existing
constructor and all completed signatures remain unchanged.

## Prepared representation-plan semantics

`PreparedRepresentationPlan` retains the exact non-null memory plan, snapshots both buffer-list
levels and the workspace list, and retains exact immutable preparation/callback references.
Callback implementations must be immutable and thread-safe because one plan may create concurrent
runs.

Validation order:

1. require `memoryPlan`, `bufferPreparations`, then `workspaceCreators` non-null;
2. require outer buffer count equal `memoryPlan.buffers().size()`;
3. require workspace count equal `memoryPlan.workspaces().size()`;
4. scan buffer positions: require each inner list non-null and non-empty, then each preparation
   non-null in representation order;
5. snapshot each inner list and then the outer list; and
6. scan workspace creators for null in order, then snapshot.

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
no component or global identity; every occurrence is one dense input position. Creator methods
declare no checked exception. Each successful callback must return one non-null fresh run-owned
representation not reused by another position or concurrent run. Cold creation validates the
non-null and within-run identity portions.

Plan construction invokes no callback, allocates no physical storage, creates no `RunState`, and
changes no validity.

## Schedule semantics

`RepresentationCreationStep` validates `representationPlan` with
`NullPointerException("representationPlan")`, retains it exactly, and returns exactly
`representationPlan.memoryPlan()`.

Existing top-level and same-plan validation order remains unchanged. After a non-null step reports
the exact schedule plan, apply this additional rule:

- a `RepresentationCreationStep` at any index other than zero fails with
  `IllegalArgumentException("steps[i] representation creation must be the first schedule occurrence")`.

There can therefore be zero or one creation step, only at index zero. Existing empty and
executable-only schedules remain valid. Repeated execution occurrences remain valid. Prepare 0003
may later require one creation prefix for a runnable final result; task 0007 does not invalidate
foundation schedules or implement that validator.

## Cold creation, ownership, and cleanup

`RunStateCreation.create` accepts caller inputs as one flat list in the encounter order of all
`CallerInput` occurrences: buffer position first, then representation position. It performs no
schedule traversal, executable binding, transfer, or execution.

Exact order:

1. require `representationPlan`, then `callerInputs` non-null;
2. count caller-input occurrences without invoking callbacks;
3. require exact caller-input count;
4. scan all caller inputs for null and repeated exact identity;
5. traverse buffer preparations, retaining caller inputs as `BORROWED` or invoking creators and
   retaining results as `RUN_OWNED`;
6. invoke workspace creators in order and retain every result as run-owned;
7. construct one `RunState` with the exact plan and completed structures; and
8. return it, transferring cleanup of created results only after construction succeeds.

Exact pre-creation failures:

- `NullPointerException("representationPlan")`
- `NullPointerException("callerInputs")`
- `IllegalArgumentException("callerInputs size must equal caller-input preparation count N")`
- `NullPointerException("callerInputs[i]")`
- `IllegalArgumentException("representation is already bound to this run")`

Null callback results fail with:

- `NullPointerException("bufferPreparations[i][j] creator result")`; or
- `NullPointerException("workspaceCreators[i] result")`.

A callback result duplicating a caller input or earlier result fails with the existing exact
`IllegalArgumentException("representation is already bound to this run")`.

Any `RuntimeException` or `Error` after callback invocation starts triggers cleanup of all
successfully created results in reverse creation order. The original failure is rethrown unchanged;
cleanup failures are suppressed in cleanup encounter order. Borrowed inputs are never closed. A
non-null duplicate callback result is not closed as a second owned object because it aliases a
borrowed or already-owned physical object.

Successful creation order is created buffers in dense buffer/representation order, then
workspaces. Existing `RunState.close()` already closes workspaces in reverse and then run-owned
buffers in reverse, so successful cleanup is also reverse creation order and skips borrowed
inputs.

Use arrays and direct iteration only: no map, reflection, string dispatch, lookup service,
synchronization, backend selection, or physical access.

## Validity, residency, and transitions

For one open state:

- every bound buffer/workspace representation is resident until closure;
- a buffer validity bit says whether that resident copy currently contains the logical slot value;
- `BORROWED` starts valid and `RUN_OWNED` starts invalid;
- multiple copies may be valid after a later copy;
- zero copies may be valid before an internal value is first produced; and
- workspaces have no validity because scratch is not a coherent logical-value copy.

The new methods first reject closure with `IllegalStateException("run state is closed")`, then
validate `bufferIndex`, then `representationIndex`. They retain the existing exact
`IndexOutOfBoundsException` messages `bufferIndex out of range: X` and
`representationIndex out of range: X`.

`isBufferRepresentationValid` returns the bit in constant time. `setBufferRepresentationValid`
stores exactly the supplied boolean in constant time. It does not inspect storage, copy data, call
a backend, change ownership, or change another bit. Repeating a value is valid.

Runtime 0008 can later require a valid source, perform an explicit backend-owned transfer, and
mark the destination valid only after success. Runner/execution work can later mark written output
copies valid and explicitly invalidate stale copies after successful computation. This task
implements none of those actions or policies.

`RunState` stays single-orchestrator-thread state. Add no synchronization, volatile/atomic state,
or permission to race validity, binding, execution, or closure. Separate runs have independent
validity arrays.

## Affected files

Expected Runtime production/Javadoc paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java` — add.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/package-info.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunStateCreation.java` — add package-private.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/package-info.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/PreparedSchedule.java`.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/package-info.java`.

Expected Runtime tests:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateCreationTest.java` — add; also lock the exact nested plan/callback surface.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateTest.java`.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/schedule/PreparedScheduleTest.java`.

Expected explanatory documentation:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation status only; no rule change.
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a contradiction is found: `AGENTS.md`, `ARCHITECTURE.md`, ADRs 0006/0010/0011,
focused architecture documents, documentation rules/profiles, planning guide, completed Runtime
0001–0006, Prepare master/tasks 0001–0002 and Draft 0003, Config master and Draft 0007, Backend
Contract master/current contracts, current Runtime and relevant Prepare/Config/Backend Contract
source/tests/generated Javadocs, Compile/Tensor/Training APIs, user and other backend guides,
Runtime/root Gradle/settings, and architecture/conformance/integration tests.

## Maximum scope

At most 18 paths: 7 Runtime production/Javadoc, 3 Runtime test, 5 explanatory documentation, and
3 planning paths.

This is the smallest credible atomic ceiling: description, schedule reachability, safe cold
creation/rollback, per-run validity, focused tests, and public/backend documentation are one
lifecycle contract. Splitting them would leave task 0007 unable either to create a complete state
safely or expose the facts required by Runtime 0008 and the runner.

No Java/test outside Runtime, Gradle, architecture contract, ADR, architecture-test,
backend-conformance, or integration path may change. Stop if another type, package, module edge,
owner, or path is needed. Do not create a later task specification.

## Acceptance criteria

- Exact planned public/package-private surfaces exist in only their named packages, with no extra
  facade, allocator/factory/manager/registry/service, ID, or helper type.
- Prepared descriptions and callbacks are immutable reusable references; mutable physical objects
  exist only as caller inputs or callback results inside a run.
- Concrete backends can implement callbacks and nominal physical representations without Runtime
  importing or inspecting them.
- Caller inputs are `BORROWED`/valid; created buffers are `RUN_OWNED`/invalid; all workspaces are
  run-owned and have no validity state.
- A buffer supports one or more resident representations and an independent bit per copy.
- Cold setup validates all caller inputs before creation and implements exact deterministic
  all-or-cleaned rollback, reverse order, suppression, and borrowed exclusion.
- Validity query/mutation is explicit, constant-time, dense-indexed, closed-state guarded, and
  free of storage action or implicit coherence.
- Runtime 0008 and runner work can use these facts without changing this contract, while this task
  implements no route, copy, materialization, kernel, execution transition, or schedule runner.
- The representation plan is reachable through `PreparedSchedule`; `PreparedExecution` remains
  exactly memory plan plus schedule.
- Existing typed cold binding and direct hot execution remain unchanged and free of creation,
  validity, map, reflection, strings, lookup, or repeated casts.
- Tests lock exact shape, validation/messages/order, immutable snapshots, callback counts,
  identity, rollback/suppression, initial/mutated validity, workspace exclusion, isolated runs,
  schedule compatibility, and forbidden mechanisms.
- Every affected contract has complete Javadoc for inputs, returns, nullability, ownership,
  identity, lifecycle, thread safety, side effects, failures, and exclusions.
- Runtime/Public API, focused boundary status, backend guide, glossary, task, master plan, and
  roadmap consistently distinguish current creation/validity from later transfers, publication,
  runner, Engine, and concrete backend behavior.
- A separate clean documentation pass finalizes Javadocs, examples, glossary, links, terminology,
  generated pages, planning evidence, and no-change conclusions without repeating successful Java
  tests absent a recorded reason.
- Architecture, dependencies, Gradle, Backend Contract, Config, Prepare Java, concrete backends,
  architecture tests, conformance, and integration remain unchanged with reasoned conclusions.
- Exact scope, task ordering/status, later-spec absence, links, anchors, terminology, fences,
  newlines, whitespace, and `git diff --check` pass.

## Tests / validation

Implementation development:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.run.RunStateCreationTest \
  --tests io.github.pho001.synaptik.runtime.run.RunStateTest \
  --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest
```

Final affected module after Java stabilizes:

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
outside the repository for local targets/anchors, unique effective anchors, balanced backtick and
tilde fences, final newlines, and trailing whitespace.

Also verify:

- exact record/nested/generic/sealed/functional-interface and schedule surfaces;
- exact package-private operation and absence of public setup;
- every specified validation, message, order, identity, call count, cleanup, suppression, initial
  state, mutation, closure, and isolation rule;
- no callback before complete caller validation;
- unchanged `PreparedExecution` source/compiled surface and unchanged `PreparedExecutable`,
  `BoundInvocation`, and their hot-path bytecode;
- no forbidden map/reflection/raw/string/registry/synchronization/concrete-backend/upstream/Config/
  Prepare/Engine/transfer/kernel/runner/publication mechanism or import;
- unchanged Runtime/root Gradle, dependencies, and Java 26 configuration;
- exact 18 paths and no Java/test outside Runtime;
- Runtime 0001–0006 and Prepare 0001–0002 remain Complete; Runtime 0007 synchronizes through
  implementation and becomes Complete only after all gates; Runtime 0008–0011, Prepare 0003, and
  Config 0007 remain Draft without detailed specs; and
- final documentation/terminology/status/whitespace gates.

Repository-wide and architecture tests are deferred to the Runtime prepared-execution checkpoint
or CI because this changes one module without a dependency, build, architecture, concrete-backend,
or end-to-end behavior change. Conformance/integration tests are not applicable yet.

The documentation context reuses successful Runtime test evidence unless it changes executable
behavior or records a concrete risk.

## Dependencies

- Runtime 0001–0006 — Complete.
- Prepare 0001–0002 — Complete; they establish geometry and dense resource coordinates, but
  Runtime imports none of their types.
- ADR 0011 — Accepted and sufficient for this ownership/creation/validity split.
- Existing Runtime dependencies and Java 26 build contract — unchanged.

Config 0007, Prepare 0003, Runtime 0008+, concrete backends, Engine, transfers, publication,
tracing, and tuning are not dependencies.

## Follow-up tasks

- Runtime 0008 remains Draft for explicit transfer/materialization steps using these coordinates
  and validity mutations.
- Runtime 0009 remains Draft for publication/result associations and ownership transitions.
- Runtime 0010 remains Draft for cold schedule consumption, caller input handoff, binding,
  execution, and explicit validity transitions.
- Runtime 0011 remains Draft for contract closure.
- Prepare 0003 remains Draft for public preparation and validation that a runnable schedule has
  exactly one compatible creation prefix.
- Concrete backend tasks later implement creator callbacks, physical representations, transfers,
  and kernels.

Do not create any follow-up specification here.

## Javadocs and documentation impact

- Fully document the representation plan and nested contracts: prepared/run distinction, dense
  order, snapshots, callback obligations, ownership, reuse, validation, and exclusions.
- Document package-private `RunStateCreation`: caller order, callback order, transfer point,
  rollback, original/suppressed failures, and non-public role.
- Update `RunState`/package Javadocs for structural residency, validity, initial state, mutation,
  workspace exclusion, closure, and unchanged single-thread/cold-binding rules.
- Update schedule Javadocs for first-only setup, same-plan identity, reachability through
  `PreparedExecution`, compatibility, and no runner behavior.
- Runtime API needs the current mental model and a focused example with borrowed valid input,
  created invalid buffer, run-owned workspace, explicit validity change, and cleanup. Label any
  future-runner use conceptual.
- Public API lists the new current contracts without claiming a public run/Prepare facade.
- Focused architecture page changes implementation status/mechanics only, not rules.
- Backend guide shows immutable typed callbacks and fresh physical results without registry,
  Runtime lookup, or hot allocation.
- Glossary adds only reusable prepared-representation-plan and buffer-validity terms and updates
  status; do not duplicate algorithms.
- Generate and inspect Runtime Javadocs for the new/nested pages, `RunState`, `PreparedSchedule`,
  and affected package summaries.

Required no-change conclusions:

- Architecture contract, lifecycle/module/dependency pages, and ADRs already assign these owners.
- Compile, Tensor, and Training APIs do not expose physical representations/validity/runner.
- Prepare Java/Javadocs/history stay unchanged because Draft Prepare 0003 is the later consumer.
- Config/Config 0007 stay unchanged because creation/validity needs no invocation/publication option.
- Backend Contract stays closed because these are Runtime/concrete-backend roles, not identity DTOs.
- Other guides stay unchanged absent a concrete contradiction; no Engine workflow/backend exists.
- Gradle/architecture tests stay unchanged because no edge/rule changes; conformance/integration
  remain inapplicable.

## Architecture impact

Expected impact: None.

The architecture and ADR 0011 already require explicit prepared creation, backend-owned physical
mechanics, Runtime-owned per-run orchestration/validity/residency/cleanup, and isolated runs. The
new schedule variant is a compatible representation of setup work and keeps the two-component
`PreparedExecution` intact.

If creators must move to Prepare/Backend Contract, Runtime must inspect a concrete backend/device,
`PreparedExecution` needs another component, a completed signature must change incompatibly, or a
module edge/architecture rule must change, stop without editing and report the decision.

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

- Use one `PreparedRepresentationPlan`, not callbacks on physical interfaces or executables, to
  keep reusable descriptions immutable and physical objects per-run.
- Nest preparations/creators because their vocabulary is plan-local.
- Use zero-component `CallerInput` occurrences; dense encounter order replaces a new ID/map.
- Put one creation step at the schedule prefix so `PreparedExecution` remains unchanged.
- Keep creation package-private; the later runner is the public consumer.
- Initialize validity from existing ownership: borrowed means caller input/current value;
  run-owned means newly created storage without a value.
- Use one boolean per representation and one setter, permitting zero/multiple valid copies without
  automatic coherence.
- Treat binding as residency because every prepared representation exists before cold binding and
  remains until close; another always-true flag would duplicate arrays.
- Validate callers before creators and compare identity with direct iteration, not a map.
- Do not close a duplicate callback result as a second object because it aliases an already
  borrowed/owned resource.

## Known limitations

- No public input names or graph/value IDs; later runner uses dense caller-input order.
- All representations are created before binding and remain resident until close; no lazy
  creation, eviction, or dynamic residency.
- Validity is an orchestration fact; Runtime cannot prove bytes were copied/computed before a bit
  is set.
- The setter permits zero/multiple valid copies and has no action-specific transition policy.
- Callback contracts cannot prove concrete immutability, thread safety, freshness, geometry, or
  physical cleanup behavior; backend tests must.
- Empty/executable-only schedules remain constructible; Prepare 0003 later validates runnable
  completeness.
- No transfer, publication/result transition, runner, production backend, or Engine lifecycle.

## Validation evidence

- Implementation context `/root` ran the focused command:

  ```bash
  ./gradlew :modules:runtime:test \
    --tests io.github.pho001.synaptik.runtime.run.RunStateCreationTest \
    --tests io.github.pho001.synaptik.runtime.run.RunStateTest \
    --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest
  ```

  Its final stabilized result was `BUILD SUCCESSFUL`; JUnit XML reported three suites and 37
  tests (`RunStateCreationTest` 10, `RunStateTest` 15, and `PreparedScheduleTest` 12), with zero
  failures, errors, or skips.
- The implementation context ran exactly one final affected-module command after executable Java
  stabilized: `./gradlew :modules:runtime:test`. It passed with `BUILD SUCCESSFUL`; JUnit XML
  reported 11 suites and 94 tests with zero failures, errors, or skips. Gradle reported nine
  actionable tasks, one executed and eight up-to-date.
- Documentation-focused context `/root/runtime_0007_docs` applied the General, API/Javadoc,
  Architecture, Backend Guide, Planning, and Example profiles. It independently read the
  architecture contract, ADRs 0006/0010/0011, focused lifecycle/module/dependency/boundary
  explanations, planning guide/roadmap, Runtime and Prepare master/task history, final Runtime
  source/tests, adjacent Runtime/Prepare/Backend Contract/Config contracts, APIs, backend guide,
  glossary, builds, and the actual dirty diff.
- The documentation pass changed Javadocs, package documentation, explanatory documentation,
  glossary entries, and planning evidence only; it changed no executable behavior or tests. It
  therefore reused the successful 37-test and 94-test implementation evidence and did not repeat
  either Java test command.
- Final Runtime Javadoc generation passed:

  ```bash
  ./gradlew :modules:runtime:javadoc
  ```

  Gradle reported `BUILD SUCCESSFUL`; five tasks were actionable, two executed and three
  up-to-date. Generated pages were inspected for `PreparedRepresentationPlan` and all five nested
  contracts, `RunState`, `PreparedSchedule`, `RepresentationCreationStep`, and the resource/run/
  schedule package summaries. They expose dense ordering, immutable reuse, callback freshness,
  ownership, nullability, initial validity, structural residency, constant-time mutation,
  first-only creation, lifecycle, rollback, suppression, and deliberate boundaries.
- The focused current Runtime API example was compiled and executed with Java 26:

  ```bash
  javac --release 26 -cp modules/runtime/build/classes/java/main \
    -d /tmp/runtime-0007-api-example /tmp/Runtime0007ApiExample.java
  java -cp modules/runtime/build/classes/java/main:/tmp/runtime-0007-api-example \
    Runtime0007ApiExample
  ```

  Both commands passed with no output. The example checks borrowed-valid and created-invalid
  initial state, one explicit validity mutation, workspace exclusion, exact plan retention,
  borrowed cleanup exclusion, run-owned cleanup, and idempotence. The future runner is labeled
  conceptual; the example does not claim a copy, kernel, or automatic transition.
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
  and tilde fences, final newlines, and trailing whitespace.
- `javap -p` confirmed the exact `PreparedRepresentationPlan` record and nested sealed/record/
  functional-interface surface; package-private `RunStateCreation.create`; `RunState`'s sole
  `boolean[][]` plus exactly two public validity methods; the sealed schedule creation variant;
  and unchanged two-component `PreparedExecution`. Bytecode inspection confirmed
  `BoundInvocation.execute` remains one state-open guard plus `executeBound`, while
  `PreparedExecutable.bind` retains its existing compatibility and direct-binding flow.
- Source and compiled-mechanism scans found no map, reflection, `ServiceLoader`, synchronization,
  atomic/concurrent state, raw `Object`, unchecked suppression, upstream module, concrete backend,
  registry/service, transfer, kernel, publication, or runner mechanism in affected production.
  Source/tests confirm exact plan validation and snapshot order, complete caller prevalidation,
  buffer-then-workspace creation, identity uniqueness, borrowed-valid/run-owned-invalid state,
  independent constant-time mutation, original-failure preservation, reverse rollback with
  suppressed cleanup failures, successful reverse cleanup, run isolation, and first-only
  creation-step compatibility.
- `git diff --exit-code` checks confirmed no change to `ARCHITECTURE.md`, ADRs 0006/0010/0011,
  lifecycle/module/dependency pages, Runtime/root builds, Prepare/Backend Contract/Config Java,
  concrete backends, Engine, or architecture/conformance/integration tests. Root configuration
  remains Java toolchain and release 26; Runtime adds no override or dependency.
- The final task inventory contains exactly 18 authorized paths: seven Runtime production/
  Javadoc paths, three Runtime test paths, five explanatory documents, and three planning paths.
  No Java/test path outside Runtime, Gradle path, architecture contract/ADR/test, backend-
  conformance path, integration path, or later task specification changed.
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
    representations, validity, or a runner.
  - Prepare Java/Javadocs/history remain unchanged because Runtime consumes established dense
    geometry and Draft Prepare 0003 is the later public construction/validation consumer.
  - Config and Draft Config 0007 remain unchanged because creation and validity consume no run or
    publication option. Backend Contract remains closed because creators and physical roles belong
    to Runtime/concrete backends rather than the identity DTO leaf.
  - Other guides and Engine remain unchanged because no public lifecycle or production backend
    exists; the backend contributor guide alone needed the current typed-callback pattern.
  - Gradle/dependencies/Java 26, architecture tests, backend conformance, integration tests, and
    concrete backends remain unchanged because no edge, architecture rule, backend implementation,
    or end-to-end behavior changed. Later specifications remain absent by progressive planning.

## Implementation notes

- Added only the exact immutable representation plan and nested preparation/creator contracts,
  package-private cold creation operation, two validity methods and backing array, and one sealed
  schedule variant specified by the task.
- Cold creation prevalidates every borrowed caller input before callbacks, creates buffers then
  workspaces, and gives a complete state cleanup ownership only after construction succeeds.
  Partial failure preserves the original unchecked failure and rolls back prior created results in
  reverse order without closing borrowed or duplicate identities.
- Binding presence is structural residency. Borrowed buffers initialize valid, created run-owned
  buffers initialize invalid, workspaces have no validity, and the setter changes exactly one bit
  without physical or coherence behavior.
- The documentation pass finalized all seven production/package Javadocs, the focused current API
  example, Runtime/Public API status, focused architecture implementation status, typed backend
  callback guidance, the two reusable glossary terms, and synchronized planning evidence.

## Completion summary

- Completed changes: implemented and documented immutable prepared representation origins,
  package-private all-or-cleaned cold setup, structural residency, independent per-copy validity,
  and first-only schedule reachability while preserving the existing aggregate and hot path.
- Files changed or created: exactly the authorized 18 paths—seven Runtime production/Javadoc,
  three Runtime test, five explanatory documentation, and three planning paths.
- Tests and validation: reused the implementation context's successful focused 37-test and final
  94-test Runtime evidence; Runtime Javadoc/generated pages, Java 26 API example, eight-file
  Markdown, exact surface/order/failure/rollback/validity/hot-path/import/mechanism/build/Java-26/
  scope/status/later-specification, final-newline/fence/whitespace, and `git diff --check` gates
  passed.
- Documentation-agent review: completed in clean context `/root/runtime_0007_docs` without
  executable behavior/test changes or repeated Java tests; implementation context was `/root`.
- Documentation impact: Runtime/Public APIs, focused architecture status, backend guide, task,
  Runtime master plan, and roadmap now distinguish current creation/residency/validity from later
  transfer, execution, publication, public Prepare, Engine, and runner work.
- Javadoc review: all seven affected production/package paths were independently finalized and
  generated pages inspected for every new/nested/current contract and package summary.
- Glossary impact: added only reusable prepared-representation-plan and buffer-validity terms and
  synchronized current residency, schedule, and run-state status.
- Unresolved issues: None.
- Follow-up required: None for this task. Runtime 0008–0011, Prepare 0003, and Config 0007 remain
  separate Draft planning work without detailed specifications.

Status: Complete
