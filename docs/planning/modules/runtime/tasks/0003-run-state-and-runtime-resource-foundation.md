# Task 0003: Run-State and Runtime Resource Foundation

## Status

Complete

## Goal

Implement the smallest Runtime-owned per-run resource foundation authorized by
[ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md): nominal
backend-implemented buffer/workspace representation roles, explicit borrowed versus run-owned
buffer bindings, array-indexed slot access, and exactly one closeable `RunState` for one complete
logical run of a prepared memory plan.

The exact public surface is:

```java
package io.github.pho001.synaptik.runtime.resource;

public interface BufferRepresentation extends AutoCloseable {
    @Override
    void close();
}

public interface WorkspaceRepresentation extends AutoCloseable {
    @Override
    void close();
}
```

```java
package io.github.pho001.synaptik.runtime.run;

public enum RunResourceOwnership {
    BORROWED,
    RUN_OWNED
}

public record BufferRepresentationBinding(
        BufferRepresentation representation,
        RunResourceOwnership ownership) {}

public final class RunState implements AutoCloseable {
    public RunState(
            PreparedMemoryPlan memoryPlan,
            List<List<BufferRepresentationBinding>> bufferBindings,
            List<WorkspaceRepresentation> workspaceRepresentations);

    public PreparedMemoryPlan memoryPlan();
    public int bufferSlotCount();
    public int bufferRepresentationCount(int bufferIndex);
    public BufferRepresentationBinding bufferRepresentation(
            int bufferIndex,
            int representationIndex);
    public int workspaceSlotCount();
    public WorkspaceRepresentation workspaceRepresentation(int workspaceIndex);
    public boolean isClosed();

    @Override
    public void close();
}
```

`bufferIndex` and `workspaceIndex` are dense zero-based positions in
`PreparedMemoryPlan.buffers()` and `PreparedMemoryPlan.workspaces()` encounter order. They are not
`BufferSlot.value()` or `WorkspaceSlot.value()`. The constructor copies the supplied binding
structure into arrays, so successful representation access is direct array indexing and retains
the exact representation/binding references.

## Scope

- Add the exact five public types above and package documentation for `runtime.resource` and
  `runtime.run`.
- Make both representation interfaces nominal lifecycle roles implemented by concrete backends.
  Their only shared behavior is unchecked, non-throw-declared physical cleanup through
  `close()`; Runtime does not inspect or access backend storage.
- Require `BufferRepresentationBinding` components to be non-null. `BORROWED` means the run may
  access but never close the representation. `RUN_OWNED` transfers cleanup responsibility to the
  successfully constructed `RunState`.
- Construct exactly one `RunState` from one exact `PreparedMemoryPlan`, one outer buffer list with
  an entry for every prepared buffer position, and one workspace representation for every
  prepared workspace position.
- Permit one or more ordered buffer representations per buffer position. This is a carrier seam
  only; this task defines no representation key, validity, residency, transfer, or coherence.
- Require exactly one workspace representation per workspace position. Workspace is run-owned
  backend-local scratch and is always closed by `RunState`.
- Copy list structure into private arrays while retaining exact binding and representation
  objects. Perform no map lookup on successful indexed access.
- Reject repeated representation object identity anywhere in one `RunState`, including across
  buffer and workspace domains, so cleanup responsibility is unambiguous and this task does not
  introduce aliasing.
- Make `close()` idempotent, mark the state closed before physical cleanup begins, close every
  still-owned representation in deterministic reverse order, and preserve all cleanup failures.
- Add focused public-surface, validation, identity-retention, ownership, cleanup, failure, and
  concurrency-contract tests.
- Finalize Javadocs, package documentation, Runtime API/status wording, focused boundary status,
  glossary impact, and planning evidence in a separate documentation-focused clean context.

## Out of scope

- `PreparedExecutable`, `PreparedUnit`, bound invocation/binding objects, or executable calls
- cold compatibility implementation, backend-specific casts, device/representation keys, or
  Prepare finalization
- representation creation, physical allocation, storage access, transfer, materialization,
  publication, `RunResult`, runner, schedule, or execution behavior
- validity/residency mutation or full multi-representation tracking/coherence
- ownership transfer/lease to a published result
- immutable persistent `PreparedExecution` resources or `PreparedExecution` itself
- pooling, reuse, aliasing, liveness analysis, distributed sharding, multi-device scheduling, or
  hidden write-back
- concrete backend representation classes or any `MemorySegment`, CUDA, Metal, native-handle, or
  backend route type
- raw `Object`, unchecked generic access, reflection in production, string dispatch, registries,
  service locators, managers, services, builders, factories, or public backend type switches
- changes to `BufferSlot`, `WorkspaceSlot`, `PreparedMemoryPlan`, Prepare, Backend Contract,
  Config, Trace, Engine, concrete backends, Gradle, architecture tests, conformance tests, or
  integration tests
- allocation, derivation, renumbering, or consumption of any slot, graph, Tensor, trace, backend,
  device, partition, or other identifier; construction has no source-ID side effect
- Runtime 0004+, Prepare 0002, or any other detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - Concrete backend modules
  - Run lifecycle
  - Dependency rules
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [ADR 0006: No runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0002](0002-prepared-memory-and-workspace-contracts.md)

## Architecture constraints

- Prepared recipes remain immutable/reusable; every active complete logical run has exactly one
  distinct mutable `RunState` covering all backend partitions in that run.
- Runtime owns logical per-run binding state, ownership, lifecycle orchestration, cleanup, and
  isolation. Concrete backends implement physical representation types and cleanup mechanics.
- Runtime source must not mention or depend on concrete physical/backend classes and must not
  choose or discover a backend.
- A caller input representation may be borrowed; internal buffer representations and all
  workspaces are run-owned. This task adds no publication ownership transition.
- Heterogeneous compatibility checking and backend-owned typed direct-reference bound invocation
  belong to Runtime 0004. This task must not add a generic catch-all access method as a substitute.
- Multiple buffer representations in this carrier do not imply that they are simultaneously
  valid, coherent, or automatically synchronized.
- Workspace is backend-local scratch, not a logical graph value or transferable publication.
- The hot-path foundation uses dense array positions, not maps, reflection, strings, global
  lookup, graph inspection, or slot-value-sized arrays.
- Runtime dependencies and all dependency directions remain unchanged.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.runtime.memory` — consume `PreparedMemoryPlan` without modification.

Packages added:

- `io.github.pho001.synaptik.runtime.resource` — public nominal physical representation roles.
- `io.github.pho001.synaptik.runtime.run` — public per-run ownership/binding and lifecycle state.

Type placement:

- `runtime.resource.BufferRepresentation` — nominal backend-implemented buffer cleanup role.
- `runtime.resource.WorkspaceRepresentation` — distinct nominal backend-implemented scratch
  cleanup role.
- `runtime.run.RunResourceOwnership` — per-run buffer borrowing/ownership vocabulary.
- `runtime.run.BufferRepresentationBinding` — exact representation plus its initial run ownership.
- `runtime.run.RunState` — the sole lifecycle owner and array-indexed accessor for one run.

Tests mirror the two production packages. No root facade or other Java package changes.

## Affected files

Expected production paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/BufferRepresentation.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/WorkspaceRepresentation.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunResourceOwnership.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/BufferRepresentationBinding.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/package-info.java`

Expected test paths:

- up to one focused representation-contract test under `runtime.resource`;
- up to one ownership/binding test under `runtime.run`; and
- one focused `RunStateTest` under `runtime.run`.

Expected documentation/planning paths:

- `docs/api/runtime-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status update only
- `docs/glossary.md`
- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: root architecture and ADR 0011, lifecycle,
module/dependency docs, backend guide, Runtime 0001–0002, Prepare 0001/master plan, current Runtime
source/build, and architecture/conformance/integration tests.

## Maximum scope

At most 16 paths:

- 7 Runtime production paths;
- 3 Runtime test paths; and
- 6 documentation/planning paths.

No Java/test path outside Runtime, Gradle path, architecture contract, ADR, architecture-test,
backend-conformance, or integration path may change. Stop if another type, package, module edge,
or path is required. Do not create a later task specification.

## Validation, ownership, and failure rules

Constructor validation occurs in this order:

1. require `memoryPlan`, then `bufferBindings`, then `workspaceRepresentations` non-null;
2. require outer buffer count equal `memoryPlan.buffers().size()`;
3. require workspace count equal `memoryPlan.workspaces().size()`;
4. scan buffer positions in increasing order: require each inner list non-null and non-empty, then
   each binding non-null, and reject the first repeated exact representation identity;
5. scan workspaces in increasing order: require each representation non-null and reject the first
   repeated exact identity, including an identity already used by a buffer binding; and
6. only after all validation succeeds, store arrays and accept responsibility for `RUN_OWNED`
   buffer representations and every workspace representation.

Exact construction failures:

- null top-level inputs: `NullPointerException("memoryPlan")`,
  `NullPointerException("bufferBindings")`, or
  `NullPointerException("workspaceRepresentations")`;
- count mismatch: `IllegalArgumentException("bufferBindings size must equal prepared buffer count N")`
  or `IllegalArgumentException("workspaceRepresentations size must equal prepared workspace count N")`;
- null/empty inner list: `NullPointerException("bufferBindings[i]")` or
  `IllegalArgumentException("bufferBindings[i] must not be empty")`;
- null binding/workspace: `NullPointerException("bufferBindings[i][j]")` or
  `NullPointerException("workspaceRepresentations[i]")`;
- repeated identity: `IllegalArgumentException("representation is already bound to this run")`.

`BufferRepresentationBinding` validates `representation` before `ownership`, using
`NullPointerException("representation")` and `NullPointerException("ownership")`.

Construction failure transfers no ownership and closes nothing. After successful construction:

- buffer/workspace counts equal prepared-plan list counts;
- each buffer position exposes its supplied non-empty ordered binding sequence;
- each workspace position exposes its exact supplied representation;
- `memoryPlan()` retains and returns the exact plan reference;
- indexed access uses private arrays and performs no map lookup;
- invalid indices fail with `IndexOutOfBoundsException` and exact messages
  `bufferIndex out of range: X`, `representationIndex out of range: X`, or
  `workspaceIndex out of range: X`; and
- representation access after closure fails first with
  `IllegalStateException("run state is closed")`. Immutable plan/count inspection and
  `isClosed()` remain available after closure.

`close()` atomically changes the logical lifecycle to closed before invoking representation
cleanup. `RunState` is not otherwise thread-safe; callers must not race access, close, or future
mutation. Close order is workspace positions in reverse order, then buffer positions in reverse
order and each position's representations in reverse order. Borrowed buffers are skipped. Every
owned representation is attempted once. The first `RuntimeException` or `Error` is rethrown after
all attempts, and later failures are added to it as suppressed exceptions in encounter order.
Repeated `close()` performs no further cleanup and throws no prior failure again.

Separate concurrent `RunState` instances may share the immutable `PreparedMemoryPlan`. Their
run-owned representation objects must be distinct. Borrowed representations may be shared only
when the caller guarantees lifetime, thread access, and external synchronization for the complete
run; Runtime neither closes nor extends their lifetime.

## Acceptance criteria

- The exact five-type public surface exists in only the two planned packages.
- Representation roles are nominally distinct, declare only non-checked `close()`, and expose no
  storage/access/backend/device method.
- Binding and `RunState` validate in the specified order with exact failures.
- Successful construction snapshots all list structure into arrays, preserves order, and retains
  exact plan, binding, and representation references.
- Multiple buffer representations are representable without claiming validity/coherence; each
  workspace position has exactly one run-owned representation.
- Indexed access is dense prepared-plan encounter order and never interprets slot numeric values.
- Focused tests prove no map/reflection/string dispatch or raw/unchecked production mechanism,
  exact lifecycle behavior, borrowed versus owned cleanup, reverse order, suppression, idempotence,
  constructor-failure ownership, and closed access.
- Tests also prove two states can share one plan while retaining distinct run-owned resources and
  that one state does not close another's resources.
- No executable/bound-invocation, residency, transfer, publication/result, schedule/runner,
  allocation, pooling, backend, Prepare, Gradle, dependency, or architecture behavior is added.
- Construction retains existing plan/binding/representation references and allocates only private
  array structure; it creates, consumes, renumbers, or rolls back no project identifier.
- Every public member has complete Javadoc for nullability, ownership, identity, index meaning,
  lifecycle, thread safety, result semantics, and caller-visible failures.
- A separate documentation-focused agent finalizes affected Javadocs/docs/glossary/status and
  records reasoned no-change conclusions for reviewed adjacent surfaces.
- Runtime 0001–0002 and Prepare 0001 remain Complete; Runtime 0004+ and Prepare 0002 remain Draft
  without detailed specs; exactly this one new Runtime specification exists.
- Final scope, Markdown, newline/fence/whitespace, and `git diff --check` gates pass.

## Tests / validation

Implementation-focused validation:

```bash
./gradlew :modules:runtime:test
```

Documentation-focused pass after final Javadocs/documentation:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/glossary.md \
  docs/planning/modules/runtime/tasks/0003-run-state-and-runtime-resource-foundation.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

Also verify the exact source/API/package surface, validation order/messages, reference retention,
array-backed implementation, absence of maps/reflection/raw or unchecked access in production,
forbidden imports/types, unchanged Runtime/root build contracts, exact 16-path ceiling, no Java
outside Runtime, synchronized statuses, and later-spec absence.

Repository-wide tests and architecture tests are deferred to the Runtime prepared-contract
checkpoint or continuous integration. This task changes no dependency direction, Gradle edge,
shared build contract, concrete backend, or end-to-end behavior. Backend conformance and
integration tests are not applicable.

The documentation context reuses the successful Runtime test evidence unless it changes
executable Java behavior or records a concrete reason to repeat it.

## Dependencies

- Runtime 0001–0002 — Complete.
- Prepare 0001 — Complete; it motivates later slot assignment but is not imported here.
- ADR 0011 — Accepted and resolves resource ownership, run isolation, and cold binding.
- Existing Runtime dependencies and Java 26 build contract — unchanged.

Runtime 0004, Prepare 0002, concrete backends, Engine, scheduling, publication/results, transfer,
residency, and physical allocation are not dependencies of this foundation.

## Follow-up tasks

- Runtime 0004 defines prepared executable/unit contracts and backend-owned typed cold-bound
  invocation objects against the nominal representation carrier.
- Prepare 0002 later finalizes analyses against assigned slots after Runtime 0004.
- Later Runtime tasks add explicit validity/residency, prepared transfers, schedules,
  publication/`RunResult`, runner behavior, and ownership transfer.

These rows remain Draft without detailed specifications. Do not create them in this task.

## Architecture impact

Expected impact: None. This task implements the foundation already authorized by ADR 0011 and the
updated architecture contract. It changes no module boundary or dependency direction. Stop if
implementation needs backend/device inspection in Runtime, a concrete backend dependency, public
unchecked access, physical allocation, executable binding, or another architecture decision.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, ADR 0011, the focused Runtime/
Prepare/backend and lifecycle/dependency docs, documentation rules and General/API-Javadoc/
Architecture/Planning profiles, the Runtime master plan, Runtime 0001–0002, current Runtime
source/tests/build, and
docs/planning/modules/runtime/tasks/0003-run-state-and-runtime-resource-foundation.md.

Implement Runtime 0003 exactly within its five-type surface and 16-path ceiling. Add only the
nominal closeable representation roles, borrowed/run-owned buffer binding, array-backed one-run
RunState lifecycle, focused Runtime tests/Javadocs, and required current-status documentation.
Do not add executable/bound invocation, Prepare finalization, allocation/access mechanics,
validity/residency, transfers, schedule/runner, publication/result, backend code, pooling,
dependency/Gradle/architecture changes, or later task specs. Stop on any architecture, package,
API, validation, or scope conflict.

Run one final Runtime module test and all source/scope/status checks. Then hand the actual diff
and exact Java evidence to a separate documentation-focused clean context. That pass must follow
documentation-rules.md, finalize Javadocs/docs/glossary/planning evidence, and not repeat Java
tests unless executable behavior changes or a concrete risk is recorded. Mark Complete only after
all gates pass and return both context IDs, exact paths, commands/results/counts, unresolved
issues, follow-up, and the repository completion status format.
```

## Local decisions

- Dense access positions are the prepared-plan entry-list indices, not slot numeric components.
- Buffer bindings carry ownership per physical representation because one logical buffer may have
  more than one explicit representation with different origins.
- Workspaces need no ownership enum in this task: every supplied workspace is run-owned, and an
  algorithm needing host plus device scratch receives separate slots.
- Full representation keys and validity/residency state wait for their prepared consumers. The
  foundation carries ordered representations without implying coherence.
- Identity-duplicate rejection prevents ambiguous cleanup and avoids introducing aliasing.
- Run-state construction transfers ownership only after complete successful validation.

## Known limitations

- No current API chooses a buffer representation by backend/device or states which representation
  is valid. Runtime 0004 and later prepared transfer/residency tasks must supply those associations.
- No publication ownership transfer, `RunResult`, persistent prepared-resource lifecycle,
  allocation, storage access, schedule, runner, or executable contract exists.
- `RunState` is single-orchestrator-thread state. It does not synchronize concurrent access or make
  a borrowed representation safe to share.

## Validation evidence

- Implementation context `/root/implement_runtime_0003` ran the focused command:

  ```bash
  ./gradlew :modules:runtime:test \
    --tests io.github.pho001.synaptik.runtime.resource.RepresentationContractTest \
    --tests io.github.pho001.synaptik.runtime.run.BufferRepresentationBindingTest \
    --tests io.github.pho001.synaptik.runtime.run.RunStateTest
  ```

  It passed with `BUILD SUCCESSFUL`; JUnit XML reported three suites and 20 tests
  (`RepresentationContractTest` 2, `BufferRepresentationBindingTest` 4, and `RunStateTest` 14),
  with zero failures, errors, and skips.
- The same implementation context ran the one final affected-module command after executable Java
  stabilized:

  ```bash
  ./gradlew :modules:runtime:test
  ```

  It passed with `BUILD SUCCESSFUL`; JUnit XML reported six suites and 45 tests: existing
  `BufferSlotTest` 4, `WorkspaceSlotTest` 5, and `PreparedMemoryPlanTest` 16 plus the new 2, 4,
  and 14-test suites, with zero failures, errors, and skips. Gradle reported nine actionable tasks,
  one executed and eight up-to-date.
- Documentation-focused Codex context `019fb892-9477-7ea3-b271-c4528917f6ca` applied the General,
  API/Javadoc, Architecture, and Planning profiles plus the example format. It read the
  architecture contract, ADR 0011, planning guide/roadmap, Runtime master/task history, Prepare
  master/0001, focused lifecycle/module/dependency/boundary/backend contracts, Runtime API,
  complete glossary, current Runtime build/source/tests, and the actual dirty diff. It changed
  Javadocs and package documentation but no executable Java, so it reused the two successful test
  results and did not repeat either Java suite.
- Final Runtime Javadoc generation passed:

  ```bash
  ./gradlew :modules:runtime:javadoc
  ```

  Gradle reported `BUILD SUCCESSFUL`; five tasks were actionable, two executed and three
  up-to-date. Generated pages expose the exact five public types and two packages and contain the
  ownership, identity, dense-index, nullability, lifecycle, thread-safety, failure, suppression,
  and unsupported-behavior contracts.
- Targeted Markdown validation passed both after the substantive documentation edits and after
  final evidence/status synchronization:

  ```bash
  python3 /tmp/validate_synaptik_markdown.py \
    docs/api/runtime-api.md \
    docs/architecture/runtime-prepare-backend-boundary.md \
    docs/glossary.md \
    docs/planning/modules/runtime/tasks/0003-run-state-and-runtime-resource-foundation.md \
    docs/planning/modules/runtime/master-plan.md \
    docs/planning/roadmap.md
  ```

  Each run reported `validated 6 Markdown files`, checking local targets and heading anchors,
  unique effective anchors, balanced backtick/tilde fences, final newlines, and trailing
  whitespace.
- The two Runtime API examples were combined in `/tmp/Runtime0003ApiExample.java`, then compiled
  and executed with:

  ```bash
  javac --release 26 -cp modules/runtime/build/classes/java/main \
    -d /tmp/runtime-0003-api-example /tmp/Runtime0003ApiExample.java
  java -cp modules/runtime/build/classes/java/main:/tmp/runtime-0003-api-example \
    Runtime0003ApiExample
  ```

  Both commands passed with no output. The example checks exact binding retention, dense counts,
  borrowed cleanup exclusion, run-owned cleanup, and idempotence without claiming allocation or
  execution.
- The final source/API/package audit confirmed exactly five new public types in exactly
  `runtime.resource` and `runtime.run`, seven production paths, three mirrored test paths, and no
  extra public member or nested type. Production imports are limited to `PreparedMemoryPlan`, the
  two representation roles, `java.util.List`, and `java.util.Objects`.
- The implementation and focused tests confirm exact top-level, count, buffer, then workspace
  validation order and messages; exact plan/binding/representation retention; identity-based
  duplicate rejection across both domains; closed-first access; reverse deterministic cleanup;
  borrowed/run-owned ownership; `RuntimeException`/`Error` suppression; idempotence; and isolated
  concurrent states. Successful access indexes private arrays directly. Production uses no map,
  reflection, string dispatch, raw `Object`, unchecked generic access, registry, service locator,
  concrete backend type, or backend/device inspection.
- The identifier audit found no identifier construction, allocation, derivation, renumbering,
  consumption, rollback, or side effect. `RunState` mentions `BufferSlot` and `WorkspaceSlot` only
  to state that dense entry positions are not their numeric components.
- The build audit confirmed no diff in root or Runtime Gradle files. Runtime dependencies remain
  Config, Backend Contract, and Trace only; root configuration alone retains Java 26 toolchain and
  release settings.
- The task-isolated audit found exactly 16 paths: seven Runtime production paths, three Runtime
  tests, and the six authorized documentation/planning paths. No Java/test path outside Runtime,
  Gradle path, architecture test, backend-conformance path, integration path, Prepare source/test,
  concrete backend, or other module changed. Seven separate pre-existing ADR 0011 review-only
  paths remained preserved and unmodified by this pass.
- Status/specification checks confirmed Runtime 0001–0003 and Prepare 0001 are Complete; Runtime
  0004–0008 and Prepare 0002 remain Draft; only Runtime 0001–0003 and Prepare 0001 have detailed
  specifications; and task, master-plan, and roadmap status are synchronized.
- `git diff --check` passed after final documentation and status synchronization with no output.
- No-change conclusions:
  - `ARCHITECTURE.md` and ADR 0011 already authorize the exact one-run ownership and cold-binding
    foundation; this task changes no architecture decision or dependency direction.
  - Lifecycle, module-boundary, dependency, and backend-integration explanations already state the
    complete architecture accurately. The focused boundary page alone needed current-versus-
    planned implementation status for Runtime 0003.
  - Runtime 0001–0002 Javadocs/tasks and Prepare 0001/master remain accurate: the new carrier
    consumes final plan order without changing slot geometry, Prepare declarations, or assignment.
  - Runtime/root build contracts need no change because no dependency or Java-toolchain edge was
    added.
  - Architecture tests need no update because dependency and prohibited-import rules are
    unchanged. Backend conformance and integration tests remain inapplicable because no concrete
    backend or end-to-end execution behavior exists.
  - Compile, Model, Config, Trace, Engine, user workflow, and other API surfaces need no change
    because this task adds no graph, configuration, trace emission, lifecycle facade, executable,
    schedule, or result behavior.
  - No later task specification was created because Runtime 0004 and Prepare 0002 require a
    separate frontier-planning step.

## Implementation notes

- Added only the exact two representation roles, two ownership/binding values, and final
  array-backed `RunState` in the planned packages, with package documentation and three focused
  suites.
- Constructor validation completes before assigning fields, so failed construction closes
  nothing and transfers no ownership. Successful construction retains the exact plan, bindings,
  and representations while copying only list structure.
- Cleanup marks the state closed before callbacks, traverses workspaces then owned buffers in
  exact reverse order, attempts every owned representation, preserves unchecked failures, and is
  inert after the first call.
- Finalized all seven production/package Javadocs and updated Runtime API, focused boundary
  implementation status, glossary, task evidence, Runtime master plan, and roadmap. No executable
  Java changed during the documentation pass.

## Completion summary

- Completed changes: implemented and documented the exact Runtime-owned one-run representation,
  ownership, direct-array access, and cleanup foundation authorized by ADR 0011.
- Files changed or created: exactly the 16 authorized task paths—seven Runtime production paths,
  three Runtime test paths, and six documentation/planning paths.
- Tests and validation: reused the implementation context's successful focused 20-test and final
  45-test Runtime evidence; Runtime Javadoc, generated pages, targeted six-file Markdown, exact
  source/API/import/mechanism/build/toolchain/scope/status/later-specification, final-newline/
  fence/whitespace, and final `git diff --check` gates passed.
- Documentation-agent review: completed in clean Codex context
  `019fb892-9477-7ea3-b271-c4528917f6ca` without executable Java changes or duplicate Java-test
  execution.
- Documentation impact: Runtime API, focused architecture status, glossary, task, master plan,
  and roadmap now distinguish the implemented carrier/lifecycle from later allocation, cold
  binding, residency, transfer, scheduling, execution, and publication.
- Javadoc review: all seven production/package paths were independently finalized for exact
  semantics, ownership, identity, nullability, lifecycle, failures, thread safety, and boundaries.
- Glossary impact: made `RunState` and the nominal representation/binding foundation current while
  retaining physical representation implementations and complete execution behavior as planned.
- Unresolved issues: None.
- Follow-up required: None for this task. Runtime 0004 and Prepare 0002 remain separate Draft
  frontier work without detailed specifications.

Status: Complete
