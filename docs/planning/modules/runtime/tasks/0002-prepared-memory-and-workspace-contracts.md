# Task 0002: Prepared Memory and Workspace Contracts

## Status

Complete

## Goal

Add the smallest immutable Runtime-owned prepared-memory geometry that shared Prepare can produce
after backend analysis without moving Prepare or Model facts into Runtime.

The exact public surface is:

```java
package io.github.pho001.synaptik.runtime.memory;

public record WorkspaceSlot(long value) {}

public record PreparedMemoryPlan(
        List<PreparedMemoryPlan.BufferEntry> buffers,
        List<PreparedMemoryPlan.WorkspaceEntry> workspaces) {
    public record BufferEntry(
            BufferSlot slot,
            long byteSize,
            long byteAlignment) {}

    public record WorkspaceEntry(
            WorkspaceSlot slot,
            long byteSize,
            long byteAlignment) {}
}
```

`WorkspaceSlot` is a non-negative identity in one prepared-memory-plan context. It is nominally
distinct from `BufferSlot` and from Prepare's analysis-local workspace requirement ID even when
their numeric values happen to match.

Each nested entry records exact non-negative byte size and positive power-of-two byte alignment
for one Runtime slot. `PreparedMemoryPlan` snapshots caller-supplied buffer and workspace entry
order and requires each slot identity to occur exactly once in its own domain.

This task does not import or retain `PreparationResourceRequirement`, `BackendPartitionAnalysis`,
`ValueId`, `LogicalMemoryRequirement`, or `PlannedPartition`. A later Prepare-owned assignment and
finalization contract will translate exact analysis requirements into this backend-neutral
Runtime geometry while retaining the source-to-slot associations that backend finalization needs.

## Rationale and mental model

```text
Prepare analysis facts                     Runtime 0002 geometry

Buffer(ValueId, bytes, alignment)      ->  BufferEntry(BufferSlot, bytes, alignment)
Workspace(local ID, bytes, alignment)  ->  WorkspaceEntry(WorkspaceSlot, bytes, alignment)

source-to-slot association: Prepare        reusable slot geometry: Runtime
physical storage and run binding: later Runtime/backend work
```

Prepare 0001 makes the required size and alignment facts concrete, but Runtime must not acquire a
dependency on Prepare or Model to represent reusable prepared state. Runtime therefore owns only
the final slot identities and their exact geometry. Shared Prepare later owns deterministic
translation, requirement coverage, and source association.

The initial translation policy is conservative: every distinct buffer value requirement receives
its own `BufferSlot`, and every workspace declaration receives its own `WorkspaceSlot`. No slot
reuse is authorized because no liveness or interference proof exists. Runtime 0002 locks the
unique final slot geometry needed by that policy; it does not itself inspect analysis results.

## Scope

- Add public final record `WorkspaceSlot` with exactly one `long value` component.
- Accept every non-negative workspace-slot value, including zero and `Long.MAX_VALUE`.
- Reject a negative workspace-slot value with `IllegalArgumentException` and exact message
  `value must be non-negative`.
- Define workspace-slot identity only within one owning prepared-memory-plan context; another plan
  may reuse the same numeric value.
- Add public final record `PreparedMemoryPlan` with exactly the ordered immutable
  `buffers` and `workspaces` list components shown above.
- Add only the public nested final records `BufferEntry` and `WorkspaceEntry` shown above.
- Require every entry slot to be non-null, every byte size to be non-negative, and every byte
  alignment to be a positive power of two.
- Accept zero byte size and alignments from `1` through `1L << 62`.
- Preserve supplied deterministic entry order in immutable list snapshots and retain exact
  immutable entry references.
- Require buffer slots to be unique among buffer entries and workspace slots to be unique among
  workspace entries.
- Keep buffer and workspace identity domains separate; equal numeric values across the two
  nominal record types are valid.
- Permit empty buffer and workspace lists, including a completely empty plan.
- Use ordinary nominal record equality, hashing, and diagnostic `toString()` semantics.
- Update `runtime.memory` package documentation, the Runtime API, the focused
  Runtime/Prepare/backend explanation, and the glossary.
- Add focused API and contract tests in the mirrored Runtime package.
- Finalize affected Javadocs and documentation in a separate clean documentation-focused context.
- Synchronize this task, the Runtime master plan, and the roadmap after implementation and
  validation.

## Out of scope

- importing, retaining, copying, or exposing `PreparationResourceRequirement`,
  `BackendPartitionAnalysis`, `PrepareContext`, `ValueId`, `LogicalMemoryRequirement`,
  `PlannedPartition`, `Operation`, or `CompiledNode`
- a Runtime dependency on Prepare, Planning, Compiler, Model, Engine, or a concrete backend
- the Prepare-owned deterministic requirement-to-slot translator, assignment aggregate,
  requirement coverage validation, or source-to-slot association
- backend finalization or changes to Prepare production/test code
- physical storage, allocation, deallocation, bytes ownership, storage handles, addresses,
  pooling, aliasing, reuse, liveness, interference intervals, or resource lifetimes
- device identity, placement, residency, transfer, materialization, or backend representation
- `RuntimeSlotTable`, access tables, per-run bindings, `RunState`, input binding, or run results
- `PreparedExecutable`, `PreparedUnit`, `PreparedPartition`, `PreparedSchedule`,
  `PreparedExecution`, runner, publication, transfer, or execution behavior
- route selection, lowering, executable construction, backend discovery, fallback, tuning,
  measurement, or cache behavior
- constructors, factories, builders, allocators, registries, services, maps, lookup APIs, or slot
  derivation beyond the exact record constructors
- serialization or a stable external numeric or text format
- module dependency, Gradle, Java toolchain, architecture-contract, ADR, architecture-test,
  backend-conformance, or integration-test changes
- other modules and later Runtime or Prepare task specifications

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core lifecycle and invariants
  - `modules/runtime`
  - `modules/prepare`
  - Dependency rules
  - Prepare lifecycle
  - Run lifecycle
- [ADR 0010: Stage backend preparation around shared slot assignment](../../../../design/decisions/0010-staged-backend-preparation.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime master plan](../master-plan.md)
- [Prepare master plan](../../prepare/master-plan.md)
- [Prepare 0001](../../prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [Runtime 0001](0001-prepared-buffer-slot-identity.md)

## Architecture constraints

- Runtime owns `BufferSlot`, `WorkspaceSlot`, `PreparedMemoryPlan`, and reusable prepared-memory
  geometry.
- Prepare owns backend analysis, exact analysis requirements, shared assignment orchestration,
  requirement coverage, source-to-slot association, backend finalization, and prepared-memory
  validation.
- Runtime must remain independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backend implementations.
- Runtime types contain no graph identity, partition, analysis, backend-plan, route, executable,
  storage, device, residency, or per-run binding.
- The hot path must not use `Operation` or `CompiledNode`; this task adds neither.
- Byte size and alignment describe requested slot geometry only. They do not allocate, own, bind,
  address, retain, close, or release bytes.
- The initial no-reuse policy belongs to shared Prepare translation. Distinct Runtime slot
  identities do not by themselves define physical non-aliasing or lifetime.
- Runtime must not discover backends, select kernels, lower work, inspect graphs, or tune settings.
- `ARCHITECTURE.md` and existing module dependency directions remain unchanged. If implementation
  needs another module type or dependency, stop and report the conflict.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.runtime.memory` — public prepared-memory identity and geometry
  contracts.

Packages added or changed:

- no new Java package; only the existing `runtime.memory` package changes.

Type placement:

- `io.github.pho001.synaptik.runtime.memory.WorkspaceSlot` — Runtime owns the stable
  prepared-plan-local workspace identity, colocated with `BufferSlot`.
- `io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan` — Runtime owns the immutable
  reusable final slot geometry.
- `PreparedMemoryPlan.BufferEntry` — nested because it has meaning only as one plan's
  buffer-slot geometry and does not justify another top-level public type.
- `PreparedMemoryPlan.WorkspaceEntry` — nested for the corresponding workspace-slot geometry.

Tests mirror the production package:

- `io.github.pho001.synaptik.runtime.memory.WorkspaceSlotTest`
- `io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlanTest`

No builder or deriver is added. Runtime lacks and must not receive the Prepare facts needed to
derive assignments. The later translator belongs with Prepare's requirement association and can
construct these public immutable Runtime values directly.

## Affected files

Expected Runtime production/test paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/WorkspaceSlot.java` —
  add.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlan.java`
  — add the exact plan and nested-entry records.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/package-info.java` —
  update current-versus-planned package scope.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/memory/WorkspaceSlotTest.java` —
  add.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlanTest.java`
  — add.

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found:

- `AGENTS.md`
- `ARCHITECTURE.md`
- ADR 0010
- `docs/architecture/lifecycle.md`
- `docs/architecture/module-boundaries.md`
- `docs/architecture/dependency-rules.md`
- `docs/backend-guide/partition-preparer.md`
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md`
- Runtime and Prepare build files
- architecture, backend-conformance, and integration tests

## Maximum scope

This task may create or modify at most 11 paths:

- 5 Runtime production/test paths;
- 3 explanatory documentation paths; and
- 3 Runtime/global planning paths.

No Java or test path outside `modules/runtime` may change. No Gradle, architecture contract, ADR,
architecture-test, backend-conformance, or integration-test path may change. If another path,
top-level type, package, module edge, or helper is needed, stop and propose a follow-up task. Do
not create a later task specification.

## Validation, ordering, immutability, and failure rules

Validation occurs in this exact order:

1. `WorkspaceSlot` rejects a negative component before construction completes.
2. Each `BufferEntry` validates `slot`, then `byteSize`, then `byteAlignment`.
3. Each `WorkspaceEntry` validates `slot`, then `byteSize`, then `byteAlignment`.
4. `PreparedMemoryPlan` validates top-level `buffers`, then `workspaces`.
5. The plan scans buffer entries in supplied order, rejecting the first null or later duplicate
   buffer slot.
6. The plan snapshots buffer entries.
7. The plan scans workspace entries in supplied order, rejecting the first null or later
   duplicate workspace slot.
8. The plan snapshots workspace entries.

Exact failures:

- negative `WorkspaceSlot.value`:
  `IllegalArgumentException("value must be non-negative")`
- null entry slot: `NullPointerException("slot")`
- negative entry byte size:
  `IllegalArgumentException("byteSize must be non-negative")`
- non-positive or non-power-of-two entry alignment:
  `IllegalArgumentException("byteAlignment must be a positive power of two")`
- null plan list: `NullPointerException("buffers")` or
  `NullPointerException("workspaces")`
- null plan entry: `NullPointerException("buffers[index]")` or
  `NullPointerException("workspaces[index]")`
- duplicate buffer slot:
  `IllegalArgumentException("buffers[index].slot duplicates BufferSlot[value=n]")`
- duplicate workspace slot:
  `IllegalArgumentException("workspaces[index].slot duplicates WorkspaceSlot[value=n]")`

The duplicate diagnostic uses the later entry's zero-based supplied index and ordinary record
text for its slot. Buffer and workspace duplicates are checked in separate nominal domains.

The plan preserves supplied order exactly, stores immutable list snapshots, and retains the exact
immutable entry objects and exact slot objects contained by those entries. It does not retain
caller list containers. Every type is deeply immutable. Record equality and hashing compare
stored values; record text is diagnostic only.

The constructors do not sort, renumber, normalize, merge, deduplicate, allocate resources, or
derive entries. Callers supply deterministic final order. Later Prepare translation must use
ordered `BackendPartitionAnalysis` results and each analysis's ordered requirements to assign
slots deterministically, but that source traversal and association are outside Runtime 0002.

## Acceptance criteria

- The only new top-level production types are the exact public records `WorkspaceSlot` and
  `PreparedMemoryPlan` in `runtime.memory`.
- `PreparedMemoryPlan` contains only the exact two public nested records and no other nested type.
- No extra public constructor, factory, method, field, interface, builder, deriver, allocator,
  registry, service, map, or lookup surface exists beyond explicit documented record members.
- `WorkspaceSlot` accepts and retains zero through `Long.MAX_VALUE`, rejects every negative value
  with the exact failure, and remains nominally distinct from `BufferSlot`.
- Entry records retain exact slot references, accept zero byte size, accept every valid positive
  power-of-two `long` alignment, and reject invalid components in the specified order with exact
  messages.
- `PreparedMemoryPlan` accepts empty lists, snapshots supplied order, retains exact entry
  references, exposes immutable lists, and rejects nulls and duplicate slots in the specified
  order with exact messages.
- Equal numeric buffer/workspace slot values are accepted because their nominal identity domains
  are separate.
- Tests lock exact public shape, component generic types, visibility, nested-type count, boundary
  values, validation order/messages, deterministic order, immutable snapshots, exact reference
  retention, separate uniqueness domains, and ordinary record semantics.
- Runtime production and test code imports no Prepare, Planning, Compiler, Model, Engine, concrete
  backend, physical-storage, or trace type. Production code imports only JDK collection and
  validation types; focused tests may use JDK reflection to lock the public API.
- Runtime Gradle dependencies and root Java 26 toolchain/release configuration are unchanged.
- No requirement association, graph/partition fact, physical storage/allocation/ownership,
  aliasing/reuse/lifetime, device/residency, runtime access/binding, executable, schedule,
  publication, transfer, execution, backend, or tuning behavior is present.
- Documentation distinguishes final Runtime slot geometry from Prepare's source requirements and
  later assignment/finalization association without claiming either is implemented by this task.
- A separate documentation-focused agent finalizes production/package Javadocs, Runtime API,
  focused architecture status, glossary impact, planning evidence/status, links, terminology, and
  no-change conclusions in the same overall change.
- Exactly the authorized paths change; Runtime 0001 and Prepare 0001 remain Complete; Runtime
  0003–0008 and Prepare 0002 remain Draft without detailed specifications; final newlines,
  whitespace, and `git diff --check` pass.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.memory.WorkspaceSlotTest \
  --tests io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlanTest
```

After executable Java stabilizes, run the affected module once:

```bash
./gradlew :modules:runtime:test
```

Documentation-focused pass after final Javadocs and documentation:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

If `/tmp/validate_synaptik_markdown.py` is absent or requires path arguments, create or invoke an
equivalent temporary validator outside the repository. Validate every changed Markdown file for
local link targets and heading anchors, unique effective heading anchors, balanced backtick and
tilde fences, final newlines, and trailing whitespace.

Required source/scope/status checks:

- verify the exact two-top-level/two-nested public record surface and generic list components from
  source and focused API tests;
- verify exact validation order, messages, immutable snapshots, supplied order, exact reference
  retention, and separate slot uniqueness domains;
- verify Runtime source/test import boundaries and absence of all out-of-scope types;
- verify `modules/runtime/build.gradle.kts` is unchanged and retains no Prepare, Planning,
  Compiler, Model, Engine, or concrete-backend dependency;
- verify Java 26 only from root `build.gradle.kts` and no Runtime override;
- verify the exact 11-path ceiling and no Java/test path outside Runtime;
- verify Runtime 0001 and Prepare 0001 remain Complete, Runtime 0002 is synchronized through
  In-progress and Complete transitions, and later rows remain Draft;
- verify Runtime 0003–0008 and Prepare 0002 have no detailed task specifications; and
- verify final newlines, trailing whitespace, and `git diff --check`.

Repository-wide tests and architecture tests are deferred to the Runtime prepared-contract
capability checkpoint or continuous integration. This task changes one module without changing a
dependency, architecture boundary, shared build contract, concrete backend, or end-to-end
behavior. Backend-conformance and integration tests are not applicable.

The documentation-focused context reuses successful Runtime Java-test evidence unless it changes
executable Java behavior or records a concrete reason to repeat it.

## Dependencies

- Runtime 0001 `BufferSlot` — Complete.
- Prepare 0001 exact `Buffer` and `Workspace` declarations — Complete and sufficient to justify
  the final size/alignment geometry, but not imported by Runtime.
- ADR 0010 staged analysis, shared assignment, and finalization — Accepted.
- Existing Runtime dependency and Java 26 build contracts — unchanged.

Prepare 0002, Runtime 0003–0008, concrete backends, Engine, physical storage, runtime binding, and
execution are not dependencies of this bounded geometry task.

## Follow-up tasks

- Runtime 0003 remains the next Runtime row after 0002 completes. A separate planning step must
  define typed per-run slot access without adding its specification here.
- Prepare 0002 later owns deterministic translation from ordered analyses and requirements,
  exact requirement-to-slot associations, complete/unique/foreign/duplicate coverage validation,
  and backend finalization against those assignments after Runtime 0003–0004 stabilize.
- The initial Prepare translation must assign one distinct `BufferSlot` per distinct declared
  buffer value and one distinct `WorkspaceSlot` per workspace declaration. Reuse/aliasing remains
  deferred until a later proved liveness/interference model.
- Runtime 0004 and later rows remain Draft without detailed specifications.

Do not create any follow-up specification in this task.

## Javadocs and documentation impact

- `WorkspaceSlot` Javadoc must explain its prepared-plan-local domain, valid range, immutability,
  ordinary record semantics, nominal distinction from `BufferSlot` and analysis-local workspace
  IDs, and lack of physical/resource semantics.
- `PreparedMemoryPlan` Javadoc must explain ordered immutable final slot geometry, uniqueness in
  separate domains, empty-plan validity, supplied ordering, and separation from source
  requirements, physical storage, and per-run binding.
- Entry Javadocs must document every component, unit, constraint, exact retention, failure,
  ownership boundary, and lack of allocation/resource semantics.
- Explicit accessors and canonical constructors must provide complete `@param`, `@return`, and
  `@throws` contracts without adding members beyond the record surface.
- Package Javadoc, Runtime API, focused architecture explanation, and glossary must use consistent
  current-versus-planned terminology.
- The documentation pass must review the backend preparer guide, lifecycle, module-boundary,
  dependency, Prepare master/task, Gradle, architecture-test, backend-conformance, and integration
  surfaces. Record a reasoned no-change conclusion unless a concrete contradiction is found.
- Architecture tests require no update because the module dependencies and boundary rules do not
  change.

## Architecture impact

Expected impact: None.

The architecture already assigns `WorkspaceSlot` and `PreparedMemoryPlan` ownership to Runtime,
exact analysis requirements and shared assignment to Prepare, and physical allocation/per-run
binding to later Runtime/backend work. This task implements only the final Runtime geometry. Stop
if implementation requires a Prepare/Model type, a dependency change, source-to-slot association,
physical semantics, or another architecture decision.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/runtime/tasks/0002-prepared-memory-and-workspace-contracts.md, the Runtime
master plan, ADR 0010, and the directly referenced focused contracts and documentation profiles.

Implement Runtime 0002 exactly within its 11-path ceiling. Add only WorkspaceSlot and
PreparedMemoryPlan with its two nested slot-geometry entries, focused Runtime tests, Javadocs,
and the specified explanatory/planning updates. Keep Runtime free of Prepare, Planning, Compiler,
Model, Engine, concrete backend, physical storage, run-state, executable, schedule, execution,
and tuning types or behavior. Stop on an architecture, dependency, package, API, or scope
conflict.

Run the focused tests, one final Runtime module test, and all source/scope/status checks. Then hand
the actual diff and exact Java evidence to a separate documentation-focused clean context. That
pass must follow documentation-rules.md; finalize all affected Javadocs and documentation,
glossary impact, task evidence/summary, master-plan/roadmap status, links, terminology, scope, and
no-change conclusions; and not repeat successful Java tests unless executable behavior changes or
a concrete risk is recorded.

Mark the task Complete only after every implementation and documentation gate passes. Return both
context IDs, exact paths, commands/results/counts, unresolved issues, follow-up, and the required
repository completion status.
```

## Local decisions

- `PreparedMemoryPlan` stores only final per-slot geometry. It retains immutable ordered list
  snapshots and exact immutable entry/slot references, while Prepare retains source requirements,
  assignment traversal, coverage, and source-to-slot associations.
- Buffer and workspace uniqueness are separate nominal domains. Equal numeric values across
  `BufferSlot` and `WorkspaceSlot` are valid; a duplicate inside either entry list fails at the
  first later occurrence in supplied order.
- The record constructors validate and snapshot buffers before workspaces. No sorting,
  normalization, derivation, deduplication, allocation, or lookup surface was added.
- Zero byte size is valid, and every positive power-of-two `long` alignment from `1` through
  `1L << 62` is valid. This preserves Prepare 0001's exact declaration range without importing a
  Prepare type.
- Explicit canonical constructors and accessors provide complete contract Javadocs while adding
  no members beyond the ordinary record surface locked by the focused tests.
- Documentation consistently describes the Runtime carrier as current, Prepare assignment and
  association as planned, and physical allocation/per-run access as later Runtime/backend work.

## Known limitations

- Runtime 0002 cannot prove which Prepare requirement produced an entry or which plan owns a
  standalone equal-valued slot. The later Prepare assignment/finalization contract must retain
  those associations.
- No lifetime, interference, aliasing, reuse, physical allocation, storage ownership, slot access,
  per-run binding, executable, schedule, publication, transfer, or execution contract exists.
- Repository-wide and architecture-test validation remains deferred to the Runtime
  prepared-contract capability checkpoint or continuous integration because this task changes no
  dependency, architecture rule, shared build contract, concrete backend, or end-to-end behavior.

## Validation evidence

- Implementation context `/root/implement_runtime_0002` ran
  `./gradlew :modules:runtime:test --tests
  io.github.pho001.synaptik.runtime.memory.WorkspaceSlotTest --tests
  io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlanTest`. It passed with
  `BUILD SUCCESSFUL`; JUnit XML reported two suites and 21 tests
  (`WorkspaceSlotTest` 5 and `PreparedMemoryPlanTest` 16), with zero failures, errors, and skips.
  Gradle reported nine actionable tasks, three executed and six up-to-date.
- The same implementation context ran the single final affected-module command
  `./gradlew :modules:runtime:test` after executable Java stabilized. It passed with
  `BUILD SUCCESSFUL`; JUnit XML reported three suites and 25 tests (`BufferSlotTest` 4,
  `WorkspaceSlotTest` 5, and `PreparedMemoryPlanTest` 16), with zero failures, errors, and skips.
  Gradle reported nine actionable tasks, one executed and eight up-to-date.
- Documentation context `/root/implement_runtime_0002/runtime_0002_docs` changed Javadocs and
  documentation but no executable Java after that evidence, so it reused both successful test
  results and did not repeat either Java suite. The implementation context's pre-handoff
  `git diff --check` also passed.
- The documentation context applied General, API/Javadoc, Architecture, and Planning profiles. It
  read the architecture contract, ADR 0010, planning guide/roadmap, Runtime and Prepare plans and
  completed predecessor tasks, documentation rules/profiles, focused architecture/lifecycle/
  boundary/dependency explanations, backend guide, Runtime API, complete glossary, final Runtime
  source/tests, build contracts, test inventories, later-spec inventory, and actual diff.
- Final Runtime Javadoc generation passed:

  ```bash
  ./gradlew :modules:runtime:javadoc
  ```

  Gradle reported `BUILD SUCCESSFUL`; five tasks were actionable, two executed and three
  up-to-date. Generated pages for `WorkspaceSlot`, `PreparedMemoryPlan`, both nested entries, and
  the package contain the identity, units, range, nullability, immutability, ordering, reference
  retention, failure, ordinary record, and non-resource boundaries.
- The Runtime API example was extracted to `/tmp/Runtime0002ApiExample.java`, then
  `javac --release 26 -cp modules/runtime/build/classes/java/main -d
  /tmp/runtime-0002-api-example /tmp/Runtime0002ApiExample.java` and
  `java -cp modules/runtime/build/classes/java/main:/tmp/runtime-0002-api-example
  Runtime0002ApiExample` passed with no output. It constructs the documented ordered buffer and
  workspace geometry and checks exact slot-reference, byte-size, and alignment retention.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md
  docs/architecture/runtime-prepare-backend-boundary.md docs/glossary.md
  docs/planning/modules/runtime/tasks/0002-prepared-memory-and-workspace-contracts.md
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md` passed after the first
  documentation edits with `validated 6 Markdown files`; the same six-path command passed again
  after final evidence and status synchronization. It checked local targets and heading anchors,
  unique effective heading anchors, balanced backtick/tilde fences, final newlines, and trailing
  whitespace.
- The final source/API/import audit confirmed exactly the new top-level public records
  `WorkspaceSlot` and `PreparedMemoryPlan`, exactly the nested public records `BufferEntry` and
  `WorkspaceEntry`, the two generic list components, and no extra helper type or public surface.
  Focused tests lock validation order/messages, boundary alignments, immutable ordered snapshots,
  exact entry/slot reference retention, separate duplicate domains, empty plans, and ordinary
  record semantics.
- Runtime 0002 production imports only `java.util.HashSet`, `java.util.List`, and
  `java.util.Objects`; `WorkspaceSlot` has no imports. Focused tests import only JUnit and JDK
  collections/reflection. The source audit found no Prepare, Planning, Compiler, Model, Engine,
  concrete-backend, graph, storage, run-state, executable, schedule, or tuning type or behavior.
- The build audit confirmed `modules/runtime/build.gradle.kts` and root `build.gradle.kts` are
  unchanged. Runtime dependencies remain Config, Backend Contract, and Trace only; root
  configuration alone sets `JavaLanguageVersion.of(26)` and `options.release.set(26)`.
- The final scope audit found exactly the 11 authorized paths: five Runtime production/test
  paths, three explanatory documentation paths, and three planning paths. No Java/test path
  outside Runtime, Gradle file, architecture contract/ADR/test, backend-conformance, integration,
  Prepare source/test, other module, or later task specification changed.
- Status/specification checks confirmed Runtime 0001 and Prepare 0001 remain Complete; Runtime
  0002 is Complete in this task, the Runtime master plan, and the roadmap while the Runtime
  project remains In progress; Runtime 0003–0008 and Prepare 0002 remain Draft; and the only
  detailed Runtime task specs are 0001 and 0002 while the only detailed Prepare task spec is
  0001.
- `git diff --check` passed after final documentation and status synchronization with no output.
- No-change conclusions:
  - `BufferSlot` Javadoc remains accurate because this task consumes its unchanged plan-local
    identity without changing its range, equality, ownership, or non-resource semantics.
  - `ARCHITECTURE.md` and ADR 0010 already assign final geometry to Runtime, analysis/assignment/
    finalization to Prepare and concrete backends, and physical/run resources to later work; no
    architecture decision or dependency rule changed.
  - Lifecycle, module-boundary, and dependency explanations remain accurate at their architecture
    level; the focused boundary page carries the needed current-versus-planned implementation
    status.
  - The backend partition-preparer guide and Prepare master/task remain analysis-focused and
    correctly defer assignment/finalization; they need no executable example or contract change
    for a dependency-neutral Runtime result carrier.
  - Runtime and Prepare Gradle files need no change because no module edge was added; root Java 26
    configuration remains sufficient.
  - Architecture tests need no update because dependency directions and prohibited semantic
    imports are unchanged. Backend-conformance and integration tests remain inapplicable because
    this task adds no concrete backend or end-to-end execution behavior.
  - Other APIs/modules need no change because there is no Engine facade, caller workflow,
    physical resource, executable, run, publication, or configuration surface to document.
  - No later Runtime or Prepare specification was created because frontier selection is a
    separate planning step after completion.

## Implementation notes

- Added only `WorkspaceSlot` and `PreparedMemoryPlan` with its exact two nested entry records in
  the existing `runtime.memory` package, plus the two focused test suites.
- Constructor validation implements the specified deterministic order and exact failures.
  Successful plans preserve supplied buffer/workspace order in immutable snapshots while retaining
  exact immutable entries and slots.
- Final Javadocs cover every public type, canonical constructor, component, accessor, return,
  caller-visible failure, unit, nullability, range, identity, equality, mutation, and resource
  boundary. Package documentation now describes the complete current memory surface.
- Runtime API, focused architecture explanation, glossary, Runtime master plan, and roadmap now
  distinguish current final Runtime geometry from planned Prepare assignment/finalization and
  later physical/run state.
- No executable Java changed during the documentation pass.

## Completion summary

- Completed changes: implemented and documented the exact nominal workspace-slot identity and
  immutable ordered per-buffer/per-workspace final byte geometry.
- Files changed or created: exactly the 11 authorized paths—five Runtime production/test paths,
  Runtime API, focused boundary explanation, glossary, this task, Runtime master plan, and
  roadmap.
- Tests and validation: reused the successful focused 21-test and final 25-test Runtime evidence;
  Runtime Javadoc, generated pages, compiled Java 26 API example, six-file Markdown, exact
  source/API/import/build/toolchain/scope/status/specification, final-newline/fence/whitespace, and
  final `git diff --check` gates passed.
- Documentation-agent review: completed in clean context
  `/root/implement_runtime_0002/runtime_0002_docs` without executable Java changes or duplicate
  Java-test execution.
- Documentation impact: the API and focused architecture page now expose the current geometry
  mental model and example while keeping assignment, source associations, allocation, binding,
  finalization, and execution explicitly later.
- Javadoc review: finalized `WorkspaceSlot`, `PreparedMemoryPlan`, both nested entries, explicit
  constructors/accessors, and package documentation. Existing `BufferSlot` Javadoc remains
  accurate without modification.
- Glossary impact: made Runtime implementation status current and added reusable
  `WorkspaceSlot` and `PreparedMemoryPlan` definitions while refining buffer-slot and memory-slot
  distinctions.
- Unresolved issues: None.
- Follow-up required: None for this task. Runtime 0003 remains Draft pending a separate frontier
  planning step; Prepare 0002 remains Draft until Runtime 0003–0004 stabilize.

Status: Complete
