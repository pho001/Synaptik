# Task 0002: Prepared Memory and Workspace Contracts

> **Editorial historical reading key.**
>
> Read this completed task as the historical introduction of Runtime’s geometry carrier, not an
> allocator or execution owner. Shared Prepare is the intended caller of the record constructors:
> backend analysis supplies buffer/workspace requirements, and Prepare translates them into ordered
> slot identities, byte sizes and alignments while retaining source-to-slot associations outside
> Runtime. Runtime defines the carrier’s validation and nominal identity rules; it neither interprets
> graph values nor frees storage. Snapshotting list membership retains exact immutable entries rather
> than copying payload bytes. The [current memory
> contract](../../../../reference/runtime/memory.md#current-prepared-memory-plan) preserves that
> distinction. Assignment and finalization described here as later work now have a [current Prepare
> handoff](../../../../reference/runtime/orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization),
> with explicit persistent-resource rollback and ownership transfer separate from geometry.

## Status

Complete

## Goal

Add the smallest [immutable](../../../../glossary.md#term-immutable) [Runtime](../../../../glossary.md#term-runtime)-owned prepared-memory geometry that shared [Prepare](../../../../glossary.md#prepare) can produce
after [backend](../../../../glossary.md#backend) analysis without moving Prepare or [Model](../../../../glossary.md#term-model) facts into Runtime.

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

[`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) is a non-negative [identity](../../../../glossary.md#term-identity-meanings) in one prepared-memory-plan context. It is nominally
distinct from [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) and from Prepare's analysis-local workspace requirement ID even when
their numeric values happen to match.

Each nested entry records exact non-negative byte size and positive power-of-two [byte alignment](../../../../glossary.md#term-byte-alignment)
for one Runtime slot. [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) snapshots caller-supplied [buffer](../../../../glossary.md#term-buffer) and workspace entry
order and requires each slot identity to occur exactly once in its own domain.

This task does not import or retain `PreparationResourceRequirement`, [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis),
[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), `LogicalMemoryRequirement`, or `PlannedPartition`. A later Prepare-owned assignment and
finalization contract will translate exact analysis requirements into this backend-neutral
Runtime geometry while retaining the source-to-slot associations that backend finalization needs.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
Prepare analysis facts                     Runtime 0002 geometry

Buffer(ValueId, bytes, alignment)      ->  BufferEntry(BufferSlot, bytes, alignment)
Workspace(local ID, bytes, alignment)  ->  WorkspaceEntry(WorkspaceSlot, bytes, alignment)

source-to-slot association: Prepare        reusable slot geometry: Runtime
physical storage and run binding: later Runtime/backend work
```

[Prepare](../../../../glossary.md#prepare) 0001 makes the required size and [alignment](../../../../glossary.md#term-alignment) facts concrete, but [Runtime](../../../../glossary.md#term-runtime) must not acquire a
[dependency](../../../../glossary.md#term-dependency) on Prepare or [Model](../../../../glossary.md#term-model) to represent reusable prepared state. Runtime therefore owns only
the final slot [identities](../../../../glossary.md#term-identity-meanings) and their exact geometry. Shared Prepare later owns deterministic
translation, requirement coverage, and source association.

The initial translation policy is conservative: every distinct [buffer](../../../../glossary.md#term-buffer) value requirement receives
its own [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot), and every workspace declaration receives its own [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot). No slot
reuse is authorized because no [liveness](../../../../glossary.md#term-liveness) or interference proof exists. Runtime 0002 locks the
unique final slot geometry needed by that policy; it does not itself inspect analysis results.

## [Scope](../../../../glossary.md#term-scope)

- Add public final record [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) with exactly one `long value` component.
- Accept every non-negative workspace-slot value, including zero and `Long.MAX_VALUE`.
- Reject a negative workspace-slot value with `IllegalArgumentException` and exact message
  `value must be non-negative`.
- Define workspace-slot identity only within one owning prepared-memory-plan context; another plan
  may reuse the same numeric value.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add public final record [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) with exactly the ordered [immutable](../../../../glossary.md#term-immutable)
  [`buffers`](../../../../glossary.md#term-buffer) and `workspaces` list components shown above.
- Add only the public nested final records `BufferEntry` and `WorkspaceEntry` shown above.
- Require every entry slot to be non-null, every byte size to be non-negative, and every byte
  alignment to be a positive power of two.<span data-glossary-navigation="true"> ([byte alignment glossary](../../../../glossary.md#term-byte-alignment))</span>
- Accept zero byte size and [alignments](../../../../glossary.md#term-alignment) from `1` through `1L << 62`.
- Preserve supplied deterministic entry order in immutable list snapshots and retain exact
  immutable entry references.
- Require buffer slots to be unique among buffer entries and workspace slots to be unique among
  workspace entries.
- Keep buffer and workspace identity domains separate; equal numeric values across the two
  nominal record types are valid.
- Permit empty buffer and workspace lists, including a completely empty plan.
- Use ordinary [nominal](../../../../glossary.md#term-nominal) record equality, hashing, and diagnostic `toString()` semantics.
- Update `runtime.memory` package documentation, the Runtime API, the focused
  Runtime/Prepare/backend explanation, and the glossary.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- Add focused [API](../../../../glossary.md#term-application-programming-interface-api) and contract tests in the mirrored [Runtime](../../../../glossary.md#term-runtime) package.
- Finalize affected Javadocs and documentation in a separate clean documentation-focused context.
- Synchronize this task, the Runtime master plan, and the roadmap after implementation and
  validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- importing, retaining, copying, or exposing `PreparationResourceRequirement`,
  [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis), [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), `LogicalMemoryRequirement`,
  `PlannedPartition`, [`Operation`](../../../../glossary.md#operation), or `CompiledNode`
- a [Runtime](../../../../glossary.md#term-runtime) [dependency](../../../../glossary.md#term-dependency) on [Prepare](../../../../glossary.md#prepare), [Planning](../../../../glossary.md#term-planning), [Compiler](../../../../glossary.md#term-compiler), [Model](../../../../glossary.md#term-model), [Engine](../../../../glossary.md#term-engine), or a [concrete backend](../../../../glossary.md#concrete-backend)
- the Prepare-owned deterministic requirement-to-slot translator, assignment aggregate,
  requirement coverage validation, or source-to-slot association<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [backend](../../../../glossary.md#backend) finalization or changes to Prepare production/test code
- physical storage, allocation, deallocation, bytes ownership, storage handles, addresses,
  pooling, aliasing, reuse, liveness, interference intervals, or resource lifetimes<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- [device](../../../../glossary.md#term-device) [identity](../../../../glossary.md#term-identity-meanings), placement, [residency](../../../../glossary.md#residency), transfer, [materialization](../../../../glossary.md#materialization), or backend representation
- `RuntimeSlotTable`, access tables, per-[run](../../../../glossary.md#term-run) [bindings](../../../../glossary.md#term-binding), [`RunState`](../../../../glossary.md#run-state--runstate), input binding, or run results
- [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), `PreparedUnit`, [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule),
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), runner, [publication](../../../../glossary.md#term-publication), transfer, or execution behavior
- route selection, lowering, executable construction, backend discovery, fallback, tuning,
  measurement, or cache behavior<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [cache glossary](../../../../glossary.md#term-cache))</span>
- constructors, factories, builders, allocators, registries, services, maps, lookup APIs, or slot
  derivation beyond the exact record constructors<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- [serialization](../../../../glossary.md#term-serialization) or a stable external numeric or text format
- module dependency, Gradle, Java toolchain, architecture-contract, ADR, architecture-test,
  backend-conformance, or integration-test changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- other [modules](../../../../glossary.md#term-module) and later Runtime or Prepare task specifications

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core [lifecycle](../../../../glossary.md#lifecycle) and invariants
  - `modules/runtime`
  - `modules/prepare`
  - [Dependency](../../../../glossary.md#term-dependency) rules
  - [Prepare](../../../../glossary.md#prepare) lifecycle
  - [Run](../../../../glossary.md#term-run) lifecycle
- [ADR 0010: Stage backend preparation around shared slot assignment](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Runtime master plan](../master-plan.md)
- [Prepare master plan](../../prepare/master-plan.md)
- [Prepare 0001](../../prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [Runtime 0001](0001-prepared-buffer-slot-identity.md)

## Architecture constraints

- [Runtime](../../../../glossary.md#term-runtime) owns [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot), [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot), [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan), and reusable prepared-memory
  geometry.
- Prepare owns backend analysis, exact analysis requirements, shared assignment orchestration,
  requirement coverage, source-to-slot association, backend finalization, and prepared-memory
  validation.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [validation glossary](../../../../glossary.md#term-validation))</span>
- Runtime must remain independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backend implementations.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Runtime types contain no graph identity, partition, analysis, backend-plan, route, executable,
  storage, device, residency, or per-run binding.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [partition glossary](../../../../glossary.md#partition); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding))</span>
- The [hot path](../../../../glossary.md#term-hot-path) must not use [`Operation`](../../../../glossary.md#operation) or `CompiledNode`; this task adds neither.
- Byte size and alignment describe requested slot geometry only. They do not allocate, own, bind,
  address, retain, close, or release bytes.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [release glossary](../../../../glossary.md#term-release))</span>
- The initial no-reuse policy belongs to shared Prepare translation. Distinct Runtime slot
  identities do not by themselves define physical non-aliasing or lifetime.<span data-glossary-navigation="true"> ([aliasing glossary](../../../../glossary.md#term-aliasing); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Runtime must not discover [backends](../../../../glossary.md#backend), select [kernels](../../../../glossary.md#kernel), lower work, inspect [graphs](../../../../glossary.md#graph), or tune settings.
- `ARCHITECTURE.md` and existing module dependency directions remain unchanged. If implementation
  needs another module type or dependency, stop and report the conflict.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.runtime.memory` — public prepared-memory identity and geometry
  contracts.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>

Packages added or changed:

- no new Java package; only the existing `runtime.memory` package changes.

Type placement:

- `io.github.pho001.synaptik.runtime.memory.WorkspaceSlot` — Runtime owns the stable
  prepared-plan-local workspace identity, colocated with [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot).
- `io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan` — Runtime owns the immutable
  reusable final slot geometry.
- `PreparedMemoryPlan.BufferEntry` — nested because it has meaning only as one plan's
  buffer-slot geometry and does not justify another top-level public type.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- `PreparedMemoryPlan.WorkspaceEntry` — nested for the corresponding workspace-slot geometry.

Tests mirror the production package:

- `io.github.pho001.synaptik.runtime.memory.WorkspaceSlotTest`
- `io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlanTest`

No builder or deriver is added. [Runtime](../../../../glossary.md#term-runtime) lacks and must not receive the [Prepare](../../../../glossary.md#prepare) facts needed to
derive assignments. The later translator belongs with Prepare's requirement association and can
construct these public [immutable](../../../../glossary.md#term-immutable) Runtime values directly.

## Affected files

Expected [Runtime](../../../../glossary.md#term-runtime) production/test paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/WorkspaceSlot.java` —
  add.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlan.java`
  — add the exact plan and nested-entry records.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/package-info.java` —
  update current-versus-planned package scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/memory/WorkspaceSlotTest.java` —
  add.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlanTest.java`
  — add.

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found:

- `AGENTS.md`
- `ARCHITECTURE.md`
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010
- `docs/architecture/lifecycle.md`
- `docs/architecture/module-boundaries.md`
- `docs/architecture/dependency-rules.md`
- `docs/backend-guide/partition-preparer.md`
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md`
- Runtime and [Prepare](../../../../glossary.md#prepare) build files
- architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration tests

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 11 paths:

- 5 [Runtime](../../../../glossary.md#term-runtime) production/test paths;
- 3 explanatory documentation paths; and
- 3 Runtime/global [planning](../../../../glossary.md#term-planning) paths.

No Java or test path outside `modules/runtime` may change. No [Gradle](../../../../glossary.md#term-gradle), [architecture contract](../../../../glossary.md#architecture-contract), [ADR](../../../../glossary.md#term-architecture-decision-record-adr),
architecture-test, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration-test path may change. If another path,
top-level type, package, [module](../../../../glossary.md#term-module) edge, or helper is needed, stop and propose a follow-up task. Do
not create a later task specification.

## [Validation](../../../../glossary.md#term-validation), ordering, immutability, and failure rules

Validation occurs in this exact order:

1. [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) rejects a negative component before construction completes.
2. Each `BufferEntry` validates `slot`, then `byteSize`, then `byteAlignment`.
3. Each `WorkspaceEntry` validates `slot`, then `byteSize`, then `byteAlignment`.
4. [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) validates top-level [`buffers`](../../../../glossary.md#term-buffer), then `workspaces`.
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
- non-positive or non-power-of-two entry [alignment](../../../../glossary.md#term-alignment):
  `IllegalArgumentException("byteAlignment must be a positive power of two")`
- null plan list: `NullPointerException("buffers")` or
  `NullPointerException("workspaces")`
- null plan entry: `NullPointerException("buffers[index]")` or
  `NullPointerException("workspaces[index]")`
- duplicate [buffer slot](../../../../glossary.md#buffer-slot--bufferslot):
  `IllegalArgumentException("buffers[index].slot duplicates BufferSlot[value=n]")`
- duplicate [workspace slot](../../../../glossary.md#workspace-slot--workspaceslot):
  `IllegalArgumentException("workspaces[index].slot duplicates WorkspaceSlot[value=n]")`

The duplicate diagnostic uses the later entry's zero-based supplied index and ordinary record
text for its slot. Buffer and workspace duplicates are checked in separate [nominal](../../../../glossary.md#term-nominal) domains.

The plan preserves supplied order exactly, stores [immutable](../../../../glossary.md#term-immutable) list snapshots, and retains the exact
immutable entry objects and exact slot objects contained by those entries. It does not retain
caller list containers. Every type is deeply immutable. Record equality and hashing compare
stored values; record text is diagnostic only.

The constructors do not sort, renumber, normalize, merge, deduplicate, allocate resources, or
derive entries. Callers supply deterministic final order. Later [Prepare](../../../../glossary.md#prepare) translation must use
ordered [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) results and each analysis's ordered requirements to assign
slots deterministically, but that source traversal and association are outside [Runtime](../../../../glossary.md#term-runtime) 0002.

## Acceptance criteria

- The only new top-level production types are the exact public records [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) and
  [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) in `runtime.memory`.
- `PreparedMemoryPlan` contains only the exact two public nested records and no other nested type.
- No extra public constructor, factory, method, field, interface, builder, deriver, allocator,
  registry, service, map, or lookup surface exists beyond explicit documented record members.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- `WorkspaceSlot` accepts and retains zero through `Long.MAX_VALUE`, rejects every negative value
  with the exact failure, and remains nominally distinct from [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot).
- Entry records retain exact slot references, accept zero byte size, accept every valid positive
  power-of-two `long` alignment, and reject invalid components in the specified order with exact
  messages.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- `PreparedMemoryPlan` accepts empty lists, snapshots supplied order, retains exact entry
  references, exposes immutable lists, and rejects nulls and duplicate slots in the specified
  order with exact messages.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Equal numeric buffer/workspace slot values are accepted because their nominal identity domains
  are separate.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [workspace slot glossary](../../../../glossary.md#workspace-slot--workspaceslot); [nominal glossary](../../../../glossary.md#term-nominal); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tests lock exact public shape, component generic types, visibility, nested-type count, boundary
  values, validation order/messages, deterministic order, immutable snapshots, exact reference
  retention, separate uniqueness domains, and ordinary record semantics.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- Runtime production and test code imports no Prepare, Planning, Compiler, Model, Engine, concrete
  backend, physical-storage, or trace type. Production code imports only JDK collection and
  validation types; focused tests may use JDK reflection to lock the public API.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [trace glossary](../../../../glossary.md#trace); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Runtime](../../../../glossary.md#term-runtime) [Gradle](../../../../glossary.md#term-gradle) dependencies and root Java 26 [toolchain](../../../../glossary.md#term-toolchain)/[release](../../../../glossary.md#term-release) configuration are unchanged.
- No requirement association, graph/partition fact, physical storage/allocation/ownership,
  aliasing/reuse/lifetime, device/residency, runtime access/binding, executable, schedule,
  publication, transfer, execution, backend, or tuning behavior is present.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing); [lifetime glossary](../../../../glossary.md#term-lifetime); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [backend glossary](../../../../glossary.md#backend))</span>
- Documentation distinguishes final Runtime slot geometry from Prepare's source requirements and
  later assignment/finalization association without claiming either is implemented by this task.
- A separate documentation-focused agent finalizes production/package Javadocs, Runtime API,
  focused architecture status, glossary impact, planning evidence/status, links, terminology, and
  no-change conclusions in the same overall change.
- Exactly the authorized paths change; Runtime 0001 and Prepare 0001 remain Complete; Runtime
  0003–0008 and Prepare 0002 remain Draft without detailed specifications; final newlines,
  whitespace, and `git diff --check` pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.memory.WorkspaceSlotTest \
  --tests io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlanTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) the affected [module](../../../../glossary.md#term-module) once:

```bash
./gradlew :modules:runtime:test
```

Documentation-focused pass after final Javadocs and documentation:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

If `/tmp/validate_synaptik_markdown.py` is absent or requires path [arguments](../../../../glossary.md#term-argument), create or invoke an
equivalent temporary validator outside the repository. Validate every changed Markdown file for
local link [targets](../../../../glossary.md#term-target) and heading anchors, unique effective heading anchors, balanced backtick and
tilde fences, final newlines, and trailing whitespace.

Required source/[scope](../../../../glossary.md#term-scope)/status checks:

- verify the exact two-top-level/two-nested public record surface and generic list components from
  source and focused API tests;<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- verify exact validation order, messages, immutable snapshots, supplied order, exact reference
  retention, and separate slot uniqueness domains;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- verify [Runtime](../../../../glossary.md#term-runtime) source/test import boundaries and absence of all out-of-scope types;
- verify `modules/runtime/build.gradle.kts` is unchanged and retains no Prepare, Planning,
  Compiler, Model, Engine, or concrete-backend dependency;<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- verify Java 26 only from root `build.gradle.kts` and no Runtime override;
- verify the exact 11-path ceiling and no Java/test path outside Runtime;
- verify Runtime 0001 and Prepare 0001 remain Complete, Runtime 0002 is synchronized through
  In-progress and Complete transitions, and later rows remain Draft;
- verify Runtime 0003–0008 and [Prepare](../../../../glossary.md#prepare) 0002 have no detailed task specifications; and
- verify final newlines, trailing whitespace, and `git diff --check`.

Repository-wide tests and architecture tests are deferred to the Runtime prepared-contract
capability [checkpoint](../../../../glossary.md#term-checkpoint) or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one module without changing a
[dependency](../../../../glossary.md#term-dependency), architecture boundary, shared build contract, [concrete backend](../../../../glossary.md#concrete-backend), or end-to-end
behavior. [Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and integration tests are not applicable.

The documentation-focused context reuses successful Runtime Java-test evidence unless it changes
executable Java behavior or records a concrete reason to repeat it.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0001 [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) — Complete.
- [Prepare](../../../../glossary.md#prepare) 0001 exact [`Buffer`](../../../../glossary.md#term-buffer) and `Workspace` declarations — Complete and sufficient to justify
  the final size/alignment geometry, but not imported by Runtime.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010 staged analysis, shared assignment, and finalization — Accepted.
- Existing Runtime [dependency](../../../../glossary.md#term-dependency) and Java 26 build contracts — unchanged.

Prepare 0002, Runtime 0003–0008, concrete [backends](../../../../glossary.md#backend), [Engine](../../../../glossary.md#term-engine), physical storage, runtime [binding](../../../../glossary.md#term-binding), and
execution are not dependencies of this bounded geometry task.

## Follow-up tasks

- Runtime 0003 remains the next Runtime row after 0002 completes. A separate planning step must
  define typed per-run slot access without adding its specification here.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [run glossary](../../../../glossary.md#term-run))</span>
- Prepare 0002 later owns deterministic translation from ordered analyses and requirements,
  exact requirement-to-slot associations, complete/unique/foreign/duplicate coverage validation,
  and backend finalization against those assignments after Runtime 0003–0004 stabilize.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend))</span>
- The initial [Prepare](../../../../glossary.md#prepare) translation must assign one distinct [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) per distinct declared
  buffer value and one distinct <span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>[`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) per workspace declaration. Reuse/aliasing remains
  deferred until a later proved liveness/interference model.<span data-glossary-navigation="true"> ([aliasing glossary](../../../../glossary.md#term-aliasing); [liveness glossary](../../../../glossary.md#term-liveness); [model glossary](../../../../glossary.md#term-model))</span>
- [Runtime](../../../../glossary.md#term-runtime) 0004 and later rows remain Draft without detailed specifications.

Do not create any follow-up specification in this task.

## Javadocs and documentation impact

- [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) Javadoc must explain its prepared-plan-local domain, valid range, immutability,
  ordinary record semantics, nominal distinction from <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [nominal glossary](../../../../glossary.md#term-nominal))</span>[`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) and analysis-local workspace
  IDs, and lack of physical/resource semantics.
- [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) Javadoc must explain ordered immutable final slot geometry, uniqueness in
  separate domains, empty-plan validity, supplied ordering, and separation from source
  requirements, physical storage, and per-run binding.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding))</span>
- Entry Javadocs must document every component, unit, constraint, exact retention, failure,
  ownership boundary, and lack of allocation/resource semantics.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Explicit accessors and [canonical](../../../../glossary.md#term-canonical-form) constructors must provide complete `@param`, `@return`, and
  `@throws` contracts without adding members beyond the record surface.
- Package Javadoc, Runtime API, focused architecture explanation, and glossary must use consistent
  current-versus-planned terminology.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The documentation pass must review the backend preparer guide, lifecycle, module-boundary,
  dependency, Prepare master/task, Gradle, architecture-test, backend-conformance, and integration
  surfaces. Record a reasoned no-change conclusion unless a concrete contradiction is found.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Prepare glossary](../../../../glossary.md#prepare); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Architecture tests require no update because the module dependencies and boundary rules do not
  change.

## Architecture impact

Expected impact: None.

The architecture already assigns [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) and [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) [ownership](../../../../glossary.md#term-ownership-meanings) to [Runtime](../../../../glossary.md#term-runtime),
exact analysis requirements and shared assignment to [Prepare](../../../../glossary.md#prepare), and physical allocation/per-[run](../../../../glossary.md#term-run)
[binding](../../../../glossary.md#term-binding) to later Runtime/[backend](../../../../glossary.md#backend) work. This task implements only the final Runtime geometry. Stop
if implementation requires a Prepare/[Model](../../../../glossary.md#term-model) type, a [dependency](../../../../glossary.md#term-dependency) change, source-to-slot association,
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

- [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) stores only final per-slot geometry. It retains immutable ordered list
  snapshots and exact immutable entry/slot references, while Prepare retains source requirements,
  assignment traversal, coverage, and source-to-slot associations.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [Buffer](../../../../glossary.md#term-buffer) and workspace uniqueness are separate [nominal](../../../../glossary.md#term-nominal) domains. Equal numeric values across
  [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) and [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) are valid; a duplicate inside either entry list fails at the
  first later occurrence in supplied order.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- The record constructors validate and snapshot buffers before workspaces. No sorting,
  normalization, derivation, deduplication, allocation, or lookup surface was added.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [normalization glossary](../../../../glossary.md#term-normalization); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Zero byte size is valid, and every positive power-of-two `long` [alignment](../../../../glossary.md#term-alignment) from `1` through
  `1L << 62` is valid. This preserves Prepare 0001's exact declaration range without importing a
  Prepare type.
- Explicit canonical constructors and accessors provide complete contract Javadocs while adding
  no members beyond the ordinary record surface locked by the focused tests.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Documentation consistently describes the Runtime carrier as current, Prepare assignment and
  association as planned, and physical allocation/per-run access as later Runtime/backend work.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [backend glossary](../../../../glossary.md#backend))</span>

## Known limitations

- Runtime 0002 cannot prove which Prepare requirement produced an entry or which plan owns a
  standalone equal-valued slot. The later Prepare assignment/finalization contract must retain
  those associations.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>
- No lifetime, interference, aliasing, reuse, physical allocation, storage ownership, slot access,
  per-run binding, executable, schedule, publication, transfer, or execution contract exists.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [aliasing glossary](../../../../glossary.md#term-aliasing); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication))</span>
- Repository-wide and architecture-test validation remains deferred to the Runtime
  prepared-contract capability checkpoint or continuous integration because this task changes no
  dependency, architecture rule, shared build contract, concrete backend, or end-to-end behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [dependency glossary](../../../../glossary.md#term-dependency); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/implement_runtime_0002` ran
  `./gradlew :modules:runtime:test --tests
  io.github.pho001.synaptik.runtime.memory.WorkspaceSlotTest --tests
  io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlanTest`. It passed with
  `BUILD SUCCESSFUL`; JUnit XML reported two suites and 21 tests
  (`WorkspaceSlotTest` 5 and `PreparedMemoryPlanTest` 16), with zero failures, errors, and skips.
  Gradle reported nine actionable tasks, three executed and six up-to-date.
- The same implementation context ran the single final affected-[module](../../../../glossary.md#term-module) command
  `./gradlew :modules:runtime:test` after executable Java stabilized. It passed with
  `BUILD SUCCESSFUL`; [JUnit](../../../../glossary.md#term-junit) XML reported three suites and 25 tests (`BufferSlotTest` 4,
  `WorkspaceSlotTest` 5, and `PreparedMemoryPlanTest` 16), with zero failures, errors, and skips.
  Gradle reported nine actionable tasks, one executed and eight up-to-date.
- Documentation context `/root/implement_runtime_0002/runtime_0002_docs` changed Javadocs and
  documentation but no executable Java after that evidence, so it reused both successful test
  results and did not repeat either Java suite. The implementation context's pre-handoff<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
  `git diff --check` also passed.
- The documentation context applied General, API/Javadoc, Architecture, and Planning profiles. It
  read the architecture contract, ADR 0010, planning guide/roadmap, Runtime and Prepare plans and
  completed predecessor tasks, documentation rules/profiles, focused architecture/lifecycle/
  boundary/dependency explanations, backend guide, Runtime API, complete glossary, final Runtime
  source/tests, build contracts, test inventories, later-spec inventory, and actual diff.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend))</span>
- Final [Runtime](../../../../glossary.md#term-runtime) [Javadoc](../../../../glossary.md#term-javadoc) generation passed:

  ```bash
  ./gradlew :modules:runtime:javadoc
  ```

  [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL`; five tasks were actionable, two executed and three
  up-to-date. Generated pages for [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot), [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan), both nested entries, and
  the package contain the identity, units, range, nullability, immutability, ordering, reference
  retention, failure, ordinary record, and non-resource boundaries.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- The Runtime [API](../../../../glossary.md#term-application-programming-interface-api) example was extracted to `/tmp/Runtime0002ApiExample.java`, then
  `javac --release 26 -cp modules/runtime/build/classes/java/main -d
  /tmp/runtime-0002-api-example /tmp/Runtime0002ApiExample.java` and
  `java -cp modules/runtime/build/classes/java/main:/tmp/runtime-0002-api-example
  Runtime0002ApiExample` passed with no output. It constructs the documented ordered buffer and
  workspace geometry and checks exact slot-reference, byte-size, and alignment retention.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md
  docs/architecture/runtime-prepare-backend-boundary.md docs/glossary.md
  docs/planning/modules/runtime/tasks/0002-prepared-memory-and-workspace-contracts.md
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md` passed after the first
  documentation edits with `validated 6 Markdown files`; the same six-path command passed again
  after final evidence and status synchronization. It checked local targets and heading anchors,
  unique effective heading anchors, balanced backtick/tilde fences, final newlines, and trailing
  whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- The final source/API/import audit confirmed exactly the new top-level public records
  `WorkspaceSlot` and `PreparedMemoryPlan`, exactly the nested public records `BufferEntry` and
  `WorkspaceEntry`, the two generic list components, and no extra helper type or public surface.
  Focused tests lock validation order/messages, boundary alignments, immutable ordered snapshots,
  exact entry/slot reference retention, separate duplicate domains, empty plans, and ordinary
  record semantics.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Runtime 0002 production imports only `java.util.HashSet`, `java.util.List`, and
  `java.util.Objects`; `WorkspaceSlot` has no imports. Focused tests import only JUnit and JDK
  collections/reflection. The source audit found no Prepare, Planning, Compiler, Model, Engine,
  concrete-backend, graph, storage, run-state, executable, schedule, or tuning type or behavior.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine); [graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run))</span>
- The build audit confirmed `modules/runtime/build.gradle.kts` and root `build.gradle.kts` are
  unchanged. Runtime dependencies remain Config, Backend Contract, and Trace only; root
  configuration alone sets <span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>`JavaLanguageVersion.of(26)` and `options.release.set(26)`.
- The final scope audit found exactly the 11 authorized paths: five Runtime production/test
  paths, three explanatory documentation paths, and three planning paths. No Java/test path
  outside Runtime, Gradle file, architecture contract/ADR/test, backend-conformance, integration,
  Prepare source/test, other module, or later task specification changed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Status/specification checks confirmed Runtime 0001 and Prepare 0001 remain Complete; Runtime
  0002 is Complete in this task, the Runtime master plan, and the roadmap while the Runtime
  project remains In progress; Runtime 0003–0008 and Prepare 0002 remain Draft; and the only
  detailed Runtime task specs are 0001 and 0002 while the only detailed Prepare task spec is
  0001.
- `git diff --check` passed after final documentation and status [synchronization](../../../../glossary.md#term-synchronization) with no output.
- No-change conclusions:
  - [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) Javadoc remains accurate because this task consumes its unchanged plan-local
    identity without changing its range, equality, ownership, or non-resource semantics.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  - `ARCHITECTURE.md` and ADR 0010 already assign final geometry to Runtime, analysis/assignment/
    finalization to Prepare and concrete backends, and physical/run resources to later work; no
    architecture decision or dependency rule changed.
  - Lifecycle, module-boundary, and dependency explanations remain accurate at their architecture
    level; the focused boundary page carries the needed current-versus-planned implementation
    status.
  - The backend partition-preparer guide and Prepare master/task remain analysis-focused and
    correctly defer assignment/finalization; they need no executable example or contract change
    for a dependency-neutral Runtime result carrier.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
  - Runtime and Prepare Gradle files need no change because no module edge was added; root Java 26
    configuration remains sufficient.
  - Architecture tests need no update because dependency directions and prohibited semantic
    imports are unchanged. Backend-conformance and integration tests remain inapplicable because
    this task adds no concrete backend or end-to-end execution behavior.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  - Other APIs/modules need no change because there is no Engine facade, caller workflow,
    physical resource, executable, run, publication, or configuration surface to document.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [publication glossary](../../../../glossary.md#term-publication))</span>
  - No later Runtime or Prepare specification was created because frontier selection is a
    separate planning step after completion.

## Implementation notes

- Added only [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) and [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) with its exact two nested entry records in
  the existing `runtime.memory` package, plus the two focused test suites.
- Constructor validation implements the specified deterministic order and exact failures.
  Successful plans preserve supplied buffer/workspace order in immutable snapshots while retaining
  exact immutable entries and slots.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [buffer glossary](../../../../glossary.md#term-buffer); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Final Javadocs cover every public type, canonical constructor, component, accessor, return,
  caller-visible failure, unit, nullability, range, identity, equality, mutation, and resource
  boundary. Package documentation now describes the complete current memory surface.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [nullability glossary](../../../../glossary.md#term-nullability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Runtime API, focused architecture explanation, glossary, Runtime master plan, and roadmap now
  distinguish current final Runtime geometry from planned Prepare assignment/finalization and
  later physical/run state.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- No executable Java changed during the documentation pass.

## Completion summary

- Completed changes: implemented and documented the exact nominal workspace-slot identity and
  immutable ordered per-buffer/per-workspace final byte geometry.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Files changed or created: exactly the 11 authorized paths—five Runtime production/test paths,
  Runtime API, focused boundary explanation, glossary, this task, Runtime master plan, and
  roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: reused the successful focused 21-test and final 25-test Runtime evidence;
  Runtime Javadoc, generated pages, compiled Java 26 API example, six-file Markdown, exact
  source/API/import/build/toolchain/scope/status/specification, final-newline/fence/whitespace, and
  final <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [toolchain glossary](../../../../glossary.md#term-toolchain); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` gates passed.
- Documentation-agent review: completed in clean context
  `/root/implement_runtime_0002/runtime_0002_docs` without executable Java changes or duplicate
  Java-test execution.
- Documentation impact: the API and focused architecture page now expose the current geometry
  mental model and example while keeping assignment, source associations, allocation, binding,
  finalization, and execution explicitly later.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: finalized [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot), [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan), both nested entries, explicit
  constructors/accessors, and package documentation. Existing [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) Javadoc remains
  accurate without modification.
- Glossary impact: made [Runtime](../../../../glossary.md#term-runtime) implementation status current and added reusable
  `WorkspaceSlot` and `PreparedMemoryPlan` definitions while refining buffer-slot and memory-slot
  distinctions.
- Unresolved issues: None.
- Follow-up required: None for this task. Runtime 0003 remains Draft pending a separate frontier
  planning step; Prepare 0002 remains Draft until Runtime 0003–0004 stabilize.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare))</span>

Status: Complete
