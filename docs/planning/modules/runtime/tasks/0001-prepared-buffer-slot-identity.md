# Task 0001: Prepared [Buffer](../../../../glossary.md#term-buffer)-Slot [Identity](../../../../glossary.md#term-identity-meanings)

> **Editorial historical reading key.**
>
> This is the historical identity-only foundation, when BufferSlot was implemented and later Runtime
> contracts were intentionally still planned. The immediate caller constructs a non-negative numeric
> token; later Prepare and backend preparation use that token within an owning memory-plan context.
> Runtime defines its nominal meaning and validation, but the record stores no plan reference, graph
> ValueId, address or resource. Equal numeric components therefore do not prove shared storage or
> cross-plan membership. No execution or cleanup responsibility crosses this constructor boundary.
>
> Read [current slot identities](../../../../reference/runtime/memory.md#current-slot-identities) for
> the continuing token contract and [current executable
> binding](../../../../reference/runtime/executables.md#current-prepared-executable-and-bound-invocation)
> for its later consumer boundary: executable selections use dense list indices, not slot numbers.
> Allocation and release belong to later concrete representations and lifecycle owners, not to
> BufferSlot. The four-test evidence and later-task Draft statements remain historical.

## Status

Complete

## Goal

Replace the [runtime](../../../../glossary.md#term-runtime) placeholder with one [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-neutral [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) [identity](../../../../glossary.md#term-identity-meanings) that
later prepared memory, prepared-unit input/output [binding](../../../../glossary.md#term-binding), runtime slot-table access, shared
[prepare](../../../../glossary.md#prepare), and [concrete backend](../../../../glossary.md#concrete-backend) preparation can use without importing [compile](../../../../glossary.md#compile)-time [graph](../../../../glossary.md#graph) objects or
physical storage implementations.

The exact public [API](../../../../glossary.md#term-application-programming-interface-api) is:

```java
package io.github.pho001.synaptik.runtime.memory;

public record BufferSlot(long value) {}
```

The [canonical](../../../../glossary.md#term-canonical-form) constructor rejects negative values. Zero through `Long.MAX_VALUE` are valid, and
no sentinel is reserved.

## Rationale

The architecture requires prepared units to connect prepared executables to input and output
slots, while the [runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path) must never consume [`Operation`](../../../../glossary.md#operation) or `CompiledNode`. A stable
`PreparedExecutable.execute(...)` [signature](../../../../glossary.md#term-signature) cannot yet be selected because the typed per-[run](../../../../glossary.md#term-run)
storage-access, resource, and [`RunState`](../../../../glossary.md#run-state--runstate) contracts do not exist. Defining that invocation seam
now would either invent those contracts or produce an unusable zero-[argument](../../../../glossary.md#term-argument) executable.

[`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) is the smallest stable prerequisite. It gives later prepared and per-run contracts a
runtime-owned [identity](../../../../glossary.md#term-identity-meanings) that is deliberately distinct from [graph](../../../../glossary.md#graph) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), while leaving physical
allocation, storage access, workspace, execution, schedules, and [mutable](../../../../glossary.md#term-mutable) state to their owning
tasks.

## [Scope](../../../../glossary.md#term-scope)

- Delete the placeholder `RuntimeModule`.
- Add public final record `io.github.pho001.synaptik.runtime.memory.BufferSlot` with exactly one
  `long value` component.
- Treat the numeric value as an opaque identity within one owning future prepared-memory-plan
  context. The same numeric value may be reused by another plan.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Accept every non-negative `long`, including zero and `Long.MAX_VALUE`.
- Reject a negative value with `IllegalArgumentException` and exact message
  `value must be non-negative`.
- Use ordinary record equality, hashing, and diagnostic `toString()` semantics.
- Add `runtime.memory` package documentation that explains the current identity-only surface and
  separates it from future physical memory and runtime access contracts.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Add focused tests for public shape, boundary values, failure behavior, equality, hashing, and
  ordinary record diagnostic text.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Update the runtime API reference, runtime/prepare/backend boundary explanation, and glossary to
  distinguish the implemented identity from later planned slot access and storage.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- Finalize affected Javadocs and documentation in a separate clean documentation-focused context.
- Synchronize this task, the Runtime master plan, and the roadmap after implementation and
  validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot)
- [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) or any physical-memory plan
- physical buffers, allocation, deallocation, pooling, aliasing, reuse, byte sizes, alignment, or
  lifetimes<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [aliasing glossary](../../../../glossary.md#term-aliasing); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- storage handles, [backend](../../../../glossary.md#backend) storage, [host storage](../../../../glossary.md#host-storage), [device](../../../../glossary.md#term-device) storage, or an untyped object carrier
- a slot allocator, factory, [registry](../../../../glossary.md#term-registry), table, map, collection, range, or count contract
- any conversion or promised numeric relationship between [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) and [graph](../../../../glossary.md#graph) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot),
  [`NodeId`](../../../../glossary.md#nodeid), [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), [trace](../../../../glossary.md#trace) identity, [backend identity](../../../../glossary.md#backend-identity--backendid), device identity, or an address
- `RuntimeSlotTable`, value access, input binding, resources, residency, transfers, or
  materialization<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency); [materialization glossary](../../../../glossary.md#materialization))</span>
- [`RunState`](../../../../glossary.md#run-state--runstate), [`RunResult`](../../../../glossary.md#run-result--runresult), [run](../../../../glossary.md#term-run) or [publication](../../../../glossary.md#term-publication) configuration, or publication behavior
- [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), `PreparedUnit`, [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), or
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution)
- backend discovery, lowering, kernel selection, fallback, concrete backend behavior, or prepare
  orchestration<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [concrete backend glossary](../../../../glossary.md#concrete-backend); [prepare glossary](../../../../glossary.md#prepare))</span>
- trace payloads, event emission, [runtime profiling](../../../../glossary.md#runtime-profiling), [model](../../../../glossary.md#term-model) tuning, or [cache](../../../../glossary.md#term-cache) access
- [Engine](../../../../glossary.md#term-engine) or public [lifecycle](../../../../glossary.md#lifecycle) facades
- module dependency, Gradle, Java toolchain, architecture-contract, ADR, architecture-test,
  backend-conformance, or integration-test changes<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- later [Runtime](../../../../glossary.md#term-runtime) task specifications

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle)
  - [Run](../../../../glossary.md#term-run) lifecycle
  - [Dependency](../../../../glossary.md#term-dependency) rules
  - Testing requirements
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend))</span>
- [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Architecture constraints

- [Runtime](../../../../glossary.md#term-runtime) owns the slot [identity](../../../../glossary.md#term-identity-meanings) and later dynamic runtime state.
- A slot identity is prepared runtime vocabulary, not [immutable](../../../../glossary.md#term-immutable) [compile](../../../../glossary.md#compile)-time [graph](../../../../glossary.md#graph) state.
- The [hot path](../../../../glossary.md#term-hot-path) must not use [`Operation`](../../../../glossary.md#operation) or `CompiledNode`; this task adds no [dependency](../../../../glossary.md#term-dependency) on either.
- [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) must not be, contain, expose, derive from, or compare itself with [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).
- The identity contains no physical buffer, storage, resource, device, backend service, residency,
  executable, or mutable state.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [device glossary](../../../../glossary.md#term-device); [backend glossary](../../../../glossary.md#backend); [residency glossary](../../../../glossary.md#residency); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Runtime must not depend on [Engine](../../../../glossary.md#term-engine) or a [concrete backend](../../../../glossary.md#concrete-backend).
- The task must not add backend discovery, lowering, kernel selection, model tuning, trace
  emission, or execution behavior.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [model glossary](../../../../glossary.md#term-model); [trace glossary](../../../../glossary.md#trace))</span>
- `ARCHITECTURE.md` remains unchanged. If implementation needs an architecture or dependency
  change, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.runtime` — remove the placeholder type only; do not add a root
  facade.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>

Packages added or changed:

- `io.github.pho001.synaptik.runtime.memory` — public prepared-memory identity vocabulary; this
  task adds only <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) and package documentation.

Type placement:

- `io.github.pho001.synaptik.runtime.memory.BufferSlot` — runtime owns prepared buffer-slot
  identity, and the <span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [buffer glossary](../../../../glossary.md#term-buffer))</span>`memory` package keeps it with the later prepared-memory and slot-access
  contracts that consume it.

Tests mirror the production package:

- `io.github.pho001.synaptik.runtime.memory.BufferSlotTest`.

No other Java package may be added or changed.

## Affected files

Expected:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/RuntimeModule.java` — delete.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/BufferSlot.java` — add.
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/package-info.java` — add.
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/memory/BufferSlotTest.java` —
  add.
- `docs/api/runtime-api.md` — make [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) current and keep all later execution contracts
  conceptual.
- `docs/architecture/runtime-prepare-backend-boundary.md` — correct implementation status and
  explain the current identity-only foundation without changing the boundary.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `docs/glossary.md` — add the implemented slot identity and distinguish it from graph values,
  storage, and residency.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [residency glossary](../../../../glossary.md#residency))</span>
- `docs/planning/modules/runtime/tasks/0001-prepared-buffer-slot-identity.md` — record decisions,
  evidence, notes, summary, and final status.
- `docs/planning/modules/runtime/master-plan.md` — synchronize task and frontier status.
- `docs/planning/roadmap.md` — synchronize the global frontier.

Review only unless a concrete contradiction is found:

- `AGENTS.md`
- `ARCHITECTURE.md`
- `docs/architecture/lifecycle.md`
- `docs/architecture/module-boundaries.md`
- `docs/architecture/dependency-rules.md`
- `docs/developer-guide/documentation-rules.md`
- `docs/developer-guide/documentation/general-style.md`
- `docs/developer-guide/documentation/api-and-javadoc-style.md`
- `docs/developer-guide/documentation/architecture-style.md`
- `docs/developer-guide/documentation/planning-style.md`
- `docs/planning/planning-guide.md`
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `modules/runtime/build.gradle.kts`
- `build.gradle.kts`
- `settings.gradle.kts`
- `testing/architecture-tests/`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create, modify, or delete at most 10 paths:

- 4 [runtime](../../../../glossary.md#term-runtime) source/test paths;
- 3 explanatory documentation paths; and
- 3 runtime/global [planning](../../../../glossary.md#term-planning) paths.

No production or test file outside `modules/runtime` may change. No [Gradle](../../../../glossary.md#term-gradle), architecture-contract,
[ADR](../../../../glossary.md#term-architecture-decision-record-adr), or architecture-test path may change.

If another path or type is needed, stop and propose a follow-up task. Do not consume scope by
creating a later task specification.

## [Validation](../../../../glossary.md#term-validation), failure, immutability, and [lifecycle](../../../../glossary.md#lifecycle) rules

- Constructor validation occurs before record construction completes.
- Every negative `long`, including `Long.MIN_VALUE`, fails with the exact
  `IllegalArgumentException` message `value must be non-negative`.
- Zero and every positive `long` are accepted without normalization, remapping, allocation, or
  reserved values.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- The record is deeply [immutable](../../../../glossary.md#term-immutable) because its only component is primitive.
- Equality and hashing are ordinary [nominal](../../../../glossary.md#term-nominal) record semantics over the exact stored value.
- Diagnostic `toString()` is not [serialization](../../../../glossary.md#term-serialization) or a stable external format.
- The numeric value has meaning only in an owning future prepared-memory-plan context; this task
  supplies no owner reference, allocator, global uniqueness, process uniqueness, or cross-plan
  comparison promise.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- A [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) may survive across runs as part of reusable prepared state, but it contains no
  per-run binding, current storage, or residency.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency))</span>
- Creation of a `BufferSlot` does not allocate, acquire, retain, close, or [release](../../../../glossary.md#term-release) any resource.

## Acceptance criteria

- `RuntimeModule` is absent.
- The only new production type is the exact one-component public record
  `io.github.pho001.synaptik.runtime.memory.BufferSlot(long value)`.
- No extra public constructor, method, field, nested type, interface, factory, allocator, or
  conversion exists beyond record-generated members and the validated canonical constructor.
- The [canonical](../../../../glossary.md#term-canonical-form) constructor accepts zero and `Long.MAX_VALUE`, retains the exact primitive value,
  and rejects negative values with the exact specified exception type and message.
- Tests lock the record surface, boundary values, failure behavior, equality, hash code, and
  diagnostic record text.<span data-glossary-navigation="true"> ([hash code glossary](../../../../glossary.md#term-hash-code))</span>
- Production and test code import no model, compiler, planning, prepare, engine, concrete backend,
  storage, or trace type. Production code imports no reflection type; the focused test may use JDK
  reflection only to lock the exact public record surface.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [trace glossary](../../../../glossary.md#trace); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
- [Runtime](../../../../glossary.md#term-runtime) [Gradle](../../../../glossary.md#term-gradle) dependencies and the Java 26 root [toolchain](../../../../glossary.md#term-toolchain)/[release](../../../../glossary.md#term-release) configuration are unchanged.
- No [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), physical-buffer, allocation, storage, device,
  residency, workspace, schedule, execution, run-state, publication, discovery, lowering,
  kernel-selection, tuning, or trace-emission behavior is present.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
- Runtime [API](../../../../glossary.md#term-application-programming-interface-api) documentation and the glossary clearly mark [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) current while keeping
  every later runtime/prepare/backend type conceptual.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- The focused architecture explanation changes implementation-status wording only; it does not
  change module ownership, dependency direction, or lifecycle rules.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A separate documentation-focused agent has finalized Javadoc, API/architecture explanations,
  glossary impact, links, terminology, and planning evidence in the same overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Exactly the authorized paths change, later Runtime rows remain Draft, no later Runtime task
  specification exists, final newlines and whitespace are valid, and `git diff --check` passes.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:runtime:test --tests io.github.pho001.synaptik.runtime.memory.BufferSlotTest
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

If `/tmp/validate_synaptik_markdown.py` is absent, create an equivalent temporary validator outside
the repository. It must validate changed Markdown link [targets](../../../../glossary.md#term-target) and heading anchors, unique
effective heading anchors, balanced backtick and tilde fences, final newlines, and trailing
whitespace.

Required source/[scope](../../../../glossary.md#term-scope)/status checks:

- verify the exact `BufferSlot(long value)` record surface from source and focused [API](../../../../glossary.md#term-application-programming-interface-api) tests;
- verify the production/test import boundary and absence of every out-of-scope [runtime](../../../../glossary.md#term-runtime) type;
- verify `modules/runtime/build.gradle.kts` is unchanged and contains no concrete-backend or Engine
  dependency;<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- verify Java 26 only from root `build.gradle.kts`: [toolchain](../../../../glossary.md#term-toolchain) language version 26 and
  `options.release` 26; runtime adds no override;
- verify the exact 10-path ceiling and no Java path outside `modules/runtime`;
- verify task/master/roadmap Ready/In-progress synchronization before implementation and Complete
  synchronization only after every implementation and documentation gate passes;<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- verify Runtime 0002–0008 remain Draft and no later Runtime task specification exists; and
- verify final newlines, trailing whitespace, and `git diff --check`.

Repository-wide tests and `testing:architecture-tests` are deferred to the selected Runtime
capability [checkpoint](../../../../glossary.md#term-checkpoint) or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes no [dependency](../../../../glossary.md#term-dependency), architecture
boundary, shared build configuration, multiple-module behavior, [backend](../../../../glossary.md#backend) behavior, or end-to-end
execution. Backend-[conformance](../../../../glossary.md#term-conformance) and integration suites are not applicable.

The documentation-focused context reuses the successful runtime test evidence unless it changes
executable Java behavior or records a concrete reason to repeat it.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0001–0006 and the Compiler project area — Complete.
- [Planning](../../../../glossary.md#term-planning) 0001–0006 and its `CLOSED` contract audit — Complete.
- [Backend](../../../../glossary.md#backend) Contract 0001–0004 and its selected milestone — Complete.
- [Trace](../../../../glossary.md#trace) 0001–0002 common envelope and [identity](../../../../glossary.md#term-identity-meanings) foundation — Complete.
- [Runtime](../../../../glossary.md#term-runtime) [module](../../../../glossary.md#term-module) build structure — present with only the removable placeholder.
- Java 26 [toolchain](../../../../glossary.md#term-toolchain) and [release](../../../../glossary.md#term-release) — configured only by root `build.gradle.kts`; runtime has no
  override.

Config 0004–0008, Trace 0003–0008, [Prepare](../../../../glossary.md#prepare), concrete backends, and [Engine](../../../../glossary.md#term-engine) are not dependencies of
this identity-only task. The task exposes no contract owned by those later areas.

## Follow-up tasks

- Runtime 0002 may define the next prepared-memory/workspace capability only after a separate
  frontier reassessment proves its exact consumer requirements.
- [Runtime](../../../../glossary.md#term-runtime) 0003 and later rows remain Draft and have no detailed task specifications.
- Prepare, concrete backend, config, trace-payload, and Engine work remain in their owning master
  plans.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend); [trace glossary](../../../../glossary.md#trace); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Engine glossary](../../../../glossary.md#term-engine))</span>

Do not create a follow-up specification in this task.

## Javadocs and documentation impact

- [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) Javadoc must explain purpose, plan-local identity domain, valid range, immutability,
  equality, lack of physical/resource semantics, and the distinction from <span data-glossary-navigation="true"> ([identity domain glossary](../../../../glossary.md#term-identity-domain))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).
- Its [canonical](../../../../glossary.md#term-canonical-form) constructor [Javadoc](../../../../glossary.md#term-javadoc) must document `value` and the negative-value failure.
- Its explicit `value()` Javadoc must document the exact non-negative result and non-address,
  non-graph meaning.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Package Javadoc must describe only the current identity foundation and label later
  prepared-memory/slot-access contracts planned.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Runtime API, the focused runtime/prepare/backend explanation, and glossary must use the same
  current-versus-planned terminology.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- Lifecycle, module-boundary, dependency, compile/public APIs, user guides, backend guides, and
  other glossary entries are review-only because this task adds no lifecycle behavior,
  dependency, executable, storage, or public facade.<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile); [facade glossary](../../../../glossary.md#term-facade))</span>
- Architecture tests need no update because no dependency rule or tested semantic boundary
  changes.

The documentation-focused pass must record reasoned no-change conclusions for every review-only
area rather than using `N/A` alone.

## Architecture impact

Expected impact: None.

The architecture already names [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) as [runtime](../../../../glossary.md#term-runtime)-owned [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) vocabulary and
requires runtime to avoid [`Operation`](../../../../glossary.md#operation), `CompiledNode`, concrete [backends](../../../../glossary.md#backend), and [Engine](../../../../glossary.md#term-engine). This task
implements only that smallest [identity](../../../../glossary.md#term-identity-meanings). If implementation requires physical-memory semantics, a
[graph](../../../../glossary.md#graph)-value relationship, another [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or any architecture change, stop and report
the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the focused runtime/prepare/module-
boundary/dependency/lifecycle architecture docs, documentation rules and General/API-Javadoc/
Architecture/Planning profiles, the Runtime master plan, and
docs/planning/modules/runtime/tasks/0001-prepared-buffer-slot-identity.md. Read every affected and
review-only path named by the task.

Implement Runtime 0001 exactly within its 10-path ceiling. Replace the placeholder with only the
exact immutable runtime.memory.BufferSlot identity, its package documentation, focused tests, and
the required current-versus-planned documentation. Do not add workspace, physical memory/storage,
graph-value conversion, slot access, RunState, prepared executable/unit/schedule/execution,
runner, publication, transfer, residency, backend, prepare orchestration, tracing emission,
Engine, dependency, Gradle, architecture, or later-task-specification work. Stop on any API,
package, scope, dependency, or architecture conflict.

Run the focused test, one final runtime module test, and required source/scope/status checks. Then
hand the actual diff and successful Java evidence to a separate documentation-focused clean
context. That pass must follow documentation-rules.md; finalize Javadocs, Runtime API, focused
architecture status, glossary, task evidence/summary, master plan, roadmap, links, terminology,
scope, and no-change conclusions; and not repeat Java tests unless executable behavior changes or
a concrete risk is recorded.

Return both context IDs, exact paths, commands/results/counts, API/validation/immutability evidence,
documentation and glossary conclusions, unresolved issues, required follow-up, and the repository
completion status format. Mark Complete only after every acceptance criterion and documentation
gate passes.
```

## Local decisions

- The record keeps one explicit [canonical](../../../../glossary.md#term-canonical-form) constructor and one explicit `value()` accessor so both
  public members have complete contract Javadoc. The accessor returns the field directly and adds
  no API beyond the record-generated surface.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The record does not retain an owning plan reference. Plan-local interpretation remains a caller
  obligation, while ordinary record equality and hashing continue to compare the exact primitive
  component.
- Focused test-only JDK reflection is used to prove the exact record modifiers, component,
  constructor, generated-member set, interfaces, and nested-type absence. The task's earlier broad
  “no reflection type” wording conflicted with that required API-shape test; the acceptance
  criterion now forbids reflection in production while permitting this bounded test
  introspection. All cross-module and storage import prohibitions remain unchanged.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [shape glossary](../../../../glossary.md#shape); [module glossary](../../../../glossary.md#term-module))</span>
- Documentation uses one current-versus-planned [model](../../../../glossary.md#term-model) consistently: only [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) is current;
  plans, workspace identities, access, binding, storage, residency, schedules, execution, and
  lifecycle facades remain planned.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Known limitations

- [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) alone cannot prove plan membership or distinguish equal numeric values interpreted
  by different plans.
- No allocator, slot count or range, prepared-memory plan, workspace identity, storage binding,
  resource lifetime, execution contract, or serialization format exists.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding); [resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Repository-wide and architecture-test validation is deferred to the Runtime capability
  checkpoint or continuous integration because this task changes no dependency, architecture,
  shared build, backend, or end-to-end behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Runtime glossary](../../../../glossary.md#term-runtime); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/implement_runtime_0001` ran
  `./gradlew :modules:runtime:test --tests io.github.pho001.synaptik.runtime.memory.BufferSlotTest`
  after implementation. It passed with `BUILD SUCCESSFUL`; JUnit XML reported one suite and four
  tests with zero skips, failures, and errors.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>
- The same implementation context ran the single final affected-[module](../../../../glossary.md#term-module) command
  `./gradlew :modules:runtime:test` after executable Java stabilized. It passed with
  `BUILD SUCCESSFUL`; JUnit XML reported one suite and four tests with zero skips, failures, and
  errors. Documentation context `019faf5b-7f3a-72c0-93b4-18543ae49532` changed no executable Java,
  so it reused both results and did not repeat either Java suite.
- The implementation context reported `git diff --check` passed before documentation handoff.
  The documentation context reran it after final documentation and it passed with no output.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- Documentation context `019faf5b-7f3a-72c0-93b4-18543ae49532` applied General,
  API/Javadoc, Architecture, and Planning profiles plus the example format. It reviewed<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>
  `AGENTS.md`, `ARCHITECTURE.md`, the current architecture index, documentation rules and selected
  profiles, the Planning Guide and roadmap, this task and Runtime master plan, final Runtime source
  and tests, Runtime API, the focused runtime/prepare/backend explanation, and the complete
  glossary.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- The same context reviewed lifecycle, module-boundary, dependency, Compile/public API,
  prepare/run user-guide, backend-integration, Prepare/Compiler/Config/Backend Contract/Trace
  planning, Runtime/root Gradle, settings, and architecture-test surfaces. No change was needed:
  the task adds no lifecycle action, dependency edge, compile/Tensor/public facade, runnable
  workflow, backend contract, Gradle/toolchain rule, or architecture-testable dependency change.
  Backend conformance and integration tests are likewise unchanged because there is no backend or
  end-to-end behavior.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency); [Compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [Compiler glossary](../../../../glossary.md#term-compiler); [Trace glossary](../../../../glossary.md#trace); [dependency edge glossary](../../../../glossary.md#term-dependency-edge); [Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade); [toolchain glossary](../../../../glossary.md#term-toolchain); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- `./gradlew :modules:runtime:javadoc` passed with `BUILD SUCCESSFUL`; five tasks were actionable,
  two executed, and three were up-to-date. Generated `BufferSlot.html` and
  `package-summary.html` contain the type, constructor, accessor, valid-range, plan-local,
  no-resource, and current-versus-planned contracts.
- `javac --release 26 -cp modules/runtime/build/classes/java/main -d
  /tmp/runtime-0001-api-example /tmp/BufferSlotApiExample.java` followed by
  `java -cp modules/runtime/build/classes/java/main:/tmp/runtime-0001-api-example
  BufferSlotApiExample` compiled and ran the Runtime API example plus its documented negative
  failure boundary with no output or error.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md
  docs/architecture/runtime-prepare-backend-boundary.md docs/glossary.md
  docs/planning/modules/runtime/tasks/0001-prepared-buffer-slot-identity.md
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md
  docs/planning/modules/compiler/master-plan.md` passed after final edits: seven Markdown files
  validated for local targets and anchors, unique effective heading anchors, balanced backtick and
  tilde fences, final newlines, and trailing whitespace. The seventh file is the untouched
  pre-existing Compiler master-plan diff.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- The final source/[API](../../../../glossary.md#term-application-programming-interface-api)/import/[scope](../../../../glossary.md#term-scope) audit confirmed the exact
  `public record BufferSlot(long value)` declaration, one validated canonical constructor,
  explicit record accessor, exact failure message, and only the three intended Runtime
  production/test files. Production has no imports; test imports are limited to JUnit and JDK
  record-surface introspection. No forbidden cross-module, storage, runtime-lifecycle, or
  out-of-scope type is present.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
- The build/toolchain audit confirmed the unchanged Runtime dependencies are only Config, Backend
  Contract, and Trace; Runtime has no Engine or concrete-backend edge. Root<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
  `build.gradle.kts` alone sets `JavaLanguageVersion.of(26)` and `options.release.set(26)`;
  Runtime declares no override.
- The final scope audit reported 11 changed paths in the shared worktree: the exact ten
  task-owned paths and the preserved pre-existing
  `docs/planning/modules/compiler/master-plan.md` change. The task changed no Java path outside
  `modules/runtime`, no [Gradle](../../../../glossary.md#term-gradle) or architecture-test path, and no [architecture contract](../../../../glossary.md#architecture-contract) or [ADR](../../../../glossary.md#term-architecture-decision-record-adr).
- The final status audit found only this Runtime task specification. Runtime 0001 is Complete in
  this file and the master plan; Runtime 0002–0008 remain Draft without specifications; the
  roadmap keeps Runtime In progress and requires a separate reassessment before selecting the next
  frontier.

## Implementation notes

- Deleted the placeholder `RuntimeModule`.
- Added only the public one-component [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) record in `runtime.memory`, its package
  documentation, and the focused four-test contract suite.
- Constructor validation retains every non-negative value exactly and rejects negative values
  before record construction completes with the required exception and message.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Final documentation explains plan-local identity, ordinary record semantics, valid range,
  immutability, and no-resource behavior without promising an owner reference, mapping, storage,
  address, allocation, access, residency, or execution.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [owner glossary](../../../../glossary.md#term-owner); [residency glossary](../../../../glossary.md#residency))</span>
- Runtime API, focused architecture status, glossary, Runtime master plan, and roadmap now
  distinguish the current identity from planned runtime/prepare/backend contracts.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- No executable Java changed during the documentation pass.

## Completion summary

- Completed changes: replaced the placeholder with the exact [immutable](../../../../glossary.md#term-immutable) prepared-plan-local
  [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) identity, focused tests, complete Javadocs/package documentation, and synchronized
  explanatory and planning documentation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: the exact ten authorized paths—four Runtime source/test paths, Runtime
  API, focused architecture explanation, glossary, this task, Runtime master plan, and roadmap.
  The separate pre-existing Compiler master-plan diff was preserved untouched and excluded from
  the task count.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Tests and validation: reused the successful focused and final four-test Runtime evidence;
  Runtime Javadoc, generated-page inspection, seven-file Markdown validation, source/API/import,
  build/toolchain, exact-scope, later-task/status, final-newline/trailing-whitespace, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [toolchain glossary](../../../../glossary.md#term-toolchain); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` gates passed.
- Documentation-agent review: clean documentation context
  `019faf5b-7f3a-72c0-93b4-18543ae49532` independently reviewed the implementation, tests,
  contracts, affected documentation, glossary, planning status, and review-only surfaces.
- Documentation impact: [Runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api) and focused architecture documentation now identify
  `BufferSlot` as current while keeping storage, access, preparation, and execution conceptual.
- Javadoc review: finalized the record, canonical constructor, accessor, and package contracts;
  executable Java behavior was unchanged.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Glossary impact: added the reusable `BufferSlot` term and corrected the memory-slot distinction
  from [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) and physical storage. Other entries remain accurate for this [identity](../../../../glossary.md#term-identity-meanings)-only task.
- Unresolved issues: None.
- Follow-up required: None. Runtime 0002 remains Draft pending a separate frontier reassessment.

Status: Complete
