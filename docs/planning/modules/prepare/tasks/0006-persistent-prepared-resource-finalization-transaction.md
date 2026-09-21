# Task 0006: Persistent Prepared-Resource Finalization Transaction

## Status

Complete

## Goal

Complete the Prepare stage of
[ADR 0013](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md).
Backend finalization must return one atomic immutable result containing its
`PreparedExecutable` and the persistent `PreparedResource` instances acquired for that result,
in physical acquisition order. Shared Prepare must own every successfully returned resource as
one transaction until it constructs the final
`PreparedExecution(memoryPlan, schedule, resources)`.

The new public value is:

```java
package io.github.pho001.synaptik.prepare;

public record BackendPartitionFinalizationResult(
        PreparedExecutable executable,
        List<PreparedResource> resources) {
    public BackendPartitionFinalizationResult(PreparedExecutable executable);
}
```

`BackendPartitionFinalizer.finalizePartition(...)` returns this value instead of returning a
bare executable. `PreparedPartition` remains exactly `PreparedPartition(partition, executable)`;
persistent resource ownership never comes from a partition, executable, or schedule occurrence.
The existing CPU finalizer uses the one-argument constructor and therefore continues to return
the same executable with an empty resource list.

## Mental model

Ownership moves only at successful method and constructor boundaries:

```text
backend finalizer while acquiring
  owns resources and rolls back before any failed return
      |
      | successful BackendPartitionFinalizationResult return
      v
shared Prepare transaction
  owns identity-unique resources in partition/acquisition order
  rolls them back after any later failure
      |
      | successful PreparedExecution(memoryPlan, schedule, resources)
      v
PreparedExecution
  is the sole persistent owner
```

This is a cold preparation transaction. It adds no run-time lookup, schedule-derived ownership,
physical allocation in shared Prepare, or backend interpretation.

## Scope

- Add the public immutable `BackendPartitionFinalizationResult` record in the Prepare root
  package.
- Give the record exactly two components in declaration order: one non-null
  `PreparedExecutable` and an immutable snapshot of a non-null acquisition-ordered
  `List<PreparedResource>` whose entries are non-null.
- Add a one-argument resource-free constructor delegating with `List.of()` so CPU and fake
  finalizers express an empty persistent-resource result without fabricating ownership.
- Change `BackendPartitionFinalizer.finalizePartition(...)` to return the atomic result.
- Require every concrete finalizer to own and roll back anything it acquires until the complete
  result returns successfully. A failed return transfers nothing to shared Prepare.
- Extend `BackendPartitionFinalizationHandoff` to consume successful result values in partition
  finalization order, extract only their executables for `PreparedPartition`, and concatenate
  their resources in result-list order.
- Track resource identity with `==`, never `equals`, and reject a repeated identity within one
  result or across results.
- On duplicate rejection, record only the first ownership occurrence so rollback closes that
  exact resource once rather than once per duplicate occurrence.
- Roll back all earlier successful finalizer results when a later finalizer, result check,
  executable-plan association, or `PreparedPartition` construction fails.
- Return the immutable ordered identity-unique resource snapshot in the package-private handoff
  result together with the existing memory plan, partitions, and buffer assignments.
- Extend `GraphPreparation` to own a successful handoff transaction through schedule assembly,
  schedule validation, and final aggregate construction.
- Construct the final Runtime owner with
  `new PreparedExecution(handoff.memoryPlan(), schedule, handoff.resources())`.
- On any failure after successful handoff and before successful aggregate construction, close
  resources once in reverse acquisition order.
- Preserve the first `RuntimeException` or `Error` as primary and add distinct cleanup failures as
  suppressed exceptions in reverse-cleanup encounter order; skip the exact primary throwable to
  avoid self-suppression.
- Keep CPU behavior resource-free by returning the existing executable with an empty resource
  list and by adapting direct CPU/test callers to extract `result.executable()`.
- Finalize every changed public and implementation Javadoc, API/boundary explanation, backend
  guide, glossary status, and planning status through the required separate clean documentation
  pass after executable Java stabilizes.

## Out of scope

- changing `PreparedPartition` components, equality, or ownership
- deriving ownership from `PreparedSchedule`, repeated executable occurrences, partition count,
  or any graph topology
- changing Runtime `PreparedExecution`, `PreparedResource`, lease, close, cleanup, runner, or
  per-run resource behavior implemented by Runtime 0016
- implementing Engine 0010 prepared-handle ownership or changing any Engine Java source
- implementing Metal 0002, compiling an MPS graph, adding a Metal resource, changing native ABI,
  or advertising a Metal capability
- changing CPU executable, generated artifact, worker-group, OpenBLAS, buffer, workspace,
  binding, execution, or cleanup behavior
- adding persistent resources to CPU finalization
- allocating, acquiring, interpreting, or physically closing a backend resource in shared Prepare
  other than invoking its nominal `PreparedResource.close()` during rollback
- changing analysis, route selection, resource declarations, slot assignment, producerless
  resource handling, schedule structure, publication, tuning, or runtime execution
- a resource registry, public resource accessor, reference-counted schedule ownership, backend
  switch, service locator, cleaner, or garbage-collection correctness mechanism
- checked cleanup exceptions, waiting close, asynchronous preparation, or retrying failed cleanup
- module dependency, Gradle, architecture-contract, ADR, architecture-test, integration-test, or
  native-source changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Prepare
  lifecycle, Runtime ownership, concrete-backend ownership, and dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Planning guide](../../../planning-guide.md) and [roadmap](../../../roadmap.md)
- [Prepare master plan](../master-plan.md) and completed Prepare tasks
  [0002](0002-backend-partition-finalization-handoff.md) and
  [0003](0003-prepare-orchestration-and-validation.md)
- [Runtime 0016](../../runtime/tasks/0016-persistent-prepared-resource-lifecycle.md)
- [Engine master plan](../../engine/master-plan.md), task 0010 row
- [Metal master plan](../../../backends/metal/master-plan.md), blocked task 0002 row

## Architecture constraints

- `PreparedExecution` is the unique final owner of immutable persistent prepared resources.
- Concrete backends own acquisition, physical resource types, release mechanics, lowering, route
  selection, and executable construction. Shared Prepare knows only the nominal Runtime cleanup
  contract.
- Backend analysis still declares every shared buffer and workspace requirement before slot
  assignment. Persistent resources are acquired only during finalization and do not revise route
  choice or shared declarations.
- A finalizer must complete its own local acquisition transaction before returning. Shared
  Prepare cannot recover a resource that was acquired but omitted because the finalizer threw.
- Shared Prepare owns only resources present in successful finalizer results. Executable sharing,
  partition associations, and repeated schedule occurrences have no ownership meaning.
- Successful `PreparedExecution` construction is the only shared-Prepare-to-Runtime ownership
  transfer point. Failed construction transfers nothing under the Runtime 0016 contract.
- Rollback is deterministic, reverse acquisition order, attempt-all, identity-aware, and does not
  use `equals`.
- Prepare remains independent of concrete backend implementations. The CPU source change is an
  implementation of the public Prepare SPI in the concrete backend, not concrete logic in
  Prepare.
- Runtime remains independent of concrete backends, and concrete backends remain independent of
  Engine.
- No dependency direction, architecture decision, or Runtime hot-path behavior changes. Stop if
  implementation requires one.

## Package impact and exact type semantics

Changed package:

- `io.github.pho001.synaptik.prepare` gains
  `BackendPartitionFinalizationResult`, updates the existing finalizer collaboration, and extends
  package-private transactional orchestration. No package is added or moved.

Consumed unchanged packages:

- `io.github.pho001.synaptik.runtime.execution` continues to own `PreparedExecutable` and the
  final `PreparedExecution` owner.
- `io.github.pho001.synaptik.runtime.resource` continues to own the nominal
  `PreparedResource` close role.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` continues to own CPU finalization and
  adopts only the changed Prepare SPI return shape.

Type placement:

- `io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult` — public backend-facing
  Prepare handoff value because both a concrete backend finalizer and shared Prepare must name it;
  it is not a Runtime aggregate or backend-private type.
- `io.github.pho001.synaptik.prepare.BackendPartitionFinalizationHandoff` — remains the
  package-private owner of complete-set finalizer invocation, cross-result identity validation,
  and rollback before a successful handoff result.
- `io.github.pho001.synaptik.prepare.GraphPreparation` — remains the public stateless facade and
  owns a successful handoff result until final aggregate construction.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer` — remains the
  concrete CPU implementation and returns no persistent resources.

`BackendPartitionFinalizationResult` is a public record with exactly these record components:

1. `PreparedExecutable executable`
2. `List<PreparedResource> resources`

The canonical constructor validates `executable`, then `resources`, then each indexed resource.
It retains the exact executable reference and snapshots the list with `List.copyOf`. The record
does not reject duplicate resource identities because shared Prepare must detect duplicates
across the complete ordered result set and close each accepted identity once. The result owns no
close method and exposes no physical payload. Ownership remains with the finalizer until the
method returns successfully, then moves to shared Prepare as a transaction.

The one-argument constructor is exactly:

```java
public BackendPartitionFinalizationResult(PreparedExecutable executable) {
    this(executable, List.of());
}
```

No factory, builder, resource lookup, backend identifier, partition, memory plan, schedule, or
generic payload is added.

## Transaction and failure semantics

### Finalizer-local transaction

A finalizer may acquire zero or more persistent resources while constructing one executable. It
must retain those resource identities in physical acquisition order. Before the complete result
returns successfully, the finalizer owns them and must reverse-close all acquired identities on
every `RuntimeException` or `Error`, including failure while constructing the result itself.

The original failure remains primary. Cleanup attempts continue in reverse order; each distinct
cleanup failure is suppressed in encounter order, and another occurrence of the exact primary
throwable is skipped. The shared layer cannot implement or infer this local rollback, so the
interface Javadoc and backend tests must make the obligation explicit.

### Complete-set handoff transaction

All existing shared validation, memory-plan construction, and typed finalization-input
construction still finish before the first finalizer call. For each partition in order:

1. invoke its finalizer exactly once;
2. require a non-null `BackendPartitionFinalizationResult`;
3. inspect resources in result-list order, using reference identity only;
4. append each first-seen identity to the transaction list;
5. if an identity was seen before, create the duplicate-identity failure without appending it a
   second time;
6. require the result executable to retain the exact shared memory plan;
7. construct the unchanged `PreparedPartition(partition, executable)`; and
8. continue to the next finalizer.

If any step fails, later finalizers are not invoked. Close the transaction list once in reverse
order and throw the original failure with cleanup failures suppressed. This ordering ensures a
duplicate identity within or across results is closed once, while every other resource from the
already successful results is attempted.

On success, the package-private handoff result snapshots the transaction list in acquisition
order. Its existing memory-plan, partition, and buffer-assignment references and ordering remain
unchanged.

### Complete graph-preparation transaction

After a successful handoff, `GraphPreparation` owns the returned resources while it:

1. constructs `PreparedScheduleContext`;
2. invokes the assembler once;
3. validates the complete schedule; and
4. constructs `PreparedExecution(memoryPlan, schedule, resources)`.

A `RuntimeException` or `Error` in steps 1–4 closes the identity-unique resources in reverse
order and preserves that failure as primary. A successful constructor call transfers ownership
to `PreparedExecution`; `GraphPreparation` must not close those resources afterward. It must not
fall back to the resource-free Runtime constructor.

### Stable failure messages

The new record uses:

- `NullPointerException("executable")`;
- `NullPointerException("resources")`; and
- `NullPointerException("resources[i]")`.

The handoff preserves `NullPointerException("entries[i].finalizer returned null")` for a null
result and the existing foreign-memory-plan failure for the result executable. A repeated exact
identity fails with:

```text
entries[i].finalizer resources[j] duplicates an earlier prepared resource identity
```

Do not specify backend-private exception messages for local acquisition failures.

## Javadoc and API effects

- `BackendPartitionFinalizationResult` Javadoc must define immutability, acquisition order,
  ownership before and after successful return, null handling, list snapshotting, permitted
  duplicates at the value boundary, and absence of a close/lookup role.
- `BackendPartitionFinalizer` Javadoc must replace the obsolete “no closeable physical
  resources” rule with the exact local transaction, successful-return transfer, ordering,
  rollback, and primary/suppressed-failure contract.
- `BackendPartitionFinalizationHandoff` Javadoc must describe returned resources, identity
  uniqueness, rollback coverage, and ownership transfer to its successful result.
- `GraphPreparation` and its package Javadoc must describe the successful-result transaction,
  schedule/aggregate rollback window, and transfer to `PreparedExecution` without claiming that
  shared Prepare performs physical allocation.
- `CpuPartitionFinalizer` Javadoc must state that CPU returns an empty persistent-resource list
  and preserves all current executable and borrowed-resource behavior.
- Runtime and Public API documentation must show the current finalization result and final
  three-argument `PreparedExecution` construction. The focused architecture boundary and backend
  guide must change Prepare 0006 wording from planned to current only when implementation passes.
- The glossary already defines prepared execution and prepared resource. Update those entries to
  describe the implemented Prepare transaction; do not add a redundant glossary entry solely for
  the Java result carrier.

## Affected files

Exact implementation and documentation allowlist (twenty-six paths):

Production and Javadoc:

1. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationResult.java` (new)
2. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizer.java`
3. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java`
4. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java`
5. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/package-info.java`
6. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`

Tests and compilation fixtures:

7. `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/FinalizationPublicShapeTest.java`
8. `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoffTest.java`
9. `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationTest.java`
10. `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationPublicShapeTest.java`
11. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
12. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutableTest.java`
13. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
14. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionDagResourceTest.java`
15. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasNativeCheckpoint.java`
16. `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`

Documentation and planning:

17. `docs/api/runtime-api.md`
18. `docs/api/public-api.md`
19. `docs/architecture/runtime-prepare-backend-boundary.md`
20. `docs/backend-guide/writing-a-backend.md`
21. `docs/glossary.md`
22. this task
23. `docs/planning/modules/prepare/master-plan.md`
24. `docs/planning/modules/engine/master-plan.md`
25. `docs/planning/backends/metal/master-plan.md`
26. `docs/planning/roadmap.md`

The six CPU and backend-conformance test/fixture paths after the focused CPU finalizer test are
mechanical direct-call adaptations required by the public return-type change. They do not widen
CPU semantics. Review without editing Runtime Java/tests, Engine Java/tests, Metal Java/native
source/tests, other concrete backends, Gradle files, architecture tests, integration tests,
Compile/Tensor/Training APIs, and all other paths.

## Maximum scope

At most the twenty-six paths above. This exceeds the usual task-size guideline because one
public backend-finalizer return-type change must atomically update the Prepare transaction, the
sole concrete finalizer, every direct compilation fixture, conformance compilation, and the
required documentation/status surfaces. Splitting the mechanical callers would leave the
repository uncompilable and splitting documentation would violate the same-change rule.

Do not add a cleanup helper source file. Keep shared rollback support package-private inside the
existing handoff/orchestration implementation. If a twenty-seventh path or a different package is
required, stop and revise the task rather than widening it implicitly.

## Acceptance criteria

- `BackendPartitionFinalizationResult` has exactly the two specified record components and the
  one specified convenience constructor, snapshots its list, rejects nulls in exact order, and
  exposes no close, lookup, backend, partition, or schedule surface.
- `BackendPartitionFinalizer.finalizePartition(...)` returns only the new atomic result and its
  Javadoc makes finalizer-local rollback before successful return mandatory.
- All existing validation and assignment completes before the first finalizer call, and
  finalizers remain ordered and invoked at most once.
- `PreparedPartition` remains byte-for-byte unchanged and contains only partition plus executable.
- A successful result transfers its resource list to shared Prepare even if its executable later
  fails exact-plan validation.
- Resource lists are concatenated in partition-finalization and within-result acquisition order.
- Identity uniqueness uses `==`; a duplicate within one result or across results is rejected,
  appears once in the rollback transaction, and is physically closed once.
- A later finalizer failure, null result, duplicate identity, foreign executable plan,
  partition-association failure, assembler failure, schedule validation failure, or aggregate
  construction failure rolls back every earlier accepted identity once in reverse acquisition
  order and never invokes a later collaborator.
- Rollback attempts every resource. The triggering `RuntimeException` or `Error` stays primary;
  distinct cleanup failures are suppressed in encounter order, and the same exact primary object
  is not self-suppressed.
- A successful `PreparedExecution` receives the exact ordered resources once and shared Prepare
  performs no later close. Closing that execution remains Runtime 0016 behavior.
- Tests demonstrate a well-behaved fake finalizer rolling back its own partially acquired
  resources before throwing; shared Prepare neither sees nor double-closes those resources.
- Resource ownership is derived only from successful finalizer results. Repeated executable and
  schedule occurrences do not change the resource list or close count.
- `CpuPartitionFinalizer` returns the exact existing executable plus an empty immutable resource
  list for every route, and all direct callers extract the executable without changing execution
  assertions.
- No Runtime, Engine, Metal, ABI, capability, dependency, build, architecture-test, integration,
  or performance behavior changes.
- Public/package shape, complete Javadocs, API/boundary/backend-guide prose, glossary status,
  task/master/roadmap status, exact scope, and current-versus-planned language are synchronized
  before completion.
- Prepare 0006 becomes `Complete` only after executable validation and the independent
  documentation pass. Engine 0010 remains Draft without a detailed specification, and Metal 0002
  remains Blocked without a detailed specification.

## Tests / validation

Focused Prepare tests must cover:

- exact record components, constructor/method descriptors, public/final shapes, generic resource
  type, immutable snapshotting, null validation, and unchanged `PreparedPartition` shape;
- ordered multi-finalizer resource collection and successful transfer to the final execution;
- finalizer-local rollback evidence through a fake finalizer that fails before returning;
- duplicate identity within one result and across results, using resources whose `equals` method
  would give a misleading answer;
- reverse attempt-all rollback and primary/suppressed/self-suppression behavior for both
  `RuntimeException` and `Error`;
- failure at every handoff stage after the first successful resource result;
- assembler and schedule-validation rollback, plus source inspection that aggregate construction
  remains inside the same rollback boundary; and
- success with repeated executable/schedule occurrences without duplicated ownership.

Focused CPU tests must prove that every route still returns its existing executable and an empty
resource list. Other listed CPU and conformance files receive only the direct-call extraction
needed for compilation and retain their current assertions.

Implementation validation after executable Java stabilizes:

```bash
./gradlew :modules:prepare:test \
  --tests io.github.pho001.synaptik.prepare.FinalizationPublicShapeTest \
  --tests io.github.pho001.synaptik.prepare.BackendPartitionFinalizationHandoffTest \
  --tests io.github.pho001.synaptik.prepare.GraphPreparationTest \
  --tests io.github.pho001.synaptik.prepare.GraphPreparationPublicShapeTest
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
./gradlew :testing:backend-conformance:test \
  --tests io.github.pho001.synaptik.testing.conformance.CpuOpenBlasRouteConformanceTest
./gradlew :modules:prepare:test :backends:cpu:test
./gradlew test
```

The repository-wide run is required once because the public Prepare SPI changes and direct
consumers span Prepare, CPU, and backend conformance. Do not repeat it in the documentation pass
unless executable Java changes after that evidence.

Documentation pass:

```bash
./gradlew :modules:prepare:javadoc :backends:cpu:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/writing-a-backend.md docs/glossary.md \
  docs/planning/modules/prepare/tasks/0006-persistent-prepared-resource-finalization-transaction.md \
  docs/planning/modules/prepare/master-plan.md \
  docs/planning/modules/engine/master-plan.md \
  docs/planning/backends/metal/master-plan.md docs/planning/roadmap.md
git diff --check
```

If the temporary validator is unavailable, use an equivalent tool outside the repository. Check
local links and heading anchors, unique effective anchors, balanced fences, final newlines,
trailing whitespace, exact twenty-six-path scope, task/master/roadmap status synchronization,
absence of an Engine 0010 or Metal 0002 task file, and unchanged excluded paths. Inspect generated
Javadoc for the result, finalizer, graph preparation, Prepare package, and CPU finalizer ownership
language.

## Dependencies

- Runtime 0016 — Complete; supplies `PreparedResource`, the unique final Runtime owner, exact
  constructor-transfer semantics, close/run leases, and reverse cleanup behavior.
- ADR 0013 — Accepted; defines the three-stage ownership transaction.
- Prepare 0001–0005 — Complete; supply staged analysis, slot assignment, finalizer invocation,
  graph orchestration, schedule validation, opaque tuning handoff, and producerless assignment.
- CPU 0010J and Engine 0009 — Complete; current direct consumers whose behavior must remain
  resource-free and otherwise unchanged.

No Metal executable route is an implementation dependency. Metal 0001 and its audit provide the
motivating native lifecycle only.

## Follow-up tasks

- Engine 0010: prepared-handle ownership and closure. It remains `Draft` and blocked on this task;
  create its detailed specification only after Prepare 0006 is Complete.
- Metal 0002: first MPSGraph prepared execution route. It remains `Blocked` without a detailed
  specification until Prepare 0006 and Engine 0010 are Complete.
- Custom Metal kernels, general asynchronous execution, bounded Metal buffer-pool implementation,
  lease-bound `MemorySegment` aliasing, and whole-plan autotuning remain later Metal tasks.

## Documentation-focused clean-context pass

After Java and the final implementation-owned test evidence stabilize, hand the exact diff,
baseline worktree status, commands/results, public-shape evidence, ownership/failure decisions,
and twenty-six-path allowlist to a distinct clean documentation-focused context. Apply the
General, API/Javadoc, Architecture, Backend Guide, and Planning profiles.

That pass must independently inspect the implementation and tests, finalize all affected
Javadocs and the five explanatory/API/glossary paths, synchronize the four planning paths, and
record reasoned no-change conclusions for Runtime Java, Engine Java, Metal/native code, CPU
behavior beyond the empty result adaptation, Compile/Tensor/Training APIs, Gradle/module edges,
architecture tests, integration tests, and performance. It reuses successful Java evidence unless
it changes executable behavior or identifies a concrete stale-evidence risk.

## Architecture impact

Expected impact: implementation of the Prepare stage already selected by ADR 0013. No new
architecture decision or dependency direction.

Stop if implementation requires changing `ARCHITECTURE.md`, ADR 0013, Runtime ownership,
`PreparedPartition`, a module edge, a concrete resource type in shared Prepare, schedule-derived
ownership, checked cleanup, waiting, Engine Java, Metal/native Java, or another public mechanism.

## Implementation prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik on Prepare task 0006. Do not use GSD, commit, or push.
Use a separate clean implementation context. Read AGENTS.md, ARCHITECTURE.md, the planning guide,
ADR 0013, the focused Runtime/Prepare/backend boundary, Runtime task 0016, Prepare master plan and
tasks 0002/0003/0005, this task, and every affected source/test named here. Inspect and preserve
the dirty worktree.

Implement and validate exactly this specification and its twenty-six-path allowlist. Stop and
report any architecture conflict, ownership ambiguity, different package need, or required
twenty-seventh path. Do not implement Engine 0010, Metal 0002, or another follow-up.

After executable Java and the recorded Prepare/CPU/conformance/repository evidence stabilize,
hand the exact diff and evidence to a distinct clean documentation-focused context. That context
must apply the specified profiles, finalize Javadocs/docs/glossary/planning, validate exact scope
and status, and avoid repeating stable Java tests unless executable behavior changes or a concrete
risk requires it. Mark Complete only after every acceptance gate finishes.
```

## Stop conditions

Stop and report instead of implementing or widening if:

- a successful finalizer result cannot carry every persistent resource needed by its executable;
- shared Prepare would need to inspect a concrete resource, acquire it, reference-count schedule
  occurrences, or infer ownership from executable topology;
- a finalizer cannot roll back its own resources before a failed return;
- failed `PreparedExecution` construction would take ownership or close resources contrary to
  Runtime 0016;
- CPU needs a non-empty persistent-resource list or changed execution lifecycle;
- Engine or Metal Java/native implementation is required for this transaction;
- a module dependency, architecture decision, Gradle edit, new package, cleanup helper source,
  or twenty-seventh path is required; or
- reverse attempt-all identity-unique rollback cannot preserve the exact primary failure.

## Local decisions

- `BackendPartitionFinalizationResult` remains a passive immutable value: it snapshots and
  validates list structure but leaves identity uniqueness to the complete-set handoff, which can
  enforce the rule across and within all results while recording only the first ownership
  occurrence.
- Shared rollback support remains package-private in
  `BackendPartitionFinalizationHandoff`; both the handoff and `GraphPreparation` use the same
  reverse attempt-all, primary/suppression, and self-suppression rules without adding a helper
  source file.
- `PreparedExecution` construction stays inside `GraphPreparation`'s rollback boundary so a
  constructor failure transfers no ownership and closes every accepted resource.
- CPU uses the one-argument result constructor. Generated artifacts, worker groups, and OpenBLAS
  invocations remain cached or borrowed state rather than new persistent-resource ownership.

## Known limitations

- This task does not make Engine prepared handles close inward Runtime executions; Engine 0010 is
  still required.
- This task does not add an actual persistent backend resource. Metal 0002 remains the first
  planned consumer after the complete shared lifecycle chain.
- General asynchronous execution and buffer-pool policy remain deferred.

## Validation evidence

- Implementation evidence was supplied by the separate implementation context and reused because
  documentation context `01a0c347-0ae0-7613-bdb5-5725eba52331` changed Javadocs and Markdown only.
  The focused four Prepare classes passed; focused `CpuPartitionFinalizerTest` passed; focused
  `CpuOpenBlasRouteConformanceTest` passed; and the corrected full
  `./gradlew :modules:prepare:test :backends:cpu:test` passed. The repository-wide
  `./gradlew test` passed after the executable implementation. After the correction that moved
  construction of the package-private successful handoff `Result` inside
  `BackendPartitionFinalizationHandoff`'s rollback boundary and added within-result duplicate plus
  `Error`-primary coverage, the focused four Prepare classes and full
  `:modules:prepare:test` passed again. Initial public-SPI adaptation compile failures were
  corrected historical diagnostics, not final failures.
- The documentation context independently reviewed all six production/Javadoc paths, the final
  four Prepare test classes, the focused CPU finalizer test and mechanical CPU/conformance
  callers, ADR 0013, Runtime 0016, the focused architecture boundary, both API pages, backend
  guide, glossary, Prepare/Engine/Metal master plans, roadmap, and this task against the General,
  API/Javadoc, Architecture, Backend Guide, and Planning profiles.
- `./gradlew :modules:prepare:javadoc :backends:cpu:javadoc` passed after final Javadoc edits.
  CPU Javadoc reported 94 existing missing-`@param` warnings in unrelated lowering/preparation
  plan constructors plus the two expected incubating-module warnings. Generated pages for
  `BackendPartitionFinalizationResult`, `BackendPartitionFinalizer`, `GraphPreparation`, the
  Prepare package, and `CpuPartitionFinalizer` were inspected and contain the required ownership,
  acquisition order, rollback, no-lookup/no-allocation, transfer, and empty-resource language.
- `python3 /tmp/validate_synaptik_markdown.py` over the exact ten documentation paths passed with
  `validated 10 Markdown files`, covering local links and heading anchors, unique effective
  anchors, balanced fences, final newlines, and trailing whitespace.
- Status and scope inspection confirmed Prepare 0006 and its master row are `Complete`; Engine
  0010 remains `Draft` with no task file and is the next frontier; Metal 0002 remains `Blocked`
  with no task file. All Prepare 0006 implementation and documentation changes are contained in
  the exact twenty-six-path allowlist. Pre-existing Runtime 0016 architecture, Java, test, and
  planning work plus the Metal strategy planning changes remain separate dirty work and were not
  edited by this documentation pass. Final `git diff --check` passed.
- Reasoned no-change conclusions: Runtime Java/tests already implement the final owner and lease
  contract and are unchanged; Engine Java/tests still lack outward handle ownership by design;
  Metal/native code still has no executable route or ABI extension; CPU behavior changes only in
  the required empty-result SPI adaptation and gains no persistent resource. Compile, Tensor, and
  Training APIs neither expose nor own this handoff. No Gradle edge, architecture contract, ADR,
  architecture test, integration test, capability advertisement, native ABI, or performance
  behavior changes, so none required an edit or repeated suite.

## Implementation notes

- Added the public immutable finalization result and changed the finalizer SPI to return its exact
  executable together with an acquisition-ordered resource snapshot.
- Extended the complete-set handoff to concatenate resources in partition/result order, enforce
  exact-identity uniqueness, keep one ownership occurrence, and roll back after every later
  handoff failure with deterministic failure composition.
- Extended graph preparation to own a successful handoff through schedule assembly, validation,
  and final aggregate construction, then transfer the exact ordered resources once to Runtime.
- Adapted CPU and its direct callers to the resource-free result without changing route,
  executable, allocation, binding, execution, or cleanup behavior.
- Finalized Javadocs, Runtime/Public API explanations, the focused architecture boundary, backend
  guidance, glossary, and planning status with implemented-versus-planned boundaries.

## Completion summary

- Completed changes: Implemented and documented the persistent prepared-resource finalization
  transaction, identity-unique rollback, final Runtime transfer, and resource-free CPU adaptation.
- Files changed or created: Exactly the twenty-six paths listed in the task allowlist.
- Tests and validation: Reused the stable focused Prepare, CPU finalizer, backend-conformance,
  combined Prepare/CPU, corrected Prepare, and repository-wide implementation evidence; passed
  final Prepare/CPU Javadoc, generated-page inspection, ten-file Markdown validation, status/file-
  absence/scope checks, and `git diff --check`.
- Documentation-agent review: Complete in clean context
  `01a0c347-0ae0-7613-bdb5-5725eba52331`; no executable Java behavior changed.
- Documentation impact: Updated both API pages, the focused architecture boundary, backend guide,
  glossary, task, three master plans, and roadmap from planned Prepare handoff to current behavior.
- Javadoc review: Finalized all six affected production contracts for ownership, snapshot/null
  semantics, local/shared rollback, identity uniqueness, failure suppression, successful
  transfer, CPU empty resources, and absence of lookup or shared physical allocation.
- Glossary impact: Updated backend finalization, graph preparation, prepared execution, and
  prepared resource entries; no redundant carrier-only term was added.
- Unresolved issues: None within Prepare 0006.
- Follow-up required: Engine 0010, then Metal 0002.

Status: Complete
