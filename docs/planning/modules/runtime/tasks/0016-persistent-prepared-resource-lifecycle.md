# Task 0016: Persistent Prepared-Resource [Lifecycle](../../../../glossary.md#lifecycle)

## Status

Complete

## Goal

Implement [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013's [Runtime](../../../../glossary.md#term-runtime)-owned [lifecycle](../../../../glossary.md#lifecycle) for [immutable](../../../../glossary.md#term-immutable) reusable [backend](../../../../glossary.md#backend)-native prepared
resources. [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) becomes the single deterministic [owner](../../../../glossary.md#term-owner), while [concrete backend](../../../../glossary.md#concrete-backend)
implementations perform physical close through one narrow [nominal](../../../../glossary.md#term-nominal) Runtime contract. [Synchronous](../../../../glossary.md#term-synchronous)
runs [lease](../../../../glossary.md#term-lease) the execution so close rejects new runs immediately and defers physical cleanup to the
last admitted [run](../../../../glossary.md#term-run) without waiting.

The task changes no concrete backend and does not yet transport resources through [Prepare](../../../../glossary.md#prepare).
Current [CPU](../../../../glossary.md#term-cpu) construction remains resource-free and behaviorally unchanged until an execution is
explicitly closed.

## [Scope](../../../../glossary.md#term-scope)

- Add public `PreparedResource extends AutoCloseable` in `runtime.resource`, with an unchecked
  `void close()` contract and no physical access methods.
- Convert [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) from a record to a public final class implementing
  `AutoCloseable`.
- Preserve `PreparedExecution(PreparedMemoryPlan, PreparedSchedule)` as the resource-free source
  compatibility constructor.
- Add a constructor accepting an ordered `List<? extends PreparedResource>` and snapshot exact
  identities without exposing that list or a resource lookup API.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [lookup glossary](../../../../glossary.md#term-lookup); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Preserve exact `memoryPlan()` and `schedule()` accessors and their reference-[identity](../../../../glossary.md#term-identity-meanings) [invariant](../../../../glossary.md#term-invariant).
- Add a narrow public nested run-lease type and acquisition method used by the runner; the lease
  exposes only idempotent <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>`close()` and no resource, plan, schedule, or [backend](../../../../glossary.md#backend) value.
- [Lease](../../../../glossary.md#term-lease) every `PreparedExecutionRunner.run(...)` call before caller-input inspection, run-state
  creation, binding, or physical work, and release after the complete synchronous call.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [release glossary](../../../../glossary.md#term-release); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- Add focused Runtime tests and update the exhaustive Runtime architecture-test source [manifest](../../../../glossary.md#term-manifest).
- Finalize affected Javadocs, Runtime/public API documentation, focused architecture explanation,
  backend guidance, glossary, and planning evidence in a separate documentation context.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Prepare finalizer return shapes, resource aggregation across partitions, transactional
  partial-prepare rollback, or changes outside Runtime production<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [rollback glossary](../../../../glossary.md#term-rollback); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Engine](../../../../glossary.md#term-engine) ordinary or advanced handle [ownership](../../../../glossary.md#term-ownership-meanings), temporary preparation cleanup, or Engine close
- Metal, CUDA, CPU, native ABI, MPSGraph, concrete resource implementations, capability claims,
  backend conformance, or integration behavior<span data-glossary-navigation="true"> ([CUDA glossary](../../../../glossary.md#term-cuda); [CPU glossary](../../../../glossary.md#term-cpu); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- ownership by [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), schedule occurrences, a backend integration, registry,
  service locator, global cache, per-run workspace, <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [cache glossary](../../../../glossary.md#term-cache); [run glossary](../../../../glossary.md#term-run))</span>`Cleaner`, finalizer, or [garbage collection](../../../../glossary.md#term-garbage-collection-gc)
- waiting close, timeout, cancellation, interruption, asynchronous execution, a public/general
  backend-resource reference-counting API, pooling, serialization, or leak detection; the private
  active-run lease count required by ADR 0013 is explicitly in scope<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [pooling glossary](../../../../glossary.md#term-pooling); [serialization glossary](../../../../glossary.md#term-serialization); [lease glossary](../../../../glossary.md#term-lease); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- a public `PreparedResourceScope`, resource accessor/count, raw `Object`, unchecked generic
  access, concrete-type switch, reflection, string dispatch, or map-based resource lookup<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- a detailed [Prepare](../../../../glossary.md#prepare) 0006, Engine 0010, or [Metal](../../../../glossary.md#term-metal) 0002 task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Runtime master plan](../master-plan.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)
- [Runtime 0006](0006-prepared-execution-aggregate.md)
- [Runtime 0010](0010-prepared-runner-and-dynamic-execution.md)
- [Runtime 0012](0012-run-state-shared-throwable-cleanup.md)
- [Runtime 0014](0014-runtime-architecture-enforcement.md)
- [Runtime 0015](0015-leased-publication-representation-access.md)

## Architecture constraints

- Recipe data and retained resource identities are immutable after construction; lifecycle state
  is explicit mutable coordination state, not recipe mutation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [lifecycle glossary](../../../../glossary.md#lifecycle); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Runtime owns lifecycle orchestration only. It imports no Metal, CUDA, CPU, native handle, Engine,
  Prepare, Compiler, Planning, Model, or backend implementation type.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model))</span>
- A [backend](../../../../glossary.md#backend) owns each [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) implementation and all physical [release](../../../../glossary.md#term-release) mechanics.
- Each resource identity has one owner and one physical close attempt. Repeated schedule or
  executable occurrences never imply another ownership entry.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [owner glossary](../../../../glossary.md#term-owner); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Every active logical [run](../../../../glossary.md#term-run) still owns one isolated [`RunState`](../../../../glossary.md#run-state--runstate); prepared-resource leases neither
  replace nor enter that state.<span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease))</span>
- The runtime hot action traversal remains direct-reference and allocation-free. Lease work is a
  bounded cold operation before state creation and after synchronous completion.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- No module dependency direction changes. The architecture test changes only because its source
  manifest is exhaustive.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [architecture test glossary](../../../../glossary.md#term-architecture-test); [manifest glossary](../../../../glossary.md#term-manifest))</span>

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.runtime.resource` — owns the nominal backend-implemented persistent
  prepared-resource lifecycle role beside the existing physical representation roles.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal); [backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `io.github.pho001.synaptik.runtime.execution` — owns the prepared root, its private resource
  aggregate, and its opaque run lease.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [run glossary](../../../../glossary.md#term-run); [lease glossary](../../../../glossary.md#term-lease))</span>
- `io.github.pho001.synaptik.runtime.run` — acquires and releases the lease around current runner
  behavior.<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>

Type placement:

- `io.github.pho001.synaptik.runtime.resource.PreparedResource` — public nominal physical
  lifecycle contract required for later concrete backend implementations.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- `io.github.pho001.synaptik.runtime.execution.PreparedExecution` — existing public root converted
  to the unique lifecycle owner; its resource aggregate remains private implementation state.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `PreparedExecution.RunLease` — public final nested opaque integration token because the existing
  public runner is in a sibling Runtime package. It is not a resource scope or user payload.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [scope glossary](../../../../glossary.md#term-scope); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>

No package is added, moved, or removed. Do not make a [Runtime](../../../../glossary.md#term-runtime)-owned resource aggregate public.

## Exact [API](../../../../glossary.md#term-application-programming-interface-api) and [lifecycle](../../../../glossary.md#lifecycle) semantics

[`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) is a public interface extending `AutoCloseable` and redeclaring:

```java
@Override
void close();
```

It declares no checked exception and no other method.

[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) is a public final class implementing `AutoCloseable`. Preserve these members:

```java
public PreparedExecution(PreparedMemoryPlan memoryPlan, PreparedSchedule schedule)
public PreparedMemoryPlan memoryPlan()
public PreparedSchedule schedule()
```

Add:

```java
public PreparedExecution(
        PreparedMemoryPlan memoryPlan,
        PreparedSchedule schedule,
        List<? extends PreparedResource> resources)
public RunLease acquireRunLease()
public boolean isClosed()
@Override public void close()
```

`RunLease` is a public final nested class implementing `AutoCloseable`. Its constructor is not
public. Its only public behavior is [idempotent](../../../../glossary.md#term-idempotent) `close()`. It exposes no enclosing execution or
resource. Exact member names may change only if the implementation pass first updates this task
and master plan without [widening](../../../../glossary.md#term-widening) the surface.

Constructor [validation](../../../../glossary.md#term-validation) order is `memoryPlan`, `schedule`, exact schedule-plan [identity](../../../../glossary.md#term-identity-meanings),
`resources`, then resource entries in supplied order. The two-[argument](../../../../glossary.md#term-argument) constructor delegates with
an empty list. A null entry reports its indexed position. A repeated exact resource identity is an
`IllegalArgumentException`; equality is not consulted. Failed construction transfers no
[ownership](../../../../glossary.md#term-ownership-meanings) and closes nothing. Successful construction snapshots the list into private storage and
transfers ownership of every listed exact resource.

The class intentionally inherits identity `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString` from `Object`.
Lifecycle-bearing [owners](../../../../glossary.md#term-owner) are not values: two executions with equal recipe components can own
different native resources and must not compare equal. Tests must record the deliberate record
reflection and equality compatibility change.

[Lease](../../../../glossary.md#term-lease) [admission](../../../../glossary.md#term-admission) and close transition are serialized. `acquireRunLease()` succeeds only while the
execution is open, increments the active count, and otherwise throws
`IllegalStateException("prepared execution is closed")`. The first `close()` marks the execution
closed to new leases before doing anything physical. `isClosed()` becomes true from that instant.

Close never waits:

- with no active lease, that close caller claims cleanup and closes resources immediately;
- with active leases, close returns after the state transition and cleanup is deferred; and
- the last lease close claims and performs cleanup.

Only one thread claims cleanup. Physical closes occur once in reverse constructor-list order and
attempt every resource. The first `RuntimeException` or `Error` is primary. Later distinct
throwables are suppressed in encounter order; another [occurrence](../../../../glossary.md#term-occurrence) of the same exact primary object
is skipped to avoid self-suppression. The cleanup-performing close or lease close throws the
primary. Repeated execution close and lease close are no-ops and do not replay a prior failure.
No [synchronization](../../../../glossary.md#term-synchronization) monitor is held during [backend](../../../../glossary.md#backend) `close()` calls.

`PreparedExecutionRunner.run(...)` acquires the lease before validating `callerInputs` or creating
a [`RunState`](../../../../glossary.md#run-state--runstate). Existing [run](../../../../glossary.md#term-run) failure remains primary and a lease cleanup failure is suppressed. If
the run otherwise produced a [`RunResult`](../../../../glossary.md#run-result--runresult) but deferred prepared-resource cleanup fails, the runner
closes that result, suppresses any distinct result-cleanup failure on the prepared-resource
failure, and returns no result. A run admitted before close may return a valid result after
successful deferred cleanup because the persistent executable is no longer needed by that
completed [synchronous](../../../../glossary.md#term-synchronous) run.

## Affected files

Expected production and tests:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedResource.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/PreparedExecutionRunner.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/package-info.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutionTest.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/PreparedExecutionRunnerTest.java`
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/RuntimeDependencyAndHotPathContractTest.java`

Architecture-decision paths already changed by the prerequisite [planning](../../../../glossary.md#term-planning) pass and retained in this
same overall change:

- `ARCHITECTURE.md`
- `docs/architecture/current-architecture-plan.md`
- `docs/design/README.md`
- `docs/design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md`

Expected documentation and planning review/finalization:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/lifecycle.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/backends/metal/master-plan.md`
- `docs/planning/roadmap.md`

Review without change unless contradicted: [Compile](../../../../glossary.md#compile), [Tensor](../../../../glossary.md#tensor), and Training APIs; [CPU](../../../../glossary.md#term-cpu) source/tests and
guide; [Metal](../../../../glossary.md#term-metal) task 0001/native [ABI](../../../../glossary.md#term-application-binary-interface-abi); [Prepare](../../../../glossary.md#prepare) finalization/orchestration; [Engine](../../../../glossary.md#term-engine) handle [lifecycle](../../../../glossary.md#lifecycle);
other architecture pages and ADRs; [Gradle](../../../../glossary.md#term-gradle) files; [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

Maximum scope is exactly 25 paths in the complete overall change:

- 9 Java/test paths: 6 Runtime production/Javadoc paths, 2 Runtime test paths, and 1
  architecture-test path; and<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- 16 architecture/explanatory/planning paths: the 4 architecture-decision paths already changed,
  6 explanatory documentation paths, and 6 planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

The implementation and documentation passes must preserve the four existing architecture-decision
paths rather than treating them as new implementation scope. No Java outside [Runtime](../../../../glossary.md#term-runtime) and the
single [architecture test](../../../../glossary.md#term-architecture-test), no [Gradle](../../../../glossary.md#term-gradle) or [dependency](../../../../glossary.md#term-dependency) file, no [concrete backend](../../../../glossary.md#concrete-backend), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine),
[conformance](../../../../glossary.md#term-conformance), or integration source may change. Stop and revise the plan if another production
type, package, or twenty-sixth path is required.

## Acceptance criteria

- Exact nominal resource and lifecycle-bearing prepared-execution surfaces exist without a public
  resource aggregate or resource lookup.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal); [lifecycle glossary](../../../../glossary.md#lifecycle); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Resource [identities](../../../../glossary.md#term-identity-meanings) are [immutable](../../../../glossary.md#term-immutable), unique by `==`, privately snapshotted, and closed once in
  deterministic reverse order.
- Close and lease races meet the exact non-waiting admission, cleanup-owner, failure, suppression,
  and idempotence rules above without holding a lock during physical close.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [admission glossary](../../../../glossary.md#term-admission); [owner glossary](../../../../glossary.md#term-owner))</span>
- Runner leases before any current validation or state creation and preserves existing creation,
  binding, execution, validity, result, and cleanup behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding))</span>
- Record-to-class compatibility and identity-equality consequences are explicitly tested and
  documented.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Resource-free [CPU](../../../../glossary.md#term-cpu)-compatible construction and concurrent isolated runs remain covered.
- [Runtime](../../../../glossary.md#term-runtime) imports no [concrete backend](../../../../glossary.md#concrete-backend) or upstream [graph](../../../../glossary.md#graph) types and adds no forbidden mechanism.
- The exhaustive Runtime architecture source [manifest](../../../../glossary.md#term-manifest) classifies [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource); dependency
  and direct-hot-path enforcement remain unchanged.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Every affected public member has complete lifecycle, ownership, concurrency, nullability,
  failure, and result Javadoc.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [nullability glossary](../../../../glossary.md#term-nullability); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A clean documentation-focused pass finalizes documentation and records reasoned no-change
  conclusions for Prepare/Engine/CPU/Metal executable code, other APIs, builds, conformance, and
  integration.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Runtime 0016 becomes `Complete` only after all validation; Prepare 0006 and Engine 0010 remain
  concise `Draft` rows without task files, and [Metal](../../../../glossary.md#term-metal) 0002 remains `Blocked` without a task file.
- The final combined worktree contains exactly the authorized 25 paths, including the four
  architecture-decision paths from the prerequisite planning pass.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutionTest \
  --tests io.github.pho001.synaptik.runtime.run.PreparedExecutionRunnerTest
./gradlew :modules:runtime:test
./gradlew :testing:architecture-tests:test \
  --tests io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest
```

Because this implements a shared architecture [lifecycle](../../../../glossary.md#lifecycle) and changes the architecture-test source
inventory, [run](../../../../glossary.md#term-run) the repository [checkpoint](../../../../glossary.md#term-checkpoint) once after executable code and documentation stabilize:

```bash
./gradlew test
```

Documentation pass:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  ARCHITECTURE.md docs/architecture/current-architecture-plan.md \
  docs/design/README.md \
  docs/design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md \
  docs/api/runtime-api.md docs/api/public-api.md \
  docs/architecture/lifecycle.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/writing-a-backend.md docs/glossary.md \
  docs/planning/modules/runtime/tasks/0016-persistent-prepared-resource-lifecycle.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/modules/prepare/master-plan.md \
  docs/planning/modules/engine/master-plan.md \
  docs/planning/backends/metal/master-plan.md docs/planning/roadmap.md
git diff --check
```

If the temporary validator is absent, create an equivalent tool outside the repository. Validate
local file links and heading anchors, unique effective anchors, balanced fences, final newlines,
trailing whitespace, task/master/roadmap status [synchronization](../../../../glossary.md#term-synchronization), the exact combined 25-path
inventory, and absence of later detailed task files. Inspect source/reflection/`javap` [shape](../../../../glossary.md#shape) for
the record-to-class change, constructors, accessors, nested [lease](../../../../glossary.md#term-lease), interfaces, [identity](../../../../glossary.md#term-identity-meanings) equality,
imports, fields, and lack of public resource access. Exercise deterministic races with bounded
latches; tests must not depend on timing sleeps.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0003–0015 — Complete.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013 — Accepted and authoritative through the coordinated architecture update.
- Metal 0001 and its 0002 blocker audit — evidence for the concrete need only; no Metal code is
  consumed or changed.

[Prepare](../../../../glossary.md#prepare) 0006, [Engine](../../../../glossary.md#term-engine) 0010, and [Metal](../../../../glossary.md#term-metal) 0002 are downstream, not implementation dependencies.

## Follow-up tasks

- [Prepare](../../../../glossary.md#prepare) 0006: persistent prepared-resource finalization [transaction](../../../../glossary.md#term-transaction) (`Draft`, required).
  Finalizer results list resources in acquisition order; Prepare concatenates partition results,
  rejects duplicate identities, closes each identity once, and preserves the primary preparation
  failure with reverse-order distinct cleanup failures suppressed.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Engine](../../../../glossary.md#term-engine) 0010: prepared-handle [ownership](../../../../glossary.md#term-ownership-meanings) and closure (`Draft`, required after Prepare 0006).
- [Metal](../../../../glossary.md#term-metal) 0002: first [MPSGraph](../../../../glossary.md#term-mpsgraph) [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) route (remains `Blocked` until all three shared
  prerequisites are Complete).

Do not create detailed specifications for those tasks in this change.

## Architecture impact

Expected impact: implementation of [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013 only. No [dependency direction](../../../../glossary.md#java-platform-and-build-tools) changes. Stop if the
implementation needs a [concrete backend](../../../../glossary.md#concrete-backend) import, public resource [scope](../../../../glossary.md#term-scope), waiting close, checked
close exception, new [module](../../../../glossary.md#term-module) edge, [Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) source edit, or different [ownership](../../../../glossary.md#term-ownership-meanings) transfer.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, ADR 0013, the focused Runtime
architecture pages, Runtime master plan and tasks 0003/0006/0010/0012/0014/0015, this task, and
the directly affected Runtime source/tests/Javadocs plus the Runtime architecture test.

Implement Runtime 0016 exactly as specified. Preserve current run-state, runner, CPU-compatible,
dependency, and hot-path behavior. Stop on architecture, ownership, package, API, or scope
conflict. Do not implement Prepare 0006, Engine 0010, Metal 0002, or another later task.

Treat the four architecture-decision paths already present in the worktree as retained parts of
the authorized exact 25-path overall change. Do not stop at the obsolete 21-path count.

Run focused Runtime and architecture tests, one final Runtime suite, and the specified repository
checkpoint. Then hand the actual diff and exact evidence to a separate clean documentation-focused
context. That context must follow docs/developer-guide/documentation-rules.md, finalize affected
Javadocs/docs/glossary/planning, validate links/anchors/fences/newlines/scope/status, and avoid
repeating stable Java tests unless executable code changes. Mark Complete only after every gate.
```

## Documentation-review prompt

```text
You are the mandatory separate clean-context documentation reviewer for Synaptik Runtime 0016.
Do not commit or push. Read AGENTS.md, ARCHITECTURE.md, ADR 0013, documentation rules and the
General/API-Javadoc/Architecture/Backend-Guide/Planning profiles, Runtime task 0016, the final
implementation diff, affected source/tests, and recorded Java-test evidence.

Independently finalize the affected Javadocs, Runtime/Public APIs, lifecycle and focused boundary
explanations, backend guide, glossary, Runtime/Prepare/Engine/Metal master-plan rows, roadmap, and
task evidence. Verify the resource owner, identity uniqueness, record-to-class equality change,
non-waiting close/lease race, reverse cleanup/failure ordering, and current-versus-planned
boundaries. Record reasoned no-change conclusions for Java outside Runtime, CPU behavior, Prepare
transaction code, Engine handles, Metal ABI/routes, Compile/Tensor/Training APIs, Gradle/module
edges, conformance, and integration. Run Runtime Javadoc and exact documentation/scope/status/
whitespace gates against the complete authorized 25-path inventory; do not rerun stable Java
suites unless executable behavior changes.
```

## Local decisions

- The central aggregate is private inside [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution); only [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) and the
  opaque run lease are public nominal contracts.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [run glossary](../../../../glossary.md#term-run); [lease glossary](../../../../glossary.md#term-lease); [nominal glossary](../../../../glossary.md#term-nominal))</span>
- Deferred release is selected over waiting close to avoid indefinite waits and close/run
  deadlocks.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
- `PreparedExecution` uses [identity](../../../../glossary.md#term-identity-meanings) equality after conversion from a record.
- The resource-free constructor preserves current [CPU](../../../../glossary.md#term-cpu) source compatibility.

## Known limitations

- [Runtime](../../../../glossary.md#term-runtime) 0016 alone cannot receive resources from [backend](../../../../glossary.md#backend) finalization; [Prepare](../../../../glossary.md#prepare) 0006 is required.
- [Engine](../../../../glossary.md#term-engine) handles do not yet close inward executions; Engine 0010 is required before [Metal](../../../../glossary.md#term-metal) 0002.
- Deferred close reports a physical cleanup failure to the last lease releaser, not the earlier
  close caller.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease))</span>
- No cleaner or leak detector is provided.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a0bea7-a4b3-7060-990f-3391712c51ab` supplied stable executable
  evidence. Focused `PreparedExecutionTest` plus `PreparedExecutionRunnerTest` passed 2 suites / 25
  tests; the final Runtime suite passed 17 suites / 154 tests; the focused Runtime architecture
  suite passed 1 suite / 3 tests; and `./gradlew test` passed 509 suites / 3,304 tests with 30
  expected skips and zero failures or errors. No executable Java changed in the later
  documentation context, so those suites were not repeated.
- The same implementation context recorded `javap` confirmation of the exact final class,
  constructors, generic resource parameter, opaque lease, and stateless runner surface, with no
  public resource accessor. Source/import inspection found no forbidden concrete-backend,
  upstream graph, reflection, service-location, map-based lookup, or dynamic-discovery mechanism.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [opaque glossary](../../../../glossary.md#term-opaque); [lease glossary](../../../../glossary.md#term-lease); [backend glossary](../../../../glossary.md#backend); [graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup))</span>
  `git diff --check` and the then-current exact nine implementation paths passed. The coordinator
  independently reviewed cleanup outside monitors, one-time lease/resource cleanup, failure
  precedence, and successful-result cleanup after deferred resource failure.
- Clean documentation context `01a0beb6-b749-7092-9b56-8721a8921bbb` independently reviewed the
  final production/package Javadocs, implementation source and focused tests, architecture test,
  ADR, architecture explanations, Runtime/Public APIs, backend guide, glossary, all four affected
  master plans, roadmap, and this task. It changed no executable Java behavior and therefore
  reused the stable Java evidence above.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- The first sandboxed `./gradlew :modules:runtime:javadoc` invocation could not open Gradle's
  user-cache lock file; the approved rerun with cache access passed after the final Javadocs.
  Generated pages for<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [cache glossary](../../../../glossary.md#term-cache))</span>
  [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource), [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), nested `RunLease`, [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), and the
  three affected packages were inspected for ownership, identity, lease, cleanup, failure, and
  no-lookup wording.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `python3 /tmp/validate_synaptik_markdown.py` over the exact 16 authorized architecture,
  explanatory, and planning paths passed: 16 Markdown files validated for local links, heading
  targets, unique effective anchors, balanced fences, final newlines, and trailing whitespace.
  The first pass exposed duplicate generic example anchors in the Runtime API and backend guide;
  the headings were made purpose-specific before the passing final run.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [targets glossary](../../../../glossary.md#term-target); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [run glossary](../../../../glossary.md#term-run))</span>
- The exact inventory command compared the worktree union with the authorized list and reported
  `exact-path-inventory=25`. Status inspection confirmed [Runtime](../../../../glossary.md#term-runtime) 0016 `Complete`, Prepare 0006 and
  Engine 0010 <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>`Draft`, and [Metal](../../../../glossary.md#term-metal) 0002 `Blocked`. No Prepare 0006, Engine 0010, Metal 0002, or later
  detailed task file exists. An initial zsh harness preserved the count result but accidentally
  overwrote zsh's special `path` variable and used an unmatched glob; the corrected harness used
  `item` and null-glob qualifiers and passed in full. Final `git diff --check` passed.
- Reasoned no-change conclusions: Java outside the six Runtime production paths and two Runtime
  tests did not need change because the nominal owner stays inside Runtime and the exhaustive
  architecture manifest is the sole cross-module test adjustment. CPU behavior remains
  resource-free through the compatible constructor. Prepare still cannot return or transactionally
  roll back persistent resources; Engine handles still do not own inward close; and Metal keeps
  ABI version 1, fail-closed capability, and no MPSGraph route. Compile, Tensor, and Training APIs
  neither expose nor own this lifecycle. No Gradle/module edge changed. Backend conformance and
  integration behavior did not change, so neither suite nor source needed an update.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal); [owner glossary](../../../../glossary.md#term-owner); [manifest glossary](../../../../glossary.md#term-manifest); [module glossary](../../../../glossary.md#term-module); [CPU glossary](../../../../glossary.md#term-cpu); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added the cleanup-only [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) role and converted [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) from a record
  to a final identity-bearing owner while preserving exact plan/schedule access and the
  resource-free constructor.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The [owner](../../../../glossary.md#term-owner) snapshots exact resource identities, rejects duplicate identities without `equals`,
  admits synchronous runs through opaque leases, and performs one reverse attempt-all cleanup
  outside lifecycle monitors without waiting.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [opaque glossary](../../../../glossary.md#term-opaque); [leases glossary](../../../../glossary.md#term-lease); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The stateless runner acquires before caller-input inspection or state creation and composes run,
  lease, and result cleanup with the specified primary/suppressed failure precedence.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Documentation now states the implemented Runtime boundary separately from planned Prepare 0006,
  Engine 0010, and blocked Metal 0002. No public resource lookup or physical access was added.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [lookup glossary](../../../../glossary.md#term-lookup))</span>

## Completion summary

- Completed changes: Implemented and documented the persistent prepared-resource owner, close/run
  lease lifecycle, runner cleanup composition, architecture decision, and synchronized planning
  status.<span data-glossary-navigation="true"> ([resource owner glossary](../../../../glossary.md#ownership-borrowing-and-lease); [run glossary](../../../../glossary.md#term-run); [lease glossary](../../../../glossary.md#term-lease); [lifecycle glossary](../../../../glossary.md#lifecycle); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Exactly the authorized 25-path union: nine implementation/test paths
  and sixteen architecture, explanatory, and planning paths listed above.
- Tests and validation: Reused the stable 25-test focused, 154-test Runtime, 3-test architecture,
  and 3,304-test repository evidence; passed final Runtime Javadoc, generated-page inspection,
  16-file Markdown validation, exact path/status/later-spec gates, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check`.
- Documentation-agent review: Complete in clean context
  `01a0beb6-b749-7092-9b56-8721a8921bbb`; no executable Java behavior changed.
- Documentation impact: Finalized Runtime/Public API, lifecycle/boundary explanations, backend
  guidance, ADR/index, master plans, roadmap, and task evidence with explicit current/planned
  boundaries.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Javadoc review: Complete; the implementation drafts already met the API/Javadoc profile and
  required no Java edit.
- Glossary impact: Updated [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) from the obsolete record description and added the
  current [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource)/downstream-[handoff](../../../../glossary.md#term-handoff) distinction.
- Unresolved issues: None within [Runtime](../../../../glossary.md#term-runtime) 0016.
- Follow-up required: [Prepare](../../../../glossary.md#prepare) 0006, then [Engine](../../../../glossary.md#term-engine) 0010, then [Metal](../../../../glossary.md#term-metal) 0002.

Status: Complete
