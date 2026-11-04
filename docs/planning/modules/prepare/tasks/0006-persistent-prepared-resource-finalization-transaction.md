# Task 0006: Persistent Prepared-Resource Finalization [Transaction](../../../../glossary.md#term-transaction)

> **Editorial historical reading key.**
>
> This task records a three-stage ownership transaction, not ownership inferred from executable
> reachability. Shared Prepare calls each backend finalizer with assigned resources; the backend
> returns an executable plus acquisition-ordered persistent resources. Before successful return, the
> backend must clean up its own acquisitions. After return, Prepare owns rollback through schedule
> assembly and aggregate construction; successful Runtime `PreparedExecution` construction transfers
> final ownership. CPU’s empty resource list is a concrete implementation fact, not a prohibition on
> other backends.
>
> Compare the [current finalization
> handoff](../../../../reference/runtime/orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization)
> and [Runtime prepared
> owner](../../../../reference/runtime/prepared-execution.md#current-prepared-execution). Preserve the
> archived duplicate-scanning algorithm as history: Prepare 0009 later repairs trailing-resource
> rollback. Likewise, Engine ownership and Metal follow-up statuses describe this checkpoint, not
> current support. Repeated schedule occurrences never multiply cleanup ownership.

## Status

Complete

## Goal

Complete the [Prepare](../../../../glossary.md#prepare) stage of
[ADR 0013](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>.
[Backend](../../../../glossary.md#backend) finalization must return one [atomic](../../../../glossary.md#resource-transaction-and-rollback) [immutable](../../../../glossary.md#term-immutable) result containing its
[`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) and the persistent [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) instances acquired for that result,
in physical acquisition order. Shared Prepare must own every successfully returned resource as
one [transaction](../../../../glossary.md#term-transaction) until it constructs the final
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
bare executable. [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) remains exactly `PreparedPartition(partition, executable)`;
persistent resource [ownership](../../../../glossary.md#term-ownership) never comes from a [partition](../../../../glossary.md#partition), executable, or schedule [occurrence](../../../../glossary.md#term-occurrence).
The existing [CPU](../../../../glossary.md#term-cpu) finalizer uses the one-[argument](../../../../glossary.md#term-argument) constructor and therefore continues to return
the same executable with an empty resource list.

## [Mental model](../../../../glossary.md#term-mental-model)

[Ownership](../../../../glossary.md#term-ownership-meanings) moves only at successful method and constructor boundaries:

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

This is a cold preparation [transaction](../../../../glossary.md#term-transaction). It adds no [run](../../../../glossary.md#term-run)-time [lookup](../../../../glossary.md#term-lookup), schedule-derived ownership,
physical allocation in shared [Prepare](../../../../glossary.md#prepare), or [backend](../../../../glossary.md#backend) interpretation.

## [Scope](../../../../glossary.md#term-scope)

- Add the public [immutable](../../../../glossary.md#term-immutable) `BackendPartitionFinalizationResult` record in the Prepare root
  package.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- Give the record exactly two components in declaration order: one non-null
  [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) and an immutable [snapshot](../../../../glossary.md#term-snapshot) of a non-null acquisition-ordered
  `List<PreparedResource>` whose entries are non-null.
- Add a one-[argument](../../../../glossary.md#term-argument) resource-free constructor delegating with `List.of()` so CPU and fake
  finalizers express an empty persistent-resource result without fabricating ownership.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Change `BackendPartitionFinalizer.finalizePartition(...)` to return the [atomic](../../../../glossary.md#resource-transaction-and-rollback) result.
- Require every concrete finalizer to own and roll back anything it acquires until the complete
  result returns successfully. A failed return transfers nothing to shared Prepare.
- Extend `BackendPartitionFinalizationHandoff` to consume successful result values in partition
  finalization order, extract only their executables for <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>[`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), and concatenate
  their resources in result-list order.
- Track resource [identity](../../../../glossary.md#term-identity-meanings) with `==`, never `equals`, and reject a repeated identity within one
  result or across results.
- On duplicate rejection, record only the first ownership occurrence so rollback closes that
  exact resource once rather than once per duplicate occurrence.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Roll back all earlier successful finalizer results when a later finalizer, result check,
  executable-plan association, or `PreparedPartition` construction fails.
- Return the immutable ordered identity-unique resource snapshot in the package-private handoff
  result together with the existing memory plan, partitions, and buffer assignments.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Extend [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) to own a successful handoff transaction through schedule assembly,
  schedule validation, and final aggregate construction.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [validation glossary](../../../../glossary.md#term-validation))</span>
- Construct the final [Runtime](../../../../glossary.md#term-runtime) [owner](../../../../glossary.md#term-owner) with
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
  pass after executable Java stabilizes.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- changing [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) components, equality, or [ownership](../../../../glossary.md#term-ownership-meanings)
- deriving ownership from [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), repeated executable occurrences, partition count,
  or any graph topology<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [graph glossary](../../../../glossary.md#graph); [topology glossary](../../../../glossary.md#term-topology))</span>
- changing [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource), lease, close, cleanup, runner, or
  per-run resource behavior implemented by Runtime 0016<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [run glossary](../../../../glossary.md#term-run))</span>
- implementing [Engine](../../../../glossary.md#term-engine) 0010 prepared-handle ownership or changing any Engine Java source
- implementing Metal 0002, compiling an MPS graph, adding a Metal resource, changing native ABI,
  or advertising a Metal capability<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [MPS glossary](../../../../glossary.md#term-mps); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- changing CPU executable, generated artifact, worker-group, OpenBLAS, buffer, workspace,
  binding, execution, or cleanup behavior<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [buffer glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding))</span>
- adding persistent resources to [CPU](../../../../glossary.md#term-cpu) finalization
- allocating, acquiring, interpreting, or physically closing a backend resource in shared Prepare
  other than invoking its nominal <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [nominal glossary](../../../../glossary.md#term-nominal))</span>`PreparedResource.close()` during [rollback](../../../../glossary.md#term-rollback)
- changing analysis, route selection, resource declarations, slot assignment, producerless
  resource handling, schedule structure, publication, tuning, or runtime execution<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- a resource registry, public resource accessor, reference-counted schedule ownership, backend
  switch, service locator, cleaner, or garbage-collection correctness mechanism<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- checked cleanup exceptions, waiting close, [asynchronous](../../../../glossary.md#term-asynchronous) preparation, or retrying failed cleanup
- module dependency, Gradle, architecture-contract, ADR, architecture-test, integration-test, or
  native-source changes<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Prepare
  lifecycle, Runtime ownership, concrete-backend ownership, and dependency rules<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend ownership glossary](../../../../glossary.md#backend-ownership); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding))</span>
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span> and [roadmap](../../../roadmap.md)
- [Prepare master plan](../master-plan.md) and completed [Prepare](../../../../glossary.md#prepare) tasks
  [0002](0002-backend-partition-finalization-handoff.md) and
  [0003](0003-prepare-orchestration-and-validation.md)
- [Runtime 0016](../../runtime/tasks/0016-persistent-prepared-resource-lifecycle.md)
- [Engine master plan](../../engine/master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>, task 0010 row
- [Metal master plan](../../../backends/metal/master-plan.md)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>, blocked task 0002 row

## Architecture constraints

- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) is the unique final [owner](../../../../glossary.md#term-owner) of [immutable](../../../../glossary.md#term-immutable) persistent prepared resources.
- Concrete backends own acquisition, physical resource types, release mechanics, lowering, route
  selection, and executable construction. Shared Prepare knows only the nominal Runtime cleanup
  contract.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [release glossary](../../../../glossary.md#term-release); [lowering glossary](../../../../glossary.md#lowering); [Prepare glossary](../../../../glossary.md#prepare); [nominal glossary](../../../../glossary.md#term-nominal); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Backend analysis still declares every shared buffer and workspace requirement before slot
  assignment. Persistent resources are acquired only during finalization and do not revise route
  choice or shared declarations.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- A finalizer must complete its own local acquisition transaction before returning. Shared
  Prepare cannot recover a resource that was acquired but omitted because the finalizer threw.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>
- Shared Prepare owns only resources present in successful finalizer results. Executable sharing,
  partition associations, and repeated schedule occurrences have no ownership meaning.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Successful `PreparedExecution` construction is the only shared-Prepare-to-Runtime ownership
  transfer point. Failed construction transfers nothing under the Runtime 0016 contract.
- Rollback is deterministic, reverse acquisition order, attempt-all, identity-aware, and does not
  use <span data-glossary-navigation="true"> ([Rollback glossary](../../../../glossary.md#term-rollback); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`equals`.
- Prepare remains independent of concrete backend implementations. The CPU source change is an
  implementation of the public Prepare SPI in the concrete backend, not concrete logic in
  Prepare.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [CPU glossary](../../../../glossary.md#term-cpu); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- Runtime remains independent of concrete backends, and concrete backends remain independent of
  Engine.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- No dependency direction, architecture decision, or Runtime hot-path behavior changes. Stop if
  implementation requires one.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Package impact and exact type semantics

Changed package:

- `io.github.pho001.synaptik.prepare` gains
  `BackendPartitionFinalizationResult`, updates the existing finalizer collaboration, and extends
  package-private transactional orchestration. No package is added or moved.

Consumed unchanged packages:

- `io.github.pho001.synaptik.runtime.execution` continues to own [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) and the
  final [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) [owner](../../../../glossary.md#term-owner).
- `io.github.pho001.synaptik.runtime.resource` continues to own the [nominal](../../../../glossary.md#term-nominal)
  [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) close role.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` continues to own CPU finalization and
  adopts only the changed Prepare SPI return shape.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [shape glossary](../../../../glossary.md#shape))</span>

Type placement:

- `io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult` — public backend-facing
  Prepare handoff value because both a concrete backend finalizer and shared Prepare must name it;
  it is not a Runtime aggregate or backend-private type.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `io.github.pho001.synaptik.prepare.BackendPartitionFinalizationHandoff` — remains the
  package-private owner of complete-set finalizer invocation, cross-result identity validation,
  and rollback before a successful handoff result.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- `io.github.pho001.synaptik.prepare.GraphPreparation` — remains the public stateless facade and
  owns a successful handoff result until final aggregate construction.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizer` — remains the
  concrete CPU implementation and returns no persistent resources.

`BackendPartitionFinalizationResult` is a public record with exactly these record components:

1. `PreparedExecutable executable`
2. `List<PreparedResource> resources`

The [canonical](../../../../glossary.md#term-canonical-form) constructor validates `executable`, then `resources`, then each indexed resource.
It retains the exact executable reference and snapshots the list with `List.copyOf`. The record
does not reject duplicate resource [identities](../../../../glossary.md#term-identity-meanings) because shared [Prepare](../../../../glossary.md#prepare) must detect duplicates
across the complete ordered result set and close each accepted [identity](../../../../glossary.md#term-identity-meanings) once. The result owns no
close method and exposes no physical [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication). [Ownership](../../../../glossary.md#term-ownership-meanings) remains with the finalizer until the
method returns successfully, then moves to shared Prepare as a [transaction](../../../../glossary.md#term-transaction).

The one-[argument](../../../../glossary.md#term-argument) constructor is exactly:

```java
public BackendPartitionFinalizationResult(PreparedExecutable executable) {
    this(executable, List.of());
}
```

No factory, builder, resource [lookup](../../../../glossary.md#term-lookup), [backend](../../../../glossary.md#backend) identifier, [partition](../../../../glossary.md#partition), memory plan, schedule, or
generic payload is added.

## [Transaction](../../../../glossary.md#term-transaction) and failure semantics

### Finalizer-local [transaction](../../../../glossary.md#term-transaction)

A finalizer may acquire zero or more persistent resources while constructing one executable. It
must retain those resource [identities](../../../../glossary.md#term-identity-meanings) in physical acquisition order. Before the complete result
returns successfully, the finalizer owns them and must reverse-close all acquired identities on
every `RuntimeException` or `Error`, including failure while constructing the result itself.

The original failure remains primary. Cleanup attempts continue in reverse order; each distinct
cleanup failure is suppressed in encounter order, and another [occurrence](../../../../glossary.md#term-occurrence) of the exact primary
throwable is skipped. The shared layer cannot implement or infer this local [rollback](../../../../glossary.md#term-rollback), so the
interface [Javadoc](../../../../glossary.md#term-javadoc) and [backend](../../../../glossary.md#backend) tests must make the obligation explicit.

### Complete-set [handoff](../../../../glossary.md#term-handoff) [transaction](../../../../glossary.md#term-transaction)

All existing shared [validation](../../../../glossary.md#term-validation), memory-plan construction, and typed finalization-input
construction still finish before the first finalizer call. For each [partition](../../../../glossary.md#partition) in order:

1. invoke its finalizer exactly once;
2. require a non-null `BackendPartitionFinalizationResult`;
3. inspect resources in result-list order, using [reference identity](../../../../glossary.md#term-reference-identity) only;
4. append each first-seen [identity](../../../../glossary.md#term-identity-meanings) to the transaction list;
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
order. Its existing memory-plan, partition, and [buffer](../../../../glossary.md#term-buffer)-assignment references and ordering remain
unchanged.

### Complete [graph](../../../../glossary.md#graph)-preparation [transaction](../../../../glossary.md#term-transaction)

After a successful [handoff](../../../../glossary.md#term-handoff), [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) owns the returned resources while it:

1. constructs [`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext);
2. invokes the assembler once;
3. validates the complete schedule; and
4. constructs `PreparedExecution(memoryPlan, schedule, resources)`.

A `RuntimeException` or `Error` in steps 1–4 closes the [identity](../../../../glossary.md#term-identity-meanings)-unique resources in reverse
order and preserves that failure as primary. A successful constructor call transfers [ownership](../../../../glossary.md#term-ownership-meanings)
to [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution); `GraphPreparation` must not close those resources afterward. It must not
fall back to the resource-free [Runtime](../../../../glossary.md#term-runtime) constructor.

### Stable failure messages

The new record uses:

- `NullPointerException("executable")`;
- `NullPointerException("resources")`; and
- `NullPointerException("resources[i]")`.

The [handoff](../../../../glossary.md#term-handoff) preserves `NullPointerException("entries[i].finalizer returned null")` for a null
result and the existing foreign-memory-plan failure for the result executable. A repeated exact
[identity](../../../../glossary.md#term-identity-meanings) fails with:

```text
entries[i].finalizer resources[j] duplicates an earlier prepared resource identity
```

Do not specify [backend](../../../../glossary.md#backend)-private exception messages for local acquisition failures.

## [Javadoc](../../../../glossary.md#term-javadoc) and [API](../../../../glossary.md#term-application-programming-interface-api) effects

- `BackendPartitionFinalizationResult` Javadoc must define immutability, acquisition order,
  ownership before and after successful return, null handling, list snapshotting, permitted
  duplicates at the value boundary, and absence of a close/lookup role.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- `BackendPartitionFinalizer` Javadoc must replace the obsolete “no closeable physical
  resources” rule with the exact local transaction, successful-return transfer, ordering,
  rollback, and primary/suppressed-failure contract.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- `BackendPartitionFinalizationHandoff` Javadoc must describe returned resources, identity
  uniqueness, rollback coverage, and ownership transfer to its successful result.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) and its package Javadoc must describe the successful-result transaction,
  schedule/aggregate rollback window, and transfer to [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) without claiming that
  shared Prepare performs physical allocation.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- `CpuPartitionFinalizer` Javadoc must state that CPU returns an empty persistent-resource list
  and preserves all current executable and borrowed-resource behavior.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Runtime and Public API documentation must show the current finalization result and final
  three-argument <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [argument glossary](../../../../glossary.md#term-argument))</span>`PreparedExecution` construction. The focused architecture boundary and backend
  guide must change Prepare 0006 wording from planned to current only when implementation passes.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- The glossary already defines prepared execution and prepared resource. Update those entries to
  describe the implemented Prepare transaction; do not add a redundant glossary entry solely for
  the Java result carrier.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource))</span>

## Affected files

Exact implementation and documentation allowlist (twenty-six paths):

Production and [Javadoc](../../../../glossary.md#term-javadoc):

1. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationResult.java` (new)
2. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizer.java`
3. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java`
4. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java`
5. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/package-info.java`
6. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`

Tests and [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) fixtures:

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

Documentation and [planning](../../../../glossary.md#term-planning):

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

The six [CPU](../../../../glossary.md#term-cpu) and [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) test/[fixture](../../../../glossary.md#term-fixture) paths after the focused CPU finalizer test are
mechanical direct-call adaptations required by the public return-type change. They do not widen
CPU semantics. Review without editing [Runtime](../../../../glossary.md#term-runtime) Java/tests, [Engine](../../../../glossary.md#term-engine) Java/tests, [Metal](../../../../glossary.md#term-metal) Java/native
source/tests, other concrete backends, [Gradle](../../../../glossary.md#term-gradle) files, architecture tests, integration tests,
[Compile](../../../../glossary.md#compile)/[Tensor](../../../../glossary.md#tensor)/Training APIs, and all other paths.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the twenty-six paths above. This exceeds the usual task-size guideline because one
public [backend](../../../../glossary.md#backend)-finalizer return-type change must atomically update the [Prepare](../../../../glossary.md#prepare) [transaction](../../../../glossary.md#term-transaction), the
sole concrete finalizer, every direct [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) [fixture](../../../../glossary.md#term-fixture), [conformance](../../../../glossary.md#term-conformance) compilation, and the
required documentation/status surfaces. Splitting the mechanical callers would leave the
repository uncompilable and splitting documentation would violate the same-change rule.

Do not add a cleanup helper source file. Keep shared [rollback](../../../../glossary.md#term-rollback) support package-private inside the
existing [handoff](../../../../glossary.md#term-handoff)/orchestration implementation. If a twenty-seventh path or a different package is
required, stop and revise the task rather than [widening](../../../../glossary.md#term-widening) it implicitly.

## Acceptance criteria

- `BackendPartitionFinalizationResult` has exactly the two specified record components and the
  one specified convenience constructor, snapshots its list, rejects nulls in exact order, and
  exposes no close, lookup, backend, partition, or schedule surface.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [backend glossary](../../../../glossary.md#backend))</span>
- `BackendPartitionFinalizer.finalizePartition(...)` returns only the new atomic result and its
  Javadoc makes finalizer-local rollback before successful return mandatory.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- All existing validation and assignment completes before the first finalizer call, and
  finalizers remain ordered and invoked at most once.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) remains byte-for-byte unchanged and contains only [partition](../../../../glossary.md#partition) plus executable.
- A successful result transfers its resource list to shared Prepare even if its executable later
  fails exact-plan validation.
- Resource lists are concatenated in partition-finalization and within-result acquisition order.
- [Identity](../../../../glossary.md#term-identity-meanings) uniqueness uses `==`; a duplicate within one result or across results is rejected,
  appears once in the rollback transaction, and is physically closed once.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>
- A later finalizer failure, null result, duplicate identity, foreign executable plan,
  partition-association failure, assembler failure, schedule validation failure, or aggregate
  construction failure rolls back every earlier accepted identity once in reverse acquisition
  order and never invokes a later collaborator.
- [Rollback](../../../../glossary.md#term-rollback) attempts every resource. The triggering `RuntimeException` or `Error` stays primary;
  distinct cleanup failures are suppressed in encounter order, and the same exact primary object
  is not self-suppressed.
- A successful [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) receives the exact ordered resources once and shared Prepare
  performs no later close. Closing that execution remains Runtime 0016 behavior.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Tests demonstrate a well-behaved fake finalizer rolling back its own partially acquired
  resources before throwing; shared Prepare neither sees nor double-closes those resources.
- Resource ownership is derived only from successful finalizer results. Repeated executable and
  schedule occurrences do not change the resource list or close count.<span data-glossary-navigation="true"> ([Resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership))</span>
- `CpuPartitionFinalizer` returns the exact existing executable plus an empty immutable resource
  list for every route, and all direct callers extract the executable without changing execution
  assertions.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- No Runtime, Engine, Metal, ABI, capability, dependency, build, architecture-test, integration,
  or performance behavior changes.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Public/package shape, complete Javadocs, API/boundary/backend-guide prose, glossary status,
  task/master/roadmap status, exact scope, and current-versus-planned language are synchronized
  before completion.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- [Prepare](../../../../glossary.md#prepare) 0006 becomes `Complete` only after executable validation and the independent
  documentation pass. Engine 0010 remains Draft without a detailed specification, and Metal 0002
  remains Blocked without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused [Prepare](../../../../glossary.md#prepare) tests must cover:

- exact record components, constructor/method descriptors, public/final shapes, generic resource
  type, immutable snapshotting, null validation, and unchanged <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable))</span>[`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) [shape](../../../../glossary.md#shape);
- ordered multi-finalizer resource collection and successful transfer to the final execution;
- finalizer-local [rollback](../../../../glossary.md#term-rollback) evidence through a fake finalizer that fails before returning;
- duplicate [identity](../../../../glossary.md#term-identity-meanings) within one result and across results, using resources whose `equals` method
  would give a misleading answer;
- reverse attempt-all rollback and primary/suppressed/self-suppression behavior for both
  `RuntimeException` and `Error`;
- failure at every [handoff](../../../../glossary.md#term-handoff) stage after the first successful resource result;
- assembler and schedule-validation rollback, plus source inspection that aggregate construction
  remains inside the same rollback boundary; and
- success with repeated executable/schedule occurrences without duplicated [ownership](../../../../glossary.md#term-ownership-meanings).

Focused [CPU](../../../../glossary.md#term-cpu) tests must prove that every route still returns its existing executable and an empty
resource list. Other listed CPU and [conformance](../../../../glossary.md#term-conformance) files receive only the direct-call extraction
needed for [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and retain their current assertions.

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

The repository-wide [run](../../../../glossary.md#term-run) is required once because the public Prepare [SPI](../../../../glossary.md#term-service-provider-interface-spi) changes and direct
consumers span Prepare, CPU, and [backend](../../../../glossary.md#backend) conformance. Do not repeat it in the documentation pass
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
trailing whitespace, exact twenty-six-path [scope](../../../../glossary.md#term-scope), task/master/roadmap status [synchronization](../../../../glossary.md#term-synchronization),
absence of an [Engine](../../../../glossary.md#term-engine) 0010 or [Metal](../../../../glossary.md#term-metal) 0002 task file, and unchanged excluded paths. Inspect generated
[Javadoc](../../../../glossary.md#term-javadoc) for the result, finalizer, [graph preparation](../../../../glossary.md#graph-preparation--graphpreparation), Prepare package, and CPU finalizer ownership
language.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0016 — Complete; supplies [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource), the unique final Runtime owner, exact
  constructor-transfer semantics, close/run leases, and reverse cleanup behavior.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [run glossary](../../../../glossary.md#term-run); [leases glossary](../../../../glossary.md#term-lease))</span>
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013 — Accepted; defines the three-stage [ownership](../../../../glossary.md#term-ownership-meanings) [transaction](../../../../glossary.md#term-transaction).
- Prepare 0001–0005 — Complete; supply staged analysis, slot assignment, finalizer invocation,
  graph orchestration, schedule validation, opaque tuning handoff, and producerless assignment.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation); [opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- CPU 0010J and Engine 0009 — Complete; current direct consumers whose behavior must remain
  resource-free and otherwise unchanged.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>

No [Metal](../../../../glossary.md#term-metal) executable route is an implementation [dependency](../../../../glossary.md#term-dependency). Metal 0001 and its audit provide the
motivating native [lifecycle](../../../../glossary.md#lifecycle) only.

## Follow-up tasks

- [Engine](../../../../glossary.md#term-engine) 0010: prepared-handle [ownership](../../../../glossary.md#term-ownership-meanings) and closure. It remains `Draft` and blocked on this task;
  create its detailed specification only after Prepare 0006 is Complete.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [Metal](../../../../glossary.md#term-metal) 0002: first [MPSGraph](../../../../glossary.md#term-mpsgraph) [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) route. It remains `Blocked` without a detailed
  specification until Prepare 0006 and Engine 0010 are Complete.
- Custom Metal kernels, general asynchronous execution, bounded Metal buffer-pool implementation,
  lease-bound <span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [buffer glossary](../../../../glossary.md#term-buffer); [lease glossary](../../../../glossary.md#term-lease))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) [aliasing](../../../../glossary.md#term-aliasing), and whole-plan [autotuning](../../../../glossary.md#autotuning--model-autotuning) remain later Metal tasks.

## Documentation-focused clean-context pass

After Java and the final implementation-owned test evidence stabilize, hand the exact diff,
baseline worktree status, commands/results, public-[shape](../../../../glossary.md#shape) evidence, [ownership](../../../../glossary.md#term-ownership-meanings)/failure decisions,
and twenty-six-path allowlist to a distinct clean documentation-focused context. Apply the
General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Architecture, [Backend](../../../../glossary.md#backend) Guide, and [Planning](../../../../glossary.md#term-planning) profiles.

That pass must independently inspect the implementation and tests, finalize all affected
Javadocs and the five explanatory/API/glossary paths, synchronize the four planning paths, and
record reasoned no-change conclusions for [Runtime](../../../../glossary.md#term-runtime) Java, [Engine](../../../../glossary.md#term-engine) Java, [Metal](../../../../glossary.md#term-metal)/[native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), [CPU](../../../../glossary.md#term-cpu)
behavior beyond the empty result adaptation, [Compile](../../../../glossary.md#compile)/[Tensor](../../../../glossary.md#tensor)/Training APIs, [Gradle](../../../../glossary.md#term-gradle)/[module](../../../../glossary.md#term-module) edges,
architecture tests, integration tests, and performance. It reuses successful Java evidence unless
it changes executable behavior or identifies a concrete stale-evidence risk.

## Architecture impact

Expected impact: implementation of the [Prepare](../../../../glossary.md#prepare) stage already selected by [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013. No new
architecture decision or [dependency direction](../../../../glossary.md#java-platform-and-build-tools).

Stop if implementation requires changing `ARCHITECTURE.md`, ADR 0013, [Runtime](../../../../glossary.md#term-runtime) [ownership](../../../../glossary.md#term-ownership-meanings),
[`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), a [module](../../../../glossary.md#term-module) edge, a concrete resource type in shared Prepare, schedule-derived
ownership, checked cleanup, waiting, [Engine](../../../../glossary.md#term-engine) Java, [Metal](../../../../glossary.md#term-metal)/native Java, or another public mechanism.

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

Stop and report instead of implementing or [widening](../../../../glossary.md#term-widening) if:

- a successful finalizer result cannot carry every persistent resource needed by its executable;
- shared Prepare would need to inspect a concrete resource, acquire it, reference-count schedule
  occurrences, or infer ownership from executable topology;<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [topology glossary](../../../../glossary.md#term-topology))</span>
- a finalizer cannot roll back its own resources before a failed return;
- failed [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) construction would take ownership or close resources contrary to
  Runtime 0016;<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [CPU](../../../../glossary.md#term-cpu) needs a non-empty persistent-resource list or changed execution [lifecycle](../../../../glossary.md#lifecycle);
- [Engine](../../../../glossary.md#term-engine) or [Metal](../../../../glossary.md#term-metal) Java/native implementation is required for this [transaction](../../../../glossary.md#term-transaction);
- a module dependency, architecture decision, Gradle edit, new package, cleanup helper source,
  or twenty-seventh path is required; or<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- reverse attempt-all [identity](../../../../glossary.md#term-identity-meanings)-unique [rollback](../../../../glossary.md#term-rollback) cannot preserve the exact primary failure.

## Local decisions

- `BackendPartitionFinalizationResult` remains a passive immutable value: it snapshots and
  validates list structure but leaves identity uniqueness to the complete-set handoff, which can
  enforce the rule across and within all results while recording only the first ownership
  occurrence.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Shared [rollback](../../../../glossary.md#term-rollback) support remains package-private in
  `BackendPartitionFinalizationHandoff`; both the [handoff](../../../../glossary.md#term-handoff) and [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) use the same
  reverse attempt-all, primary/suppression, and self-suppression rules without adding a helper
  source file.
- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) construction stays inside `GraphPreparation`'s rollback boundary so a
  constructor failure transfers no ownership and closes every accepted resource.
- CPU uses the one-argument result constructor. Generated artifacts, worker groups, and OpenBLAS
  invocations remain cached or borrowed state rather than new persistent-resource ownership.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [argument glossary](../../../../glossary.md#term-argument); [artifacts glossary](../../../../glossary.md#term-artifact); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>

## Known limitations

- This task does not make Engine prepared handles close inward Runtime executions; Engine 0010 is
  still required.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- This task does not add an actual persistent backend resource. Metal 0002 remains the first
  planned consumer after the complete shared lifecycle chain.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- General [asynchronous](../../../../glossary.md#term-asynchronous) execution and [buffer](../../../../glossary.md#term-buffer)-pool policy remain deferred.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation evidence was supplied by the separate implementation context and reused because
  documentation context `01a0c347-0ae0-7613-bdb5-5725eba52331` changed Javadocs and Markdown only.
  The focused four Prepare classes passed; focused `CpuPartitionFinalizerTest` passed; focused
  `CpuOpenBlasRouteConformanceTest` passed; and the corrected full
  `./gradlew :modules:prepare:test :backends:cpu:test` passed. The repository-wide
  `./gradlew test` passed after the executable implementation. After the correction that moved
  construction of the package-private successful handoff <span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>`Result` inside
  `BackendPartitionFinalizationHandoff`'s [rollback](../../../../glossary.md#term-rollback) boundary and added within-result duplicate plus
  `Error`-primary coverage, the focused four [Prepare](../../../../glossary.md#prepare) classes and full
  `:modules:prepare:test` passed again. Initial public-SPI adaptation compile failures were
  corrected historical diagnostics, not final failures.<span data-glossary-navigation="true"> ([SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [compile glossary](../../../../glossary.md#compile))</span>
- The documentation context independently reviewed all six production/Javadoc paths, the final
  four Prepare test classes, the focused CPU finalizer test and mechanical CPU/conformance
  callers, ADR 0013, Runtime 0016, the focused architecture boundary, both API pages, backend
  guide, glossary, Prepare/Engine/Metal master plans, roadmap, and this task against the General,
  API/Javadoc, Architecture, Backend Guide, and Planning profiles.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [Planning glossary](../../../../glossary.md#term-planning))</span>
- `./gradlew :modules:prepare:javadoc :backends:cpu:javadoc` passed after final Javadoc edits.
  CPU Javadoc reported 94 existing missing-`@param` warnings in unrelated lowering/preparation
  plan constructors plus the two expected incubating-module warnings. Generated pages for<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [module glossary](../../../../glossary.md#term-module))</span>
  `BackendPartitionFinalizationResult`, `BackendPartitionFinalizer`, [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), the
  Prepare package, and `CpuPartitionFinalizer` were inspected and contain the required ownership,
  acquisition order, rollback, no-lookup/no-allocation, transfer, and empty-resource language.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- `python3 /tmp/validate_synaptik_markdown.py` over the exact ten documentation paths passed with
  `validated 10 Markdown files`, covering local links and heading anchors, unique effective
  anchors, balanced fences, final newlines, and trailing whitespace.
- Status and [scope](../../../../glossary.md#term-scope) inspection confirmed Prepare 0006 and its master row are `Complete`; Engine
  0010 remains `Draft` with no task file and is the next frontier; [Metal](../../../../glossary.md#term-metal) 0002 remains `Blocked`
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
  behavior changes, so none required an edit or repeated suite.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lease glossary](../../../../glossary.md#term-lease); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [Tensor glossary](../../../../glossary.md#tensor); [Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [architecture test glossary](../../../../glossary.md#term-architecture-test); [integration test glossary](../../../../glossary.md#term-integration-test))</span>

## Implementation notes

- Added the public immutable finalization result and changed the finalizer SPI to return its exact
  executable together with an acquisition-ordered resource snapshot.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Extended the complete-set handoff to concatenate resources in partition/result order, enforce
  exact-identity uniqueness, keep one ownership occurrence, and roll back after every later
  handoff failure with deterministic failure composition.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [partition glossary](../../../../glossary.md#partition); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Extended graph preparation to own a successful handoff through schedule assembly, validation,
  and final aggregate construction, then transfer the exact ordered resources once to Runtime.<span data-glossary-navigation="true"> ([graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [validation glossary](../../../../glossary.md#term-validation); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Adapted CPU and its direct callers to the resource-free result without changing route,
  executable, allocation, binding, execution, or cleanup behavior.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [binding glossary](../../../../glossary.md#term-binding))</span>
- Finalized Javadocs, Runtime/Public API explanations, the focused architecture boundary, backend
  guidance, glossary, and planning status with implemented-versus-planned boundaries.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented the persistent prepared-resource finalization
  transaction, identity-unique rollback, final Runtime transfer, and resource-free CPU adaptation.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [identity glossary](../../../../glossary.md#term-identity-meanings); [rollback glossary](../../../../glossary.md#term-rollback); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Files changed or created: Exactly the twenty-six paths listed in the task allowlist.
- Tests and validation: Reused the stable focused Prepare, CPU finalizer, backend-conformance,
  combined Prepare/CPU, corrected Prepare, and repository-wide implementation evidence; passed
  final Prepare/CPU Javadoc, generated-page inspection, ten-file Markdown validation, status/file-
  absence/scope checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: Complete in clean context
  `01a0c347-0ae0-7613-bdb5-5725eba52331`; no executable Java behavior changed.
- Documentation impact: Updated both API pages, the focused architecture boundary, backend guide,
  glossary, task, three master plans, and roadmap from planned Prepare handoff to current behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Javadoc review: Finalized all six affected production contracts for ownership, snapshot/null
  semantics, local/shared rollback, identity uniqueness, failure suppression, successful
  transfer, CPU empty resources, and absence of lookup or shared physical allocation.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Glossary impact: Updated backend finalization, graph preparation, prepared execution, and
  prepared resource entries; no redundant carrier-only term was added.<span data-glossary-navigation="true"> ([graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource))</span>
- Unresolved issues: None within [Prepare](../../../../glossary.md#prepare) 0006.
- Follow-up required: [Engine](../../../../glossary.md#term-engine) 0010, then [Metal](../../../../glossary.md#term-metal) 0002.

Status: Complete
