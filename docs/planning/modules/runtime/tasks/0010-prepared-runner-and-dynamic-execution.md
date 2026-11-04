# Task 0010: Prepared Runner and Dynamic Execution

> **Editorial historical reading key.**
>
> “Dynamic execution” here means consuming prepared occurrences for one invocation, not a promise of
> symbolic shapes or runtime backend selection. `PreparedExecutionRunner.run` accepts a prepared
> recipe and dense borrowed representations, creates one isolated state, and binds every action before
> traversal. Backend executable declarations provide read/write authority; Runtime checks reads before
> invalidating every copy of each output buffer, then validates only declared writes after success.
> Backend actions own physical computation, not Runtime validity or resource cleanup. A post-creation
> failure leaves cleanup with the runner; successful result construction transfers the whole-state
> lease.
>
> See the [current runner
> contract](../../../../reference/runtime/orchestration.md#current-prepared-runner) for present
> orchestration. The task's unchanged, close-free aggregate is historical: the [current prepared
> owner](../../../../reference/runtime/prepared-execution.md#current-prepared-execution) adds
> persistent resources and lease-controlled closure. No physical rollback follows from logical
> invalidation.

## Status

Complete

## Goal

Make the existing [immutable](../../../../glossary.md#term-immutable) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) recipe runnable through one narrow [Runtime](../../../../glossary.md#term-runtime)-owned
runner:

```text
caller inputs -> one RunState -> cold-bound direct occurrences -> ordered traversal
              -> publication -> RunResult lease or failure cleanup
```

Preserve `PreparedExecution` exactly as [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) plus [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule). Add explicit
read/write meaning to current executable [buffer](../../../../glossary.md#term-buffer) selections so Runtime can validate reads,
invalidate stale output copies before invocation, and validate exact written copies only after
success.

## [Scope](../../../../glossary.md#term-scope)

- Add public stateless `runtime.run.PreparedExecutionRunner` for one complete [synchronous](../../../../glossary.md#term-synchronous) [run](../../../../glossary.md#term-run).
- Hand dense [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) caller inputs to existing package-private `RunStateCreation`.
- Permit no creation [occurrence](../../../../glossary.md#term-occurrence) only for a completely empty memory plan and empty caller list.
- Cold-bind every executable, transfer, and [publication occurrence](../../../../glossary.md#publication-occurrence) before the first action.
- Traverse a private direct-reference bound-step array in schedule encounter order.
- Add [immutable](../../../../glossary.md#term-immutable) `READ_ONLY`, `WRITE_ONLY`, and `READ_WRITE` access declarations aligned with
  `PreparedExecutable.BufferSelection` occurrences.
- Before invocation, validate declared reads, then invalidate every copy of every output [buffer](../../../../glossary.md#term-buffer).
- After success, validate only exact declared written representations; after failure, leave all
  output copies invalid.
- Preserve transfer, [publication](../../../../glossary.md#term-publication), and whole-state [`RunResult`](../../../../glossary.md#run-result--runresult) contracts unchanged.
- Close the one state after any post-creation failure, preserving the original unchecked failure
  and suppressing cleanup failure.
- Cover empty schedules/results, repetition, read/write overlap, [aliasing](../../../../glossary.md#term-aliasing), and run isolation.
- Finalize Javadocs/documentation in the mandatory separate clean documentation context.
- Synchronize only this task, the [Runtime](../../../../glossary.md#term-runtime) master plan, and the roadmap during [planning](../../../../glossary.md#term-planning).

## Out of [scope](../../../../glossary.md#term-scope)

- [graph](../../../../glossary.md#graph) inspection; [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning), [Model](../../../../glossary.md#term-model), or [Prepare](../../../../glossary.md#prepare) [identities](../../../../glossary.md#term-identity-meanings)
- [backend](../../../../glossary.md#backend) discovery, registration, service [lookup](../../../../glossary.md#term-lookup), [lowering](../../../../glossary.md#lowering), route/[kernel](../../../../glossary.md#kernel) search, or fallback
- dynamic allocation, representation lookup, maps, reflection, string dispatch, boxing lookup,
  synchronization, retry, branching, or parallel scheduling in the hot traversal<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- physical allocation/access, [lazy](../../../../glossary.md#term-lazy-evaluation) creation, eviction, [pooling](../../../../glossary.md#term-pooling), [alias](../../../../glossary.md#aliasing-and-arena) discovery, or coherence
- public [Engine](../../../../glossary.md#term-engine) or [Tensor](../../../../glossary.md#tensor)/value result access, conversion, download, export, or [host storage](../../../../glossary.md#host-storage)
- Prepare translation/orchestration; Config policy or `RunOptions`
- new [Trace](../../../../glossary.md#trace) payloads, events, sinks, emission, profiling, or clocks
- concrete backend behavior; dependency, Gradle, architecture, ADR, architecture-test,
  conformance, or integration changes<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changing [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), `RunStateCreation`, [`RunResult`](../../../../glossary.md#run-result--runresult), transfer, or
  publication contracts<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- [Runtime](../../../../glossary.md#term-runtime) 0011 or Prepare 0003 implementation/specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [ADR 0006](../../../../design/decisions/0006-no-runtime-service-locator.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Runtime master plan](../master-plan.md)
- completed [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md),
  [0004](0004-prepared-executable-and-bound-invocation.md),
  [0005](0005-prepared-schedule-contract.md), [0006](0006-prepared-execution-aggregate.md),
  [0007](0007-representation-creation-and-residency-foundation.md),
  [0008](0008-prepared-buffer-transfer-and-materialization-schedule.md), and
  [0009](0009-publication-and-result-schedule-steps.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>, [Trace 0001](../../trace/tasks/0001-core-trace-event-envelope.md),
  and [Trace 0002](../../trace/tasks/0002-model-correlation-identifiers.md)
- [Prepare master plan](../../prepare/master-plan.md) and [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Prepared recipes remain [immutable](../../../../glossary.md#term-immutable)/reusable; every call receives exactly one distinct [mutable](../../../../glossary.md#term-mutable)
  [`RunState`](../../../../glossary.md#run-state--runstate) for the complete [heterogeneous](../../../../glossary.md#term-heterogeneous) [run](../../../../glossary.md#term-run).
- Runtime owns caller handoff, state creation, binding, validity, traversal, cleanup, publication,
  and result lease. Backends own physical representations, callbacks, direct typed actions, and
  physical mechanics.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [binding glossary](../../../../glossary.md#term-binding); [lease glossary](../../../../glossary.md#term-lease))</span>
- The runner belongs to `runtime.run`, which already owns package-private state creation,
  validity, publication, result leasing, and cleanup.
- All occurrence binding finishes before traversal. The hot loop uses direct bound references and
  precomputed primitive coordinates only.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Read validity is checked before output invalidation, preserving in-place read/write input.
- Every copy of an output buffer is invalidated before backend work because a prepared slot may
  now represent a new logical value. Success validates exact declared writes only.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- Failure after possible output mutation leaves all copies of that output invalid and performs no
  physical rollback or retry.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Transfer and [publication](../../../../glossary.md#term-publication) retain their completed explicit, no-fallback transitions.
- The runner consumes no Trace Java. Trace 0001–0002 are preserved; Trace 0006 run payloads remain
  Draft, so this task invents and emits nothing.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Runtime](../../../../glossary.md#term-runtime) stays independent of [Model](../../../../glossary.md#term-model), [Planning](../../../../glossary.md#term-planning), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), and concrete [backends](../../../../glossary.md#backend).
- Stop if implementation needs graph facts, physical alias policy, a new module edge, mutable
  prepared state, a third <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [alias glossary](../../../../glossary.md#aliasing-and-arena); [module glossary](../../../../glossary.md#term-module))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) component, or an architecture change.

## Package impact

Existing packages changed:

- `runtime.execution` — explicit access declarations on the existing executable recipe.
- `runtime.run` — complete dynamic runner beside its existing [lifecycle](../../../../glossary.md#lifecycle) contracts.

`runtime.schedule` is consumed unchanged. No package is added, moved, or removed.

Type placement:

- `PreparedExecutable.BufferAccess` is nested because it describes one executable selection.
- [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner) is the narrow public Runtime integration seam for later explicit
  Engine composition; it is not a service, manager, registry, or module-root facade.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [registry glossary](../../../../glossary.md#term-registry); [module glossary](../../../../glossary.md#term-module); [facade glossary](../../../../glossary.md#term-facade))</span>
- Bound-step implementations remain private runner details, not another schedule [model](../../../../glossary.md#term-model).

## Exact [API](../../../../glossary.md#term-application-programming-interface-api) additions

Extend [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) compatibly:

```java
public enum BufferAccess { READ_ONLY, WRITE_ONLY, READ_WRITE }

protected PreparedExecutable(
        PreparedMemoryPlan memoryPlan,
        List<PreparedExecutable.BufferSelection> bufferSelections,
        List<PreparedExecutable.WorkspaceSelection> workspaceSelections,
        List<PreparedExecutable.BufferAccess> bufferAccesses);

public final int bufferSelectionCount();
public final PreparedExecutable.BufferSelection bufferSelection(int selectionIndex);
public final PreparedExecutable.BufferAccess bufferAccess(int selectionIndex);
```

The existing three-[argument](../../../../glossary.md#term-argument) constructor remains and assigns `READ_ONLY` to every [buffer](../../../../glossary.md#term-buffer)
selection. Existing [binding](../../../../glossary.md#term-binding), hooks, records, and failures remain unchanged.

Add exactly:

```java
package io.github.pho001.synaptik.runtime.run;

public final class PreparedExecutionRunner {
    public PreparedExecutionRunner();
    public RunResult run(
            PreparedExecution execution,
            List<BufferRepresentation> callerInputs);
}
```

The runner has no field, interface, builder, factory, overload, nested public type, configuration,
[trace](../../../../glossary.md#trace) input, or output accessor.

## [Prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) access contract

- `READ_ONLY` requires the selected copy valid and declares no write.
- `WRITE_ONLY` requires no old value and declares an exact successful write.
- `READ_WRITE` requires the old selected copy valid and declares it written.

The four-[argument](../../../../glossary.md#term-argument) constructor validates in this exact order:

1. require `memoryPlan`, `bufferSelections`, `workspaceSelections`, then `bufferAccesses` non-null;
2. preserve existing [buffer](../../../../glossary.md#term-buffer) [validation](../../../../glossary.md#term-validation)/[snapshot](../../../../glossary.md#term-snapshot), then workspace validation/snapshot;
3. require access count equal buffer-selection count; and
4. reject the first null access and snapshot access order.

New failures are:

- `NullPointerException("bufferAccesses")`;
- `IllegalArgumentException("bufferAccesses size must equal buffer selection count N")`; and
- `NullPointerException("bufferAccesses[i]")`.

The old constructor must preserve its prior top-level and indexed validation order/messages.
Empty lists and repeated selections remain valid. Accessors allocate nothing and retain exact
selection/access values. Either indexed accessor rejects an invalid position with
`IndexOutOfBoundsException("selectionIndex out of range: X")`.

## Runner [validation](../../../../glossary.md#term-validation) and [lifecycle](../../../../glossary.md#lifecycle) order

[`run`](../../../../glossary.md#term-run) acts exactly as follows:

1. require `execution`, then `callerInputs` non-null;
2. if schedule position zero is a creation [occurrence](../../../../glossary.md#term-occurrence), call existing `RunStateCreation.create`;
3. otherwise require both prepared buffer/workspace counts and caller-input count zero, then
   construct one empty state against the exact plan;<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
4. bind every remaining schedule occurrence in encounter order, snapshotting direct bound steps
   and bound publications before any action;
5. traverse bound steps in exact encounter order;
6. construct [`RunResult`](../../../../glossary.md#run-result--runresult) from the exact state and dense bound-[publication](../../../../glossary.md#term-publication) order; and
7. return without closing the leased state.

Runner-owned failures:

- `NullPointerException("execution")`;
- `NullPointerException("callerInputs")`;
- `IllegalArgumentException("non-empty prepared memory plan requires a representation creation occurrence")`;
- `IllegalArgumentException("callerInputs size must equal caller-input preparation count 0")`; and
- `IllegalStateException("executable buffer selection i requires a valid input representation")`
  for the first invalid read in original selection order.

Existing creation/[binding](../../../../glossary.md#term-binding)/action/result failures propagate unchanged. Schedule construction
already proves exact plan [identity](../../../../glossary.md#term-identity-meanings), first-only creation, and the dense publication suffix; the
runner must not duplicate that prepared validation.

## Cold [binding](../../../../glossary.md#term-binding) and hot traversal

Cold binding produces one private bound object per executable, transfer, or [publication](../../../../glossary.md#term-publication). An
executable bound object retains its direct [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) plus primitive arrays for:

- reads with original selection indices;
- distinct output [buffer](../../../../glossary.md#term-buffer) indices in first-write encounter order; and
- distinct written buffer/representation pairs in first-write encounter order.

Use bounded primitive [linear](../../../../glossary.md#term-linear-meanings) deduplication only—no map, set, boxing, reflection, or prepared-step
retention for hot redispatch. Transfer/publication objects retain one direct
[`BoundBufferTransfer`](../../../../glossary.md#bound-buffer-transfer--boundbuffertransfer)/[`BoundPublication`](../../../../glossary.md#bound-publication--boundpublication). Binding may allocate bound objects and primitive arrays
but acquires no closeable auxiliary resource.

Each bound executable action:

1. queries reads in original selection order, failing before mutation/work on the first invalid;
2. invalidates every resident representation of every distinct output buffer;
3. calls `BoundInvocation.execute()` once; and
4. after normal return, validates every distinct exact write [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix).

A [backend](../../../../glossary.md#backend) `RuntimeException` or `Error` propagates unchanged and leaves every output copy invalid.
Repeated reads remain ordered. Repeated writes coalesce. Multiple declared written copies of one
buffer all become valid after success. For input/output overlap, [Runtime](../../../../glossary.md#term-runtime) validates the old input,
then invalidates logical copies while the bound backend retains its direct physical references,
and validates the written copy only after success. Runtime performs no cross-buffer physical [alias](../../../../glossary.md#aliasing-and-arena)
discovery.

After traversal begins, allocate nothing and perform no list/map/boxing [lookup](../../../../glossary.md#term-lookup), compatibility
cast, resource resolution, backend discovery, [graph](../../../../glossary.md#graph) inspection, route/[kernel](../../../../glossary.md#kernel) search, config/tuning,
or [Trace](../../../../glossary.md#trace) emission.

## Result and failure cleanup

- Transfers and publications execute exactly at their bound schedule positions with unchanged
  semantics.
- Bound publications are collected in dense suffix order; an empty suffix creates a zero-result
  lease.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease))</span>
- Empty plan/schedule/result without creation is valid. A non-empty plan requires creation. A
  creation-only or zero-publication run returns a zero-result lease.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Successful [`RunResult`](../../../../glossary.md#run-result--runresult) construction alone transfers the whole-state close obligation.
- Any [binding](../../../../glossary.md#term-binding), traversal, [publication](../../../../glossary.md#term-publication), or result-construction failure after state creation calls
  `RunState.close()` exactly once. Rethrow the original unchecked failure unchanged and attach a
  distinct cleanup failure as suppressed; avoid self-suppression. Construct no result.
- A `RunStateCreation` failure already owns [rollback](../../../../glossary.md#term-rollback); propagate it without another cleanup pass.
- Separate calls share immutable recipes only. Their state, validity, bound objects, publication
  flags, created resources, workspaces, results, and cleanup remain isolated.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>

## Performance and side effects

- The runner is stateless/[thread-safe](../../../../glossary.md#term-thread-safe); one call is [synchronous](../../../../glossary.md#term-synchronous) and single-orchestrator-thread.
- Cold setup is linear in schedule/selections/representation counts, with bounded quadratic
  primitive output deduplication inside one executable.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Hot traversal is one dense direct-object array pass; executable transitions use primitive
  coordinates and <span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>[`RunState`](../../../../glossary.md#run-state--runstate) arrays.
- The runner itself performs no physical allocation/access. Prepared backend callbacks/actions do
  the physical work already authorized by completed contracts.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Caller inputs remain borrowed. Created buffers/workspaces remain run-owned until failure cleanup
  or the returned result closes the state.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [buffers glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run))</span>

## Affected files

Expected [Runtime](../../../../glossary.md#term-runtime) production/[Javadoc](../../../../glossary.md#term-javadoc):

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutable.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/PreparedExecutionRunner.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/package-info.java`

Expected Runtime tests:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutableTest.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/PreparedExecutionRunnerTest.java`

Expected explanatory documentation:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning):

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless contradicted: all mandatory architecture/planning/docs rules; completed Runtime
0001–0009; [Trace](../../../../glossary.md#trace) master/0001–0002/source; [Prepare](../../../../glossary.md#prepare) master/0001–0002 and Draft 0003 row; [Backend](../../../../glossary.md#backend)
Contract, Config, and [Engine](../../../../glossary.md#term-engine) master plans; current Runtime source/tests/Javadocs; [Compile](../../../../glossary.md#compile)/[Tensor](../../../../glossary.md#tensor)/
Training APIs; builds; architecture, [conformance](../../../../glossary.md#term-conformance), and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 14 paths: 4 [Runtime](../../../../glossary.md#term-runtime) production/[Javadoc](../../../../glossary.md#term-javadoc), 2 Runtime tests, 5 explanatory documents, and 3
[planning](../../../../glossary.md#term-planning) documents. No Java/test outside Runtime or any [Gradle](../../../../glossary.md#term-gradle), architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/test, [Prepare](../../../../glossary.md#prepare),
Config, [Trace](../../../../glossary.md#trace), [Backend](../../../../glossary.md#backend) Contract, [Engine](../../../../glossary.md#term-engine), concrete-backend, [conformance](../../../../glossary.md#term-conformance), or integration path may
change. Stop if another type, package, edge, behavior [owner](../../../../glossary.md#term-owner), or path is required. Do not create a
later specification.

## Acceptance criteria

- Exact [API](../../../../glossary.md#term-application-programming-interface-api)/package surface above; [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) remains [bytecode](../../../../glossary.md#term-bytecode)-exactly two components.
- Old executable constructor behavior is preserved; new access validation and immutable inspection
  match exact order/messages.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- One isolated state; all-bound-before-hot order; direct primitive bound representation.
- Exact read, full-output invalidation, success-only write, failure, overlap, duplicate, and
  multi-copy semantics.
- Exact transfer/[publication](../../../../glossary.md#term-publication) order and zero/aliased result [lease](../../../../glossary.md#term-lease) behavior.
- Every post-state failure closes once, preserves original identity, suppresses cleanup, skips
  borrowed inputs, and produces no result.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Repeated/concurrent calls share no [mutable](../../../../glossary.md#term-mutable) [run state](../../../../glossary.md#run-state--runstate)/resource.
- No hot allocation/lookup/dispatch mechanism forbidden above and no forbidden imports/Trace
  emission.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [Trace glossary](../../../../glossary.md#trace))</span>
- No Config/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine)/[backend](../../../../glossary.md#backend)/[dependency](../../../../glossary.md#term-dependency)/architecture behavior or public output access.
- Detailed Javadocs cover inputs/results/nullability, identity, access, side effects, ownership,
  cleanup, failures, concurrency, performance, and exclusions.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Mandatory clean documentation pass completes in the same change.
- Exact 14 paths; Runtime 0001–0009 and Prepare 0001–0002 stay Complete; Runtime 0011 and Prepare
  0003 stay Draft without specs; documentation/whitespace gates pass.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutableTest \
  --tests io.github.pho001.synaptik.runtime.run.PreparedExecutionRunnerTest
```

Exactly one final affected-[module](../../../../glossary.md#term-module) command after Java stabilizes:

```bash
./gradlew :modules:runtime:test
```

Documentation pass, reusing successful Java evidence unless executable Java changes:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/writing-a-backend.md docs/glossary.md \
  docs/planning/modules/runtime/tasks/0010-prepared-runner-and-dynamic-execution.md \
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md
git diff --check
```

If needed, create an equivalent temporary Markdown validator outside the repository. Validate
local links/anchors, unique effective anchors, fences, final newlines, and trailing whitespace.
Also inspect exact source/compiled/reflection surfaces, old/new validation, creation/[binding](../../../../glossary.md#term-binding)/action
order, direct fields/primitive arrays, hot [bytecode](../../../../glossary.md#term-bytecode), validity/overlap, cleanup/suppression,
empty/[alias](../../../../glossary.md#aliasing-and-arena)/isolation behavior, imports/mechanisms, unchanged completed surfaces/builds/Java 26,
exact [scope](../../../../glossary.md#term-scope), status, and later-spec absence.

Repository-wide and architecture validation is deferred to [Runtime](../../../../glossary.md#term-runtime) 0011, [CI](../../../../glossary.md#term-continuous-integration-ci), or the prepared-
execution [checkpoint](../../../../glossary.md#term-checkpoint). No module edge, shared build, architecture rule, [concrete backend](../../../../glossary.md#concrete-backend), public
[Engine](../../../../glossary.md#term-engine), or end-to-end consumer changes. [Conformance](../../../../glossary.md#term-conformance)/integration remain inapplicable.

## Mandatory clean documentation [handoff](../../../../glossary.md#term-handoff)

After the final [Runtime](../../../../glossary.md#term-runtime) test, hand this task, actual diff, affected [API](../../../../glossary.md#term-application-programming-interface-api)/behavior, architecture
constraints, exact 14 paths, and Java evidence to a separate clean documentation-focused agent.
It must read `AGENTS.md`, `ARCHITECTURE.md`, documentation rules, General/API-[Javadoc](../../../../glossary.md#term-javadoc)/Architecture/
[Backend](../../../../glossary.md#backend)-Guide/[Planning](../../../../glossary.md#term-planning) profiles, final source/tests, and actual diff; finalize both affected
packages/classes and five explanatory docs; inspect generated runner/access Javadocs; validate
examples, glossary, links/anchors/fences/newlines/whitespace/[scope](../../../../glossary.md#term-scope)/status; and record exact evidence
and limitations without repeating stable Java tests.

Required no-change conclusions: architecture/ADRs/general [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency); unchanged
Runtime memory/resource/schedule/state-creation/transfer/[publication](../../../../glossary.md#term-publication)/result contracts; [Trace](../../../../glossary.md#trace) (no
[run](../../../../glossary.md#term-run) [DTO](../../../../glossary.md#data-transfer-object--dto)); [Prepare](../../../../glossary.md#prepare) 0003; [Compile](../../../../glossary.md#compile)/[Tensor](../../../../glossary.md#tensor)/Training APIs; Config 0007; Backend Contract, [Engine](../../../../glossary.md#term-engine),
concrete backends, other guides, builds, architecture tests, [conformance](../../../../glossary.md#term-conformance), and integration.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0001–0009 and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0011 — Complete/accepted.
- Runtime 0004 supplies checked direct executable binding; 0007 supplies creation/validity; 0008
  transfer; 0009 publication/result lease.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [lease glossary](../../../../glossary.md#term-lease))</span>
- Trace 0001–0002 are stable preservation constraints only. Trace 0006 run payloads are Draft, so
  Runtime consumes no Trace Java.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Existing Runtime dependencies and Java 26 build remain unchanged.

[Prepare](../../../../glossary.md#prepare) 0003, Runtime 0011, Config 0007, [Trace](../../../../glossary.md#trace) 0003–0008, [Engine](../../../../glossary.md#term-engine), concrete [backends](../../../../glossary.md#backend), output access,
[conformance](../../../../glossary.md#term-conformance)/integration, and tuning are not dependencies.

## Follow-up tasks

- [Runtime](../../../../glossary.md#term-runtime) 0011: Draft contract closure and prepared-execution [checkpoint](../../../../glossary.md#term-checkpoint).
- [Prepare](../../../../glossary.md#prepare) 0003: Draft translation/orchestration and complete candidate [validation](../../../../glossary.md#term-validation).
- [Trace](../../../../glossary.md#trace) 0006: later typed passive [run](../../../../glossary.md#term-run) payloads after [producer](../../../../glossary.md#term-producer) assessment.
- Config 0007: later declarative run/[publication](../../../../glossary.md#term-publication) policy.
- Concrete [backends](../../../../glossary.md#backend): physical creators/executables/transfers and [conformance](../../../../glossary.md#term-conformance).
- [Engine](../../../../glossary.md#term-engine): public [Tensor](../../../../glossary.md#tensor)/value composition over this Runtime seam.

Do not create any follow-up detailed specification.

## Decisions and rejected alternatives

- Runner in `runtime.run`: source [ownership](../../../../glossary.md#term-ownership-meanings) avoids [widening](../../../../glossary.md#term-widening)/duplicating `RunStateCreation`.
- Public stateless instance: narrow later [Engine](../../../../glossary.md#term-engine) composition, not static global/[service locator](../../../../glossary.md#term-service-locator).
- Keep [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) unchanged: schedule already reaches every recipe.
- Explicit aligned access enum: current selections cannot safely infer roles from order, validity,
  graph facts, or physical types.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Old constructor is all-read-only: source-compatible and fail-closed for fresh invalid outputs;
  writers must opt in.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Validate reads before output invalidation for in-place input/output overlap.
- Invalidate all output-[buffer](../../../../glossary.md#term-buffer) copies before work; validate exact writes after success only.
- Bind everything before execution so a late compatibility failure precedes all physical effects.
- Private direct bound steps/primitive arrays instead of schedule reinterpretation or maps.
- Require creation for non-empty memory; permit the established truly empty [run](../../../../glossary.md#term-run).
- [Lease](../../../../glossary.md#term-lease) only after full success; no [publication](../../../../glossary.md#term-publication) [rollback](../../../../glossary.md#term-rollback)/individual detachment.
- Preserve [Trace](../../../../glossary.md#trace) rather than invent generic/string payloads before Trace 0006.

## Known limitations

- Old-constructor writers must adopt explicit access [metadata](../../../../glossary.md#term-metadata).
- [Runtime](../../../../glossary.md#term-runtime) trusts [backend](../../../../glossary.md#backend) access declarations and cannot verify bytes; backend tests must.
- Conservative invalidation may discard a physically intact old copy after failure.
- [Synchronous](../../../../glossary.md#term-synchronous) sequential runner only; no [cancellation](../../../../glossary.md#term-cancellation), retry, timeout, parallelism, or partial result.
- All representations remain resident until state/result closure; whole-state lease retains
  internal resources and borrowed-input lifetime obligations.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [`RunResult`](../../../../glossary.md#run-result--runresult) exposes no value; [Prepare](../../../../glossary.md#prepare) completeness and [Engine](../../../../glossary.md#term-engine) access remain future work.
- No [Trace](../../../../glossary.md#trace) event, production backend, or Engine consumer exists yet.
- Repository/architecture/[conformance](../../../../glossary.md#term-conformance)/integration [validation](../../../../glossary.md#term-validation) remains deferred as specified.

## Documentation impact

- [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), and the affected execution/run package
  Javadocs become the precise API contract for access declarations, cold binding, traversal,
  validity transitions, cleanup, result lease, threading, and performance.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [binding glossary](../../../../glossary.md#term-binding); [lease glossary](../../../../glossary.md#term-lease))</span>
- Runtime API gains the current complete internal run mental model and a focused example covering
  caller handoff, read/write output production, publication, result closure, and failure cleanup.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [handoff glossary](../../../../glossary.md#term-handoff); [publication glossary](../../../../glossary.md#term-publication))</span>
- Public API lists the runner as a Runtime integration contract without claiming an Engine,
  Tensor/value result access, Config policy, concrete backend, or end-to-end user workflow.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Tensor glossary](../../../../glossary.md#tensor); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- The focused Runtime/Prepare/Backend page receives implementation-status/mechanics wording only;
  architecture authority and module ownership do not change.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Backend guidance documents correct immutable access declarations and direct bound fields; it
  must not suggest runtime discovery, hot allocation, or implicit coherence.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- The glossary adds or updates only reusable runner/access/validity terms; ordinary implementation
  words are not added.
- The clean documentation pass records the reasoned no-change conclusions required above and
  generates final Runtime Javadoc after its edits.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Architecture impact

Expected impact: None. The architecture already assigns prepared runner, validity, cleanup,
[publication](../../../../glossary.md#term-publication), and result [ownership](../../../../glossary.md#term-ownership-meanings) to [Runtime](../../../../glossary.md#term-runtime). Stop if implementation requires [graph](../../../../glossary.md#graph)-derived roles,
physical [alias](../../../../glossary.md#aliasing-and-arena) policy, another [module](../../../../glossary.md#term-module) edge, [Trace](../../../../glossary.md#trace) [schema](../../../../glossary.md#term-schema), [mutable](../../../../glossary.md#term-mutable) prepared state, public output
access, or a third aggregate component.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the focused Runtime architecture
and ADRs, documentation rules/profiles, Runtime/Trace/Prepare master plans, completed Runtime
0001–0009, this task, and directly relevant Runtime source/tests/Javadocs/docs.

Implement Runtime 0010 exactly within its 14-path ceiling. Preserve PreparedExecution as memory
plan plus schedule and every completed contract except the specified compatible executable-access
additions. Stop on architecture/package/API/validation/ownership/scope conflict. Do not add graph
inspection, discovery/search, hot allocation/lookup, output access, Prepare/Engine/Config, Trace
payload/emission, backend, dependency/Gradle/architecture, or later-spec work.

Run focused tests and exactly one final Runtime module test after Java stabilizes, then all exact
surface/mechanism/scope/status checks. Hand the actual diff/evidence to a separate clean docs
context, which must finalize Javadocs/docs/examples/glossary/planning, inspect generated pages,
record no-change conclusions, and not repeat stable Java tests. Mark Complete only after all gates.
Return context IDs, exact paths, commands/counts, conclusions, issues/follow-up, and status.
```

## Local decisions

- Exact package/[API](../../../../glossary.md#term-application-programming-interface-api)/order/access/validity/cleanup/[Trace](../../../../glossary.md#trace) decisions above are fixed.
- Private bound-step names may vary; direct retained facts and hot behavior may not.
- Bounded cold primitive deduplication is selected over boxing/maps.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/runtime0010_impl` completed the Java work. The focused command
  passed 26 tests; the single final `./gradlew :modules:runtime:test` command passed 17 suites and
  143 tests with no failures, errors, or skips.
- Reflection and `javap` preserved the exact two-component [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), confirmed the
  requested executable additions, and confirmed the runner's sole public constructor and [`run`](../../../../glossary.md#term-run)
  method with no field, interface, or public nested type.
- Source, import, field, and [bytecode](../../../../glossary.md#term-bytecode) inspection confirmed direct [`RunState`](../../../../glossary.md#run-state--runstate), bound-action, and
  primitive-array traversal without forbidden project imports, reflection, maps, sets, service
  lookup, hot compatibility casts, or hot allocation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Clean documentation context `/root/runtime0010_docs` finalized the Javadocs, five explanatory
  documents, and three planning documents without changing executable behavior or repeating Java
  tests. Runtime Javadoc, generated pages, eight-file Markdown validation, exact 14-path scope,
  status, and whitespace checks passed.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Runtime glossary](../../../../glossary.md#term-runtime); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Java 26 and build/dependency configuration remain unchanged. Repository-wide, architecture,
  conformance, and integration validation remains deferred as specified.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- The compatible constructor defaults selections to `READ_ONLY`; writers opt into aligned access.
- Cold setup binds every occurrence. Bound executables retain direct invocations and primitive
  coordinates. Reads precede conservative invalidation; only exact successful writes become
  valid. Failures leave output copies invalid and close the state.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- The runner is stateless and supports isolated concurrent calls; each run is synchronous and
  leases its complete state only after full success.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [synchronous glossary](../../../../glossary.md#term-synchronous); [leases glossary](../../../../glossary.md#term-lease))</span>
- No architecture or ADR changed. Existing lifecycle/module/dependency rules already assign this
  work to Runtime. Memory, resource, schedule, state-creation, transfer, publication, and result
  contracts remain unchanged.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication))</span>
- Trace has no current run DTO, so no emission was added. Prepare 0003 remains Draft without a
  specification. Compile, Tensor, Training, Config 0007, Backend Contract, Engine, concrete
  backends, other guides, builds, architecture tests, conformance, and integration need no change:
  this task adds no semantics, policy, backend implementation, edge, public value access, or
  end-to-end composition.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [DTO glossary](../../../../glossary.md#data-transfer-object--dto); [Prepare glossary](../../../../glossary.md#prepare); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [Backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Completion summary

- Completed changes: executable access declarations and stateless all-bound-before-traversal
  runner with exact validity, cleanup, result, and isolation behavior.
- Files changed or created: exactly the 14 production, test, explanatory, and [planning](../../../../glossary.md#term-planning) paths.
- Tests and validation: focused 26 tests and final Runtime 17 suites/143 tests passed; Javadoc,
  generated pages, Markdown, surface/mechanism, build, scope, status, and whitespace checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Runtime glossary](../../../../glossary.md#term-runtime); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `/root/runtime0010_docs` completed the independent
  finalization and required no-change review.
- Documentation impact: Runtime/Public APIs, boundary, backend guide, glossary, and planning now
  describe the implemented lifecycle and planned boundaries.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: access and runner/package contracts cover ordering, ownership, failure,
  concurrency, performance, and exclusions.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: executable access and the [prepared execution runner](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner) are defined.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
