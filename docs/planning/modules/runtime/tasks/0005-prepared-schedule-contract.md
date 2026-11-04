# Task 0005: [Prepared Schedule](../../../../glossary.md#prepared-schedule--preparedschedule) Contract

> **Editorial historical reading key.**
>
> At this milestone, Prepare’s finalized partition associations supply exact Runtime executable
> references; the schedule constructor accepts only their ordered `ExecutionStep` occurrences and one
> shared memory plan. Runtime validates exact plan identity and snapshots membership without binding,
> executing or acquiring resources. List position carries orchestration order: retaining one
> executable twice requests two occurrences, not two cleanup owners or two numerical implementations.
> Backend executables remain responsible for region work; a later runner was to bind them to per-run
> resources. The [current schedule
> contract](../../../../reference/runtime/schedules.md#current-prepared-schedule) extends this
> historical execution-only sealed family with creation, transfer and publication occurrences. The
> [current runner](../../../../reference/runtime/orchestration.md#current-prepared-runner) now
> cold-binds work before traversal and manages isolated state and failure cleanup. Neither development
> changes the archival constructor’s no-action meaning.

## Status

Complete

## Goal

Implement the smallest [immutable](../../../../glossary.md#term-immutable) [Runtime](../../../../glossary.md#term-runtime)-owned [prepared schedule](../../../../glossary.md#prepared-schedule--preparedschedule) recipe that orders already-
[prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) work for later cold consumption by a runner. One [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule) retains
one exact [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) and an immutable ordered [snapshot](../../../../glossary.md#term-snapshot) of Runtime-owned steps. The only
current step executes one exact [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) whose memory-plan reference is that same
schedule plan.

The exact public surface is:

```java
package io.github.pho001.synaptik.runtime.schedule;

public record PreparedSchedule(
        PreparedMemoryPlan memoryPlan,
        List<PreparedSchedule.Step> steps) {

    public sealed interface Step permits ExecutionStep {
        PreparedMemoryPlan memoryPlan();
    }

    public record ExecutionStep(PreparedExecutable executable) implements Step {
        @Override
        public PreparedMemoryPlan memoryPlan();
    }
}
```

`ExecutionStep.memoryPlan()` returns `executable.memoryPlan()` by exact reference. The [sealed](../../../../glossary.md#term-sealed) step
contract makes the schedule-wide plan [invariant](../../../../glossary.md#term-invariant) explicit and supplies a type-safe Runtime-owned
extension point for later architecture-required work. This task permits only `ExecutionStep`; it
does not guess transfer, [materialization](../../../../glossary.md#materialization), or [publication](../../../../glossary.md#term-publication) facts.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
Prepare-owned ordered finalized partitions
  -> extract their exact Runtime PreparedExecutable references
  -> construct Runtime ExecutionStep values
  -> construct one schedule against the exact shared PreparedMemoryPlan

later cold runner preparation
  -> traverse steps once in encounter order
  -> bind each executable to one exact-plan RunState
  -> retain dense direct BoundInvocation references

later hot execution
  -> invoke already-bound work in deterministic order
```

The schedule is a reusable prepared recipe, not per-[run state](../../../../glossary.md#run-state--runstate) or a hot-path interpreter. It
contains no [graph](../../../../glossary.md#graph), [partition](../../../../glossary.md#partition), [backend](../../../../glossary.md#backend)-selection, allocation, or physical-resource facts. A future
runner can consume typed values during cold setup without [`Operation`](../../../../glossary.md#operation), `CompiledNode`, backend
rediscovery, maps, reflection, generic payloads, or string [dispatch](../../../../glossary.md#term-dispatch).

`PreparedUnit` remains unnecessary. Current [Prepare](../../../../glossary.md#prepare) finalization associates a Prepare-owned
`PlannedPartition` with one [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable); Runtime cannot retain that association
without importing forbidden [Planning](../../../../glossary.md#term-planning)/Prepare facts. Within Runtime, an executable step adds only
ordered [occurrence](../../../../glossary.md#term-occurrence) and exact-plan [validation](../../../../glossary.md#term-validation). A separate wrapper has no additional [invariant](../../../../glossary.md#term-invariant) or
consumer.

## [Scope](../../../../glossary.md#term-scope)

- Add the exact one-top-level/two-nested public surface above in
  `io.github.pho001.synaptik.runtime.schedule`.
- Require one exact non-null [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) reference for the complete schedule.
- [Snapshot](../../../../glossary.md#term-snapshot) the supplied step list with `List.copyOf`, retaining exact element/executable references
  in supplied encounter order.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Require every step to report the same exact plan reference as the schedule.
- Represent current schedulable work only as `ExecutionStep(PreparedExecutable)`.
- Permit an empty schedule, including against a non-empty plan. Emptiness implies no publication
  or resource-lifecycle policy.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Permit repeated step or executable references. Each list position is one explicit execution
  occurrence and does not duplicate immutable prepared-resource ownership.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- Add focused tests for shape, validation, identity, immutability, order, emptiness, repetition,
  ownership, concurrency, and mechanism exclusions.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Finalize affected Javadocs and explanatory documentation in a separate clean documentation
  context.

## Out of [scope](../../../../glossary.md#term-scope)

- execution, executable [binding](../../../../glossary.md#term-binding), [`RunState`](../../../../glossary.md#run-state--runstate) construction, cold binding, or runner implementation
- resource allocation/representation creation, physical access, or [ownership](../../../../glossary.md#term-ownership-meanings) transfer
- transfer mechanics, [residency](../../../../glossary.md#residency)/validity/coherence, or a transfer step
- [materialization](../../../../glossary.md#materialization) mechanics or a materialization step
- [publication](../../../../glossary.md#term-publication)/result delivery, [`RunResult`](../../../../glossary.md#run-result--runresult), [targets](../../../../glossary.md#term-target)/policies, or a publication step
- `PreparedUnit`, [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), `PreparationResourceAssignment`,
  [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis), `PlannedPartition`, or [Prepare](../../../../glossary.md#prepare) orchestration
- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), its [lifecycle](../../../../glossary.md#lifecycle), persistent prepared resources, or [run](../../../../glossary.md#term-run) configuration
- concrete [backends](../../../../glossary.md#backend), routes, [kernels](../../../../glossary.md#kernel), native handles, backend configuration, or tuning/[cache](../../../../glossary.md#term-cache) work
- [Compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning) changes or [Runtime](../../../../glossary.md#term-runtime) representations of `PublicationPlan`, publication
  bindings, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [`TensorId`](../../../../glossary.md#tensorid), or `LogicalMemoryRequirement`
- [`Operation`](../../../../glossary.md#operation), `CompiledNode`, graph inspection, backend discovery/selection, fallback, service
  lookup, allocation, profiling selection, or trace emission<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup); [trace glossary](../../../../glossary.md#trace))</span>
- step IDs, stored indices, [dependency](../../../../glossary.md#term-dependency) edges, jumps, branches, barriers, retries, or priorities
- maps, registries, manager/service/util types, `Object` payloads, string dispatch, reflection,
  unchecked casts/generics, or another step abstraction<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Runtime 0006+, Prepare 0003, or another detailed task specification
- Gradle, dependency, architecture/ADR, architecture-test, backend-conformance, or integration-test
  changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Runtime master plan](../master-plan.md)
- [Runtime 0002](0002-prepared-memory-and-workspace-contracts.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)
- [Runtime 0004](0004-prepared-executable-and-bound-invocation.md)
- [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Schedule, steps, and executables are immutable reusable prepared state and may be consumed
  concurrently for distinct logical runs.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- The schedule owns only its immutable list snapshot. It does not own or close the plan, steps,
  executables, run states, representations, invocations, or backend resources.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [run glossary](../../../../glossary.md#term-run); [backend glossary](../../../../glossary.md#backend))</span>
- One schedule associates with exactly one [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) by reference identity. Value-equal
  geometry from another plan is not the same prepared association.<span data-glossary-navigation="true"> ([reference identity glossary](../../../../glossary.md#term-reference-identity))</span>
- Runtime production must not import Prepare, Planning, Compiler, Model, Engine, or concrete-
  backend types, including every forbidden type named in the goal and exclusions.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Prepare may construct Runtime schedule values by extracting Runtime executable references; that
  inward dependency does not authorize Runtime to retain Prepare associations.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- The sealed family is semantic type dispatch, not a payload registry. Runtime may add a variant
  only after stable Runtime-owned facts and a concrete consumer establish its invariant.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [dispatch glossary](../../../../glossary.md#term-dispatch); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [registry glossary](../../../../glossary.md#term-registry); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- A future runner must cold-bind execution occurrences to direct [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) references
  before its hot loop. Step traversal, allocation, backend discovery, graph inspection, and
  resource lookup must not move into <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup))</span>`BoundInvocation.execute()`.
- If another module edge, generic payload, or new ownership decision is required, stop and report
  an architecture conflict rather than expanding the task.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Package impact

Existing packages consumed without modification:

- `runtime.memory` supplies the exact [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) association.
- `runtime.execution` supplies [immutable](../../../../glossary.md#term-immutable) [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) recipes.

Added package:

- `runtime.schedule` owns the immutable schedule and closed current semantic step family.

[`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule) is the only top-level type. `Step` is nested because it is meaningful only as a
schedule entry; its sole method exposes the plan association required by schedule [validation](../../../../glossary.md#term-validation).
`ExecutionStep` is nested and final by record semantics. No package-private [API](../../../../glossary.md#term-application-programming-interface-api) or root [facade](../../../../glossary.md#term-facade) is
added.

## Exact [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape)

[`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule) is a public record with exactly these components, in order:

1. `PreparedMemoryPlan memoryPlan`
2. `List<PreparedSchedule.Step> steps`

Its compact [canonical](../../../../glossary.md#term-canonical-form) constructor applies the rules below. Generated accessors return the exact
plan and [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot). Add no factory, builder, overload, mutator, executor, index accessor,
array [view](../../../../glossary.md#term-view-meanings), equality override, or [serialization](../../../../glossary.md#term-serialization) contract.

`PreparedSchedule.Step` is a public nested [sealed](../../../../glossary.md#term-sealed) interface permitting exactly `ExecutionStep`.
It declares only `PreparedMemoryPlan memoryPlan()`. It has no ID, kind enum, [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), execution
method, visitor, or default method.

`ExecutionStep` is a public nested record with exactly one component,
`PreparedExecutable executable`. Its compact constructor rejects null. Its explicit
`memoryPlan()` returns `executable.memoryPlan()` without caching, copying, or equality conversion.
It adds no bind/execute behavior. Package [Javadoc](../../../../glossary.md#term-javadoc) describes immutable prepared scheduling recipes
and says that current scheduling contains executable occurrences only.

## [Validation](../../../../glossary.md#term-validation), order, and failure rules

`ExecutionStep` construction first and only requires `executable` non-null. The exact failure is
`NullPointerException("executable")`; success retains the exact reference.

[`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule) construction validates in this exact order:

1. require `memoryPlan` non-null;
2. require `steps` non-null;
3. [scan](../../../../glossary.md#term-cumulative-scan) steps in supplied order, rejecting the first null [element](../../../../glossary.md#term-element);
4. for each non-null step in that scan, require `step.memoryPlan() == memoryPlan`; and
5. after the complete scan succeeds, assign `steps = List.copyOf(steps)`.

Exact failures are:

- `NullPointerException("memoryPlan")`;
- `NullPointerException("steps")`;
- `NullPointerException("steps[i]")`; and
- `IllegalArgumentException("steps[i] memory plan does not match schedule memory plan")`.

Fail on the first invalid [occurrence](../../../../glossary.md#term-occurrence). Do not compare plans with `equals`, deduplicate, sort,
group, flatten, bind, or execute. Call `memoryPlan()` exactly once per encountered valid step
before the first failure. [Snapshot](../../../../glossary.md#term-snapshot) only after full validation, so construction fails closed.

Empty steps is valid. Repeated exact steps, equal steps, and executable references are valid and
preserve every occurrence. [Immutable](../../../../glossary.md#term-immutable) list encounter order is deterministic execution order.

List position is sufficient occurrence order. No current consumer needs branching, random access,
retry, correlation, or stable [identity](../../../../glossary.md#term-identity-meanings) across schedules. A later runner may derive a transient
encounter index for validation or [trace](../../../../glossary.md#trace) translation without storing identity here.

## Immutability, [ownership](../../../../glossary.md#term-ownership-meanings), thread safety, and performance

- Construction is linear in step count, performs one structural snapshot, and performs no
  physical resource operation.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot); [operation glossary](../../../../glossary.md#operation))</span>
- Exact [immutable](../../../../glossary.md#term-immutable) [elements](../../../../glossary.md#term-element) are retained; steps and executables are not cloned.
- Schedule and steps contain no [mutable](../../../../glossary.md#term-mutable) per-[run state](../../../../glossary.md#run-state--runstate), are not `AutoCloseable`, and transfer no
  cleanup responsibility.
- One schedule may be traversed concurrently for distinct run preparations. This does not make
  one future <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>[`RunState`](../../../../glossary.md#run-state--runstate) or [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) safe for concurrent use.
- Production contains no map, [registry](../../../../glossary.md#term-registry), reflection, class-name test, `Object` payload, string
  dispatch, service lookup, graph/backend reference, or identifier allocation.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Repeated work uses dense list occurrences, not [lookup](../../../../glossary.md#term-lookup) tables or IDs.
- This task adds no hot-path method. Future cold preparation retains direct typed invocation
  references; eventual execution performs no plan equality scan, slot lookup, backend selection,
  allocation, transfer/materialization decision, or publication binding.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding))</span>

## Transfer, [materialization](../../../../glossary.md#materialization), and [publication](../../../../glossary.md#term-publication) decision

These variants cannot be defined stably in this task:

- Transfer needs Runtime-owned source/destination representation and validity/residency facts.
  Current <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>[`RunState`](../../../../glossary.md#run-state--runstate) defines neither residency nor coherence, while executable selections are
  executable-local rather than transfer recipes.
- Materialization needs Runtime-owned source/result representation facts and a creation/copy
  consumer. [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) supplies geometry only; allocation, physical access, and copy
  mechanics are absent.
- Publication needs a Runtime association from prepared resources to a delivery target and
  ownership policy. Compiler publication bindings retain Compiler/Model identities and boundary
  membership only; they define no Runtime delivery target and cannot be imported by Runtime.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Compiler glossary](../../../../glossary.md#term-compiler); [bindings glossary](../../../../glossary.md#term-binding); [Model glossary](../../../../glossary.md#term-model))</span>

These remain separate Draft work after their foundations stabilize. The [sealed](../../../../glossary.md#term-sealed) step contract
reserves no later [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) beyond the exact schedule-plan association every future step must obey.

## Affected files

Expected [Runtime](../../../../glossary.md#term-runtime) production paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/PreparedSchedule.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/package-info.java`

Expected Runtime test path:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/schedule/PreparedScheduleTest.java`

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — status only; no rule change
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a contradiction is found: architecture/ADRs, Runtime 0001–0004, [Prepare](../../../../glossary.md#prepare)
0001–0002, all current Runtime and relevant Prepare production/tests, [Compiler](../../../../glossary.md#term-compiler) [publication](../../../../glossary.md#term-publication) and
Planning logical-memory contracts, current APIs/guides/glossary, build files, and architecture/
[conformance](../../../../glossary.md#term-conformance)/integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 11 paths: two [Runtime](../../../../glossary.md#term-runtime) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, one Runtime test path, five explanatory
documentation paths, and three [planning](../../../../glossary.md#term-planning) paths. No Java/test path outside Runtime, [Gradle](../../../../glossary.md#term-gradle),
[dependency](../../../../glossary.md#term-dependency), architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr), architecture-test, [conformance](../../../../glossary.md#term-conformance), or integration path may change.
Stop if another type, package, edge, behavior [owner](../../../../glossary.md#term-owner), or path is required. Do not create a later
detailed task specification.

## Test requirements

`PreparedScheduleTest` must cover:

- exact record/nested shape, modifiers, components, permitted subclass, method, constructors, and
  absence of extra public/package-private declarations;<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- exact step null failure and executable/plan reference retention;
- top-level null failures in required order;
- indexed first-null and first-plan-mismatch failures, exact messages, order, and fail-closed
  behavior;<span data-glossary-navigation="true"> ([fail-closed behavior glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- [reference identity](../../../../glossary.md#term-reference-identity) rather than equality using two value-equal distinct plans;
- structural [snapshot](../../../../glossary.md#term-snapshot) isolation from a [mutable](../../../../glossary.md#term-mutable) source list;
- exact [element](../../../../glossary.md#term-element) retention and deterministic order;
- valid empty schedule and repeated step/executable references;
- absence of [binding](../../../../glossary.md#term-binding), execution, resource action, [ownership](../../../../glossary.md#term-ownership-meanings) transfer, or closing at construction;
- concurrent read/traversal without mutation; and
- forbidden import, surface, and mechanism absence.

Tests may use minimal local [immutable](../../../../glossary.md#term-immutable) [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) subclasses only to create plan-
associated recipes. They must not implement a runner, bind/execute, allocate representations, or
define deferred semantics. No architecture, [conformance](../../../../glossary.md#term-conformance), or [integration test](../../../../glossary.md#term-integration-test) is added because no
[module](../../../../glossary.md#term-module) edge or executable [backend](../../../../glossary.md#backend)/end-to-end behavior changes.

## Acceptance criteria

- The exact one-top-level/two-nested surface exists only in `runtime.schedule`.
- `Step` permits exactly `ExecutionStep`; no other variant, kind, visitor, payload, or registry
  exists.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [registry glossary](../../../../glossary.md#term-registry))</span>
- Every schedule retains one exact plan and an immutable ordered snapshot whose steps report that
  same reference.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Validation order/messages, empty/repetition policy, and exact reference retention match this
  task.
- No `PreparedUnit`, step ID, or stored index exists.
- [Runtime](../../../../glossary.md#term-runtime) production imports no forbidden upstream or concrete type.
- Construction performs no binding, execution, allocation, representation creation, transfer,
  residency, materialization, publication, discovery, tuning, or tracing.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>
- Javadocs cover invariants, inputs, results, failures, ownership, reuse, and thread safety without
  promising deferred behavior.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The five explanatory docs and three planning docs consistently distinguish current executable
  scheduling from Draft later work.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Runtime 0001–0004 and Prepare 0001–0002 remain Complete; Runtime 0006+ and Prepare 0003 remain
  Draft without specifications.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- All [validation](../../../../glossary.md#term-validation), [scope](../../../../glossary.md#term-scope), and whitespace checks pass.

## [Validation](../../../../glossary.md#term-validation)

Validation tier: **2 — [module](../../../../glossary.md#term-module) and focused documentation validation**.

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest
./gradlew :modules:runtime:test
```

After Java stabilizes, the documentation context runs without repeating successful tests unless
it changes executable Java:

```bash
./gradlew :modules:runtime:javadoc
```

Also validate [Javadoc](../../../../glossary.md#term-javadoc) warnings, Markdown links/anchors/fences/final newlines, exact status and
surface, [sealed](../../../../glossary.md#term-sealed)-family reflection/import checks, forbidden mechanisms/dependencies, absence of a
later specification, the exact maximum changed-path set, and `git diff --check`. Repository-wide
validation is deferred to the prepared-[runtime](../../../../glossary.md#term-runtime) [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes no
[dependency](../../../../glossary.md#term-dependency)/build/architecture boundary or [concrete backend](../../../../glossary.md#concrete-backend)/end-to-end behavior.

## Dependencies

- Complete [Runtime 0002](0002-prepared-memory-and-workspace-contracts.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span> supplies plan geometry.
- Complete [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md) establishes the
  per-run exact-plan association a future runner consumes.
- Complete [Runtime 0004](0004-prepared-executable-and-bound-invocation.md) supplies the exact
  schedulable recipe and cold-bound invocation boundary.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- Complete [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  proves finalized executables retain one shared exact plan and preserves partition order.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- ADRs 0010 and 0011 establish staged preparation and [immutable](../../../../glossary.md#term-immutable)-prepared/per-[run](../../../../glossary.md#term-run) [ownership](../../../../glossary.md#term-ownership-meanings).

## Follow-up tasks

- [Runtime](../../../../glossary.md#term-runtime) 0006: Draft [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) aggregate and [lifecycle](../../../../glossary.md#lifecycle).
- Runtime 0007: Draft representation creation and [residency](../../../../glossary.md#residency)/validity foundation.
- Runtime 0008: Draft transfer and [materialization](../../../../glossary.md#materialization) steps after Runtime 0007.
- Runtime 0009: Draft [publication](../../../../glossary.md#term-publication)/result associations and step after delivery/[ownership](../../../../glossary.md#term-ownership-meanings) stabilizes.
- Runtime 0010: Draft runner, cold schedule consumption, and dynamic execution.
- Runtime 0011: Draft Runtime contract-closure audit.
- [Prepare](../../../../glossary.md#prepare) 0003: Draft orchestration and complete prepared-result [validation](../../../../glossary.md#term-validation).

No follow-up detailed specification may be created here.

## Decisions

- Use one top-level record with a nested sealed semantic family: this preserves known
  heterogeneous scheduling without inventing later payloads.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [heterogeneous glossary](../../../../glossary.md#term-heterogeneous))</span>
- Implement only `ExecutionStep`, the sole stable [Runtime](../../../../glossary.md#term-runtime)-owned work fact with a current [producer](../../../../glossary.md#term-producer).
- Omit `PreparedUnit`; it adds no [invariant](../../../../glossary.md#term-invariant) beyond executable and schedule [occurrence](../../../../glossary.md#term-occurrence).
- Use exact plan [reference identity](../../../../glossary.md#term-reference-identity) consistently with current Runtime/[Prepare](../../../../glossary.md#prepare) contracts.
- Use list position for occurrence order; add no [identity](../../../../glossary.md#term-identity-meanings) or stored index.
- Allow empty schedules and repeated executable occurrences without changing [ownership](../../../../glossary.md#term-ownership-meanings).
- Split transfer/materialization/publication from the runner frontier until their Runtime facts
  exist.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>

## Known limitations

- Current schedules describe executable occurrences only and cannot yet express required
  transfer, materialization, or publication work.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>
- No representation creation, input/result slot association, complete [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), public
  Prepare orchestration, runner, or Engine lifecycle exists.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Each later justified semantic variant requires an explicit [Runtime](../../../../glossary.md#term-runtime) source/specification change.

## Architecture impact

None. This implements existing [Runtime](../../../../glossary.md#term-runtime) schedule [ownership](../../../../glossary.md#term-ownership-meanings), preserves [dependency](../../../../glossary.md#term-dependency) rules, and adds no
[module](../../../../glossary.md#term-module) edge, architecture rule, or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) decision. Splitting under-founded variants into later Draft
tasks is [planning](../../../../glossary.md#term-planning) [decomposition](../../../../glossary.md#term-decomposition), not an architecture change.

## Implementation prompt

Execute in a fresh implementation context. Read `AGENTS.md`, `ARCHITECTURE.md`, the referenced
architecture/ADRs and [planning](../../../../glossary.md#term-planning) documents, documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/Example
profiles, [Runtime](../../../../glossary.md#term-runtime) 0002–0004, [Prepare](../../../../glossary.md#prepare) 0001–0002, current Runtime/relevant Prepare source and tests,
[Compiler](../../../../glossary.md#term-compiler) [publication](../../../../glossary.md#term-publication) and Planning logical-memory contracts, current APIs/guides/glossary, and
root/runtime/prepare [Gradle](../../../../glossary.md#term-gradle) files in full.

Implement exactly this surface, [validation](../../../../glossary.md#term-validation), [ownership](../../../../glossary.md#term-ownership-meanings), performance, [scope](../../../../glossary.md#term-scope), and exclusions. Do not
add `PreparedUnit`, another variant, runner, allocation, transfer, [materialization](../../../../glossary.md#materialization), publication,
upstream [identity](../../../../glossary.md#term-identity-meanings), generic [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), [registry](../../../../glossary.md#term-registry), [dependency](../../../../glossary.md#term-dependency)/architecture change, or later task spec.
Stop if another type, path, dependency, or ownership decision is required. After Java stabilizes,
use a distinct clean documentation context to finalize permitted Javadocs/docs. [Run](../../../../glossary.md#term-run) the specified
validation once, verify exact scope, and do not [commit](../../../../glossary.md#term-commit) or push.

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context `/root/runtime_0005_impl` ran
  `./gradlew :modules:runtime:test --tests
  io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest`: one suite and 11 tests passed
  with no skips, failures, or errors.
- Exact API, reflection, and bytecode inspection confirmed the two record shapes, the nested sealed
  family with only <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [bytecode glossary](../../../../glossary.md#term-bytecode); [sealed glossary](../../../../glossary.md#term-sealed))</span>`ExecutionStep` permitted, the exact constructors/methods, one
  `Step.memoryPlan()` call per encountered step, reference comparison, and snapshot creation only
  after a successful scan.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Source/import/mechanism scans confirmed the required validation messages and order, no forbidden
  upstream or concrete-backend imports, and no binding, execution, allocation, ownership,
  transfer, materialization, publication, registry, reflection, or generic-payload mechanism.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [binding glossary](../../../../glossary.md#term-binding); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication); [registry glossary](../../../../glossary.md#term-registry); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- The clean documentation context `/root/runtime_0005_docs` applied the General, API/Javadoc,
  Example, Backend Guide, Architecture, and Planning profiles. It made no executable Java change
  and therefore did not repeat the successful focused Java test.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>
- `./gradlew :modules:runtime:javadoc` passed without warnings. Standalone Java 26 compilation and
  execution validated the schedule examples in the Runtime API and backend guide.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Markdown links and anchors, fences, final newlines, exact public surface, imports and excluded
  mechanisms, current/Draft status, later-specification absence, exact 11-path maximum scope, and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` all passed.
- Final coordinator validation ran `./gradlew :modules:runtime:test`: `BUILD SUCCESSFUL`, with nine
  suites and 74 tests passing with no failures, errors, or skips. Gradle reported nine actionable
  tasks, one executed and eight up-to-date, and reused the configuration cache.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [cache glossary](../../../../glossary.md#term-cache))</span>

## Implementation notes

- Added [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule) as the sole top-level type in `runtime.schedule`, with an exact-plan
  record component and an immutable ordered <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`List<Step>` [snapshot](../../../../glossary.md#term-snapshot).
- Added the nested [sealed](../../../../glossary.md#term-sealed) `Step` contract and its sole current `ExecutionStep` record. The step
  retains the exact executable and derives its plan directly from that executable.
- Construction validates nulls, encounter order, indexed failures, and exact plan reference
  identity before snapshotting. Empty schedules and repeated occurrences remain valid.<span data-glossary-navigation="true"> ([reference identity glossary](../../../../glossary.md#term-reference-identity))</span>
- Added focused tests for exact shape, validation and failure order, reference identity,
  snapshot/order/repetition behavior, ownership and action absence, concurrent reads, and
  forbidden mechanisms.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Finalized both production/package Javadocs, Runtime and public API guides, the focused boundary
  status, backend guide, glossary, and synchronized task/master/roadmap records.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend))</span>

## Local decisions

- Retained the implementation context's production Javadocs after independent review because they
  already document every invariant, input, result, failure, ownership boundary, reuse rule, and
  concurrency limit required by the selected profiles.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Used concrete executable-only examples that demonstrate exact plan identity, immutable list
  snapshotting, encounter order, and repeated occurrences without implying runner or resource
  behavior.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Kept Compile API, Prepare API/docs, lifecycle and module-boundary explanations, user guides,
  ADRs, Gradle files, and architecture/conformance/integration tests unchanged: the new type adds
  no public cross-module facade, dependency or architecture rule, executable backend behavior,
  end-to-end behavior, configuration, or build change.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [facade glossary](../../../../glossary.md#term-facade); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend))</span>
- Kept other modules unchanged because current scheduling owns only Runtime types and deliberately
  defers orchestration and schedule consumption.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

- Completed changes: implemented the exact immutable executable-only prepared schedule contract,
  its closed current step family, focused contract tests, complete Javadocs, explanatory docs, and
  synchronized planning status.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: the two `runtime.schedule` production/Javadoc files, one focused test,
  five explanatory documents, and this task plus the Runtime master plan and roadmap—exactly the
  permitted 11 paths.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Validation: focused tests passed one suite/11 tests; exact surface, reflection/bytecode, imports,
  mechanisms, Javadoc, Java 26 examples, Markdown, status, later-specification absence, exact
  scope, and whitespace gates passed. Final <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`./gradlew :modules:runtime:test` [validation](../../../../glossary.md#term-validation) was
  `BUILD SUCCESSFUL`: nine suites/74 tests, no failures, errors, or skips; nine actionable tasks,
  one executed and eight up-to-date; configuration cache reused.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Documentation review: Compile API, Prepare API/docs, user guides, architecture/ADRs, build files,
  and architecture/conformance/integration tests require no change for the reasons recorded above.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Unresolved issues: none within Runtime 0005. Deferred representation/residency,
  transfer/materialization, publication/result, runner, and Prepare orchestration work remains in
  Draft Runtime 0006–0011 and Prepare 0003.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>
- Required follow-up: none for this task.

Status: Complete
