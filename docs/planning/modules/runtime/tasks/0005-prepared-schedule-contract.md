# Task 0005: Prepared Schedule Contract

## Status

Complete

## Goal

Implement the smallest immutable Runtime-owned prepared schedule recipe that orders already-
prepared executable work for later cold consumption by a runner. One `PreparedSchedule` retains
one exact `PreparedMemoryPlan` and an immutable ordered snapshot of Runtime-owned steps. The only
current step executes one exact `PreparedExecutable` whose memory-plan reference is that same
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

`ExecutionStep.memoryPlan()` returns `executable.memoryPlan()` by exact reference. The sealed step
contract makes the schedule-wide plan invariant explicit and supplies a type-safe Runtime-owned
extension point for later architecture-required work. This task permits only `ExecutionStep`; it
does not guess transfer, materialization, or publication facts.

## Rationale and mental model

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

The schedule is a reusable prepared recipe, not per-run state or a hot-path interpreter. It
contains no graph, partition, backend-selection, allocation, or physical-resource facts. A future
runner can consume typed values during cold setup without `Operation`, `CompiledNode`, backend
rediscovery, maps, reflection, generic payloads, or string dispatch.

`PreparedUnit` remains unnecessary. Current Prepare finalization associates a Prepare-owned
`PlannedPartition` with one Runtime `PreparedExecutable`; Runtime cannot retain that association
without importing forbidden Planning/Prepare facts. Within Runtime, an executable step adds only
ordered occurrence and exact-plan validation. A separate wrapper has no additional invariant or
consumer.

## Scope

- Add the exact one-top-level/two-nested public surface above in
  `io.github.pho001.synaptik.runtime.schedule`.
- Require one exact non-null `PreparedMemoryPlan` reference for the complete schedule.
- Snapshot the supplied step list with `List.copyOf`, retaining exact element/executable references
  in supplied encounter order.
- Require every step to report the same exact plan reference as the schedule.
- Represent current schedulable work only as `ExecutionStep(PreparedExecutable)`.
- Permit an empty schedule, including against a non-empty plan. Emptiness implies no publication
  or resource-lifecycle policy.
- Permit repeated step or executable references. Each list position is one explicit execution
  occurrence and does not duplicate immutable prepared-resource ownership.
- Add focused tests for shape, validation, identity, immutability, order, emptiness, repetition,
  ownership, concurrency, and mechanism exclusions.
- Finalize affected Javadocs and explanatory documentation in a separate clean documentation
  context.

## Out of scope

- execution, executable binding, `RunState` construction, cold binding, or runner implementation
- resource allocation/representation creation, physical access, or ownership transfer
- transfer mechanics, residency/validity/coherence, or a transfer step
- materialization mechanics or a materialization step
- publication/result delivery, `RunResult`, targets/policies, or a publication step
- `PreparedUnit`, `PreparedPartition`, `PreparationResourceAssignment`,
  `BackendPartitionAnalysis`, `PlannedPartition`, or Prepare orchestration
- `PreparedExecution`, its lifecycle, persistent prepared resources, or run configuration
- concrete backends, routes, kernels, native handles, backend configuration, or tuning/cache work
- Compiler/planning changes or Runtime representations of `PublicationPlan`, publication
  bindings, `ValueId`, `TensorId`, or `LogicalMemoryRequirement`
- `Operation`, `CompiledNode`, graph inspection, backend discovery/selection, fallback, service
  lookup, allocation, profiling selection, or trace emission
- step IDs, stored indices, dependency edges, jumps, branches, barriers, retries, or priorities
- maps, registries, manager/service/util types, `Object` payloads, string dispatch, reflection,
  unchecked casts/generics, or another step abstraction
- Runtime 0006+, Prepare 0003, or another detailed task specification
- Gradle, dependency, architecture/ADR, architecture-test, backend-conformance, or integration-test
  changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0002](0002-prepared-memory-and-workspace-contracts.md)
- [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)
- [Runtime 0004](0004-prepared-executable-and-bound-invocation.md)
- [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Schedule, steps, and executables are immutable reusable prepared state and may be consumed
  concurrently for distinct logical runs.
- The schedule owns only its immutable list snapshot. It does not own or close the plan, steps,
  executables, run states, representations, invocations, or backend resources.
- One schedule associates with exactly one `PreparedMemoryPlan` by reference identity. Value-equal
  geometry from another plan is not the same prepared association.
- Runtime production must not import Prepare, Planning, Compiler, Model, Engine, or concrete-
  backend types, including every forbidden type named in the goal and exclusions.
- Prepare may construct Runtime schedule values by extracting Runtime executable references; that
  inward dependency does not authorize Runtime to retain Prepare associations.
- The sealed family is semantic type dispatch, not a payload registry. Runtime may add a variant
  only after stable Runtime-owned facts and a concrete consumer establish its invariant.
- A future runner must cold-bind execution occurrences to direct `BoundInvocation` references
  before its hot loop. Step traversal, allocation, backend discovery, graph inspection, and
  resource lookup must not move into `BoundInvocation.execute()`.
- If another module edge, generic payload, or new ownership decision is required, stop and report
  an architecture conflict rather than expanding the task.

## Package impact

Existing packages consumed without modification:

- `runtime.memory` supplies the exact `PreparedMemoryPlan` association.
- `runtime.execution` supplies immutable `PreparedExecutable` recipes.

Added package:

- `runtime.schedule` owns the immutable schedule and closed current semantic step family.

`PreparedSchedule` is the only top-level type. `Step` is nested because it is meaningful only as a
schedule entry; its sole method exposes the plan association required by schedule validation.
`ExecutionStep` is nested and final by record semantics. No package-private API or root facade is
added.

## Exact API shape

`PreparedSchedule` is a public record with exactly these components, in order:

1. `PreparedMemoryPlan memoryPlan`
2. `List<PreparedSchedule.Step> steps`

Its compact canonical constructor applies the rules below. Generated accessors return the exact
plan and immutable snapshot. Add no factory, builder, overload, mutator, executor, index accessor,
array view, equality override, or serialization contract.

`PreparedSchedule.Step` is a public nested sealed interface permitting exactly `ExecutionStep`.
It declares only `PreparedMemoryPlan memoryPlan()`. It has no ID, kind enum, payload, execution
method, visitor, or default method.

`ExecutionStep` is a public nested record with exactly one component,
`PreparedExecutable executable`. Its compact constructor rejects null. Its explicit
`memoryPlan()` returns `executable.memoryPlan()` without caching, copying, or equality conversion.
It adds no bind/execute behavior. Package Javadoc describes immutable prepared scheduling recipes
and says that current scheduling contains executable occurrences only.

## Validation, order, and failure rules

`ExecutionStep` construction first and only requires `executable` non-null. The exact failure is
`NullPointerException("executable")`; success retains the exact reference.

`PreparedSchedule` construction validates in this exact order:

1. require `memoryPlan` non-null;
2. require `steps` non-null;
3. scan steps in supplied order, rejecting the first null element;
4. for each non-null step in that scan, require `step.memoryPlan() == memoryPlan`; and
5. after the complete scan succeeds, assign `steps = List.copyOf(steps)`.

Exact failures are:

- `NullPointerException("memoryPlan")`;
- `NullPointerException("steps")`;
- `NullPointerException("steps[i]")`; and
- `IllegalArgumentException("steps[i] memory plan does not match schedule memory plan")`.

Fail on the first invalid occurrence. Do not compare plans with `equals`, deduplicate, sort,
group, flatten, bind, or execute. Call `memoryPlan()` exactly once per encountered valid step
before the first failure. Snapshot only after full validation, so construction fails closed.

Empty steps is valid. Repeated exact steps, equal steps, and executable references are valid and
preserve every occurrence. Immutable list encounter order is deterministic execution order.

List position is sufficient occurrence order. No current consumer needs branching, random access,
retry, correlation, or stable identity across schedules. A later runner may derive a transient
encounter index for validation or trace translation without storing identity here.

## Immutability, ownership, thread safety, and performance

- Construction is linear in step count, performs one structural snapshot, and performs no
  physical resource operation.
- Exact immutable elements are retained; steps and executables are not cloned.
- Schedule and steps contain no mutable per-run state, are not `AutoCloseable`, and transfer no
  cleanup responsibility.
- One schedule may be traversed concurrently for distinct run preparations. This does not make
  one future `RunState` or `BoundInvocation` safe for concurrent use.
- Production contains no map, registry, reflection, class-name test, `Object` payload, string
  dispatch, service lookup, graph/backend reference, or identifier allocation.
- Repeated work uses dense list occurrences, not lookup tables or IDs.
- This task adds no hot-path method. Future cold preparation retains direct typed invocation
  references; eventual execution performs no plan equality scan, slot lookup, backend selection,
  allocation, transfer/materialization decision, or publication binding.

## Transfer, materialization, and publication decision

These variants cannot be defined stably in this task:

- Transfer needs Runtime-owned source/destination representation and validity/residency facts.
  Current `RunState` defines neither residency nor coherence, while executable selections are
  executable-local rather than transfer recipes.
- Materialization needs Runtime-owned source/result representation facts and a creation/copy
  consumer. `PreparedMemoryPlan` supplies geometry only; allocation, physical access, and copy
  mechanics are absent.
- Publication needs a Runtime association from prepared resources to a delivery target and
  ownership policy. Compiler publication bindings retain Compiler/Model identities and boundary
  membership only; they define no Runtime delivery target and cannot be imported by Runtime.

These remain separate Draft work after their foundations stabilize. The sealed step contract
reserves no later payload beyond the exact schedule-plan association every future step must obey.

## Affected files

Expected Runtime production paths:

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

Expected planning paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a contradiction is found: architecture/ADRs, Runtime 0001–0004, Prepare
0001–0002, all current Runtime and relevant Prepare production/tests, Compiler publication and
Planning logical-memory contracts, current APIs/guides/glossary, build files, and architecture/
conformance/integration tests.

## Maximum scope

At most 11 paths: two Runtime production/Javadoc paths, one Runtime test path, five explanatory
documentation paths, and three planning paths. No Java/test path outside Runtime, Gradle,
dependency, architecture/ADR, architecture-test, conformance, or integration path may change.
Stop if another type, package, edge, behavior owner, or path is required. Do not create a later
detailed task specification.

## Test requirements

`PreparedScheduleTest` must cover:

- exact record/nested shape, modifiers, components, permitted subclass, method, constructors, and
  absence of extra public/package-private declarations;
- exact step null failure and executable/plan reference retention;
- top-level null failures in required order;
- indexed first-null and first-plan-mismatch failures, exact messages, order, and fail-closed
  behavior;
- reference identity rather than equality using two value-equal distinct plans;
- structural snapshot isolation from a mutable source list;
- exact element retention and deterministic order;
- valid empty schedule and repeated step/executable references;
- absence of binding, execution, resource action, ownership transfer, or closing at construction;
- concurrent read/traversal without mutation; and
- forbidden import, surface, and mechanism absence.

Tests may use minimal local immutable `PreparedExecutable` subclasses only to create plan-
associated recipes. They must not implement a runner, bind/execute, allocate representations, or
define deferred semantics. No architecture, conformance, or integration test is added because no
module edge or executable backend/end-to-end behavior changes.

## Acceptance criteria

- The exact one-top-level/two-nested surface exists only in `runtime.schedule`.
- `Step` permits exactly `ExecutionStep`; no other variant, kind, visitor, payload, or registry
  exists.
- Every schedule retains one exact plan and an immutable ordered snapshot whose steps report that
  same reference.
- Validation order/messages, empty/repetition policy, and exact reference retention match this
  task.
- No `PreparedUnit`, step ID, or stored index exists.
- Runtime production imports no forbidden upstream or concrete type.
- Construction performs no binding, execution, allocation, representation creation, transfer,
  residency, materialization, publication, discovery, tuning, or tracing.
- Javadocs cover invariants, inputs, results, failures, ownership, reuse, and thread safety without
  promising deferred behavior.
- The five explanatory docs and three planning docs consistently distinguish current executable
  scheduling from Draft later work.
- Runtime 0001–0004 and Prepare 0001–0002 remain Complete; Runtime 0006+ and Prepare 0003 remain
  Draft without specifications.
- All validation, scope, and whitespace checks pass.

## Validation

Validation tier: **2 — module and focused documentation validation**.

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

Also validate Javadoc warnings, Markdown links/anchors/fences/final newlines, exact status and
surface, sealed-family reflection/import checks, forbidden mechanisms/dependencies, absence of a
later specification, the exact maximum changed-path set, and `git diff --check`. Repository-wide
validation is deferred to the prepared-runtime checkpoint or CI because this task changes no
dependency/build/architecture boundary or concrete backend/end-to-end behavior.

## Dependencies

- Complete [Runtime 0002](0002-prepared-memory-and-workspace-contracts.md) supplies plan geometry.
- Complete [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md) establishes the
  per-run exact-plan association a future runner consumes.
- Complete [Runtime 0004](0004-prepared-executable-and-bound-invocation.md) supplies the exact
  schedulable recipe and cold-bound invocation boundary.
- Complete [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)
  proves finalized executables retain one shared exact plan and preserves partition order.
- ADRs 0010 and 0011 establish staged preparation and immutable-prepared/per-run ownership.

## Follow-up tasks

- Runtime 0006: Draft `PreparedExecution` aggregate and lifecycle.
- Runtime 0007: Draft representation creation and residency/validity foundation.
- Runtime 0008: Draft transfer and materialization steps after Runtime 0007.
- Runtime 0009: Draft publication/result associations and step after delivery/ownership stabilizes.
- Runtime 0010: Draft runner, cold schedule consumption, and dynamic execution.
- Runtime 0011: Draft Runtime contract-closure audit.
- Prepare 0003: Draft orchestration and complete prepared-result validation.

No follow-up detailed specification may be created here.

## Decisions

- Use one top-level record with a nested sealed semantic family: this preserves known
  heterogeneous scheduling without inventing later payloads.
- Implement only `ExecutionStep`, the sole stable Runtime-owned work fact with a current producer.
- Omit `PreparedUnit`; it adds no invariant beyond executable and schedule occurrence.
- Use exact plan reference identity consistently with current Runtime/Prepare contracts.
- Use list position for occurrence order; add no identity or stored index.
- Allow empty schedules and repeated executable occurrences without changing ownership.
- Split transfer/materialization/publication from the runner frontier until their Runtime facts
  exist.

## Known limitations

- Current schedules describe executable occurrences only and cannot yet express required
  transfer, materialization, or publication work.
- No representation creation, input/result slot association, complete `PreparedExecution`, public
  Prepare orchestration, runner, or Engine lifecycle exists.
- Each later justified semantic variant requires an explicit Runtime source/specification change.

## Architecture impact

None. This implements existing Runtime schedule ownership, preserves dependency rules, and adds no
module edge, architecture rule, or ADR decision. Splitting under-founded variants into later Draft
tasks is planning decomposition, not an architecture change.

## Implementation prompt

Execute in a fresh implementation context. Read `AGENTS.md`, `ARCHITECTURE.md`, the referenced
architecture/ADRs and planning documents, documentation rules and General/API-Javadoc/Example
profiles, Runtime 0002–0004, Prepare 0001–0002, current Runtime/relevant Prepare source and tests,
Compiler publication and Planning logical-memory contracts, current APIs/guides/glossary, and
root/runtime/prepare Gradle files in full.

Implement exactly this surface, validation, ownership, performance, scope, and exclusions. Do not
add `PreparedUnit`, another variant, runner, allocation, transfer, materialization, publication,
upstream identity, generic payload, registry, dependency/architecture change, or later task spec.
Stop if another type, path, dependency, or ownership decision is required. After Java stabilizes,
use a distinct clean documentation context to finalize permitted Javadocs/docs. Run the specified
validation once, verify exact scope, and do not commit or push.

## Validation evidence

- The implementation context `/root/runtime_0005_impl` ran
  `./gradlew :modules:runtime:test --tests
  io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest`: one suite and 11 tests passed
  with no skips, failures, or errors.
- Exact API, reflection, and bytecode inspection confirmed the two record shapes, the nested sealed
  family with only `ExecutionStep` permitted, the exact constructors/methods, one
  `Step.memoryPlan()` call per encountered step, reference comparison, and snapshot creation only
  after a successful scan.
- Source/import/mechanism scans confirmed the required validation messages and order, no forbidden
  upstream or concrete-backend imports, and no binding, execution, allocation, ownership,
  transfer, materialization, publication, registry, reflection, or generic-payload mechanism.
- The clean documentation context `/root/runtime_0005_docs` applied the General, API/Javadoc,
  Example, Backend Guide, Architecture, and Planning profiles. It made no executable Java change
  and therefore did not repeat the successful focused Java test.
- `./gradlew :modules:runtime:javadoc` passed without warnings. Standalone Java 26 compilation and
  execution validated the schedule examples in the Runtime API and backend guide.
- Markdown links and anchors, fences, final newlines, exact public surface, imports and excluded
  mechanisms, current/Draft status, later-specification absence, exact 11-path maximum scope, and
  `git diff --check` all passed.
- Final coordinator validation ran `./gradlew :modules:runtime:test`: `BUILD SUCCESSFUL`, with nine
  suites and 74 tests passing with no failures, errors, or skips. Gradle reported nine actionable
  tasks, one executed and eight up-to-date, and reused the configuration cache.

## Implementation notes

- Added `PreparedSchedule` as the sole top-level type in `runtime.schedule`, with an exact-plan
  record component and an immutable ordered `List<Step>` snapshot.
- Added the nested sealed `Step` contract and its sole current `ExecutionStep` record. The step
  retains the exact executable and derives its plan directly from that executable.
- Construction validates nulls, encounter order, indexed failures, and exact plan reference
  identity before snapshotting. Empty schedules and repeated occurrences remain valid.
- Added focused tests for exact shape, validation and failure order, reference identity,
  snapshot/order/repetition behavior, ownership and action absence, concurrent reads, and
  forbidden mechanisms.
- Finalized both production/package Javadocs, Runtime and public API guides, the focused boundary
  status, backend guide, glossary, and synchronized task/master/roadmap records.

## Local decisions

- Retained the implementation context's production Javadocs after independent review because they
  already document every invariant, input, result, failure, ownership boundary, reuse rule, and
  concurrency limit required by the selected profiles.
- Used concrete executable-only examples that demonstrate exact plan identity, immutable list
  snapshotting, encounter order, and repeated occurrences without implying runner or resource
  behavior.
- Kept Compile API, Prepare API/docs, lifecycle and module-boundary explanations, user guides,
  ADRs, Gradle files, and architecture/conformance/integration tests unchanged: the new type adds
  no public cross-module facade, dependency or architecture rule, executable backend behavior,
  end-to-end behavior, configuration, or build change.
- Kept other modules unchanged because current scheduling owns only Runtime types and deliberately
  defers orchestration and schedule consumption.

## Completion summary

- Completed changes: implemented the exact immutable executable-only prepared schedule contract,
  its closed current step family, focused contract tests, complete Javadocs, explanatory docs, and
  synchronized planning status.
- Files changed or created: the two `runtime.schedule` production/Javadoc files, one focused test,
  five explanatory documents, and this task plus the Runtime master plan and roadmap—exactly the
  permitted 11 paths.
- Validation: focused tests passed one suite/11 tests; exact surface, reflection/bytecode, imports,
  mechanisms, Javadoc, Java 26 examples, Markdown, status, later-specification absence, exact
  scope, and whitespace gates passed. Final `./gradlew :modules:runtime:test` validation was
  `BUILD SUCCESSFUL`: nine suites/74 tests, no failures, errors, or skips; nine actionable tasks,
  one executed and eight up-to-date; configuration cache reused.
- Documentation review: Compile API, Prepare API/docs, user guides, architecture/ADRs, build files,
  and architecture/conformance/integration tests require no change for the reasons recorded above.
- Unresolved issues: none within Runtime 0005. Deferred representation/residency,
  transfer/materialization, publication/result, runner, and Prepare orchestration work remains in
  Draft Runtime 0006–0011 and Prepare 0003.
- Required follow-up: none for this task.

Status: Complete
