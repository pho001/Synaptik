# Task 0010: Prepared Runner and Dynamic Execution

## Status

Complete

## Goal

Make the existing immutable `PreparedExecution` recipe runnable through one narrow Runtime-owned
runner:

```text
caller inputs -> one RunState -> cold-bound direct occurrences -> ordered traversal
              -> publication -> RunResult lease or failure cleanup
```

Preserve `PreparedExecution` exactly as `PreparedMemoryPlan` plus `PreparedSchedule`. Add explicit
read/write meaning to current executable buffer selections so Runtime can validate reads,
invalidate stale output copies before invocation, and validate exact written copies only after
success.

## Scope

- Add public stateless `runtime.run.PreparedExecutionRunner` for one complete synchronous run.
- Hand dense borrowed caller inputs to existing package-private `RunStateCreation`.
- Permit no creation occurrence only for a completely empty memory plan and empty caller list.
- Cold-bind every executable, transfer, and publication occurrence before the first action.
- Traverse a private direct-reference bound-step array in schedule encounter order.
- Add immutable `READ_ONLY`, `WRITE_ONLY`, and `READ_WRITE` access declarations aligned with
  `PreparedExecutable.BufferSelection` occurrences.
- Before invocation, validate declared reads, then invalidate every copy of every output buffer.
- After success, validate only exact declared written representations; after failure, leave all
  output copies invalid.
- Preserve transfer, publication, and whole-state `RunResult` contracts unchanged.
- Close the one state after any post-creation failure, preserving the original unchecked failure
  and suppressing cleanup failure.
- Cover empty schedules/results, repetition, read/write overlap, aliasing, and run isolation.
- Finalize Javadocs/documentation in the mandatory separate clean documentation context.
- Synchronize only this task, the Runtime master plan, and the roadmap during planning.

## Out of scope

- graph inspection; `Operation`, `CompiledNode`, Compiler, Planning, Model, or Prepare identities
- backend discovery, registration, service lookup, lowering, route/kernel search, or fallback
- dynamic allocation, representation lookup, maps, reflection, string dispatch, boxing lookup,
  synchronization, retry, branching, or parallel scheduling in the hot traversal
- physical allocation/access, lazy creation, eviction, pooling, alias discovery, or coherence
- public Engine or Tensor/value result access, conversion, download, export, or host storage
- Prepare translation/orchestration; Config policy or `RunOptions`
- new Trace payloads, events, sinks, emission, profiling, or clocks
- concrete backend behavior; dependency, Gradle, architecture, ADR, architecture-test,
  conformance, or integration changes
- changing `PreparedExecution`, `PreparedSchedule`, `RunStateCreation`, `RunResult`, transfer, or
  publication contracts
- Runtime 0011 or Prepare 0003 implementation/specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [ADR 0006](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Runtime master plan](../master-plan.md)
- completed [Runtime 0003](0003-run-state-and-runtime-resource-foundation.md),
  [0004](0004-prepared-executable-and-bound-invocation.md),
  [0005](0005-prepared-schedule-contract.md), [0006](0006-prepared-execution-aggregate.md),
  [0007](0007-representation-creation-and-residency-foundation.md),
  [0008](0008-prepared-buffer-transfer-and-materialization-schedule.md), and
  [0009](0009-publication-and-result-schedule-steps.md)
- [Trace master plan](../../trace/master-plan.md), [Trace 0001](../../trace/tasks/0001-core-trace-event-envelope.md),
  and [Trace 0002](../../trace/tasks/0002-model-correlation-identifiers.md)
- [Prepare master plan](../../prepare/master-plan.md) and [Prepare 0002](../../prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Prepared recipes remain immutable/reusable; every call receives exactly one distinct mutable
  `RunState` for the complete heterogeneous run.
- Runtime owns caller handoff, state creation, binding, validity, traversal, cleanup, publication,
  and result lease. Backends own physical representations, callbacks, direct typed actions, and
  physical mechanics.
- The runner belongs to `runtime.run`, which already owns package-private state creation,
  validity, publication, result leasing, and cleanup.
- All occurrence binding finishes before traversal. The hot loop uses direct bound references and
  precomputed primitive coordinates only.
- Read validity is checked before output invalidation, preserving in-place read/write input.
- Every copy of an output buffer is invalidated before backend work because a prepared slot may
  now represent a new logical value. Success validates exact declared writes only.
- Failure after possible output mutation leaves all copies of that output invalid and performs no
  physical rollback or retry.
- Transfer and publication retain their completed explicit, no-fallback transitions.
- The runner consumes no Trace Java. Trace 0001–0002 are preserved; Trace 0006 run payloads remain
  Draft, so this task invents and emits nothing.
- Runtime stays independent of Model, Planning, Compiler, Prepare, Engine, and concrete backends.
- Stop if implementation needs graph facts, physical alias policy, a new module edge, mutable
  prepared state, a third `PreparedExecution` component, or an architecture change.

## Package impact

Existing packages changed:

- `runtime.execution` — explicit access declarations on the existing executable recipe.
- `runtime.run` — complete dynamic runner beside its existing lifecycle contracts.

`runtime.schedule` is consumed unchanged. No package is added, moved, or removed.

Type placement:

- `PreparedExecutable.BufferAccess` is nested because it describes one executable selection.
- `PreparedExecutionRunner` is the narrow public Runtime integration seam for later explicit
  Engine composition; it is not a service, manager, registry, or module-root facade.
- Bound-step implementations remain private runner details, not another schedule model.

## Exact API additions

Extend `PreparedExecutable` compatibly:

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

The existing three-argument constructor remains and assigns `READ_ONLY` to every buffer
selection. Existing binding, hooks, records, and failures remain unchanged.

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
trace input, or output accessor.

## Prepared executable access contract

- `READ_ONLY` requires the selected copy valid and declares no write.
- `WRITE_ONLY` requires no old value and declares an exact successful write.
- `READ_WRITE` requires the old selected copy valid and declares it written.

The four-argument constructor validates in this exact order:

1. require `memoryPlan`, `bufferSelections`, `workspaceSelections`, then `bufferAccesses` non-null;
2. preserve existing buffer validation/snapshot, then workspace validation/snapshot;
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

## Runner validation and lifecycle order

`run` acts exactly as follows:

1. require `execution`, then `callerInputs` non-null;
2. if schedule position zero is a creation occurrence, call existing `RunStateCreation.create`;
3. otherwise require both prepared buffer/workspace counts and caller-input count zero, then
   construct one empty state against the exact plan;
4. bind every remaining schedule occurrence in encounter order, snapshotting direct bound steps
   and bound publications before any action;
5. traverse bound steps in exact encounter order;
6. construct `RunResult` from the exact state and dense bound-publication order; and
7. return without closing the leased state.

Runner-owned failures:

- `NullPointerException("execution")`;
- `NullPointerException("callerInputs")`;
- `IllegalArgumentException("non-empty prepared memory plan requires a representation creation occurrence")`;
- `IllegalArgumentException("callerInputs size must equal caller-input preparation count 0")`; and
- `IllegalStateException("executable buffer selection i requires a valid input representation")`
  for the first invalid read in original selection order.

Existing creation/binding/action/result failures propagate unchanged. Schedule construction
already proves exact plan identity, first-only creation, and the dense publication suffix; the
runner must not duplicate that prepared validation.

## Cold binding and hot traversal

Cold binding produces one private bound object per executable, transfer, or publication. An
executable bound object retains its direct `BoundInvocation` plus primitive arrays for:

- reads with original selection indices;
- distinct output buffer indices in first-write encounter order; and
- distinct written buffer/representation pairs in first-write encounter order.

Use bounded primitive linear deduplication only—no map, set, boxing, reflection, or prepared-step
retention for hot redispatch. Transfer/publication objects retain one direct
`BoundBufferTransfer`/`BoundPublication`. Binding may allocate bound objects and primitive arrays
but acquires no closeable auxiliary resource.

Each bound executable action:

1. queries reads in original selection order, failing before mutation/work on the first invalid;
2. invalidates every resident representation of every distinct output buffer;
3. calls `BoundInvocation.execute()` once; and
4. after normal return, validates every distinct exact write coordinate.

A backend `RuntimeException` or `Error` propagates unchanged and leaves every output copy invalid.
Repeated reads remain ordered. Repeated writes coalesce. Multiple declared written copies of one
buffer all become valid after success. For input/output overlap, Runtime validates the old input,
then invalidates logical copies while the bound backend retains its direct physical references,
and validates the written copy only after success. Runtime performs no cross-buffer physical alias
discovery.

After traversal begins, allocate nothing and perform no list/map/boxing lookup, compatibility
cast, resource resolution, backend discovery, graph inspection, route/kernel search, config/tuning,
or Trace emission.

## Result and failure cleanup

- Transfers and publications execute exactly at their bound schedule positions with unchanged
  semantics.
- Bound publications are collected in dense suffix order; an empty suffix creates a zero-result
  lease.
- Empty plan/schedule/result without creation is valid. A non-empty plan requires creation. A
  creation-only or zero-publication run returns a zero-result lease.
- Successful `RunResult` construction alone transfers the whole-state close obligation.
- Any binding, traversal, publication, or result-construction failure after state creation calls
  `RunState.close()` exactly once. Rethrow the original unchecked failure unchanged and attach a
  distinct cleanup failure as suppressed; avoid self-suppression. Construct no result.
- A `RunStateCreation` failure already owns rollback; propagate it without another cleanup pass.
- Separate calls share immutable recipes only. Their state, validity, bound objects, publication
  flags, created resources, workspaces, results, and cleanup remain isolated.

## Performance and side effects

- The runner is stateless/thread-safe; one call is synchronous and single-orchestrator-thread.
- Cold setup is linear in schedule/selections/representation counts, with bounded quadratic
  primitive output deduplication inside one executable.
- Hot traversal is one dense direct-object array pass; executable transitions use primitive
  coordinates and `RunState` arrays.
- The runner itself performs no physical allocation/access. Prepared backend callbacks/actions do
  the physical work already authorized by completed contracts.
- Caller inputs remain borrowed. Created buffers/workspaces remain run-owned until failure cleanup
  or the returned result closes the state.

## Affected files

Expected Runtime production/Javadoc:

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

Expected planning:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless contradicted: all mandatory architecture/planning/docs rules; completed Runtime
0001–0009; Trace master/0001–0002/source; Prepare master/0001–0002 and Draft 0003 row; Backend
Contract, Config, and Engine master plans; current Runtime source/tests/Javadocs; Compile/Tensor/
Training APIs; builds; architecture, conformance, and integration tests.

## Maximum scope

At most 14 paths: 4 Runtime production/Javadoc, 2 Runtime tests, 5 explanatory documents, and 3
planning documents. No Java/test outside Runtime or any Gradle, architecture/ADR/test, Prepare,
Config, Trace, Backend Contract, Engine, concrete-backend, conformance, or integration path may
change. Stop if another type, package, edge, behavior owner, or path is required. Do not create a
later specification.

## Acceptance criteria

- Exact API/package surface above; `PreparedExecution` remains bytecode-exactly two components.
- Old executable constructor behavior is preserved; new access validation and immutable inspection
  match exact order/messages.
- One isolated state; all-bound-before-hot order; direct primitive bound representation.
- Exact read, full-output invalidation, success-only write, failure, overlap, duplicate, and
  multi-copy semantics.
- Exact transfer/publication order and zero/aliased result lease behavior.
- Every post-state failure closes once, preserves original identity, suppresses cleanup, skips
  borrowed inputs, and produces no result.
- Repeated/concurrent calls share no mutable run state/resource.
- No hot allocation/lookup/dispatch mechanism forbidden above and no forbidden imports/Trace
  emission.
- No Config/Prepare/Engine/backend/dependency/architecture behavior or public output access.
- Detailed Javadocs cover inputs/results/nullability, identity, access, side effects, ownership,
  cleanup, failures, concurrency, performance, and exclusions.
- Mandatory clean documentation pass completes in the same change.
- Exact 14 paths; Runtime 0001–0009 and Prepare 0001–0002 stay Complete; Runtime 0011 and Prepare
  0003 stay Draft without specs; documentation/whitespace gates pass.

## Tests / validation

Focused development:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutableTest \
  --tests io.github.pho001.synaptik.runtime.run.PreparedExecutionRunnerTest
```

Exactly one final affected-module command after Java stabilizes:

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
Also inspect exact source/compiled/reflection surfaces, old/new validation, creation/binding/action
order, direct fields/primitive arrays, hot bytecode, validity/overlap, cleanup/suppression,
empty/alias/isolation behavior, imports/mechanisms, unchanged completed surfaces/builds/Java 26,
exact scope, status, and later-spec absence.

Repository-wide and architecture validation is deferred to Runtime 0011, CI, or the prepared-
execution checkpoint. No module edge, shared build, architecture rule, concrete backend, public
Engine, or end-to-end consumer changes. Conformance/integration remain inapplicable.

## Mandatory clean documentation handoff

After the final Runtime test, hand this task, actual diff, affected API/behavior, architecture
constraints, exact 14 paths, and Java evidence to a separate clean documentation-focused agent.
It must read `AGENTS.md`, `ARCHITECTURE.md`, documentation rules, General/API-Javadoc/Architecture/
Backend-Guide/Planning profiles, final source/tests, and actual diff; finalize both affected
packages/classes and five explanatory docs; inspect generated runner/access Javadocs; validate
examples, glossary, links/anchors/fences/newlines/whitespace/scope/status; and record exact evidence
and limitations without repeating stable Java tests.

Required no-change conclusions: architecture/ADRs/general lifecycle/module/dependency; unchanged
Runtime memory/resource/schedule/state-creation/transfer/publication/result contracts; Trace (no
run DTO); Prepare 0003; Compile/Tensor/Training APIs; Config 0007; Backend Contract, Engine,
concrete backends, other guides, builds, architecture tests, conformance, and integration.

## Dependencies

- Runtime 0001–0009 and ADR 0011 — Complete/accepted.
- Runtime 0004 supplies checked direct executable binding; 0007 supplies creation/validity; 0008
  transfer; 0009 publication/result lease.
- Trace 0001–0002 are stable preservation constraints only. Trace 0006 run payloads are Draft, so
  Runtime consumes no Trace Java.
- Existing Runtime dependencies and Java 26 build remain unchanged.

Prepare 0003, Runtime 0011, Config 0007, Trace 0003–0008, Engine, concrete backends, output access,
conformance/integration, and tuning are not dependencies.

## Follow-up tasks

- Runtime 0011: Draft contract closure and prepared-execution checkpoint.
- Prepare 0003: Draft translation/orchestration and complete candidate validation.
- Trace 0006: later typed passive run payloads after producer assessment.
- Config 0007: later declarative run/publication policy.
- Concrete backends: physical creators/executables/transfers and conformance.
- Engine: public Tensor/value composition over this Runtime seam.

Do not create any follow-up detailed specification.

## Decisions and rejected alternatives

- Runner in `runtime.run`: source ownership avoids widening/duplicating `RunStateCreation`.
- Public stateless instance: narrow later Engine composition, not static global/service locator.
- Keep `PreparedExecution` unchanged: schedule already reaches every recipe.
- Explicit aligned access enum: current selections cannot safely infer roles from order, validity,
  graph facts, or physical types.
- Old constructor is all-read-only: source-compatible and fail-closed for fresh invalid outputs;
  writers must opt in.
- Validate reads before output invalidation for in-place input/output overlap.
- Invalidate all output-buffer copies before work; validate exact writes after success only.
- Bind everything before execution so a late compatibility failure precedes all physical effects.
- Private direct bound steps/primitive arrays instead of schedule reinterpretation or maps.
- Require creation for non-empty memory; permit the established truly empty run.
- Lease only after full success; no publication rollback/individual detachment.
- Preserve Trace rather than invent generic/string payloads before Trace 0006.

## Known limitations

- Old-constructor writers must adopt explicit access metadata.
- Runtime trusts backend access declarations and cannot verify bytes; backend tests must.
- Conservative invalidation may discard a physically intact old copy after failure.
- Synchronous sequential runner only; no cancellation, retry, timeout, parallelism, or partial result.
- All representations remain resident until state/result closure; whole-state lease retains
  internal resources and borrowed-input lifetime obligations.
- `RunResult` exposes no value; Prepare completeness and Engine access remain future work.
- No Trace event, production backend, or Engine consumer exists yet.
- Repository/architecture/conformance/integration validation remains deferred as specified.

## Documentation impact

- `PreparedExecutable`, `PreparedExecutionRunner`, and the affected execution/run package
  Javadocs become the precise API contract for access declarations, cold binding, traversal,
  validity transitions, cleanup, result lease, threading, and performance.
- Runtime API gains the current complete internal run mental model and a focused example covering
  caller handoff, read/write output production, publication, result closure, and failure cleanup.
- Public API lists the runner as a Runtime integration contract without claiming an Engine,
  Tensor/value result access, Config policy, concrete backend, or end-to-end user workflow.
- The focused Runtime/Prepare/Backend page receives implementation-status/mechanics wording only;
  architecture authority and module ownership do not change.
- Backend guidance documents correct immutable access declarations and direct bound fields; it
  must not suggest runtime discovery, hot allocation, or implicit coherence.
- The glossary adds or updates only reusable runner/access/validity terms; ordinary implementation
  words are not added.
- The clean documentation pass records the reasoned no-change conclusions required above and
  generates final Runtime Javadoc after its edits.

## Architecture impact

Expected impact: None. The architecture already assigns prepared runner, validity, cleanup,
publication, and result ownership to Runtime. Stop if implementation requires graph-derived roles,
physical alias policy, another module edge, Trace schema, mutable prepared state, public output
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

- Exact package/API/order/access/validity/cleanup/Trace decisions above are fixed.
- Private bound-step names may vary; direct retained facts and hot behavior may not.
- Bounded cold primitive deduplication is selected over boxing/maps.

## Validation evidence

- Implementation context `/root/runtime0010_impl` completed the Java work. The focused command
  passed 26 tests; the single final `./gradlew :modules:runtime:test` command passed 17 suites and
  143 tests with no failures, errors, or skips.
- Reflection and `javap` preserved the exact two-component `PreparedExecution`, confirmed the
  requested executable additions, and confirmed the runner's sole public constructor and `run`
  method with no field, interface, or public nested type.
- Source, import, field, and bytecode inspection confirmed direct `RunState`, bound-action, and
  primitive-array traversal without forbidden project imports, reflection, maps, sets, service
  lookup, hot compatibility casts, or hot allocation.
- Clean documentation context `/root/runtime0010_docs` finalized the Javadocs, five explanatory
  documents, and three planning documents without changing executable behavior or repeating Java
  tests. Runtime Javadoc, generated pages, eight-file Markdown validation, exact 14-path scope,
  status, and whitespace checks passed.
- Java 26 and build/dependency configuration remain unchanged. Repository-wide, architecture,
  conformance, and integration validation remains deferred as specified.

## Implementation notes

- The compatible constructor defaults selections to `READ_ONLY`; writers opt into aligned access.
- Cold setup binds every occurrence. Bound executables retain direct invocations and primitive
  coordinates. Reads precede conservative invalidation; only exact successful writes become
  valid. Failures leave output copies invalid and close the state.
- The runner is stateless and supports isolated concurrent calls; each run is synchronous and
  leases its complete state only after full success.
- No architecture or ADR changed. Existing lifecycle/module/dependency rules already assign this
  work to Runtime. Memory, resource, schedule, state-creation, transfer, publication, and result
  contracts remain unchanged.
- Trace has no current run DTO, so no emission was added. Prepare 0003 remains Draft without a
  specification. Compile, Tensor, Training, Config 0007, Backend Contract, Engine, concrete
  backends, other guides, builds, architecture tests, conformance, and integration need no change:
  this task adds no semantics, policy, backend implementation, edge, public value access, or
  end-to-end composition.

## Completion summary

- Completed changes: executable access declarations and stateless all-bound-before-traversal
  runner with exact validity, cleanup, result, and isolation behavior.
- Files changed or created: exactly the 14 production, test, explanatory, and planning paths.
- Tests and validation: focused 26 tests and final Runtime 17 suites/143 tests passed; Javadoc,
  generated pages, Markdown, surface/mechanism, build, scope, status, and whitespace checks passed.
- Documentation-agent review: clean context `/root/runtime0010_docs` completed the independent
  finalization and required no-change review.
- Documentation impact: Runtime/Public APIs, boundary, backend guide, glossary, and planning now
  describe the implemented lifecycle and planned boundaries.
- Javadoc review: access and runner/package contracts cover ordering, ownership, failure,
  concurrency, performance, and exclusions.
- Glossary impact: executable access and the prepared execution runner are defined.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
