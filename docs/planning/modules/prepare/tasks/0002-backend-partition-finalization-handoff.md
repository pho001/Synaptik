# Task 0002: Backend Partition Finalization Handoff

## Status

Complete

## Goal

Implement the smallest Prepare-owned handoff that consumes a complete ordered set of current
`BackendPartitionAnalysis` results, assigns stable Runtime buffer and workspace slots only after
all declarations are known, constructs one exact `PreparedMemoryPlan`, and calls each owning
backend to create a current `PreparedExecutable` against those assignments.

The task also adds the smallest reusable association required by the next schedule frontier:

```java
package io.github.pho001.synaptik.prepare;

public sealed interface PreparationResourceAssignment {
    record Buffer(
            PreparationResourceRequirement.Buffer requirement,
            BufferSlot slot,
            int planIndex) implements PreparationResourceAssignment {}

    record Workspace(
            PreparationResourceRequirement.Workspace requirement,
            WorkspaceSlot slot,
            int planIndex) implements PreparationResourceAssignment {}
}

public record BackendPartitionFinalization<P extends BackendPreparationPlan>(
        BackendPartitionAnalysis<P> analysis,
        PreparedMemoryPlan memoryPlan,
        List<PreparationResourceAssignment> assignments) {}

public interface BackendPartitionFinalizer<P extends BackendPreparationPlan> {
    BackendId backendId();
    PreparedExecutable finalizePartition(BackendPartitionFinalization<P> finalization);
}

public record PreparedPartition(
        PlannedPartition partition,
        PreparedExecutable executable) {}
```

One package-private `BackendPartitionFinalizationHandoff` operation owns complete-set validation,
assignment, finalizer invocation, and the immutable batch result. It receives the independently
expected ordered planned partitions plus ordered typed entries containing the exact
`PrepareContext`, `BackendPartitionAnalysis`, and matching `BackendPartitionFinalizer`. The
package-private entry preserves the backend-specific input and selected-plan generic types
without a raw type, `Object`, generic map parameter, registry, or service lookup.

`PreparedUnit` is not needed in this task. The current executable already retains its exact
memory-plan association and dense resource selections; `PreparedPartition(partition, executable)`
adds only the missing compile/prepare partition association. Runtime 0005 must decide whether an
ordered schedule step needs a distinct `PreparedUnit` invariant.

## Rationale and mental model

```text
ordered expected partitions + ordered complete analyses/finalizers
  -> validate all sources, ownership, references, and declarations
  -> assign all shared slots and construct one PreparedMemoryPlan
  -> construct every typed BackendPartitionFinalization
  -> call owning finalizers in partition order
  -> immutable ordered PreparedPartition values
```

All shared decisions complete before the first backend finalizer runs. A finalizer receives its
exact opaque analysis plan and exact source assignments, but it cannot return a replacement plan
or declarations. It returns only a `PreparedExecutable`, whose exact memory-plan reference is
validated before the prepared partition is accepted.

## Scope

- Add the exact four public root-package declarations above and root package documentation.
- Add one package-private finalization handoff with package-private typed `Entry` and `Result`
  records; add no public orchestration facade.
- Accept an independently supplied ordered `List<PlannedPartition>` as the expected complete
  coverage and an equally sized ordered list of finalization entries.
- Require each entry's context partition and analysis partition to be the exact expected
  partition reference at that index.
- Require the finalizer's non-null `BackendId` to equal the expected partition owner.
- Validate every buffer declaration against the entry's projected values and matching logical
  memory requirement before assignment.
- When the same `ValueId` is declared by more than one analysis, require the projections to retain
  the same exact `GraphValue` and `LogicalMemoryRequirement` references.
- Traverse analyses in expected partition order and each analysis's requirements in stored order.
- Assign `BufferSlot` values densely from zero in first-declaration order, one slot per distinct
  declared `ValueId` across the complete set.
- Give a repeated buffer value one plan entry whose byte size is the maximum declared size and
  whose alignment is the maximum declared alignment. Because all alignments are powers of two,
  the maximum satisfies every declaration without inventing another alignment rule.
- Assign `WorkspaceSlot` values densely from zero in declaration encounter order, one distinct
  slot for every workspace declaration. Workspace IDs remain local to their analysis and are
  never merged across partitions.
- Add no lifetime, interference, aliasing, or reuse model.
- Construct buffer plan entries in first buffer-declaration order and workspace entries in
  workspace-declaration order. Preserve those exact slot and entry references in every source
  assignment.
- Give each analysis one immutable assignment list in exact requirement order. Each assignment
  retains the exact requirement object, assigned slot, and dense index into the corresponding
  `PreparedMemoryPlan` list.
- Construct and validate every `BackendPartitionFinalization` before invoking any backend.
- Invoke finalizers exactly once each in expected partition order after assignment completes.
- Require each returned executable to be non-null and retain the exact shared memory-plan
  reference.
- Produce immutable ordered `PreparedPartition` values retaining the exact planned-partition and
  executable references.
- Add focused contract, assignment, finalization, failure-order, and public-shape tests with fake
  typed backend inputs, opaque plans, finalizers, and executables.
- Finalize Javadocs and affected explanatory documentation in a separate clean documentation
  context after implementation.

## Out of scope

- physical allocation, storage access, representation creation, native resource acquisition, or
  executable-resource cleanup
- `RunState` creation, input binding, cold invocation binding, or execution
- `PreparedUnit`, `PreparedSchedule`, schedule steps, scheduling, or Runtime 0005
- transfer, residency, validity, materialization, publication, `RunResult`, or runner behavior
- `PreparedExecution`, immutable persistent prepared-resource ownership, or lifecycle closing
- a public prepare orchestration facade, Engine composition, registration, discovery, registry,
  service locator, builder, manager, or service
- a concrete backend implementation, concrete executable, route, kernel, native bridge, or
  physical representation
- changing, repeating, tuning, searching, or scoring the selected route
- adding declarations during finalization or changing the analysis plan
- model-autotuning measurement, candidate enumeration, cache lookup, or cache mutation
- graph capture, graph optimization, compiler behavior, planning behavior, or compile artifacts
- dynamic dimension binding or a run-dynamic resource contract
- pooling, slot lifetime reuse, aliasing, interference analysis, distributed sharding, or
  multi-device scheduling
- Config, Trace, Backend Contract, Model, Planning, Compiler, Runtime, Engine, concrete-backend,
  Gradle, dependency, architecture-contract, ADR, architecture-test, conformance-test, or
  integration-test changes
- Prepare 0003, Runtime 0005, or any later detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core lifecycle and invariants
  - `modules/runtime`
  - `modules/prepare`
  - Concrete backend modules
  - Prepare lifecycle
  - Run lifecycle
  - Dependency rules
- [ADR 0010: Stage backend preparation around shared slot assignment](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)

## Architecture constraints

- Shared Prepare owns complete-set validation, deterministic assignment, exact source
  associations, backend finalization coordination, and `PreparedPartition`.
- Runtime owns `BufferSlot`, `WorkspaceSlot`, `PreparedMemoryPlan`, and `PreparedExecutable`.
  Runtime remains free of Prepare, Model, Planning, and Compiler facts.
- Concrete backends own the opaque selected plan and executable subclass. Shared Prepare never
  inspects, downcasts, copies, or interprets backend-private plan fields.
- Backend analysis and route choice are complete before this handoff. Finalization receives the
  exact analysis and cannot replace its route, plan, or declarations.
- All assignment and shared validation complete before any finalizer call. No backend may observe
  a partial shared assignment.
- Prepared recipes are immutable and reusable. This task creates no per-run mutable state.
- Runtime hot-path code remains free of `Operation`, `CompiledNode`, maps, graph inspection,
  backend discovery, and selection work. The new Model/Planning associations stay in Prepare.
- Prepare remains independent of concrete backend implementations. A typed finalizer is supplied
  explicitly; no lookup or public backend switch is added.
- No new module edge or architecture rule is permitted. Stop if implementation requires one.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.prepare.analysis` — consume the current typed analysis contracts
  without changing their public shape or semantics.
- `io.github.pho001.synaptik.prepare` — the master plan reserves this root for the finalized
  prepared-partition result and narrow shared handoff/validation contracts.

Packages added or changed:

- no package beyond the planned Prepare root; `prepare.analysis` receives Javadoc review only
  unless a concrete contradiction is found.

Type placement:

- `PreparationResourceAssignment` — public exact declaration-to-Runtime-slot association that a
  concrete backend finalizer consumes.
- `BackendPartitionFinalization` — public typed immutable finalization input for one analysis.
- `BackendPartitionFinalizer` — public concrete-backend-implemented typed collaboration.
- `PreparedPartition` — public minimal association of one exact planned partition with its one
  finalized executable.
- `BackendPartitionFinalizationHandoff` — package-private shared complete-set operation; its
  nested typed `Entry` and immutable `Result` remain internal until Prepare 0003 supplies public
  orchestration.

Tests mirror `io.github.pho001.synaptik.prepare` so they can exercise the package-private handoff.
No `util`, `service`, `registry`, or generic orchestration package is added.

## Exact public and package-private API shape

The four public declarations are exactly the surface shown in the Goal. Constructors and record
accessors are explicit only where complete Javadoc is required; they add no member beyond the
ordinary record surface.

The package-private shape is:

```java
final class BackendPartitionFinalizationHandoff {
    static Result finalizePartitions(
            List<PlannedPartition> partitions,
            List<? extends Entry<?, ?>> entries);

    record Entry<
            I extends BackendAnalysisInputs,
            P extends BackendPreparationPlan>(
            PrepareContext<I> context,
            BackendPartitionAnalysis<P> analysis,
            BackendPartitionFinalizer<P> finalizer) {}

    record Result(
            PreparedMemoryPlan memoryPlan,
            List<PreparedPartition> partitions) {}
}
```

The operation may use private helpers and private implementation-only collections inside the same
file. It must not expose a map parameter or result, use a raw type or `Object`, or add another
top-level type. Wildcard capture must preserve each entry's `P` association when constructing the
typed finalization and calling its finalizer.

## Affected files

Expected Prepare production paths:

- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparationResourceAssignment.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalization.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizer.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedPartition.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/package-info.java`

Expected Prepare test paths:

- up to one exact public/package-private shape test under the mirrored root package;
- up to one finalization-value validation test; and
- up to one complete handoff/assignment/failure-order test.

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status clarification
  only; no architecture rule change
- `docs/backend-guide/partition-preparer.md`
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs 0010/0011,
lifecycle/module/dependency docs, Prepare 0001, Runtime 0001–0004, all current Prepare and relevant
Runtime source/tests/Javadocs, Compile API, Training API, current backend guides, build files, and
architecture/conformance/integration tests.

## Maximum scope

At most 18 paths:

- 6 Prepare production paths;
- 3 Prepare test paths;
- 5 explanatory documentation paths; and
- 4 Prepare/Runtime/global planning paths.

No Java/test path outside `modules/prepare`, Gradle path, architecture contract, ADR,
architecture-test, backend-conformance, or integration-test path may change. If another public
type, package, module edge, lifecycle owner, or path is required, stop and report a follow-up.
Do not create Prepare 0003, Runtime 0005, or another detailed task specification.

## Assignment semantics

The handoff validates all source facts before constructing assignments. It then performs one
complete deterministic traversal in expected partition order and requirement order.

For each first-seen buffer `ValueId`, create a new `BufferSlot` whose numeric value is the current
buffer count. Later equal IDs reuse that exact slot reference. The final `BufferEntry` uses the
maximum declared byte size and maximum declared byte alignment for the value. The entry appears at
the value's first declaration position relative to other distinct buffer values.

For every workspace declaration, create a fresh `WorkspaceSlot` whose numeric value is the
current workspace count. Its `WorkspaceEntry` copies that declaration's exact size and alignment.
Even equal analysis-local requirement IDs in different analyses receive distinct slots.

Each source declaration receives exactly one assignment in the owning analysis's requirement
order. A buffer assignment retains its exact `Buffer` requirement, the exact shared slot, and the
buffer entry's dense index. A workspace assignment retains its exact `Workspace` requirement, its
exact distinct slot, and the workspace entry's dense index. The plan and assignment lists are
immutable snapshots. No source collection container is retained.

## Validation, ownership, order, and failure rules

### Value constructors

`PreparationResourceAssignment.Buffer` validates `requirement`, then `slot`, then non-negative
`planIndex`. `Workspace` uses the same order. Exact failures are
`NullPointerException("requirement")`, `NullPointerException("slot")`, and
`IllegalArgumentException("planIndex must be non-negative")`.

`BackendPartitionFinalization` validates `analysis`, `memoryPlan`, then `assignments` non-null. It
then requires the assignment count to equal the analysis requirement count and scans assignments
in order. For each index it requires:

1. a non-null assignment;
2. the assignment to retain the exact `analysis.requirements().get(index)` object;
3. the dense plan index to be in range;
4. the plan entry to retain the exact assigned slot reference; and
5. buffer geometry to be at least the declaration's size/alignment, or workspace geometry to
   equal its declaration exactly.

Exact failures are:

- `NullPointerException("analysis")`, `NullPointerException("memoryPlan")`, or
  `NullPointerException("assignments")`;
- `IllegalArgumentException("assignments size must equal analysis requirement count N")`;
- `NullPointerException("assignments[i]")`;
- `IllegalArgumentException("assignments[i].requirement does not match analysis.requirements[i]")`;
- `IllegalArgumentException("assignments[i] buffer planIndex out of range: X")` or the
  corresponding `workspace` message;
- `IllegalArgumentException("assignments[i] buffer slot does not match memoryPlan.buffers[X]")`
  or the corresponding workspace message; and
- `IllegalArgumentException("assignments[i] buffer geometry does not satisfy requirement")` or
  `IllegalArgumentException("assignments[i] workspace geometry does not match requirement")`.

After validation, the record snapshots the assignment list and retains every exact immutable
analysis, plan, assignment, requirement, and slot reference.

`BackendPartitionFinalizer.backendId()` must return a non-null immutable identity. The handoff
calls it once during complete-set validation. `finalizePartition` receives one non-null validated
finalization and returns one non-null immutable `PreparedExecutable`. It may validate backend
state and construct ordinary immutable Java recipe state, but this task does not permit physical
allocation, closeable/native resource acquisition, route reselection, new shared requirements,
or mutation of analysis/cache state.

`PreparedPartition` validates `partition` before `executable`, snapshots nothing, and retains both
exact immutable references. Exact null messages are `partition` and `executable`. Backend
ownership is obtained from `partition.owner()`; the record does not duplicate `BackendId`.
Executable-to-plan association is validated by the handoff rather than by this independently
constructible two-component value.

### Complete handoff

The package-private operation validates in this exact order:

1. require `partitions`, then `entries` non-null;
2. scan expected partitions in order for the first null or later equal duplicate;
3. scan entries in order for the first null and validate each entry's `context`, `analysis`, and
   `finalizer` components in declaration order;
4. require equal list sizes;
5. for each index, require exact context-partition and analysis-partition reference identity;
6. call `backendId()` once, reject null, and require value equality with `partition.owner()`;
7. validate every buffer source in requirement order against the context's unique projected
   value and logical requirement;
8. for repeated cross-analysis buffer IDs, require exact projected `GraphValue` and
   `LogicalMemoryRequirement` reference identity with the first declaration;
9. only after all entries pass, derive every assignment and the complete memory plan;
10. construct every typed finalization and validate it before any finalizer call;
11. call each finalizer once in partition order, stopping at the first thrown unchecked failure;
12. reject a null executable, then require `executable.memoryPlan() == memoryPlan`; and
13. construct and snapshot the ordered prepared partitions and return the exact memory plan.

Exact handoff failures are:

- `NullPointerException("partitions")` or `NullPointerException("entries")`;
- `NullPointerException("partitions[i]")`;
- `IllegalArgumentException("partitions[i] duplicates <partition>")`;
- `NullPointerException("entries[i]")`, followed by component messages `context`, `analysis`, or
  `finalizer` when constructing an entry;
- `IllegalArgumentException("entries size must equal partitions size N")`;
- `IllegalArgumentException("entries[i].context.partition does not match partitions[i]")`;
- `IllegalArgumentException("entries[i].analysis.partition does not match context partition")`;
- `NullPointerException("entries[i].finalizer.backendId")`;
- `IllegalArgumentException("entries[i].finalizer backendId <actual> does not match partition owner <expected>")`;
- `IllegalArgumentException("entries[i].requirements[j] buffer valueId is absent from context.values: <id>")`;
- `IllegalArgumentException("entries[i].requirements[j] projected value reference does not match first declaration for <id>")`;
- `IllegalArgumentException("entries[i].requirements[j] logical requirement reference does not match first declaration for <id>")`;
- `NullPointerException("entries[i].finalizer returned null")`; and
- `IllegalArgumentException("entries[i] executable memory plan does not match assigned memory plan")`.

An empty complete set is valid and produces an empty `PreparedMemoryPlan` and empty prepared
partition list without calling a finalizer. All list structures are immutable snapshots. The
operation retains exact immutable elements and does not mutate supplied lists, contexts,
analyses, plans, requirements, finalizers, or executables.

No finalizer is called when any shared validation, coverage, source-reference, assignment, or
finalization-construction check fails. When a finalizer itself throws `RuntimeException` or
`Error`, the exact failure propagates and later finalizers are not called. Earlier returned
executables own no closeable resource under this task, so no rollback or cleanup protocol is
introduced.

## Acceptance criteria

- The exact four public declarations and one package-private handoff surface exist in the planned
  Prepare root package; no facade, registry, service, builder, manager, or `PreparedUnit` exists.
- Generic bounds preserve each opaque backend plan's association with its finalizer without raw
  types, `Object`, unchecked public access, or a generic map parameter.
- Complete-set coverage, exact partition references, backend ownership, buffer source membership,
  and repeated-source references fail closed in the specified order and with exact messages.
- Assignment begins only after every analysis passes shared validation; every finalization is
  constructed before any backend is called.
- Buffer slots are dense and deterministic in first declaration order. Repeated declared values
  share the exact slot and use maximum size/alignment geometry.
- Workspace slots are dense and deterministic in declaration order. Every declaration receives a
  distinct exact slot with unchanged geometry.
- Every requirement has one exact source assignment in requirement order, and every assignment's
  dense index, slot reference, and plan geometry are validated.
- Finalizers are called once in partition order only after assignment, see the exact typed
  analysis/plan and assignments, and cannot return changed route/declaration data.
- Null, foreign-backend, foreign-partition, foreign-source, foreign-slot, out-of-range, geometry,
  null-executable, and foreign-plan cases fail exactly as specified.
- `PreparedPartition` is only the exact two-reference partition/executable association. Its
  backend identity comes from the partition and its memory plan from the executable.
- Tests prove empty input, list snapshots, exact reference retention, stable order, maximum
  geometry, separate slot domains, no reuse, no finalizer-before-validation, stop-on-first
  finalizer failure, and exact output order.
- No physical resource, `RunState`, invocation, execution, scheduling, transfer, residency,
  materialization, publication, concrete backend, tuning, cache mutation, graph/compiler/planning
  behavior, dependency, Gradle, architecture, conformance, or integration change is added.
- Every public and protected member has meaningful Javadoc covering input constraints,
  nullability, result, exact-reference ownership, immutability, order, failure, and lifecycle.
- A separate clean documentation-focused pass finalizes affected Javadocs, Runtime/Public APIs,
  focused boundary and backend guide status, glossary impact, and planning evidence without
  repeating successful Java tests unless executable behavior changes.
- Prepare 0001 and Runtime 0001–0004 remain Complete. Prepare 0002 becomes Complete only after all
  implementation/documentation gates. Prepare 0003 and Runtime 0005–0008 remain Draft without
  detailed specifications.
- Exact scope, link, anchor, fence, newline, whitespace, and `git diff --check` gates pass.

## Tests / validation

Implementation-focused validation:

```bash
./gradlew :modules:prepare:test
```

Run focused new suites while developing, then run the command above once after executable Java
stabilizes. Existing Runtime tests are not repeated because Runtime Java does not change.

Documentation-focused pass after final Javadocs and documentation:

```bash
./gradlew :modules:prepare:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md \
  docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/partition-preparer.md \
  docs/glossary.md \
  docs/planning/modules/prepare/tasks/0002-backend-partition-finalization-handoff.md \
  docs/planning/modules/prepare/master-plan.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

If the temporary Markdown validator is absent or requires another invocation form, create or use
an equivalent validator outside the repository. Validate every changed Markdown file for local
targets and heading anchors, unique effective anchors, balanced backtick/tilde fences, final
newlines, and trailing whitespace.

Required source/scope/status checks:

- exact public/package-private/nested surface, generic bounds, modifiers, packages, and no
  `PreparedUnit`;
- exact validation order/messages, list snapshots, assignment traversal, first-declaration
  ordering, maximum buffer geometry, workspace non-reuse, exact references, and finalizer calls;
- production scan for no raw `Object`, unchecked generic access, public/generic map parameter,
  reflection, string dispatch, registry/service locator, concrete backend, or hot-path behavior;
- no modification to existing Prepare analysis semantics or Runtime contracts;
- unchanged root/Prepare/Runtime Gradle files and Java 26 root toolchain/release configuration;
- exact 18-path ceiling and no Java/test outside Prepare;
- Prepare 0001, Runtime 0001–0004, task/master/roadmap status synchronization, and absence of
  Prepare 0003, Runtime 0005+, or other later specifications; and
- final newline, trailing whitespace, and `git diff --check`.

Repository-wide and architecture tests are deferred to the Prepare contract checkpoint or
continuous integration. This task changes one module without a module edge, dependency rule,
shared build contract, concrete backend, or end-to-end execution. Backend conformance and
integration tests are not applicable.

The documentation context reuses successful Prepare test evidence unless it changes executable
Java behavior or records a concrete reason to repeat it.

## Dependencies

- Prepare 0001 — Complete.
- Runtime 0001–0004 — Complete.
- Planning 0001–0006 — Complete.
- Compiler 0001–0006 — Complete.
- Backend Contract 0001–0004 — Complete and closed.
- ADR 0010 staged preparation and ADR 0011 runtime ownership/cold binding — Accepted.
- Current Prepare dependencies already include Runtime, Planning, Compiler, Config, Backend
  Contract, and Trace. No build change is needed.

## Follow-up tasks

- Prepare 0003 will compose public prepare orchestration plus partition, prepared-memory, and
  later schedule validation after Runtime 0005 supplies the schedule contract.
- Runtime 0005 will define the prepared schedule and decide whether an actual schedule-step
  invariant justifies `PreparedUnit`.
- Runtime 0006–0008 remain Draft for the prepared aggregate, runner/dynamic execution, and closure
  audit.
- Concrete backend implementation remains in each backend plan after the shared contracts are
  complete.

All follow-ups remain Draft without detailed specifications. Do not create them in this task.

## Architecture impact

Expected impact: None.

The architecture and ADR 0010 already assign shared slot assignment and source associations to
Prepare, final executable construction to the owning backend after assignment, Runtime geometry
and executable contracts to Runtime, and `PreparedPartition` to Prepare. This task implements that
boundary without changing module ownership or dependency direction. Stop if implementation needs
a concrete backend dependency, Runtime knowledge of Prepare facts, route/declaration mutation,
physical allocation, a prepared-resource cleanup lifecycle, or another architecture decision.

## Javadocs and documentation impact

- Document every new public type, record component, explicit constructor/accessor, finalizer
  method, generic bound, failure, order, nullability, exact-reference rule, immutability
  obligation, and lifecycle exclusion.
- Root package documentation must distinguish current analysis contracts, this finalization
  handoff, and later public orchestration/schedule work.
- Runtime and Public API references must make assignment, finalization, and `PreparedPartition`
  current only after implementation; schedules, `PreparedUnit`, `PreparedExecution`, allocation,
  residency, transfer, publication, and the runner remain planned.
- The backend partition-preparer guide must show the typed analysis-to-assignment-to-finalization
  sequence without presenting a production backend or public orchestration facade.
- The focused architecture page receives implementation-status wording only. The architecture
  contract, lifecycle, module-boundary, dependency, ADR, Compile API, Training API, and other
  explanatory documents remain review-only because this task changes no architecture rule,
  compile/training contract, or public runnable lifecycle.
- Update the glossary only for reusable current terms or changed current/planned status. Do not
  duplicate the task's validation algorithm in the glossary.
- Architecture tests need no change because module edges and prohibited dependency rules remain
  unchanged. Backend-conformance and integration documentation remain unchanged because no
  concrete backend or end-to-end execution is implemented.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, ADRs 0010/0011, the focused
Runtime/Prepare/backend lifecycle, module, dependency, and boundary docs, documentation rules and
General/API-Javadoc/Backend-Guide/Architecture/Planning profiles, the Prepare and Runtime master
plans, Prepare 0001, Runtime 0001–0004, current Prepare and directly relevant Runtime source/tests,
and docs/planning/modules/prepare/tasks/0002-backend-partition-finalization-handoff.md.

Implement Prepare 0002 exactly within its public/package-private surface and 18-path ceiling. Add
only deterministic conservative assignment, exact source associations, typed backend
finalization, the minimal PreparedPartition association, focused Prepare tests/Javadocs, and the
specified documentation/status updates. Do not add PreparedUnit, public orchestration, physical
allocation/access, closeable prepared resources, RunState/invocation/execution, schedule,
transfer/residency/materialization/publication, concrete backend, tuning/cache mutation,
graph/compiler/planning behavior, dependency/Gradle/architecture changes, or later task specs.
Stop on any architecture, package, API, validation, or scope conflict.

Run focused tests and one final Prepare module test, then all source/scope/status checks. Hand the
actual diff and exact Java evidence to a separate clean documentation-focused context. That pass
must follow documentation-rules.md, independently finalize affected Javadocs/docs/glossary/
planning evidence, and not repeat successful Java tests unless executable behavior changes or a
concrete risk is recorded. Mark Complete only after every gate passes and return both context IDs,
exact paths, commands/results/counts, unresolved issues, follow-up, and the repository completion
status format.
```

## Local decisions

- The finalizer is a separate typed collaboration from the analyzer. Its `BackendId` is validated
  against partition ownership; shared Prepare does not require analyzer and finalizer object
  identity or add backend registration.
- `BackendPartitionFinalization` carries the exact analysis rather than only the opaque plan, so
  the backend receives typed selected state while the constructor can prove assignment coverage
  against the unchanged declaration list.
- Assignments retain exact requirement and slot references plus dense plan indices. This gives a
  finalizer the source association and the coordinate required by current
  `PreparedExecutable.BufferSelection`/`WorkspaceSelection` without map lookup or slot-number
  interpretation.
- Buffer declarations for one `ValueId` share one slot. Maximum size and power-of-two alignment
  conservatively satisfy multiple backend representations after all declarations are known.
- Every workspace declaration receives a distinct slot. No workspace identity is globalized and
  no lifetime field or reuse policy is introduced.
- `PreparedPartition` stores only partition and executable. Duplicating backend ID or memory plan
  would create competing association sources.
- `PreparedUnit` remains deferred. The current prepared-partition consumer has no invariant beyond
  the existing executable recipe and the new partition association.
- The batch handoff and its result remain package-private because Prepare 0003, not this task,
  owns the eventual public orchestration boundary.
- Finalization in this task may create ordinary immutable executable recipe state only. A
  closeable/native prepared-resource lifecycle is deferred because current Runtime contracts do
  not provide cleanup or partial-failure rollback for it.

## Known limitations

- Only fully static contexts from Prepare 0001 can reach this handoff.
- Buffer geometry is conservative and has no liveness/interference optimization. Workspace
  declarations never reuse slots.
- The finalizer contract cannot mechanically prove a concrete implementation's immutability or
  that it retained its private route unchanged. Its input/output shape prevents shared Prepare
  from authorizing a replacement plan or declaration, and fake tests lock the intended pattern.
- No current production backend implements the finalizer.
- No schedule consumes `PreparedPartition` yet; Runtime 0005 is the next dependent frontier.
- No closeable persistent prepared-resource lifecycle exists, so this task forbids acquisition of
  such resources during finalization.

## Notes

- This specification is non-authoritative planning. `ARCHITECTURE.md` and ADRs 0010/0011
  remain authoritative when wording differs.
- Preserve completed Prepare 0001 and Runtime 0001–0004 history and evidence. Implementation of
  this task must add new evidence rather than rewriting predecessor results.

## Validation evidence

Implementation context: `/root/implement_prepare_0002`.

- Production Java stabilized before focused testing, and no executable Java changed afterward.
  The implementation context ran `./gradlew :modules:prepare:compileJava`; it passed.
- The implementation context ran
  `./gradlew :modules:prepare:test --tests io.github.pho001.synaptik.prepare.FinalizationPublicShapeTest --tests io.github.pho001.synaptik.prepare.BackendPartitionFinalizationTest --tests io.github.pho001.synaptik.prepare.BackendPartitionFinalizationHandoffTest`
  on each of three development runs. Every run passed 3 suites and 11 tests with 0 skipped,
  failures, or errors.
- The implementation context ran the single final
  `./gradlew :modules:prepare:test`; it passed 7 suites and 22 tests with 0 skipped, failures, or
  errors. Its `javap` public/package-private/generic checks, forbidden-mechanism and no-
  `PreparedUnit` scans, no-Java-outside-Prepare check, unchanged Gradle/build/Runtime/other-module
  checks, no-later-specification check, and `git diff --check` also passed.
- Documentation context: `/root/prepare_0002_docs`. It selected General and API/Javadoc profiles
  for the Java/API contracts, Backend Guide plus Developer Guide and Example profiles for the
  backend contributor example, Architecture profile for implementation-status wording, and
  Planning profile for the task, master plans, and roadmap.
- The documentation context read and reviewed the authoritative architecture, ADRs 0010/0011,
  planning guide, task and master plans, complete diff, all six new production/package files,
  all three new tests, current Prepare analysis contracts, directly relevant Runtime memory,
  executable, resource, and run contracts, and all nine changed Markdown files. It independently
  confirmed complete-set validation before assignment, deterministic first-declaration buffer
  assignment with maximum geometry, distinct workspace assignment, construction of every typed
  finalization before backend invocation, exact shared-plan retention, immutable result order,
  and the absence of physical or per-run behavior.
- The documentation context finalized the package-internal handoff Javadoc and independently
  retained the other five new production/package Javadocs because they already document the exact
  nullability, order, exact-reference, immutability, failure, and lifecycle contracts. No
  executable Java changed in this pass.
- `./gradlew :modules:prepare:javadoc` passed after final Javadoc edits: 9 actionable tasks, 2
  executed and 7 up-to-date. Successful Java tests were not repeated because the documentation
  pass changed no executable behavior.
- A Java 26 finalization example was compiled and executed with
  `javac -cp modules/prepare/build/classes/java/main:modules/runtime/build/classes/java/main:modules/planning/build/classes/java/main:modules/model/build/classes/java/main:modules/backend-contract/build/classes/java/main -d /tmp/prepare-finalization-example /tmp/PrepareFinalizationExample.java`
  followed by `java` with the same classes plus the temporary output directory. Both commands
  passed with no output; the example checked exact partition and memory-plan reference retention.
- `python3 /tmp/validate_synaptik_markdown.py` over the exact nine changed Markdown paths passed
  with `validated 9 Markdown files`. It checked local targets and heading anchors, unique
  effective anchors, balanced backtick/tilde fences, final newlines, and trailing whitespace.
- `javap -public` over the assignment variants, finalization, finalizer, and prepared partition,
  plus `javap -p` over the package-private handoff, entry, and result, passed. The output confirmed
  exactly four public root types, the sealed two-variant assignment family, typed
  `BackendPreparationPlan` bounds, exact record components, and a package-private static batch
  operation/result.
- Source scans confirmed no `PreparedUnit`, raw `Object` API, unchecked generic access, public or
  generic map parameter, reflection, string dispatch, registry/service locator, concrete backend,
  `RunState`, invocation/execution, allocation, transfer, residency, materialization,
  publication, or schedule behavior. The handoff's only map is a private typed implementation
  collection used during complete-set validation and assignment.
- Combined `git diff --name-only` plus untracked-file inspection reported exactly 18 task paths:
  6 Prepare production paths, 3 Prepare test paths, 5 explanatory documentation paths, and 4
  planning paths. No Java/test path outside Prepare changed.
- `git diff --exit-code` checks passed for root/Prepare/Runtime Gradle and build logic,
  `ARCHITECTURE.md`, ADRs 0010/0011, Runtime source, existing Prepare analysis source, and all
  architecture, backend-conformance, and integration tests. Root Java 26 configuration remains
  `JavaLanguageVersion.of(26)` with release 26.
- Status and frontier inspection confirmed Prepare 0001 and Runtime 0001–0004 Complete, this task
  and its Prepare master row Complete, Prepare 0003 and Runtime 0005–0008 Draft, and no detailed
  Prepare 0003 or Runtime 0005–0008 specifications.
- Final `git diff --check` passed.

Documentation decisions and no-change conclusions:

- Runtime/Public API references now mark assignment, typed finalization, and `PreparedPartition`
  current while keeping public orchestration, physical resources, schedules, publication, and
  runner behavior planned.
- The focused architecture page changes implementation status only and introduces no rule. The
  backend guide now follows analysis through a current typed finalizer example without presenting
  package-private assignment as public orchestration. The glossary adds concise reusable entries
  for preparation resource assignment, backend partition finalization, and prepared partition.
- Compile API and Training API remain unchanged because this task changes neither compilation nor
  training behavior or public contracts. Lifecycle, module-boundary, dependency, and ADR text
  remain unchanged because the implementation realizes the existing staged-preparation decision
  without changing ownership or module edges.
- Runtime source/Javadocs remain unchanged because Prepare consumes the established Runtime
  geometry/executable contracts without changing their semantics. Architecture tests do not need
  an update because dependency rules are unchanged; backend-conformance and integration tests do
  not apply because no concrete backend or end-to-end execution exists. Other backend guides,
  Gradle/build files, and other modules likewise remain unchanged.

## Implementation notes

- Added the exact four public root contracts and one package-private typed complete-set handoff.
- Assignment validates all source facts before construction, shares buffers by declared
  `ValueId` with maximum geometry, never merges workspace declarations, constructs all typed
  finalizations before calling a backend, and validates exact executable plan identity.
- Documentation preserves the current-versus-planned boundary and records the next Draft frontier
  without creating a later detailed specification.

## Completion summary

- Completed changes: deterministic shared assignment, typed backend finalization, minimal
  prepared-partition association, focused tests, finalized Javadocs, explanatory documentation,
  glossary terminology, and synchronized planning status.
- Files changed or created: exactly 18 paths — 6 Prepare production, 3 Prepare tests, 5
  explanatory documentation, and 4 planning paths.
- Tests and validation: reused the implementation context's passing compile, three focused
  11-test runs, and final 22-test module run; documentation Javadoc, Java 26 example, Markdown,
  API shape, mechanism, scope, unchanged-boundary, status, and whitespace gates passed.
- Documentation-agent review: `/root/prepare_0002_docs` completed the independent clean-context
  pass without changing executable Java or repeating successful Java tests.
- Documentation impact: Runtime/Public APIs, focused boundary status, backend contributor guide,
  glossary, task, Prepare/Runtime master plans, and roadmap finalized.
- Javadoc review: all six new production/package files reviewed; the internal handoff was expanded
  and the remaining five were retained as accurate. Prepare Javadoc generation passed.
- Glossary impact: added current assignment, backend-finalization, and prepared-partition entries
  and synchronized implementation-status wording.
- Unresolved issues: None.
- Follow-up required: None for this task. Draft Runtime 0005 is the next planning frontier; Draft
  Prepare 0003 remains dependent on its schedule contract.

Status: Complete
