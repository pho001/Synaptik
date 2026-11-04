# Task 0002: [Backend Partition Finalization](../../../../glossary.md#backend-partition-finalization) [Handoff](../../../../glossary.md#term-handoff)

## Status

Complete

## Goal

Implement the smallest [Prepare](../../../../glossary.md#prepare)-owned [handoff](../../../../glossary.md#term-handoff) that consumes a complete ordered set of current
[`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) results, assigns stable [Runtime](../../../../glossary.md#term-runtime) [buffer](../../../../glossary.md#term-buffer) and workspace slots only after
all declarations are known, constructs one exact [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan), and calls each owning
[backend](../../../../glossary.md#backend) to create a current [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) against those assignments.

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

One package-private `BackendPartitionFinalizationHandoff` [operation](../../../../glossary.md#operation) owns complete-set [validation](../../../../glossary.md#term-validation),
assignment, finalizer invocation, and the [immutable](../../../../glossary.md#term-immutable) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) result. It receives the independently
expected ordered planned partitions plus ordered typed entries containing the exact
[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), `BackendPartitionAnalysis`, and matching `BackendPartitionFinalizer`. The
package-private entry preserves the backend-specific input and selected-plan generic types
without a raw type, `Object`, generic map [parameter](../../../../glossary.md#term-parameter), [registry](../../../../glossary.md#term-registry), or service [lookup](../../../../glossary.md#term-lookup).

`PreparedUnit` is not needed in this task. The current executable already retains its exact
memory-plan association and dense resource selections; `PreparedPartition(partition, executable)`
adds only the missing [compile](../../../../glossary.md#compile)/prepare [partition](../../../../glossary.md#partition) association. Runtime 0005 must decide whether an
ordered schedule step needs a distinct `PreparedUnit` [invariant](../../../../glossary.md#term-invariant).

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
ordered expected partitions + ordered complete analyses/finalizers
  -> validate all sources, ownership, references, and declarations
  -> assign all shared slots and construct one PreparedMemoryPlan
  -> construct every typed BackendPartitionFinalization
  -> call owning finalizers in partition order
  -> immutable ordered PreparedPartition values
```

All shared decisions complete before the first [backend](../../../../glossary.md#backend) finalizer runs. A finalizer receives its
exact [opaque](../../../../glossary.md#term-opaque) analysis plan and exact source assignments, but it cannot return a replacement plan
or declarations. It returns only a [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), whose exact memory-plan reference is
validated before the [prepared partition](../../../../glossary.md#prepared-partition--preparedpartition) is accepted.

## [Scope](../../../../glossary.md#term-scope)

- Add the exact four public root-package declarations above and root package documentation.
- Add one package-private finalization [handoff](../../../../glossary.md#term-handoff) with package-private typed `Entry` and `Result`
  records; add no public orchestration facade.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- Accept an independently supplied ordered `List<PlannedPartition>` as the expected complete
  coverage and an equally sized ordered list of finalization entries.
- Require each entry's context partition and analysis partition to be the exact expected
  partition reference at that index.
- Require the finalizer's non-null [`BackendId`](../../../../glossary.md#backend-identity--backendid) to equal the expected [partition](../../../../glossary.md#partition) [owner](../../../../glossary.md#term-owner).
- Validate every buffer declaration against the entry's projected values and matching logical
  memory requirement before assignment.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- When the same [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) is declared by more than one analysis, require the projections to retain
  the same exact `GraphValue` and `LogicalMemoryRequirement` references.
- Traverse analyses in expected partition order and each analysis's requirements in stored order.
- Assign [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) values densely from zero in first-declaration order, one slot per distinct
  declared `ValueId` across the complete set.
- Give a repeated buffer value one plan entry whose byte size is the maximum declared size and
  whose alignment is the maximum declared alignment. Because all alignments are powers of two,
  the maximum satisfies every declaration without inventing another alignment rule.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- Assign [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) values densely from zero in declaration encounter order, one distinct
  slot for every workspace declaration. Workspace IDs remain local to their analysis and are
  never merged across partitions.
- Add no [lifetime](../../../../glossary.md#term-lifetime), interference, [aliasing](../../../../glossary.md#term-aliasing), or reuse [model](../../../../glossary.md#term-model).
- Construct buffer plan entries in first buffer-declaration order and workspace entries in
  workspace-declaration order. Preserve those exact slot and entry references in every source
  assignment.
- Give each analysis one immutable assignment list in exact requirement order. Each assignment
  retains the exact requirement object, assigned slot, and dense index into the corresponding
  [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) list.
- Construct and validate every `BackendPartitionFinalization` before invoking any [backend](../../../../glossary.md#backend).
- Invoke finalizers exactly once each in expected partition order after assignment completes.
- Require each returned executable to be non-null and retain the exact shared memory-plan
  reference.
- Produce [immutable](../../../../glossary.md#term-immutable) ordered [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) values retaining the exact planned-partition and
  executable references.
- Add focused contract, assignment, finalization, failure-order, and public-shape tests with fake
  typed backend inputs, opaque plans, finalizers, and executables.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- Finalize Javadocs and affected explanatory documentation in a separate clean documentation
  context after implementation.

## Out of [scope](../../../../glossary.md#term-scope)

- physical allocation, storage access, representation creation, native resource acquisition, or
  executable-resource cleanup
- [`RunState`](../../../../glossary.md#run-state--runstate) creation, input [binding](../../../../glossary.md#term-binding), cold invocation binding, or execution
- `PreparedUnit`, [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), schedule steps, scheduling, or [Runtime](../../../../glossary.md#term-runtime) 0005
- transfer, [residency](../../../../glossary.md#residency), validity, [materialization](../../../../glossary.md#materialization), [publication](../../../../glossary.md#term-publication), [`RunResult`](../../../../glossary.md#run-result--runresult), or runner behavior
- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [immutable](../../../../glossary.md#term-immutable) persistent prepared-resource [ownership](../../../../glossary.md#term-ownership), or [lifecycle](../../../../glossary.md#lifecycle) closing
- a public prepare orchestration facade, Engine composition, registration, discovery, registry,
  service locator, builder, manager, or service<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [Engine glossary](../../../../glossary.md#term-engine); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- a concrete backend implementation, concrete executable, route, kernel, native bridge, or
  physical representation<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [kernel glossary](../../../../glossary.md#kernel); [native bridge glossary](../../../../glossary.md#term-native-bridge))</span>
- changing, repeating, tuning, searching, or scoring the selected route
- adding declarations during finalization or changing the analysis plan
- [model](../../../../glossary.md#term-model)-[autotuning](../../../../glossary.md#autotuning--model-autotuning) measurement, candidate enumeration, [cache](../../../../glossary.md#term-cache) [lookup](../../../../glossary.md#term-lookup), or cache mutation
- [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), [graph optimization](../../../../glossary.md#forward-graph-optimization), [compiler](../../../../glossary.md#term-compiler) behavior, [planning](../../../../glossary.md#term-planning) behavior, or [compile artifacts](../../../../glossary.md#compile-artifacts)
- dynamic [dimension](../../../../glossary.md#dimension) binding or a [run](../../../../glossary.md#term-run)-dynamic resource contract
- pooling, slot lifetime reuse, aliasing, interference analysis, distributed sharding, or
  multi-device scheduling<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [lifetime glossary](../../../../glossary.md#term-lifetime); [aliasing glossary](../../../../glossary.md#term-aliasing); [device glossary](../../../../glossary.md#term-device))</span>
- Config, Trace, Backend Contract, Model, Planning, Compiler, Runtime, Engine, concrete-backend,
  Gradle, dependency, architecture-contract, ADR, architecture-test, conformance-test, or
  integration-test changes<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [Prepare](../../../../glossary.md#prepare) 0003, Runtime 0005, or any later detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core [lifecycle](../../../../glossary.md#lifecycle) and invariants
  - `modules/runtime`
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [Prepare](../../../../glossary.md#prepare) lifecycle
  - [Run](../../../../glossary.md#term-run) lifecycle
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [ADR 0010: Stage backend preparation around shared slot assignment](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)

## Architecture constraints

- Shared Prepare owns complete-set validation, deterministic assignment, exact source
  associations, backend finalization coordination, and <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend))</span>[`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition).
- [Runtime](../../../../glossary.md#term-runtime) owns [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot), [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot), [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan), and [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable).
  Runtime remains free of Prepare, Model, Planning, and Compiler facts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Concrete backends own the opaque selected plan and executable subclass. Shared Prepare never
  inspects, downcasts, copies, or interprets backend-private plan fields.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- Backend analysis and route choice are complete before this handoff. Finalization receives the
  exact analysis and cannot replace its route, plan, or declarations.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- All assignment and shared validation complete before any finalizer call. No backend may observe
  a partial shared assignment.
- Prepared recipes are [immutable](../../../../glossary.md#term-immutable) and reusable. This task creates no per-[run](../../../../glossary.md#term-run) [mutable](../../../../glossary.md#term-mutable) state.
- Runtime hot-path code remains free of [`Operation`](../../../../glossary.md#operation), `CompiledNode`, maps, graph inspection,
  backend discovery, and selection work. The new Model/Planning associations stay in Prepare.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Prepare remains independent of concrete backend implementations. A typed finalizer is supplied
  explicitly; no lookup or public backend switch is added.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- No new [module](../../../../glossary.md#term-module) edge or architecture rule is permitted. Stop if implementation requires one.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.prepare.analysis` — consume the current typed analysis contracts
  without changing their public shape or semantics.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- `io.github.pho001.synaptik.prepare` — the master plan reserves this root for the finalized
  prepared-partition result and narrow shared handoff/validation contracts.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation))</span>

Packages added or changed:

- no package beyond the planned [Prepare](../../../../glossary.md#prepare) root; `prepare.analysis` receives Javadoc review only
  unless a concrete contradiction is found.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Type placement:

- `PreparationResourceAssignment` — public exact declaration-to-Runtime-slot association that a
  concrete backend finalizer consumes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- `BackendPartitionFinalization` — public typed [immutable](../../../../glossary.md#term-immutable) finalization input for one analysis.
- `BackendPartitionFinalizer` — public concrete-[backend](../../../../glossary.md#backend)-implemented typed collaboration.
- [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) — public minimal association of one exact planned partition with its one
  finalized executable.
- `BackendPartitionFinalizationHandoff` — package-private shared complete-set operation; its
  nested typed <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`Entry` and immutable `Result` remain internal until Prepare 0003 supplies public
  orchestration.

Tests mirror `io.github.pho001.synaptik.prepare` so they can exercise the package-private [handoff](../../../../glossary.md#term-handoff).
No `util`, `service`, [`registry`](../../../../glossary.md#term-registry), or generic orchestration package is added.

## Exact public and package-private [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape)

The four public declarations are exactly the surface shown in the Goal. Constructors and record
accessors are explicit only where complete [Javadoc](../../../../glossary.md#term-javadoc) is required; they add no member beyond the
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

The [operation](../../../../glossary.md#operation) may use private helpers and private implementation-only collections inside the same
file. It must not expose a map [parameter](../../../../glossary.md#term-parameter) or result, use a raw type or `Object`, or add another
top-level type. Wildcard [capture](../../../../glossary.md#term-capture-meanings) must preserve each entry's `P` association when constructing the
typed finalization and calling its finalizer.

## Affected files

Expected [Prepare](../../../../glossary.md#prepare) production paths:

- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparationResourceAssignment.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalization.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizer.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedPartition.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/package-info.java`

Expected Prepare test paths:

- up to one exact public/package-private [shape](../../../../glossary.md#shape) test under the mirrored root package;
- up to one finalization-value [validation](../../../../glossary.md#term-validation) test; and
- up to one complete [handoff](../../../../glossary.md#term-handoff)/assignment/failure-order test.

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status clarification
  only; no architecture rule change
- `docs/backend-guide/partition-preparer.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs 0010/0011,
[lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) docs, Prepare 0001, [Runtime](../../../../glossary.md#term-runtime) 0001–0004, all current Prepare and relevant
Runtime source/tests/Javadocs, [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training API, current [backend](../../../../glossary.md#backend) guides, build files, and
architecture/[conformance](../../../../glossary.md#term-conformance)/integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 18 paths:

- 6 [Prepare](../../../../glossary.md#prepare) production paths;
- 3 Prepare test paths;
- 5 explanatory documentation paths; and
- 4 Prepare/[Runtime](../../../../glossary.md#term-runtime)/global [planning](../../../../glossary.md#term-planning) paths.

No Java/test path outside `modules/prepare`, [Gradle](../../../../glossary.md#term-gradle) path, [architecture contract](../../../../glossary.md#architecture-contract), [ADR](../../../../glossary.md#term-architecture-decision-record-adr),
architecture-test, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration-test path may change. If another public
type, package, [module](../../../../glossary.md#term-module) edge, [lifecycle](../../../../glossary.md#lifecycle) [owner](../../../../glossary.md#term-owner), or path is required, stop and report a follow-up.
Do not create Prepare 0003, Runtime 0005, or another detailed task specification.

## Assignment semantics

The [handoff](../../../../glossary.md#term-handoff) validates all source facts before constructing assignments. It then performs one
complete deterministic traversal in expected [partition](../../../../glossary.md#partition) order and requirement order.

For each first-seen [buffer](../../../../glossary.md#term-buffer) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), create a new [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot) whose numeric value is the current
buffer count. Later equal IDs reuse that exact slot reference. The final `BufferEntry` uses the
maximum declared byte size and maximum declared [byte alignment](../../../../glossary.md#term-byte-alignment) for the value. The entry appears at
the value's first declaration position relative to other distinct buffer values.

For every workspace declaration, create a fresh [`WorkspaceSlot`](../../../../glossary.md#workspace-slot--workspaceslot) whose numeric value is the
current workspace count. Its `WorkspaceEntry` copies that declaration's exact size and [alignment](../../../../glossary.md#term-alignment).
Even equal analysis-local requirement IDs in different analyses receive distinct slots.

Each source declaration receives exactly one assignment in the owning analysis's requirement
order. A buffer assignment retains its exact `Buffer` requirement, the exact shared slot, and the
buffer entry's dense index. A workspace assignment retains its exact `Workspace` requirement, its
exact distinct slot, and the workspace entry's dense index. The plan and assignment lists are
[immutable](../../../../glossary.md#term-immutable) snapshots. No source collection container is retained.

## [Validation](../../../../glossary.md#term-validation), [ownership](../../../../glossary.md#term-ownership-meanings), order, and failure rules

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
   equal its declaration exactly.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [alignment glossary](../../../../glossary.md#term-alignment))</span>

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

After [validation](../../../../glossary.md#term-validation), the record snapshots the assignment list and retains every exact [immutable](../../../../glossary.md#term-immutable)
analysis, plan, assignment, requirement, and slot reference.

`BackendPartitionFinalizer.backendId()` must return a non-null immutable [identity](../../../../glossary.md#term-identity-meanings). The [handoff](../../../../glossary.md#term-handoff)
calls it once during complete-set validation. `finalizePartition` receives one non-null validated
finalization and returns one non-null immutable [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable). It may validate [backend](../../../../glossary.md#backend)
state and construct ordinary immutable Java recipe state, but this task does not permit physical
allocation, closeable/native resource acquisition, route reselection, new shared requirements,
or mutation of analysis/[cache](../../../../glossary.md#term-cache) state.

[`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) validates [`partition`](../../../../glossary.md#partition) before `executable`, snapshots nothing, and retains both
exact immutable references. Exact null messages are `partition` and `executable`. [Backend
ownership](../../../../glossary.md#backend-ownership) is obtained from `partition.owner()`; the record does not duplicate [`BackendId`](../../../../glossary.md#backend-identity--backendid).
Executable-to-plan association is validated by the handoff rather than by this independently
constructible two-component value.

### Complete [handoff](../../../../glossary.md#term-handoff)

The package-private [operation](../../../../glossary.md#operation) validates in this exact order:

1. require `partitions`, then `entries` non-null;
2. [scan](../../../../glossary.md#term-cumulative-scan) expected partitions in order for the first null or later equal duplicate;
3. scan entries in order for the first null and validate each entry's `context`, `analysis`, and
   `finalizer` components in declaration order;
4. require equal list sizes;
5. for each index, require exact context-[partition](../../../../glossary.md#partition) and analysis-partition [reference identity](../../../../glossary.md#term-reference-identity);
6. call `backendId()` once, reject null, and require value equality with `partition.owner()`;
7. validate every buffer source in requirement order against the context's unique projected
   value and logical requirement;
8. for repeated cross-analysis [buffer](../../../../glossary.md#term-buffer) IDs, require exact projected `GraphValue` and
   `LogicalMemoryRequirement` reference identity with the first declaration;
9. only after all entries pass, derive every assignment and the complete memory plan;
10. construct every typed finalization and validate it before any finalizer call;
11. call each finalizer once in partition order, stopping at the first thrown unchecked failure;
12. reject a null executable, then require `executable.memoryPlan() == memoryPlan`; and
13. construct and [snapshot](../../../../glossary.md#term-snapshot) the ordered prepared partitions and return the exact memory plan.

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

An empty complete set is valid and produces an empty [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) and empty [prepared
partition](../../../../glossary.md#prepared-partition--preparedpartition) list without calling a finalizer. All list structures are [immutable](../../../../glossary.md#term-immutable) snapshots. The
operation retains exact immutable [elements](../../../../glossary.md#term-element) and does not mutate supplied lists, contexts,
analyses, plans, requirements, finalizers, or executables.

No finalizer is called when any shared [validation](../../../../glossary.md#term-validation), coverage, source-reference, assignment, or
finalization-construction check fails. When a finalizer itself throws `RuntimeException` or
`Error`, the exact failure propagates and later finalizers are not called. Earlier returned
executables own no closeable resource under this task, so no [rollback](../../../../glossary.md#term-rollback) or cleanup protocol is
introduced.

## Acceptance criteria

- The exact four public declarations and one package-private handoff surface exist in the planned
  Prepare root package; no facade, registry, service, builder, manager, or <span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [Prepare glossary](../../../../glossary.md#prepare); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry))</span>`PreparedUnit` exists.
- Generic bounds preserve each opaque backend plan's association with its finalizer without raw
  types, <span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend))</span>`Object`, unchecked public access, or a generic map [parameter](../../../../glossary.md#term-parameter).
- Complete-set coverage, exact partition references, backend ownership, buffer source membership,
  and repeated-source references fail closed in the specified order and with exact messages.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [backend ownership glossary](../../../../glossary.md#backend-ownership); [buffer glossary](../../../../glossary.md#term-buffer); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Assignment begins only after every analysis passes shared validation; every finalization is
  constructed before any backend is called.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Buffer slots are dense and deterministic in first declaration order. Repeated declared values
  share the exact slot and use maximum size/alignment geometry.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- Workspace slots are dense and deterministic in declaration order. Every declaration receives a
  distinct exact slot with unchanged geometry.
- Every requirement has one exact source assignment in requirement order, and every assignment's
  dense index, slot reference, and plan geometry are validated.
- Finalizers are called once in partition order only after assignment, see the exact typed
  analysis/plan and assignments, and cannot return changed route/declaration data.
- Null, foreign-backend, foreign-partition, foreign-source, foreign-slot, out-of-range, geometry,
  null-executable, and foreign-plan cases fail exactly as specified.
- [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) is only the exact two-reference partition/executable association. Its
  backend identity comes from the partition and its memory plan from the executable.<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
- Tests prove empty input, list snapshots, exact reference retention, stable order, maximum
  geometry, separate slot domains, no reuse, no finalizer-before-validation, stop-on-first
  finalizer failure, and exact output order.
- No physical resource, [`RunState`](../../../../glossary.md#run-state--runstate), invocation, execution, scheduling, transfer, residency,
  materialization, publication, concrete backend, tuning, cache mutation, graph/compiler/planning
  behavior, dependency, Gradle, architecture, conformance, or integration change is added.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication); [concrete backend glossary](../../../../glossary.md#concrete-backend); [cache glossary](../../../../glossary.md#term-cache); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Every public and protected member has meaningful Javadoc covering input constraints,
  nullability, result, exact-reference ownership, immutability, order, failure, and lifecycle.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs, Runtime/Public APIs,
  focused boundary and backend guide status, glossary impact, and planning evidence without
  repeating successful Java tests unless executable behavior changes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Prepare 0001 and Runtime 0001–0004 remain Complete. Prepare 0002 becomes Complete only after all
  implementation/documentation gates. Prepare 0003 and Runtime 0005–0008 remain Draft without
  detailed specifications.
- Exact [scope](../../../../glossary.md#term-scope), link, anchor, fence, newline, whitespace, and `git diff --check` gates pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused validation:

```bash
./gradlew :modules:prepare:test
```

[Run](../../../../glossary.md#term-run) focused new suites while developing, then run the command above once after executable Java
stabilizes. Existing [Runtime](../../../../glossary.md#term-runtime) tests are not repeated because Runtime Java does not change.

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
[targets](../../../../glossary.md#term-target) and heading anchors, unique effective anchors, balanced backtick/tilde fences, final
newlines, and trailing whitespace.

Required source/[scope](../../../../glossary.md#term-scope)/status checks:

- exact public/package-private/nested surface, generic bounds, modifiers, packages, and no
  `PreparedUnit`;
- exact validation order/messages, list snapshots, assignment traversal, first-declaration
  ordering, maximum buffer geometry, workspace non-reuse, exact references, and finalizer calls;<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- production [scan](../../../../glossary.md#term-cumulative-scan) for no raw `Object`, unchecked generic access, public/generic map parameter,
  reflection, string dispatch, registry/service locator, concrete backend, or hot-path behavior;<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- no modification to existing [Prepare](../../../../glossary.md#prepare) analysis semantics or Runtime contracts;
- unchanged root/Prepare/Runtime [Gradle](../../../../glossary.md#term-gradle) files and Java 26 root [toolchain](../../../../glossary.md#term-toolchain)/[release](../../../../glossary.md#term-release) configuration;
- exact 18-path ceiling and no Java/test outside Prepare;
- Prepare 0001, Runtime 0001–0004, task/master/roadmap status synchronization, and absence of
  Prepare 0003, Runtime 0005+, or other later specifications; and<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- final newline, trailing whitespace, and `git diff --check`.

Repository-wide and architecture tests are deferred to the Prepare contract [checkpoint](../../../../glossary.md#term-checkpoint) or
[continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one [module](../../../../glossary.md#term-module) without a module edge, [dependency](../../../../glossary.md#term-dependency) rule,
shared build contract, [concrete backend](../../../../glossary.md#concrete-backend), or end-to-end execution. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and
integration tests are not applicable.

The documentation context reuses successful Prepare test evidence unless it changes executable
Java behavior or records a concrete reason to repeat it.

## Dependencies

- [Prepare](../../../../glossary.md#prepare) 0001 — Complete.
- [Runtime](../../../../glossary.md#term-runtime) 0001–0004 — Complete.
- [Planning](../../../../glossary.md#term-planning) 0001–0006 — Complete.
- [Compiler](../../../../glossary.md#term-compiler) 0001–0006 — Complete.
- [Backend](../../../../glossary.md#backend) Contract 0001–0004 — Complete and closed.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010 staged preparation and ADR 0011 runtime [ownership](../../../../glossary.md#term-ownership-meanings)/cold [binding](../../../../glossary.md#term-binding) — Accepted.
- Current Prepare dependencies already include Runtime, Planning, Compiler, Config, Backend
  Contract, and Trace. No build change is needed.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>

## Follow-up tasks

- Prepare 0003 will compose public prepare orchestration plus partition, prepared-memory, and
  later schedule validation after Runtime 0005 supplies the schedule contract.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Runtime 0005 will define the prepared schedule and decide whether an actual schedule-step
  invariant justifies <span data-glossary-navigation="true"> ([prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule); [invariant glossary](../../../../glossary.md#term-invariant))</span>`PreparedUnit`.
- Runtime 0006–0008 remain Draft for the prepared aggregate, runner/dynamic execution, and closure
  audit.
- Concrete backend implementation remains in each backend plan after the shared contracts are
  complete.<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [backend glossary](../../../../glossary.md#backend))</span>

All follow-ups remain Draft without detailed specifications. Do not create them in this task.

## Architecture impact

Expected impact: None.

The architecture and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010 already assign shared slot assignment and source associations to
[Prepare](../../../../glossary.md#prepare), final executable construction to the owning [backend](../../../../glossary.md#backend) after assignment, [Runtime](../../../../glossary.md#term-runtime) geometry
and executable contracts to Runtime, and [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) to Prepare. This task implements that
boundary without changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) or [dependency direction](../../../../glossary.md#java-platform-and-build-tools). Stop if implementation needs
a [concrete backend](../../../../glossary.md#concrete-backend) [dependency](../../../../glossary.md#term-dependency), Runtime knowledge of Prepare facts, route/declaration mutation,
physical allocation, a prepared-resource cleanup [lifecycle](../../../../glossary.md#lifecycle), or another architecture decision.

## Javadocs and documentation impact

- Document every new public type, record component, explicit constructor/accessor, finalizer
  method, generic bound, failure, order, nullability, exact-reference rule, immutability
  obligation, and lifecycle exclusion.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Root package documentation must distinguish current analysis contracts, this finalization
  handoff, and later public orchestration/schedule work.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- [Runtime](../../../../glossary.md#term-runtime) and Public [API](../../../../glossary.md#term-application-programming-interface-api) references must make assignment, finalization, and [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition)
  current only after implementation; schedules, `PreparedUnit`, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), allocation,
  residency, transfer, publication, and the runner remain planned.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication))</span>
- The backend partition-preparer guide must show the typed analysis-to-assignment-to-finalization
  sequence without presenting a production backend or public orchestration facade.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [partition glossary](../../../../glossary.md#partition); [facade glossary](../../../../glossary.md#term-facade))</span>
- The focused architecture page receives implementation-status wording only. The architecture
  contract, lifecycle, module-boundary, dependency, ADR, Compile API, Training API, and other
  explanatory documents remain review-only because this task changes no architecture rule,
  compile/training contract, or public runnable lifecycle.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Compile glossary](../../../../glossary.md#compile))</span>
- Update the glossary only for reusable current terms or changed current/planned status. Do not
  duplicate the task's validation algorithm in the glossary.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Architecture tests need no change because module edges and prohibited dependency rules remain
  unchanged. Backend-conformance and integration documentation remain unchanged because no
  concrete backend or end-to-end execution is implemented.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>

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

- The finalizer is a separate typed collaboration from the analyzer. Its [`BackendId`](../../../../glossary.md#backend-identity--backendid) is validated
  against partition ownership; shared Prepare does not require analyzer and finalizer object
  identity or add backend registration.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- `BackendPartitionFinalization` carries the exact analysis rather than only the opaque plan, so
  the backend receives typed selected state while the constructor can prove assignment coverage
  against the unchanged declaration list.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- Assignments retain exact requirement and slot references plus dense plan indices. This gives a
  finalizer the source association and the coordinate required by current<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
  `PreparedExecutable.BufferSelection`/`WorkspaceSelection` without map lookup or slot-number
  interpretation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Buffer](../../../../glossary.md#term-buffer) declarations for one [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) share one slot. Maximum size and power-of-two alignment
  conservatively satisfy multiple backend representations after all declarations are known.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- Every workspace declaration receives a distinct slot. No workspace identity is globalized and
  no lifetime field or reuse policy is introduced.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) stores only partition and executable. Duplicating backend ID or memory plan
  would create competing association sources.
- `PreparedUnit` remains deferred. The current prepared-partition consumer has no invariant beyond
  the existing executable recipe and the new partition association.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>
- The batch handoff and its result remain package-private because Prepare 0003, not this task,
  owns the eventual public orchestration boundary.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Finalization in this task may create ordinary immutable executable recipe state only. A
  closeable/native prepared-resource lifecycle is deferred because current Runtime contracts do
  not provide cleanup or partial-failure rollback for it.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## Known limitations

- Only fully static contexts from [Prepare](../../../../glossary.md#prepare) 0001 can reach this [handoff](../../../../glossary.md#term-handoff).
- Buffer geometry is conservative and has no liveness/interference optimization. Workspace
  declarations never reuse slots.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- The finalizer contract cannot mechanically prove a concrete implementation's immutability or
  that it retained its private route unchanged. Its input/output shape prevents shared Prepare
  from authorizing a replacement plan or declaration, and fake tests lock the intended pattern.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- No current production [backend](../../../../glossary.md#backend) implements the finalizer.
- No schedule consumes [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) yet; [Runtime](../../../../glossary.md#term-runtime) 0005 is the next dependent frontier.
- No closeable persistent prepared-resource lifecycle exists, so this task forbids acquisition of
  such resources during finalization.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Notes

- This specification is non-authoritative [planning](../../../../glossary.md#term-planning). `ARCHITECTURE.md` and ADRs 0010/0011
  remain authoritative when wording differs.
- Preserve completed Prepare 0001 and Runtime 0001–0004 history and evidence. Implementation of
  this task must add new evidence rather than rewriting predecessor results.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context: `/root/implement_prepare_0002`.

- Production Java stabilized before focused testing, and no executable Java changed afterward.
  The implementation context ran `./gradlew :modules:prepare:compileJava`; it passed.
- The implementation context ran
  `./gradlew :modules:prepare:test --tests io.github.pho001.synaptik.prepare.FinalizationPublicShapeTest --tests io.github.pho001.synaptik.prepare.BackendPartitionFinalizationTest --tests io.github.pho001.synaptik.prepare.BackendPartitionFinalizationHandoffTest`
  on each of three development runs. Every run passed 3 suites and 11 tests with 0 skipped,
  failures, or errors.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The implementation context ran the single final
  `./gradlew :modules:prepare:test`; it passed 7 suites and 22 tests with 0 skipped, failures, or
  errors. Its `javap` public/package-private/generic checks, forbidden-mechanism and no-
  `PreparedUnit` scans, no-Java-outside-Prepare check, unchanged Gradle/build/Runtime/other-module
  checks, no-later-specification check, and <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`git diff --check` also passed.
- Documentation context: `/root/prepare_0002_docs`. It selected General and API/Javadoc profiles
  for the Java/API contracts, Backend Guide plus Developer Guide and Example profiles for the
  backend contributor example, Architecture profile for implementation-status wording, and
  Planning profile for the task, master plans, and roadmap.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation context read and reviewed the authoritative architecture, ADRs 0010/0011,
  planning guide, task and master plans, complete diff, all six new production/package files,
  all three new tests, current Prepare analysis contracts, directly relevant Runtime memory,
  executable, resource, and run contracts, and all nine changed Markdown files. It independently
  confirmed complete-set validation before assignment, deterministic first-declaration buffer
  assignment with maximum geometry, distinct workspace assignment, construction of every typed
  finalization before backend invocation, exact shared-plan retention, immutable result order,
  and the absence of physical or per-run behavior.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- The documentation context finalized the package-internal handoff Javadoc and independently
  retained the other five new production/package Javadocs because they already document the exact
  nullability, order, exact-reference, immutability, failure, and lifecycle contracts. No
  executable Java changed in this pass.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [nullability glossary](../../../../glossary.md#term-nullability); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `./gradlew :modules:prepare:javadoc` passed after final Javadoc edits: 9 actionable tasks, 2
  executed and 7 up-to-date. Successful Java tests were not repeated because the documentation
  pass changed no executable behavior.
- A Java 26 finalization example was compiled and executed with
  `javac -cp modules/prepare/build/classes/java/main:modules/runtime/build/classes/java/main:modules/planning/build/classes/java/main:modules/model/build/classes/java/main:modules/backend-contract/build/classes/java/main -d /tmp/prepare-finalization-example /tmp/PrepareFinalizationExample.java`
  followed by `java` with the same classes plus the temporary output directory. Both commands
  passed with no output; the example checked exact partition and memory-plan reference retention.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- `python3 /tmp/validate_synaptik_markdown.py` over the exact nine changed Markdown paths passed
  with `validated 9 Markdown files`. It checked local targets and heading anchors, unique
  effective anchors, balanced backtick/tilde fences, final newlines, and trailing whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `javap -public` over the assignment variants, finalization, finalizer, and prepared partition,
  plus <span data-glossary-navigation="true"> ([prepared partition glossary](../../../../glossary.md#prepared-partition--preparedpartition))</span>`javap -p` over the package-private handoff, entry, and result, passed. The output confirmed
  exactly four public root types, the sealed two-variant assignment family, typed<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>
  `BackendPreparationPlan` bounds, exact record components, and a package-private static batch
  operation/result.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [operation glossary](../../../../glossary.md#operation))</span>
- Source scans confirmed no `PreparedUnit`, raw `Object` API, unchecked generic access, public or
  generic map parameter, reflection, string dispatch, registry/service locator, concrete backend,<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  [`RunState`](../../../../glossary.md#run-state--runstate), invocation/execution, allocation, transfer, residency, materialization,
  publication, or schedule behavior. The handoff's only map is a private typed implementation
  collection used during complete-set validation and assignment.<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>
- Combined `git diff --name-only` plus untracked-file inspection reported exactly 18 task paths:
  6 Prepare production paths, 3 Prepare test paths, 5 explanatory documentation paths, and 4
  planning paths. No Java/test path outside Prepare changed.
- `git diff --exit-code` checks passed for root/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) [Gradle](../../../../glossary.md#term-gradle) and build logic,
  `ARCHITECTURE.md`, ADRs 0010/0011, Runtime source, existing Prepare analysis source, and all
  architecture, backend-conformance, and integration tests. Root Java 26 configuration remains<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
  `JavaLanguageVersion.of(26)` with [release](../../../../glossary.md#term-release) 26.
- Status and frontier inspection confirmed Prepare 0001 and Runtime 0001–0004 Complete, this task
  and its Prepare master row Complete, Prepare 0003 and Runtime 0005–0008 Draft, and no detailed
  Prepare 0003 or Runtime 0005–0008 specifications.
- Final `git diff --check` passed.

Documentation decisions and no-change conclusions:

- Runtime/Public [API](../../../../glossary.md#term-application-programming-interface-api) references now mark assignment, typed finalization, and [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition)
  current while keeping public orchestration, physical resources, schedules, publication, and
  runner behavior planned.
- The focused architecture page changes implementation status only and introduces no rule. The
  backend guide now follows analysis through a current typed finalizer example without presenting
  package-private assignment as public orchestration. The glossary adds concise reusable entries
  for preparation resource assignment, backend partition finalization, and prepared partition.<span data-glossary-navigation="true"> ([preparation resource assignment glossary](../../../../glossary.md#preparation-resource-assignment); [backend partition finalization glossary](../../../../glossary.md#backend-partition-finalization))</span>
- Compile API and Training API remain unchanged because this task changes neither compilation nor
  training behavior or public contracts. Lifecycle, module-boundary, dependency, and ADR text
  remain unchanged because the implementation realizes the existing staged-preparation decision
  without changing ownership or module edges.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Runtime source/Javadocs remain unchanged because Prepare consumes the established Runtime
  geometry/executable contracts without changing their semantics. Architecture tests do not need
  an update because dependency rules are unchanged; backend-conformance and integration tests do
  not apply because no concrete backend or end-to-end execution exists. Other backend guides,
  Gradle/build files, and other modules likewise remain unchanged.

## Implementation notes

- Added the exact four public root contracts and one package-private typed complete-set [handoff](../../../../glossary.md#term-handoff).
- Assignment validates all source facts before construction, shares [buffers](../../../../glossary.md#term-buffer) by declared
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) with maximum geometry, never merges workspace declarations, constructs all typed
  finalizations before calling a backend, and validates exact executable plan identity.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Documentation preserves the current-versus-planned boundary and records the next Draft frontier
  without creating a later detailed specification.

## Completion summary

- Completed changes: deterministic shared assignment, typed backend finalization, minimal
  prepared-partition association, focused tests, finalized Javadocs, explanatory documentation,
  glossary terminology, and synchronized planning status.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [partition glossary](../../../../glossary.md#partition); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly 18 paths — 6 Prepare production, 3 Prepare tests, 5
  explanatory documentation, and 4 planning paths.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- Tests and validation: reused the implementation context's passing compile, three focused
  11-test runs, and final 22-test module run; documentation Javadoc, Java 26 example, Markdown,
  API shape, mechanism, scope, unchanged-boundary, status, and whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compile glossary](../../../../glossary.md#compile); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/prepare_0002_docs` completed the independent clean-context
  pass without changing executable Java or repeating successful Java tests.
- Documentation impact: Runtime/Public APIs, focused boundary status, backend contributor guide,
  glossary, task, Prepare/Runtime master plans, and roadmap finalized.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: all six new production/package files reviewed; the internal handoff was expanded
  and the remaining five were retained as accurate. Prepare Javadoc generation passed.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- Glossary impact: added current assignment, backend-finalization, and prepared-partition entries
  and synchronized implementation-status wording.
- Unresolved issues: None.
- Follow-up required: None for this task. Draft Runtime 0005 is the next planning frontier; Draft
  Prepare 0003 remains dependent on its schedule contract.

Status: Complete
