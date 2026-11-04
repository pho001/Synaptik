# Task 0003: [Prepare](../../../../glossary.md#prepare) Orchestration and [Validation](../../../../glossary.md#term-validation)

## Status

Complete

## Goal

Close the shared [Prepare](../../../../glossary.md#prepare) milestone with one explicit orchestration [operation](../../../../glossary.md#operation) that consumes one
complete [immutable](../../../../glossary.md#term-immutable) `CompileArtifacts`, one positionally matching typed [backend](../../../../glossary.md#backend) preparation for
each planned [partition](../../../../glossary.md#partition), and one Prepare-owned schedule assembler. The operation projects every
partition context, performs all backend analyses, assigns the shared memory plan, finalizes every
partition, assembles one complete [Runtime](../../../../glossary.md#term-runtime) schedule, validates that schedule against [Compiler](../../../../glossary.md#term-compiler) and
Prepare facts, and returns one exact [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution).

The new public surface is deliberately small:

```java
package io.github.pho001.synaptik.prepare;

public record PartitionPreparation<
        I extends BackendAnalysisInputs,
        P extends BackendPreparationPlan>(
        I backendInputs,
        BackendPartitionPreparer<I, P> preparer,
        BackendPartitionFinalizer<P> finalizer) {}

public record PreparedBufferAssignment(
        ValueId valueId,
        BufferSlot slot,
        int planIndex) {}

public record PreparedScheduleContext(
        CompileArtifacts artifacts,
        PreparedMemoryPlan memoryPlan,
        List<PreparedPartition> partitions,
        List<PreparedBufferAssignment> bufferAssignments) {}

@FunctionalInterface
public interface PreparedScheduleAssembler {
    PreparedSchedule assemble(PreparedScheduleContext context);
}

public final class GraphPreparation {
    public static PreparedExecution prepare(
            CompileArtifacts artifacts,
            List<? extends PartitionPreparation<?, ?>> preparations,
            PreparedScheduleAssembler scheduleAssembler);
}
```

The task also makes one minimal consumer-driven extension to the existing Runtime representation
origin family:

```java
public record InitializedBuffer(BufferCreator creator) implements BufferPreparation {}
```

`InitializedBuffer` is [run](../../../../glossary.md#term-run)-owned like `CreatedBuffer`, invokes the same concrete-backend-owned
creator during cold state construction, and differs only by declaring that the returned
representation already contains the correct logical value and is initially valid. Prepare permits
it only for an exact [compile-time constant source](../../../../glossary.md#compile-time-constant-source). Runtime does not receive [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), inspect
bytes, or materialize the value itself.

[`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) has no instances or fields. `PreparedScheduleAssembler` is a narrow
Prepare-owned composition seam, not a backend service, [registry](../../../../glossary.md#term-registry), discovery mechanism, or public
[Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade). A [concrete backend](../../../../glossary.md#concrete-backend) still sees only its typed [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), analysis, and
finalization contracts; it never receives `CompileArtifacts` or another Compiler aggregate.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
CompileArtifacts
  + one explicit typed PartitionPreparation per planned partition
  + one explicit PreparedScheduleAssembler
  -> project and validate every PrepareContext
  -> analyze every partition in order
  -> assign one shared PreparedMemoryPlan
  -> finalize one PreparedExecutable per partition
  -> expose immutable schedule-construction facts once
  -> assemble and validate one complete PreparedSchedule
  -> PreparedExecution(memoryPlan, schedule)
```

Tasks 0001 and 0002 deliberately stopped before orchestration. [Runtime](../../../../glossary.md#term-runtime) now supplies the complete
[immutable](../../../../glossary.md#term-immutable) memory, representation, transfer, execution, [publication](../../../../glossary.md#term-publication), schedule, and
[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) vocabulary. This task joins those existing contracts without moving [Compiler](../../../../glossary.md#term-compiler)
facts into Runtime, exposing Compiler aggregates to concrete [backends](../../../../glossary.md#backend), or inventing a [concrete
backend](../../../../glossary.md#concrete-backend) implementation in shared [Prepare](../../../../glossary.md#prepare).

The schedule assembler exists because shared Prepare cannot invent backend-owned physical
representation creators, transfer recipes, or executable selections. It receives one immutable
validated context after every finalization and returns the complete reusable Runtime recipe in
one call. Prepare then proves that the recipe covers the [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) boundary and prepared
partitions exactly. The future [Engine](../../../../glossary.md#term-engine) [composition root](../../../../glossary.md#term-composition-root) will explicitly supply this collaboration;
it will not discover it globally.

The initialized-[buffer](../../../../glossary.md#term-buffer) variant closes one prerequisite exposed only by this composition: compiler
logical splats are not caller-bindable inputs, while ordinary created buffers are intentionally
invalid until written. A backend-owned creator may materialize the splat directly into a fresh
[run](../../../../glossary.md#term-run)-owned representation; Runtime needs only the generic initial-validity fact to execute the
already-selected recipe correctly.

## [Scope](../../../../glossary.md#term-scope)

- Add the exact five public [Prepare](../../../../glossary.md#prepare) root-package declarations shown above.
- Extend the existing [Runtime](../../../../glossary.md#term-runtime) `BufferPreparation` [sealed](../../../../glossary.md#term-sealed) family with exactly
  `InitializedBuffer(BufferCreator)`, preserving `CallerInput` and `CreatedBuffer` unchanged.
- Extend cold [`RunState`](../../../../glossary.md#run-state--runstate) creation so an initialized buffer uses the existing creator ownership,
  order, rollback, alias, and null-result rules but starts valid instead of invalid.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [rollback glossary](../../../../glossary.md#term-rollback); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Extend the package-private task-0002 [handoff](../../../../glossary.md#term-handoff) result with the [immutable](../../../../glossary.md#term-immutable) first-declaration
  [`PreparedBufferAssignment`](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment) list used to translate [Compiler](../../../../glossary.md#term-compiler) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) facts to dense Runtime
  buffer positions.
- Add one package-private orchestration implementation behind `GraphPreparation.prepare`.
- Validate and [snapshot](../../../../glossary.md#term-snapshot) the complete [partition](../../../../glossary.md#partition)-preparation list before analysis.
- Project one exact [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) per [compile](../../../../glossary.md#compile) partition from `CompileArtifacts` and its
  positionally supplied backend inputs.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Construct every context before invoking the first backend preparer, so malformed global input
  fails before backend-visible work begins.
- Invoke each preparer exactly once in partition order and retain the typed analysis-to-finalizer
  association without raw types, `Object`, reflection, or a [registry](../../../../glossary.md#term-registry).
- Invoke the existing complete-set finalization handoff exactly once after all analyses succeed.
- Invoke the schedule assembler exactly once only after all prepared partitions and buffer
  assignments exist.
- Validate the returned schedule's exact memory-plan identity, representation-creation facts,
  caller-input order, execution coverage/order, representation coordinates, and publication
  boundary before constructing <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [publication glossary](../../../../glossary.md#term-publication))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution).
- Preserve repeated or aliased gradient publication values as separate ordered publication
  occurrences while mapping them to the same logical buffer assignment when appropriate.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Add focused public-shape, immutable-context, successful orchestration, failure-order, and
  schedule-rejection tests using only fake typed backend roles and current Runtime recipe types.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Finalize all affected Javadocs and explanatory documentation in a separate clean documentation
  context after executable implementation.

## Out of [scope](../../../../glossary.md#term-scope)

- a public [Engine](../../../../glossary.md#term-engine) [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run) [facade](../../../../glossary.md#term-facade) or any `modules/engine` change
- backend registration, discovery, provider lookup, service loading, a backend switch, manager,
  registry, or service locator<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- a concrete [CPU](../../../../glossary.md#term-cpu), [OpenBLAS](../../../../glossary.md#term-openblas), [Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), or test [backend](../../../../glossary.md#backend) production implementation
- backend lowering, route selection, kernel selection, creator implementation, transfer
  implementation, executable implementation, or physical representation implementation<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
- executing a prepared recipe, creating [`RunState`](../../../../glossary.md#run-state--runstate), binding inputs, allocation, publication, or
  resource cleanup
- changing the existing `CallerInput` or `CreatedBuffer` contract, runner traversal, transfer,
  executable, publication, residency, ownership, or hot-path behavior beyond consuming the new
  initial-validity fact during cold state creation<span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- tuning, measurement, candidate search, scoring, compatible-[cache](../../../../glossary.md#term-cache) [lookup](../../../../glossary.md#term-lookup), or cache [persistence](../../../../glossary.md#term-persistence)
- dynamic-[dimension](../../../../glossary.md#dimension) [binding](../../../../glossary.md#term-binding), dynamic allocation geometry, or run-dynamic preparation
- inferring byte size, [alignment](../../../../glossary.md#term-alignment), representation choice, transfer route, or [publication](../../../../glossary.md#term-publication) format
- changing Compiler capture, validation, optimization, autograd, partitioning, logical memory,
  constants, publication bindings, diagnostics, or derivative metadata<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [autograd glossary](../../../../glossary.md#autograd); [constants glossary](../../../../glossary.md#term-constant); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- exposing `CompileArtifacts` through a concrete-backend-facing contract
- adding a general callback/event/builder framework, map-based parameter language, or lifecycle
  state machine<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [parameter glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Config, Trace, Backend Contract, Model, Planning, Compiler, Engine, concrete-backend, Gradle,
  dependency, architecture-contract, ADR, architecture-test, backend-conformance, or
  integration-test changes; no Runtime change beyond the exact initialized-buffer extension and
  its focused contract test/documentation<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [Runtime glossary](../../../../glossary.md#term-runtime); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- a Prepare 0004 or another later detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core [lifecycle](../../../../glossary.md#lifecycle) and invariants
  - `modules/prepare`
  - `modules/runtime`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [Prepare](../../../../glossary.md#prepare) lifecycle
  - [Run](../../../../glossary.md#term-run) lifecycle
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [ADR 0010: Stage backend preparation around shared slot assignment](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership); [binding glossary](../../../../glossary.md#term-binding))</span>

## Architecture constraints

- Prepare owns compile-to-prepared orchestration, exact projection, complete-set validation,
  deterministic slot assignment, finalization coordination, schedule validation, and construction
  of the final immutable <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [compile glossary](../../../../glossary.md#compile); [validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) recipe.
- [Compiler](../../../../glossary.md#term-compiler) owns `CompileArtifacts`, graph values, partitions, logical memory, source roles,
  publication bindings, diagnostics, and derivative metadata. Prepare consumes but does not
  mutate, reconstruct, or reinterpret those facts.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [bindings glossary](../../../../glossary.md#term-binding); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Runtime owns all physical-plan and runnable-recipe vocabulary. Runtime remains free of Model,
  Compiler, Planning, and Prepare facts. Its new initialized-buffer origin carries only an
  existing <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [buffer glossary](../../../../glossary.md#term-buffer))</span>`BufferCreator` and the generic initial-validity meaning; it carries no [scalar](../../../../glossary.md#term-scalar),
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [graph](../../../../glossary.md#graph) fact, or [backend identity](../../../../glossary.md#backend-identity--backendid).
- Concrete backends own typed analysis inputs, opaque selected plans, physical creators,
  transfers, executables, and finalization mechanics. They receive no Compiler aggregate.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- The future Engine owns explicit collaborator assembly. This task adds no global lookup and no
  Engine behavior.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- All returned recipes and contexts are immutable and reusable. This task creates no mutable
  per-run state and performs no hot-path action.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- No dependency direction, package ownership, or architecture rule changes. Stop rather than
  weakening one of these boundaries.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Package impact

Changed package:

- `io.github.pho001.synaptik.prepare` — gains the public complete-graph orchestration input,
  schedule-assembly context/collaboration, logical-to-physical buffer association, and facade.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [buffer glossary](../../../../glossary.md#term-buffer); [facade glossary](../../../../glossary.md#term-facade))</span>

Consumed unchanged packages:

- `io.github.pho001.synaptik.prepare.analysis` — existing backend-facing context, preparer,
  analysis, opaque plan, and resource requirements.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- Compiler, Model, and Planning types are consumed only by root Prepare orchestration and its
  immutable schedule context.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Existing Runtime executable, transfer, publication, schedule, and result recipe types are
  composed and validated without behavioral change.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication))</span>

Narrowly extended package:

- `io.github.pho001.synaptik.runtime.resource` and package-private cold creation in
  `runtime.run` — add only the generic run-owned initialized-buffer origin and its initial-validity
  handling.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

No new subpackage, generic `service`/`manager`/`util` package, or [module](../../../../glossary.md#term-module) edge is introduced.

## Exact public and internal [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape)

The five public [Prepare](../../../../glossary.md#prepare) declarations are exactly those in the Goal. Record accessors and [canonical](../../../../glossary.md#term-canonical-form)
constructors may be declared explicitly only to supply complete [Javadoc](../../../../glossary.md#term-javadoc) and [validation](../../../../glossary.md#term-validation). They add
no extra public member.

`PreparedRepresentationPlan.BufferPreparation` permits exactly `CallerInput`, `CreatedBuffer`,
and `InitializedBuffer`. The new record has exactly one `BufferCreator creator` component, validates
it with `NullPointerException("creator")`, and retains the exact creator reference. Cold creation
invokes it in the same deterministic [buffer](../../../../glossary.md#term-buffer)/representation order as `CreatedBuffer`; a successful
non-null result is [run](../../../../glossary.md#term-run)-owned and initially valid. It participates unchanged in attempt-all reverse
[rollback](../../../../glossary.md#term-rollback), duplicate-object rejection, [borrowed](../../../../glossary.md#ownership-borrowing-and-lease)/run-owned [alias](../../../../glossary.md#aliasing-and-arena) rejection, and final [`RunState`](../../../../glossary.md#run-state--runstate)
[ownership](../../../../glossary.md#term-ownership-meanings). Constructing a plan or the record never invokes the creator.

`PartitionPreparation` validates `backendInputs`, `preparer`, and `finalizer` in component order
with exact `NullPointerException` messages matching those names. It retains each exact [immutable](../../../../glossary.md#term-immutable)
reference.

[`PreparedBufferAssignment`](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment) validates `valueId`, then `slot`, then non-negative `planIndex` with
exact messages `valueId`, `slot`, and `planIndex must be non-negative`. It is an immutable
Prepare-only translation fact, not a public physical [binding](../../../../glossary.md#term-binding) or [Runtime](../../../../glossary.md#term-runtime) state.

[`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext) validates [`artifacts`](../../../../glossary.md#term-artifact), `memoryPlan`, `partitions`, and
`bufferAssignments` in component order, validates indexed [elements](../../../../glossary.md#term-element), snapshots both lists, and
requires:

1. `partitions.size()` equals `artifacts.partitions().size()`;
2. every [prepared partition](../../../../glossary.md#prepared-partition--preparedpartition) retains the exact [compile](../../../../glossary.md#compile) [partition](../../../../glossary.md#partition) at the same position;
3. every executable retains the exact supplied memory plan;
4. `bufferAssignments.size()` equals `memoryPlan.buffers().size()`;
5. assignment `i` has `planIndex == i`, retains the exact `memoryPlan.buffers().get(i).slot()`,
   uses a unique [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), and refers to a value in `artifacts.graph()`.

`PreparedScheduleAssembler.assemble` accepts one non-null complete context and returns one
non-null immutable [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule). It performs no execution, allocation, search, mutation, or
resource acquisition. Its implementation belongs to the explicit composition wiring, not a
[concrete backend](../../../../glossary.md#concrete-backend).

[`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) is `final`, has one private constructor, no fields, and exactly the one public
static method shown in the Goal. Private implementation helpers may remain in its source file.

The task-0002 package-private result becomes exactly:

```java
record Result(
        PreparedMemoryPlan memoryPlan,
        List<PreparedPartition> partitions,
        List<PreparedBufferAssignment> bufferAssignments) {}
```

The [handoff](../../../../glossary.md#term-handoff) produces one buffer assignment for each distinct declared `ValueId`, in the same
first-declaration order used by `memoryPlan.buffers()`. It snapshots the list and retains the
exact assigned `ValueId`, [`BufferSlot`](../../../../glossary.md#buffer-slot--bufferslot), and plan index. Existing assignment, finalization, and
failure semantics otherwise remain unchanged.

## Projection and orchestration semantics

`GraphPreparation.prepare` validates top-level inputs in [parameter](../../../../glossary.md#term-parameter) order. It then validates every
`preparations[index]` non-null before checking that its count equals the [compile](../../../../glossary.md#compile) [partition](../../../../glossary.md#partition) count.
An empty compile [graph](../../../../glossary.md#graph) therefore requires an empty preparation list.

Before the first [backend](../../../../glossary.md#backend) call, it indexes [immutable](../../../../glossary.md#term-immutable) compile facts locally and creates all
contexts in partition order:

- [nodes](../../../../glossary.md#node) are the exact graph `CompiledNode` references named by the partition, in partition order;
- values are the exact graph `GraphValue` references that are an input or output of at least one
  partition node, filtered in graph-value order without duplicates;
- logical requirements are the exact `artifacts.memory().requirements()` references for those
  values, in the same projected value order;
- [constants](../../../../glossary.md#term-constant) are the exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) references from `artifacts.constants().constantSources()`
  whose [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) occurs in the projected values, in compile constant-source encounter order;
- backend inputs are the exact positional `PartitionPreparation.backendInputs()` reference.

The current [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) fully-static rule remains authoritative. A dynamic projected [Shape](../../../../glossary.md#shape)
fails during complete context construction before any preparer is invoked. This task adds no
[binding](../../../../glossary.md#term-binding) substitute.

After every context exists, the [operation](../../../../glossary.md#operation) calls each positional preparer once in partition order.
It rejects a null analysis and requires the analysis to retain the exact context partition. It
then constructs typed [handoff](../../../../glossary.md#term-handoff) entries with the corresponding finalizer and invokes the existing
handoff once. The handoff remains responsible for [owner](../../../../glossary.md#term-owner) [validation](../../../../glossary.md#term-validation), declaration validation,
assignment, memory-plan construction, and ordered finalization.

Only after all finalizers succeed does [Prepare](../../../../glossary.md#prepare) create one [`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext), invoke the
assembler once, and validate the returned schedule. No [callback](../../../../glossary.md#term-callback) observes a partial [prepared
partition](../../../../glossary.md#prepared-partition--preparedpartition) set.

## Complete schedule [validation](../../../../glossary.md#term-validation)

The schedule must retain the exact [handoff](../../../../glossary.md#term-handoff) memory plan. If the plan has any [buffer](../../../../glossary.md#term-buffer) or workspace
entry, step zero must be the sole `RepresentationCreationStep`; an empty plan may omit it. When a
creation step exists, its [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) must retain the exact plan.

The creation plan is validated against [Compiler](../../../../glossary.md#term-compiler) source roles:

- every representation list [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) is already structurally valid by [Runtime](../../../../glossary.md#term-runtime) construction;
- every [bindable input](../../../../glossary.md#bindable-input)'s assigned buffer has exactly one `CallerInput`, and no other buffer has a
  caller-input occurrence;
- caller-input occurrences, scanned by buffer index then representation index, map in exact
  `artifacts.constants().bindableInputs()` order;
- every [compile-time constant source](../../../../glossary.md#compile-time-constant-source)'s assigned buffer has at least one `InitializedBuffer`, has
  no `CallerInput`, and may additionally have ordinary initially-invalid created representations;
- no buffer that is not a compile-time constant source may have an `InitializedBuffer`;
- the initialized creator [captures](../../../../glossary.md#term-capture-meanings) [backend](../../../../glossary.md#backend)-owned [materialization](../../../../glossary.md#materialization) behavior; Runtime receives no
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) and shared [Prepare](../../../../glossary.md#prepare) never invokes or inspects the creator;
- every workspace already has exactly one creator by Runtime contract.

Execution occurrences, ignoring the optional creation prefix, interleaved transfers, and final
[publication](../../../../glossary.md#term-publication) suffix, must equal the [prepared partition](../../../../glossary.md#prepared-partition--preparedpartition) count. [Occurrence](../../../../glossary.md#term-occurrence) `i` must retain the exact
`partitions().get(i).executable()` reference. No [partition](../../../../glossary.md#partition) may be omitted, repeated, substituted,
or reordered.

Every executable selection, transfer endpoint, and publication coordinate must use an existing
representation position in the one creation plan. Prepare performs only structural coordinate
validation; it does not infer [residency](../../../../glossary.md#residency), transfer routes, creator [ownership](../../../../glossary.md#term-ownership-meanings), or [kernel](../../../../glossary.md#kernel) support.

Expected publication values are the ordered forward [binding](../../../../glossary.md#term-binding) values followed by every ordered
[gradient](../../../../glossary.md#term-gradient) binding value. The publication suffix count must equal that occurrence count. [Publication
occurrence](../../../../glossary.md#publication-occurrence) `i` must select the prepared buffer index assigned to expected value `i`, retain the
exact memory plan, use an in-range representation position, and use Runtime's already-required
dense `resultIndex == i`. Repeated gradient values and forward/gradient [aliases](../../../../glossary.md#aliasing-and-arena) therefore produce
distinct result positions selecting the same buffer when requested.

Finally, [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) constructs [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) from the exact memory plan and exact
validated schedule. It returns no wrapper and retains no collaborator.

## [Validation](../../../../glossary.md#term-validation) order and failures

The implementation must preserve this observable phase order:

1. top-level nulls: [`artifacts`](../../../../glossary.md#term-artifact), `preparations`, `scheduleAssembler`;
2. indexed preparation nulls;
3. preparation-count equality;
4. complete context projection for every [partition](../../../../glossary.md#partition);
5. ordered [backend](../../../../glossary.md#backend) analysis and analysis-result [identity](../../../../glossary.md#term-identity-meanings) checks;
6. existing [handoff](../../../../glossary.md#term-handoff) validation, assignment, and finalization;
7. [`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext) validation;
8. schedule-assembler invocation and null result;
9. exact plan, creation/source, execution, [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), and [publication](../../../../glossary.md#term-publication) validation;
10. final [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) construction.

Task-specific exact failures are:

- `preparations size must equal compile partition count <count>`;
- `preparations[index].preparer returned null`;
- `preparations[index].analysis partition does not match compile partition`;
- `scheduleAssembler returned null`;
- `schedule memory plan does not match prepared memory plan`;
- `non-empty prepared memory plan requires a representation creation occurrence`;
- `caller-input occurrence count must equal bindable input count <count>`;
- `caller-input occurrence <index> does not match bindableInputs[<index>] <valueId>`;
- `constant source <valueId> requires an initialized buffer representation`;
- `constant source <valueId> must not use a caller-input representation`;
- `buffer <valueId> has an initialized representation but is not a constant source`;
- `execution occurrence count must equal prepared partition count <count>`;
- `execution occurrence <index> does not match preparedPartitions[<index>].executable`;
- `steps[<stepIndex>] <role> representationIndex out of creation-plan range: <index>`;
- `publication occurrence count must equal requested result count <count>`;
- `publication occurrence <index> bufferIndex does not match requested value <valueId>`; and
- `requested value has no prepared buffer assignment: <valueId>`.

Existing constructors may fail first when malformed supplied [Runtime](../../../../glossary.md#term-runtime) values cannot be constructed.
No backend preparer is called before all contexts exist; no finalizer is called before all
analyses succeed; no assembler is called before all finalizers succeed.

## Known limitations

- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) still rejects dynamic [Shapes](../../../../glossary.md#shape). [Binding](../../../../glossary.md#term-binding)-aware preparation remains future work.
- A zero-node pass-through graph cannot currently declare buffer geometry through backend
  analysis. If it contains a requested/bindable value with no prepared buffer assignment,
  preparation fails closed with the exact missing-assignment failure instead of inventing bytes,
  alignment, or a physical representation.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [buffer glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend); [prepared buffer assignment glossary](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Structural representation-coordinate validation does not prove semantic device residency,
  transfer route correctness, or creator/executable backend compatibility. Concrete backend and
  later Engine/conformance work own those proofs.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [validation glossary](../../../../glossary.md#term-validation); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [Concrete backend glossary](../../../../glossary.md#concrete-backend); [Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The assembler is a synchronous immutable recipe-construction seam only. It is not a public
  tuning or execution extension point.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- This closes the shared Prepare contract, but does not make the repository end-to-end runnable
  until a concrete backend and Engine supply the explicit collaborators.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>

## Affected files

[Prepare](../../../../glossary.md#prepare) production:

- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PartitionPreparation.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedBufferAssignment.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedScheduleContext.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedScheduleAssembler.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/package-info.java`

[Runtime](../../../../glossary.md#term-runtime) production:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunStateCreation.java`

Prepare tests:

- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationPublicShapeTest.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/PreparedScheduleContextTest.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationTest.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/FinalizationPublicShapeTest.java`
  — update only the exact package-private handoff-result component shape required by this task<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [shape glossary](../../../../glossary.md#shape))</span>

Runtime test:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateCreationTest.java`

Explanatory documentation:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — current-status clarification only
- `docs/backend-guide/partition-preparer.md`
- `docs/glossary.md`

[Planning](../../../../glossary.md#term-planning):

- this task
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs 0010/0011,
architecture index/[lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) documents, Prepare tasks 0001–0002, current
Prepare/[Compiler](../../../../glossary.md#term-compiler)/Planning/Runtime source/tests/Javadocs, [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training API, Runtime and
Compiler master plans, concrete-[backend](../../../../glossary.md#backend)/[Engine](../../../../glossary.md#term-engine) master plans, build files, and architecture,
[conformance](../../../../glossary.md#term-conformance), or integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 24 paths:

- 7 [Prepare](../../../../glossary.md#prepare) production paths;
- 3 [Runtime](../../../../glossary.md#term-runtime) production paths;
- 4 Prepare test paths;
- 1 Runtime test path;
- 5 explanatory documentation paths; and
- 4 [planning](../../../../glossary.md#term-planning) paths.

No other Java/test path, [Gradle](../../../../glossary.md#term-gradle) file, authoritative [architecture contract](../../../../glossary.md#architecture-contract), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [architecture test](../../../../glossary.md#term-architecture-test),
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) test, [integration test](../../../../glossary.md#term-integration-test), or concrete-backend/[Engine](../../../../glossary.md#term-engine) path may change. Stop if
another type, path, [module](../../../../glossary.md#term-module) edge, [lifecycle](../../../../glossary.md#lifecycle) [owner](../../../../glossary.md#term-owner), or public abstraction is required.

## Acceptance criteria

- [x] The public [API](../../../../glossary.md#term-application-programming-interface-api) is exactly the five [Prepare](../../../../glossary.md#prepare) declarations plus the one nested [Runtime](../../../../glossary.md#term-runtime)
      `InitializedBuffer` record specified above and adds no other public member.
- [x] Initialized-buffer cold creation retains ordinary creator ownership/order/rollback rules and
      starts only successful initialized representations valid.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- [x] All constructors validate, [snapshot](../../../../glossary.md#term-snapshot), and retain exact references in the specified order.
- [x] Every [partition](../../../../glossary.md#partition) context is projected from exact [compile](../../../../glossary.md#compile) references before analysis begins.
- [x] Every preparer and finalizer is called exactly once in compile partition order.
- [x] [Backend](../../../../glossary.md#backend) generic types remain associated without raw types, `Object`, reflection, or casts
      exposed in the public API.
- [x] The task-0002 handoff returns exact first-declaration buffer assignments without changing
      completed assignment/finalization behavior.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- [x] The assembler is called once after all finalizers and is never retained.
- [x] Caller inputs, execution occurrences, representation coordinates, and publication
      occurrences are validated exactly as specified.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [publication glossary](../../../../glossary.md#term-publication))</span>
- [x] Compile-time constants require initialized non-caller representations, and initialized
      representations are rejected for every nonconstant value.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
- [x] Repeated gradient publication values and forward/gradient aliases retain separate ordered
      result occurrences over the same prepared buffer.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [x] All [validation](../../../../glossary.md#term-validation) failures occur before the next side-effect phase and use exact messages.
- [x] The successful result is exactly one [immutable](../../../../glossary.md#term-immutable) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) retaining the handoff
      plan and validated schedule by reference.
- [x] No backend lookup, physical work, execution, tuning, dynamic binding, or cross-layer
      behavior is introduced.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [binding glossary](../../../../glossary.md#term-binding))</span>
- [x] Existing Prepare tests remain green and the final capability [checkpoint](../../../../glossary.md#term-checkpoint) passes.
- [x] Javadocs and affected explanatory/[planning](../../../../glossary.md#term-planning) documentation are independently finalized.
- [x] Prepare 0003 is marked Complete only after both passes and all evidence succeed.
- [x] No Prepare 0004 detailed specification exists.

## Test plan

Implementation-focused tests must cover:

- exact public/package-private [bytecode](../../../../glossary.md#term-bytecode) and reflection surface;
- record null/order/[snapshot](../../../../glossary.md#term-snapshot)/reference/value semantics;
- exact initialized-buffer sealed/public shape, lazy creator invocation, initial validity,
  ownership, duplicate/alias rejection, rollback, and failure preservation;<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- complete multi-[partition](../../../../glossary.md#partition) projection order and [constant](../../../../glossary.md#term-constant) filtering;
- all contexts constructed before the first preparer call;
- ordered one-call analysis, shared assignment, and ordered one-call finalization;
- task-0002 [buffer](../../../../glossary.md#term-buffer)-assignment order, repeated-value reuse, and unchanged workspace behavior;
- successful empty [graph](../../../../glossary.md#graph)/empty plan/schedule with no publications;
- successful execution/transfer/[publication](../../../../glossary.md#term-publication) schedule with exact reference retention;
- caller-input ordering and constant-source exclusion;
- required constant initialization, forbidden constant caller input, and forbidden initialized
  nonconstant representation;
- repeated [gradient](../../../../glossary.md#term-gradient)/aliased publication values;
- wrong counts, identities, plans, creation prefix, execution coverage/order, representation
  coordinates, publication count/order/buffer, null callback results, and missing assignments;<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [callback glossary](../../../../glossary.md#term-callback))</span>
- no later phase called after each representative earlier failure;
- dynamic [Shape](../../../../glossary.md#shape) rejection before [backend](../../../../glossary.md#backend) analysis;
- [immutable](../../../../glossary.md#term-immutable)/reusable result and fresh independent [prepare](../../../../glossary.md#prepare) calls.

[Run](../../../../glossary.md#term-run) focused tests while executable behavior stabilizes. After Java stabilizes, run exactly one
final affected-[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :modules:runtime:test :modules:prepare:test
```

After the implementation and clean documentation passes are both stable, run this single Prepare
milestone capability [checkpoint](../../../../glossary.md#term-checkpoint) because the task closes a cross-[lifecycle](../../../../glossary.md#lifecycle) public contract:

```bash
./gradlew test :testing:architecture-tests:test
```

The documentation pass reuses successful Java evidence unless it changes executable Java or
records a concrete stale-evidence risk. It runs:

```bash
./gradlew :modules:runtime:javadoc :modules:prepare:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md \
  docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/backend-guide/partition-preparer.md \
  docs/glossary.md \
  docs/planning/modules/prepare/tasks/0003-prepare-orchestration-and-validation.md \
  docs/planning/modules/prepare/master-plan.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

It must also inspect generated Javadocs; validate Markdown links, anchors, fences, whitespace,
terminology, examples, exact path [scope](../../../../glossary.md#term-scope), public/package-private imports and surfaces, synchronized
status, preserved completed history, absence of a Prepare 0004 spec, unchanged [Gradle](../../../../glossary.md#term-gradle)/[dependency](../../../../glossary.md#term-dependency)
edges, and reasoned no-change conclusions for [Compiler](../../../../glossary.md#term-compiler)/Training APIs, [Runtime](../../../../glossary.md#term-runtime) behavior, concrete
backends, [Engine](../../../../glossary.md#term-engine), architecture tests, backend [conformance](../../../../glossary.md#term-conformance), and integration tests.

## Dependencies

- [Prepare](../../../../glossary.md#prepare) 0001 and 0002 — Complete
- [Compiler](../../../../glossary.md#term-compiler) milestone through 0006 — Complete
- [Planning](../../../../glossary.md#term-planning) milestone through 0006 — Complete
- [Runtime](../../../../glossary.md#term-runtime) 0002 and 0005–0014 — Complete
- ADRs 0010 and 0011 — Accepted/current

No unfinished [dependency](../../../../glossary.md#term-dependency) blocks implementation.

## Follow-up

- After completion, the next ordered project frontier is the OpenBLAS provider foundation, then
  the concrete CPU backend and Engine composition.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Dynamic binding-aware preparation remains a future capability to specify only when an actual
  consumer and geometry contract exist.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Concrete schedule assembly, representation creators, transfers, executables, and conformance
  evidence arrive with concrete backend and Engine work.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Do not create a detailed follow-up specification in this task.

## Architecture impact

No architecture-contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), or architecture-test change is expected. The task
implements the already assigned [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) and narrowly completes the [Runtime](../../../../glossary.md#term-runtime)
representation-origin vocabulary needed for [compiler](../../../../glossary.md#term-compiler)-declared [constants](../../../../glossary.md#term-constant). The generic initialized
origin does not move splat data or [materialization](../../../../glossary.md#materialization) into Runtime. If implementation requires moving
schedule construction into Runtime, exposing Compiler facts to a [concrete backend](../../../../glossary.md#concrete-backend), adding
discovery/registration to shared Prepare, or carrying [graph](../../../../glossary.md#graph)/[scalar](../../../../glossary.md#term-scalar) facts in Runtime, stop and
request an architecture decision.

## Documentation impact

The clean documentation pass must:

- finalize Javadocs for every new/changed production declaration and all failure/ownership/
  lifecycle semantics;<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- explain the complete compile-to-prepared handoff in Runtime and Public APIs without claiming a
  concrete backend or public Engine facade exists;<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [handoff glossary](../../../../glossary.md#term-handoff); [Runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade))</span>
- update the [Prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend) boundary status and [partition](../../../../glossary.md#partition)-preparer guide;
- define new terms in the glossary only where they are actually public/useful;
- record why Compile API, Tensor API, Training API, architecture rules, Runtime hot paths,
  concrete backends, and Engine behavior do not change;<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor))</span>
- synchronize this task, Prepare and Runtime master plans, and roadmap only after final evidence
  exists.

## Implementation prompt

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, the Prepare master plan, tasks 0001–0003, completed
Compiler/Planning/Runtime contracts named by task 0003, and every affected/review-only source,
test, API, guide, and glossary path named there.

Implement Prepare task 0003 exactly inside its twenty-four authorized paths. Add only the
explicit typed per-partition input, first-declaration buffer association, immutable schedule
context, Prepare-owned schedule assembler, complete graph preparation facade, and the exact
Runtime initialized-buffer origin needed by compile-time constants. Preserve backend-facing
Compiler isolation, task-0001 projection, task-0002 assignment/finalization, all other Runtime
recipes, source/publication order, creator ownership/rollback, immutable references, validation
order, and all architecture boundaries. Add no backend registry/discovery, concrete
backend/Engine behavior, physical execution in Prepare, tuning, dynamic binding,
dependency/build/architecture change, or later task. Stop on scope, architecture,
completed-contract, or representation-proof conflict.

Run focused tests while developing and exactly one final combined Runtime/Prepare module command
after executable Java stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Runtime/Public APIs, boundary guide, glossary, Prepare and
Runtime master plans, task/roadmap status, and documentation validation while reusing successful Java evidence
unless executable behavior changes. Run the recorded Prepare milestone checkpoint only after both
passes. Mark 0003 Complete only after every criterion succeeds. Do not create a Prepare 0004
detailed specification.
```

## Local decisions

- [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) keeps the public operation stateless and uses private typed wildcard capture
  to preserve each <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>`PartitionPreparation<I,P>` association without exposing a raw type, cast,
  registry, or generic map.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- Complete compile projection precedes the first preparer invocation. Analysis remains ordered,
  and task 0002 remains the sole owner of complete-set assignment and finalization.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [owner glossary](../../../../glossary.md#term-owner))</span>
- The task-0002 [handoff](../../../../glossary.md#term-handoff) result exposes one [immutable](../../../../glossary.md#term-immutable) first-declaration
  [`PreparedBufferAssignment`](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment) per prepared buffer. The association is Prepare context only and is
  not added to Runtime geometry or state.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The schedule assembler is called once with the complete immutable context. Prepare validates
  source roles, execution order, representation coordinates, and publication order after assembly
  rather than interpreting backend-specific recipe details.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [publication glossary](../../../../glossary.md#term-publication); [backend glossary](../../../../glossary.md#backend))</span>
- `InitializedBuffer` reuses the existing backend creator and run-owned cleanup lifecycle. Its
  only distinct Runtime meaning is initial validity after successful cold creation.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Known implementation limitations

- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) still requires fully static projected Shapes; no binding or run-dynamic
  geometry contract exists.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run))</span>
- Requested values in a zero-node graph fail closed when no backend declaration supplied buffer
  geometry.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [backend glossary](../../../../glossary.md#backend); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Structural coordinate validation cannot prove backend/device compatibility, semantic transfer
  correctness, or a concrete creator's materialized value. Concrete backends and later
  conformance/integration work own those proofs.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [validation glossary](../../../../glossary.md#term-validation); [device glossary](../../../../glossary.md#term-device); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- No production [concrete backend](../../../../glossary.md#concrete-backend), production schedule assembler, or [Engine](../../../../glossary.md#term-engine) composition exists.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context: `/root/prepare_0003_impl`.

- Executable Java stabilized before the documentation [handoff](../../../../glossary.md#term-handoff). The implementation context ran
  `./gradlew :modules:runtime:test :modules:prepare:test` exactly once after stabilization; it
  passed Runtime 17 suites/146 tests and Prepare 10 suites/35 tests with zero skipped tests,
  failures, or errors. Documentation context <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>`/root/prepare_0003_docs` reused this evidence and
  changed no executable Java.
- Documentation context `/root/prepare_0003_docs` selected General and API/Javadoc profiles for
  Java/API contracts; Backend Guide, Developer Guide, and Example profiles for the backend guide;
  Architecture profile for current-status clarification only; and Planning profile for this task,
  both master plans, and the roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation pass reviewed `AGENTS.md`, the complete architecture contract, current
  architecture plan, planning guide and roadmap, documentation rules and selected profiles,
  Prepare master plan and tasks 0001–0003, the complete 24-path diff, all affected production and
  test files, directly consumed Compiler/Planning/Runtime contracts, Runtime/Public/Compile/
  Tensor/Training API boundaries, the focused lifecycle/module/dependency/boundary explanations,
  backend guide, glossary, Runtime/Compiler/Engine/backend plans, build files, and architecture,
  conformance, and integration test scope.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Compiler glossary](../../../../glossary.md#term-compiler); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [scope glossary](../../../../glossary.md#term-scope))</span>
- `./gradlew :modules:runtime:javadoc :modules:prepare:javadoc` passed after final Javadocs: 10
  actionable tasks, 2 executed and 8 up-to-date. Generated pages were inspected for all five new
  Prepare types, the updated package surface, and Runtime `InitializedBuffer`; signatures, links,
  parameter/result/failure text, and current-versus-planned boundaries rendered correctly.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- `python3 /tmp/validate_synaptik_markdown.py` with the exact nine Markdown paths specified by
  this task passed with `validated 9 Markdown files`. It checked local links and anchors, unique
  effective anchors, balanced backtick/tilde fences, final newlines, and trailing whitespace.
- Public/package-private surface and import inspection confirmed exactly five new public Prepare
  declarations, exactly one new nested Runtime variant, the three-variant sealed origin family,
  the exact package-private handoff-result extension, compiler imports confined to root Prepare
  orchestration/context, and no Compiler aggregate in a backend-facing analysis contract.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>
- Source and status scans confirmed no raw `Object` or public unchecked generic access,
  reflection, registry, service locator, string-dispatch parameter language, concrete backend,
  Engine behavior, creator invocation or execution in Prepare, dynamic binding, tuning, or
  physical-resource implementation. Prepare 0001–0002 and all completed Runtime history remain
  unchanged; this task/master/roadmap are synchronized as Complete; no Prepare 0004 exists.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [dispatch glossary](../../../../glossary.md#term-dispatch); [concrete backend glossary](../../../../glossary.md#concrete-backend); [binding glossary](../../../../glossary.md#term-binding))</span>
- Exact scope inspection reported 24 authorized paths: 7 Prepare production, 3 Runtime
  production, 4 Prepare tests, 1 Runtime test, 5 explanatory documents, and 4 planning paths. No
  Gradle/build, dependency-edge, architecture-contract, ADR, architecture-test,
  backend-conformance, integration, concrete-backend, or Engine path changed. Root Java 26
  configuration and existing module dependency declarations remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- `git diff --check` passed. After both implementation and documentation passes stabilized, the
  single milestone checkpoint <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`./gradlew test :testing:architecture-tests:test` passed: 209
  suites/1,549 tests with zero skipped tests, failures, or errors, including 4 architecture-test
  suites/6 tests with zero skipped tests, failures, or errors. Gradle reported 52 actionable
  tasks, 1 executed and 51 up-to-date.

No-change conclusions:

- Compile API and compiler behavior remain unchanged because Prepare consumes the existing
  immutable <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`CompileArtifacts` without changing compilation, constants, publication bindings,
  diagnostics, derivative metadata, or the package-private compiler entry.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [constants glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Tensor API and Training API remain unchanged because this task adds neither Tensor/model
  semantics nor training orchestration, gradients, optimizer behavior, or public lifecycle
  methods.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [gradients glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Architecture rules, ADRs, dependency explanations, and architecture-test source remain
  unchanged because the implementation realizes the existing Prepare and Runtime ownership
  contract without a new module edge. The focused boundary page changes implementation status
  only.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Runtime behavior outside the exact initialized origin remains unchanged. Caller inputs,
  ordinary created buffers, workspaces, cold creation order, alias rejection, rollback, cleanup,
  runner traversal, transfers, execution, publication, and hot-path boundaries retain their
  completed contracts.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [alias glossary](../../../../glossary.md#aliasing-and-arena); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Concrete backends and Engine remain unchanged because this task supplies only shared typed
  collaborations and validation. No production creator, transfer, executable, schedule assembler,
  registration, discovery, or public Engine facade is implemented.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- Backend-conformance and integration tests remain unchanged because no concrete backend or
  end-to-end Engine path exists. Gradle and dependency declarations remain unchanged because all
  consumed types are available through existing authorized edges.

## Implementation notes

- Added typed per-partition orchestration inputs, dense logical-to-buffer assignments, immutable
  schedule context, one explicit assembler collaboration, and stateless complete graph
  preparation.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [buffer glossary](../../../../glossary.md#term-buffer); [immutable glossary](../../../../glossary.md#term-immutable); [graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation))</span>
- Extended task 0002's result with first-declaration buffer assignments while preserving its
  assignment and finalization behavior.
- Added the Runtime initialized-buffer origin and cold initial-validity handling without changing
  ordinary creator ownership, order, alias, rollback, or cleanup rules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Finalized all affected Javadocs, Runtime/Public APIs, focused boundary status, backend guide,
  glossary, and synchronized planning evidence without changing executable behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: complete shared graph preparation and schedule validation plus the generic
  initialized-buffer origin required by compile-time logical splats.<span data-glossary-navigation="true"> ([graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [validation glossary](../../../../glossary.md#term-validation); [buffer glossary](../../../../glossary.md#term-buffer); [compile glossary](../../../../glossary.md#compile))</span>
- Files changed or created: exactly 24 authorized paths — 7 Prepare production, 3 Runtime
  production, 4 Prepare tests, 1 Runtime test, 5 explanatory documentation, and 4 planning paths.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the implementation context's passing Runtime 17-suite/146-test and
  Prepare 10-suite/35-test result; both affected-module Javadocs, generated pages, exact nine-file
  Markdown, public/internal surface, imports, mechanisms, scope, synchronized status, preserved
  history, no-0004, unchanged build/dependencies, and whitespace gates passed. The milestone
  checkpoint passed after both passes stabilized with 209 suites/1,549 tests and zero skipped
  tests, failures, or errors, including 4 architecture-test suites/6 tests.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [scope glossary](../../../../glossary.md#term-scope); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Documentation-agent review: `/root/prepare_0003_docs` completed the independent clean-context
  pass without executable Java changes or repeated affected-module tests.
- Documentation impact: Runtime/Public APIs, focused boundary status, backend contributor guide,
  glossary, task, Prepare/Runtime master plans, and roadmap now describe current orchestration and
  initialized constants without claiming a concrete backend or Engine.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [constants glossary](../../../../glossary.md#term-constant); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Javadoc review: every changed production declaration and package contract was reviewed;
  Runtime and Prepare Javadoc generation and generated-page inspection passed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Glossary impact: added reusable definitions for prepared buffer assignment and graph
  preparation and updated prepared-origin/current-status distinctions.<span data-glossary-navigation="true"> ([prepared buffer assignment glossary](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment))</span>
- Unresolved issues: None.
- Follow-up required: None for this task. The next ordered project area is the OpenBLAS provider;
  no detailed follow-up was created here.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>

Status: Complete
