# Task 0003: Prepare Orchestration and Validation

## Status

Complete

## Goal

Close the shared Prepare milestone with one explicit orchestration operation that consumes one
complete immutable `CompileArtifacts`, one positionally matching typed backend preparation for
each planned partition, and one Prepare-owned schedule assembler. The operation projects every
partition context, performs all backend analyses, assigns the shared memory plan, finalizes every
partition, assembles one complete Runtime schedule, validates that schedule against Compiler and
Prepare facts, and returns one exact `PreparedExecution`.

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

`InitializedBuffer` is run-owned like `CreatedBuffer`, invokes the same concrete-backend-owned
creator during cold state construction, and differs only by declaring that the returned
representation already contains the correct logical value and is initially valid. Prepare permits
it only for an exact compile-time constant source. Runtime does not receive `ScalarValue`, inspect
bytes, or materialize the value itself.

`GraphPreparation` has no instances or fields. `PreparedScheduleAssembler` is a narrow
Prepare-owned composition seam, not a backend service, registry, discovery mechanism, or public
Engine facade. A concrete backend still sees only its typed `PrepareContext`, analysis, and
finalization contracts; it never receives `CompileArtifacts` or another Compiler aggregate.

## Rationale and mental model

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

Tasks 0001 and 0002 deliberately stopped before orchestration. Runtime now supplies the complete
immutable memory, representation, transfer, execution, publication, schedule, and
`PreparedExecution` vocabulary. This task joins those existing contracts without moving Compiler
facts into Runtime, exposing Compiler aggregates to concrete backends, or inventing a concrete
backend implementation in shared Prepare.

The schedule assembler exists because shared Prepare cannot invent backend-owned physical
representation creators, transfer recipes, or executable selections. It receives one immutable
validated context after every finalization and returns the complete reusable Runtime recipe in
one call. Prepare then proves that the recipe covers the compiled graph boundary and prepared
partitions exactly. The future Engine composition root will explicitly supply this collaboration;
it will not discover it globally.

The initialized-buffer variant closes one prerequisite exposed only by this composition: compiler
logical splats are not caller-bindable inputs, while ordinary created buffers are intentionally
invalid until written. A backend-owned creator may materialize the splat directly into a fresh
run-owned representation; Runtime needs only the generic initial-validity fact to execute the
already-selected recipe correctly.

## Scope

- Add the exact five public Prepare root-package declarations shown above.
- Extend the existing Runtime `BufferPreparation` sealed family with exactly
  `InitializedBuffer(BufferCreator)`, preserving `CallerInput` and `CreatedBuffer` unchanged.
- Extend cold `RunState` creation so an initialized buffer uses the existing creator ownership,
  order, rollback, alias, and null-result rules but starts valid instead of invalid.
- Extend the package-private task-0002 handoff result with the immutable first-declaration
  `PreparedBufferAssignment` list used to translate Compiler `ValueId` facts to dense Runtime
  buffer positions.
- Add one package-private orchestration implementation behind `GraphPreparation.prepare`.
- Validate and snapshot the complete partition-preparation list before analysis.
- Project one exact `PrepareContext` per compile partition from `CompileArtifacts` and its
  positionally supplied backend inputs.
- Construct every context before invoking the first backend preparer, so malformed global input
  fails before backend-visible work begins.
- Invoke each preparer exactly once in partition order and retain the typed analysis-to-finalizer
  association without raw types, `Object`, reflection, or a registry.
- Invoke the existing complete-set finalization handoff exactly once after all analyses succeed.
- Invoke the schedule assembler exactly once only after all prepared partitions and buffer
  assignments exist.
- Validate the returned schedule's exact memory-plan identity, representation-creation facts,
  caller-input order, execution coverage/order, representation coordinates, and publication
  boundary before constructing `PreparedExecution`.
- Preserve repeated or aliased gradient publication values as separate ordered publication
  occurrences while mapping them to the same logical buffer assignment when appropriate.
- Add focused public-shape, immutable-context, successful orchestration, failure-order, and
  schedule-rejection tests using only fake typed backend roles and current Runtime recipe types.
- Finalize all affected Javadocs and explanatory documentation in a separate clean documentation
  context after executable implementation.

## Out of scope

- a public Engine compile/prepare/run facade or any `modules/engine` change
- backend registration, discovery, provider lookup, service loading, a backend switch, manager,
  registry, or service locator
- a concrete CPU, OpenBLAS, Metal, CUDA, or test backend production implementation
- backend lowering, route selection, kernel selection, creator implementation, transfer
  implementation, executable implementation, or physical representation implementation
- executing a prepared recipe, creating `RunState`, binding inputs, allocation, publication, or
  resource cleanup
- changing the existing `CallerInput` or `CreatedBuffer` contract, runner traversal, transfer,
  executable, publication, residency, ownership, or hot-path behavior beyond consuming the new
  initial-validity fact during cold state creation
- tuning, measurement, candidate search, scoring, compatible-cache lookup, or cache persistence
- dynamic-dimension binding, dynamic allocation geometry, or run-dynamic preparation
- inferring byte size, alignment, representation choice, transfer route, or publication format
- changing Compiler capture, validation, optimization, autograd, partitioning, logical memory,
  constants, publication bindings, diagnostics, or derivative metadata
- exposing `CompileArtifacts` through a concrete-backend-facing contract
- adding a general callback/event/builder framework, map-based parameter language, or lifecycle
  state machine
- Config, Trace, Backend Contract, Model, Planning, Compiler, Engine, concrete-backend, Gradle,
  dependency, architecture-contract, ADR, architecture-test, backend-conformance, or
  integration-test changes; no Runtime change beyond the exact initialized-buffer extension and
  its focused contract test/documentation
- a Prepare 0004 or another later detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core lifecycle and invariants
  - `modules/prepare`
  - `modules/runtime`
  - Concrete backend modules
  - Prepare lifecycle
  - Run lifecycle
  - Dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [ADR 0010: Stage backend preparation around shared slot assignment](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)

## Architecture constraints

- Prepare owns compile-to-prepared orchestration, exact projection, complete-set validation,
  deterministic slot assignment, finalization coordination, schedule validation, and construction
  of the final immutable `PreparedExecution` recipe.
- Compiler owns `CompileArtifacts`, graph values, partitions, logical memory, source roles,
  publication bindings, diagnostics, and derivative metadata. Prepare consumes but does not
  mutate, reconstruct, or reinterpret those facts.
- Runtime owns all physical-plan and runnable-recipe vocabulary. Runtime remains free of Model,
  Compiler, Planning, and Prepare facts. Its new initialized-buffer origin carries only an
  existing `BufferCreator` and the generic initial-validity meaning; it carries no scalar,
  `ValueId`, graph fact, or backend identity.
- Concrete backends own typed analysis inputs, opaque selected plans, physical creators,
  transfers, executables, and finalization mechanics. They receive no Compiler aggregate.
- The future Engine owns explicit collaborator assembly. This task adds no global lookup and no
  Engine behavior.
- All returned recipes and contexts are immutable and reusable. This task creates no mutable
  per-run state and performs no hot-path action.
- No dependency direction, package ownership, or architecture rule changes. Stop rather than
  weakening one of these boundaries.

## Package impact

Changed package:

- `io.github.pho001.synaptik.prepare` — gains the public complete-graph orchestration input,
  schedule-assembly context/collaboration, logical-to-physical buffer association, and facade.

Consumed unchanged packages:

- `io.github.pho001.synaptik.prepare.analysis` — existing backend-facing context, preparer,
  analysis, opaque plan, and resource requirements.
- Compiler, Model, and Planning types are consumed only by root Prepare orchestration and its
  immutable schedule context.
- Existing Runtime executable, transfer, publication, schedule, and result recipe types are
  composed and validated without behavioral change.

Narrowly extended package:

- `io.github.pho001.synaptik.runtime.resource` and package-private cold creation in
  `runtime.run` — add only the generic run-owned initialized-buffer origin and its initial-validity
  handling.

No new subpackage, generic `service`/`manager`/`util` package, or module edge is introduced.

## Exact public and internal API shape

The five public Prepare declarations are exactly those in the Goal. Record accessors and canonical
constructors may be declared explicitly only to supply complete Javadoc and validation. They add
no extra public member.

`PreparedRepresentationPlan.BufferPreparation` permits exactly `CallerInput`, `CreatedBuffer`,
and `InitializedBuffer`. The new record has exactly one `BufferCreator creator` component, validates
it with `NullPointerException("creator")`, and retains the exact creator reference. Cold creation
invokes it in the same deterministic buffer/representation order as `CreatedBuffer`; a successful
non-null result is run-owned and initially valid. It participates unchanged in attempt-all reverse
rollback, duplicate-object rejection, borrowed/run-owned alias rejection, and final `RunState`
ownership. Constructing a plan or the record never invokes the creator.

`PartitionPreparation` validates `backendInputs`, `preparer`, and `finalizer` in component order
with exact `NullPointerException` messages matching those names. It retains each exact immutable
reference.

`PreparedBufferAssignment` validates `valueId`, then `slot`, then non-negative `planIndex` with
exact messages `valueId`, `slot`, and `planIndex must be non-negative`. It is an immutable
Prepare-only translation fact, not a public physical binding or Runtime state.

`PreparedScheduleContext` validates `artifacts`, `memoryPlan`, `partitions`, and
`bufferAssignments` in component order, validates indexed elements, snapshots both lists, and
requires:

1. `partitions.size()` equals `artifacts.partitions().size()`;
2. every prepared partition retains the exact compile partition at the same position;
3. every executable retains the exact supplied memory plan;
4. `bufferAssignments.size()` equals `memoryPlan.buffers().size()`;
5. assignment `i` has `planIndex == i`, retains the exact `memoryPlan.buffers().get(i).slot()`,
   uses a unique `ValueId`, and refers to a value in `artifacts.graph()`.

`PreparedScheduleAssembler.assemble` accepts one non-null complete context and returns one
non-null immutable `PreparedSchedule`. It performs no execution, allocation, search, mutation, or
resource acquisition. Its implementation belongs to the explicit composition wiring, not a
concrete backend.

`GraphPreparation` is `final`, has one private constructor, no fields, and exactly the one public
static method shown in the Goal. Private implementation helpers may remain in its source file.

The task-0002 package-private result becomes exactly:

```java
record Result(
        PreparedMemoryPlan memoryPlan,
        List<PreparedPartition> partitions,
        List<PreparedBufferAssignment> bufferAssignments) {}
```

The handoff produces one buffer assignment for each distinct declared `ValueId`, in the same
first-declaration order used by `memoryPlan.buffers()`. It snapshots the list and retains the
exact assigned `ValueId`, `BufferSlot`, and plan index. Existing assignment, finalization, and
failure semantics otherwise remain unchanged.

## Projection and orchestration semantics

`GraphPreparation.prepare` validates top-level inputs in parameter order. It then validates every
`preparations[index]` non-null before checking that its count equals the compile partition count.
An empty compile graph therefore requires an empty preparation list.

Before the first backend call, it indexes immutable compile facts locally and creates all
contexts in partition order:

- nodes are the exact graph `CompiledNode` references named by the partition, in partition order;
- values are the exact graph `GraphValue` references that are an input or output of at least one
  partition node, filtered in graph-value order without duplicates;
- logical requirements are the exact `artifacts.memory().requirements()` references for those
  values, in the same projected value order;
- constants are the exact `ScalarValue` references from `artifacts.constants().constantSources()`
  whose `ValueId` occurs in the projected values, in compile constant-source encounter order;
- backend inputs are the exact positional `PartitionPreparation.backendInputs()` reference.

The current `PrepareContext` fully-static rule remains authoritative. A dynamic projected Shape
fails during complete context construction before any preparer is invoked. This task adds no
binding substitute.

After every context exists, the operation calls each positional preparer once in partition order.
It rejects a null analysis and requires the analysis to retain the exact context partition. It
then constructs typed handoff entries with the corresponding finalizer and invokes the existing
handoff once. The handoff remains responsible for owner validation, declaration validation,
assignment, memory-plan construction, and ordered finalization.

Only after all finalizers succeed does Prepare create one `PreparedScheduleContext`, invoke the
assembler once, and validate the returned schedule. No callback observes a partial prepared
partition set.

## Complete schedule validation

The schedule must retain the exact handoff memory plan. If the plan has any buffer or workspace
entry, step zero must be the sole `RepresentationCreationStep`; an empty plan may omit it. When a
creation step exists, its `PreparedRepresentationPlan` must retain the exact plan.

The creation plan is validated against Compiler source roles:

- every representation list coordinate is already structurally valid by Runtime construction;
- every bindable input's assigned buffer has exactly one `CallerInput`, and no other buffer has a
  caller-input occurrence;
- caller-input occurrences, scanned by buffer index then representation index, map in exact
  `artifacts.constants().bindableInputs()` order;
- every compile-time constant source's assigned buffer has at least one `InitializedBuffer`, has
  no `CallerInput`, and may additionally have ordinary initially-invalid created representations;
- no buffer that is not a compile-time constant source may have an `InitializedBuffer`;
- the initialized creator captures backend-owned materialization behavior; Runtime receives no
  `ScalarValue` and shared Prepare never invokes or inspects the creator;
- every workspace already has exactly one creator by Runtime contract.

Execution occurrences, ignoring the optional creation prefix, interleaved transfers, and final
publication suffix, must equal the prepared partition count. Occurrence `i` must retain the exact
`partitions().get(i).executable()` reference. No partition may be omitted, repeated, substituted,
or reordered.

Every executable selection, transfer endpoint, and publication coordinate must use an existing
representation position in the one creation plan. Prepare performs only structural coordinate
validation; it does not infer residency, transfer routes, creator ownership, or kernel support.

Expected publication values are the ordered forward binding values followed by every ordered
gradient binding value. The publication suffix count must equal that occurrence count. Publication
occurrence `i` must select the prepared buffer index assigned to expected value `i`, retain the
exact memory plan, use an in-range representation position, and use Runtime's already-required
dense `resultIndex == i`. Repeated gradient values and forward/gradient aliases therefore produce
distinct result positions selecting the same buffer when requested.

Finally, `GraphPreparation` constructs `PreparedExecution` from the exact memory plan and exact
validated schedule. It returns no wrapper and retains no collaborator.

## Validation order and failures

The implementation must preserve this observable phase order:

1. top-level nulls: `artifacts`, `preparations`, `scheduleAssembler`;
2. indexed preparation nulls;
3. preparation-count equality;
4. complete context projection for every partition;
5. ordered backend analysis and analysis-result identity checks;
6. existing handoff validation, assignment, and finalization;
7. `PreparedScheduleContext` validation;
8. schedule-assembler invocation and null result;
9. exact plan, creation/source, execution, coordinate, and publication validation;
10. final `PreparedExecution` construction.

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

Existing constructors may fail first when malformed supplied Runtime values cannot be constructed.
No backend preparer is called before all contexts exist; no finalizer is called before all
analyses succeed; no assembler is called before all finalizers succeed.

## Known limitations

- `PrepareContext` still rejects dynamic Shapes. Binding-aware preparation remains future work.
- A zero-node pass-through graph cannot currently declare buffer geometry through backend
  analysis. If it contains a requested/bindable value with no prepared buffer assignment,
  preparation fails closed with the exact missing-assignment failure instead of inventing bytes,
  alignment, or a physical representation.
- Structural representation-coordinate validation does not prove semantic device residency,
  transfer route correctness, or creator/executable backend compatibility. Concrete backend and
  later Engine/conformance work own those proofs.
- The assembler is a synchronous immutable recipe-construction seam only. It is not a public
  tuning or execution extension point.
- This closes the shared Prepare contract, but does not make the repository end-to-end runnable
  until a concrete backend and Engine supply the explicit collaborators.

## Affected files

Prepare production:

- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PartitionPreparation.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedBufferAssignment.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedScheduleContext.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedScheduleAssembler.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/package-info.java`

Runtime production:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/package-info.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunStateCreation.java`

Prepare tests:

- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationPublicShapeTest.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/PreparedScheduleContextTest.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationTest.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/FinalizationPublicShapeTest.java`
  — update only the exact package-private handoff-result component shape required by this task

Runtime test:

- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateCreationTest.java`

Explanatory documentation:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — current-status clarification only
- `docs/backend-guide/partition-preparer.md`
- `docs/glossary.md`

Planning:

- this task
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs 0010/0011,
architecture index/lifecycle/module/dependency documents, Prepare tasks 0001–0002, current
Prepare/Compiler/Planning/Runtime source/tests/Javadocs, Compile API, Training API, Runtime and
Compiler master plans, concrete-backend/Engine master plans, build files, and architecture,
conformance, or integration tests.

## Maximum scope

At most 24 paths:

- 7 Prepare production paths;
- 3 Runtime production paths;
- 4 Prepare test paths;
- 1 Runtime test path;
- 5 explanatory documentation paths; and
- 4 planning paths.

No other Java/test path, Gradle file, authoritative architecture contract, ADR, architecture test,
backend-conformance test, integration test, or concrete-backend/Engine path may change. Stop if
another type, path, module edge, lifecycle owner, or public abstraction is required.

## Acceptance criteria

- [x] The public API is exactly the five Prepare declarations plus the one nested Runtime
      `InitializedBuffer` record specified above and adds no other public member.
- [x] Initialized-buffer cold creation retains ordinary creator ownership/order/rollback rules and
      starts only successful initialized representations valid.
- [x] All constructors validate, snapshot, and retain exact references in the specified order.
- [x] Every partition context is projected from exact compile references before analysis begins.
- [x] Every preparer and finalizer is called exactly once in compile partition order.
- [x] Backend generic types remain associated without raw types, `Object`, reflection, or casts
      exposed in the public API.
- [x] The task-0002 handoff returns exact first-declaration buffer assignments without changing
      completed assignment/finalization behavior.
- [x] The assembler is called once after all finalizers and is never retained.
- [x] Caller inputs, execution occurrences, representation coordinates, and publication
      occurrences are validated exactly as specified.
- [x] Compile-time constants require initialized non-caller representations, and initialized
      representations are rejected for every nonconstant value.
- [x] Repeated gradient publication values and forward/gradient aliases retain separate ordered
      result occurrences over the same prepared buffer.
- [x] All validation failures occur before the next side-effect phase and use exact messages.
- [x] The successful result is exactly one immutable `PreparedExecution` retaining the handoff
      plan and validated schedule by reference.
- [x] No backend lookup, physical work, execution, tuning, dynamic binding, or cross-layer
      behavior is introduced.
- [x] Existing Prepare tests remain green and the final capability checkpoint passes.
- [x] Javadocs and affected explanatory/planning documentation are independently finalized.
- [x] Prepare 0003 is marked Complete only after both passes and all evidence succeed.
- [x] No Prepare 0004 detailed specification exists.

## Test plan

Implementation-focused tests must cover:

- exact public/package-private bytecode and reflection surface;
- record null/order/snapshot/reference/value semantics;
- exact initialized-buffer sealed/public shape, lazy creator invocation, initial validity,
  ownership, duplicate/alias rejection, rollback, and failure preservation;
- complete multi-partition projection order and constant filtering;
- all contexts constructed before the first preparer call;
- ordered one-call analysis, shared assignment, and ordered one-call finalization;
- task-0002 buffer-assignment order, repeated-value reuse, and unchanged workspace behavior;
- successful empty graph/empty plan/schedule with no publications;
- successful execution/transfer/publication schedule with exact reference retention;
- caller-input ordering and constant-source exclusion;
- required constant initialization, forbidden constant caller input, and forbidden initialized
  nonconstant representation;
- repeated gradient/aliased publication values;
- wrong counts, identities, plans, creation prefix, execution coverage/order, representation
  coordinates, publication count/order/buffer, null callback results, and missing assignments;
- no later phase called after each representative earlier failure;
- dynamic Shape rejection before backend analysis;
- immutable/reusable result and fresh independent prepare calls.

Run focused tests while executable behavior stabilizes. After Java stabilizes, run exactly one
final affected-module command:

```bash
./gradlew :modules:runtime:test :modules:prepare:test
```

After the implementation and clean documentation passes are both stable, run this single Prepare
milestone capability checkpoint because the task closes a cross-lifecycle public contract:

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
terminology, examples, exact path scope, public/package-private imports and surfaces, synchronized
status, preserved completed history, absence of a Prepare 0004 spec, unchanged Gradle/dependency
edges, and reasoned no-change conclusions for Compiler/Training APIs, Runtime behavior, concrete
backends, Engine, architecture tests, backend conformance, and integration tests.

## Dependencies

- Prepare 0001 and 0002 — Complete
- Compiler milestone through 0006 — Complete
- Planning milestone through 0006 — Complete
- Runtime 0002 and 0005–0014 — Complete
- ADRs 0010 and 0011 — Accepted/current

No unfinished dependency blocks implementation.

## Follow-up

- After completion, the next ordered project frontier is the OpenBLAS provider foundation, then
  the concrete CPU backend and Engine composition.
- Dynamic binding-aware preparation remains a future capability to specify only when an actual
  consumer and geometry contract exist.
- Concrete schedule assembly, representation creators, transfers, executables, and conformance
  evidence arrive with concrete backend and Engine work.
- Do not create a detailed follow-up specification in this task.

## Architecture impact

No architecture-contract, ADR, dependency, or architecture-test change is expected. The task
implements the already assigned Prepare lifecycle ownership and narrowly completes the Runtime
representation-origin vocabulary needed for compiler-declared constants. The generic initialized
origin does not move splat data or materialization into Runtime. If implementation requires moving
schedule construction into Runtime, exposing Compiler facts to a concrete backend, adding
discovery/registration to shared Prepare, or carrying graph/scalar facts in Runtime, stop and
request an architecture decision.

## Documentation impact

The clean documentation pass must:

- finalize Javadocs for every new/changed production declaration and all failure/ownership/
  lifecycle semantics;
- explain the complete compile-to-prepared handoff in Runtime and Public APIs without claiming a
  concrete backend or public Engine facade exists;
- update the Prepare/backend boundary status and partition-preparer guide;
- define new terms in the glossary only where they are actually public/useful;
- record why Compile API, Tensor API, Training API, architecture rules, Runtime hot paths,
  concrete backends, and Engine behavior do not change;
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

- `GraphPreparation` keeps the public operation stateless and uses private typed wildcard capture
  to preserve each `PartitionPreparation<I,P>` association without exposing a raw type, cast,
  registry, or generic map.
- Complete compile projection precedes the first preparer invocation. Analysis remains ordered,
  and task 0002 remains the sole owner of complete-set assignment and finalization.
- The task-0002 handoff result exposes one immutable first-declaration
  `PreparedBufferAssignment` per prepared buffer. The association is Prepare context only and is
  not added to Runtime geometry or state.
- The schedule assembler is called once with the complete immutable context. Prepare validates
  source roles, execution order, representation coordinates, and publication order after assembly
  rather than interpreting backend-specific recipe details.
- `InitializedBuffer` reuses the existing backend creator and run-owned cleanup lifecycle. Its
  only distinct Runtime meaning is initial validity after successful cold creation.

## Known implementation limitations

- `PrepareContext` still requires fully static projected Shapes; no binding or run-dynamic
  geometry contract exists.
- Requested values in a zero-node graph fail closed when no backend declaration supplied buffer
  geometry.
- Structural coordinate validation cannot prove backend/device compatibility, semantic transfer
  correctness, or a concrete creator's materialized value. Concrete backends and later
  conformance/integration work own those proofs.
- No production concrete backend, production schedule assembler, or Engine composition exists.

## Validation evidence

Implementation context: `/root/prepare_0003_impl`.

- Executable Java stabilized before the documentation handoff. The implementation context ran
  `./gradlew :modules:runtime:test :modules:prepare:test` exactly once after stabilization; it
  passed Runtime 17 suites/146 tests and Prepare 10 suites/35 tests with zero skipped tests,
  failures, or errors. Documentation context `/root/prepare_0003_docs` reused this evidence and
  changed no executable Java.
- Documentation context `/root/prepare_0003_docs` selected General and API/Javadoc profiles for
  Java/API contracts; Backend Guide, Developer Guide, and Example profiles for the backend guide;
  Architecture profile for current-status clarification only; and Planning profile for this task,
  both master plans, and the roadmap.
- The documentation pass reviewed `AGENTS.md`, the complete architecture contract, current
  architecture plan, planning guide and roadmap, documentation rules and selected profiles,
  Prepare master plan and tasks 0001–0003, the complete 24-path diff, all affected production and
  test files, directly consumed Compiler/Planning/Runtime contracts, Runtime/Public/Compile/
  Tensor/Training API boundaries, the focused lifecycle/module/dependency/boundary explanations,
  backend guide, glossary, Runtime/Compiler/Engine/backend plans, build files, and architecture,
  conformance, and integration test scope.
- `./gradlew :modules:runtime:javadoc :modules:prepare:javadoc` passed after final Javadocs: 10
  actionable tasks, 2 executed and 8 up-to-date. Generated pages were inspected for all five new
  Prepare types, the updated package surface, and Runtime `InitializedBuffer`; signatures, links,
  parameter/result/failure text, and current-versus-planned boundaries rendered correctly.
- `python3 /tmp/validate_synaptik_markdown.py` with the exact nine Markdown paths specified by
  this task passed with `validated 9 Markdown files`. It checked local links and anchors, unique
  effective anchors, balanced backtick/tilde fences, final newlines, and trailing whitespace.
- Public/package-private surface and import inspection confirmed exactly five new public Prepare
  declarations, exactly one new nested Runtime variant, the three-variant sealed origin family,
  the exact package-private handoff-result extension, compiler imports confined to root Prepare
  orchestration/context, and no Compiler aggregate in a backend-facing analysis contract.
- Source and status scans confirmed no raw `Object` or public unchecked generic access,
  reflection, registry, service locator, string-dispatch parameter language, concrete backend,
  Engine behavior, creator invocation or execution in Prepare, dynamic binding, tuning, or
  physical-resource implementation. Prepare 0001–0002 and all completed Runtime history remain
  unchanged; this task/master/roadmap are synchronized as Complete; no Prepare 0004 exists.
- Exact scope inspection reported 24 authorized paths: 7 Prepare production, 3 Runtime
  production, 4 Prepare tests, 1 Runtime test, 5 explanatory documents, and 4 planning paths. No
  Gradle/build, dependency-edge, architecture-contract, ADR, architecture-test,
  backend-conformance, integration, concrete-backend, or Engine path changed. Root Java 26
  configuration and existing module dependency declarations remain unchanged.
- `git diff --check` passed. After both implementation and documentation passes stabilized, the
  single milestone checkpoint `./gradlew test :testing:architecture-tests:test` passed: 209
  suites/1,549 tests with zero skipped tests, failures, or errors, including 4 architecture-test
  suites/6 tests with zero skipped tests, failures, or errors. Gradle reported 52 actionable
  tasks, 1 executed and 51 up-to-date.

No-change conclusions:

- Compile API and compiler behavior remain unchanged because Prepare consumes the existing
  immutable `CompileArtifacts` without changing compilation, constants, publication bindings,
  diagnostics, derivative metadata, or the package-private compiler entry.
- Tensor API and Training API remain unchanged because this task adds neither Tensor/model
  semantics nor training orchestration, gradients, optimizer behavior, or public lifecycle
  methods.
- Architecture rules, ADRs, dependency explanations, and architecture-test source remain
  unchanged because the implementation realizes the existing Prepare and Runtime ownership
  contract without a new module edge. The focused boundary page changes implementation status
  only.
- Runtime behavior outside the exact initialized origin remains unchanged. Caller inputs,
  ordinary created buffers, workspaces, cold creation order, alias rejection, rollback, cleanup,
  runner traversal, transfers, execution, publication, and hot-path boundaries retain their
  completed contracts.
- Concrete backends and Engine remain unchanged because this task supplies only shared typed
  collaborations and validation. No production creator, transfer, executable, schedule assembler,
  registration, discovery, or public Engine facade is implemented.
- Backend-conformance and integration tests remain unchanged because no concrete backend or
  end-to-end Engine path exists. Gradle and dependency declarations remain unchanged because all
  consumed types are available through existing authorized edges.

## Implementation notes

- Added typed per-partition orchestration inputs, dense logical-to-buffer assignments, immutable
  schedule context, one explicit assembler collaboration, and stateless complete graph
  preparation.
- Extended task 0002's result with first-declaration buffer assignments while preserving its
  assignment and finalization behavior.
- Added the Runtime initialized-buffer origin and cold initial-validity handling without changing
  ordinary creator ownership, order, alias, rollback, or cleanup rules.
- Finalized all affected Javadocs, Runtime/Public APIs, focused boundary status, backend guide,
  glossary, and synchronized planning evidence without changing executable behavior.

## Completion summary

- Completed changes: complete shared graph preparation and schedule validation plus the generic
  initialized-buffer origin required by compile-time logical splats.
- Files changed or created: exactly 24 authorized paths — 7 Prepare production, 3 Runtime
  production, 4 Prepare tests, 1 Runtime test, 5 explanatory documentation, and 4 planning paths.
- Tests and validation: reused the implementation context's passing Runtime 17-suite/146-test and
  Prepare 10-suite/35-test result; both affected-module Javadocs, generated pages, exact nine-file
  Markdown, public/internal surface, imports, mechanisms, scope, synchronized status, preserved
  history, no-0004, unchanged build/dependencies, and whitespace gates passed. The milestone
  checkpoint passed after both passes stabilized with 209 suites/1,549 tests and zero skipped
  tests, failures, or errors, including 4 architecture-test suites/6 tests.
- Documentation-agent review: `/root/prepare_0003_docs` completed the independent clean-context
  pass without executable Java changes or repeated affected-module tests.
- Documentation impact: Runtime/Public APIs, focused boundary status, backend contributor guide,
  glossary, task, Prepare/Runtime master plans, and roadmap now describe current orchestration and
  initialized constants without claiming a concrete backend or Engine.
- Javadoc review: every changed production declaration and package contract was reviewed;
  Runtime and Prepare Javadoc generation and generated-page inspection passed.
- Glossary impact: added reusable definitions for prepared buffer assignment and graph
  preparation and updated prepared-origin/current-status distinctions.
- Unresolved issues: None.
- Follow-up required: None for this task. The next ordered project area is the OpenBLAS provider;
  no detailed follow-up was created here.

Status: Complete
