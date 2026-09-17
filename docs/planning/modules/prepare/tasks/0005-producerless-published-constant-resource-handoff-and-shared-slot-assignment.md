# Task 0005: Producerless Published-Constant Resource Handoff and Shared Slot Assignment

## Status

Complete

## Goal

Add the smallest Prepare-owned complete-graph resource contribution needed for a fully static
source-only published compile-time splat constant. Compiler 0006B5 now supplies that graph input
and graph output with a resolved logical descriptor; Planning continues to supply the exact
producerless, consumerless, graph-output-required `LogicalMemoryRequirement`. This task carries
those exact references plus one externally supplied physical buffer declaration into Prepare's
complete resource handoff and assigns the value one deterministic shared buffer slot.

The new public value and overload are:

```java
package io.github.pho001.synaptik.prepare;

public record ProducerlessPublishedConstantResource(
        GraphValue value,
        LogicalMemoryRequirement logicalRequirement,
        long byteSize,
        long byteAlignment) {}

public final class GraphPreparation {
    public static PreparedExecution prepare(
            CompileArtifacts artifacts,
            List<? extends PartitionPreparation<?, ?>> preparations,
            List<ProducerlessPublishedConstantResource> producerlessResources,
            PreparedScheduleAssembler scheduleAssembler);
}
```

The existing three-argument `GraphPreparation.prepare(...)` remains source-compatible and means
that the caller contributes no producerless resources. It preserves the current path for ordinary
partition-connected graphs. The new value is complete-graph composition input in the Prepare
root package, not a `PrepareContext` value, partition-analysis requirement, backend selector,
representation recipe, or Runtime object.

## Current defect and mental model

Current `GraphPreparation.Projection` includes only values occurring in a planned partition's
node inputs or outputs. Current `BackendPartitionFinalizationHandoff` assigns buffers only from
ordered `BackendPartitionAnalysis.requirements()`. A source-only published constant has neither a
node producer nor a node consumer, so it is correctly absent from every partition projection and
cannot be declared by ordinary partition analysis. Later schedule validation nevertheless
requires every compile-time constant source and every requested publication to have a
`PreparedBufferAssignment`; preparation therefore fails closed with
`requested value has no prepared buffer assignment`.

The repaired flow is:

```text
Compiler-resolved GraphValue
  + exact producerless/consumerless graph-output LogicalMemoryRequirement
  + concrete-composition-supplied byte size and alignment
  -> validate exact source-only published-constant role against CompileArtifacts
  -> canonicalize contributions in final graph-value encounter order
  -> append after all ordinary partition-analysis buffer declarations
  -> assign one dense shared BufferSlot and PreparedBufferAssignment
  -> existing PreparedScheduleContext and schedule validation
```

The contribution supplies physical byte size and alignment; Prepare does not derive them. CPU
0010H will be the first concrete producer of that declaration and the initialized representation
recipe. This task deliberately stops before CPU integration, value materialization, schedule
assembly changes, or Engine use.

## Scope

- Add exactly `ProducerlessPublishedConstantResource` in the public Prepare root package.
- Add one four-argument `GraphPreparation.prepare(...)` overload accepting a list of those
  contributions between the positional preparations and schedule assembler.
- Retain the existing three-argument overload and delegate it to the same implementation with an
  empty contribution list.
- Validate and snapshot the supplied contribution list before constructing a partition context or
  invoking backend analysis.
- Validate each contribution's exact graph-value and logical-requirement references against the
  supplied `CompileArtifacts`, and validate its exact source-only published-constant role.
- Require complete, unique contribution coverage for every artifact value with that role when the
  four-argument path is used; the three-argument path supplies the empty set and therefore keeps
  unsupported source-only publication fail-closed.
- Canonicalize valid contributions by final `CompiledGraphModel.values()` encounter order rather
  than caller list order.
- Extend the package-private complete-set handoff to validate the canonical contributions and add
  their supplied buffer geometry after every ordinary partition-analysis requirement has been
  traversed.
- Preserve ordinary per-finalizer assignment lists exactly. A producerless contribution has no
  owning partition and is never inserted into `BackendPartitionFinalization.assignments()`.
- Return the contributed association through the existing plan-ordered
  `PreparedBufferAssignment` list and `PreparedScheduleContext` only.
- Add focused public-shape, record-contract, orchestration, failure-order, complete-coverage,
  deterministic-order, assignment, and compatibility tests.
- Finalize affected Javadocs and the focused Prepare explanations through the mandatory separate
  clean documentation pass after executable Java stabilizes.

## Out of scope

- CPU 0010H production, tests, Javadocs, or detailed task specification
- Engine 0006 production, tests, Javadocs, or API behavior
- Runtime production, tests, API, representation, validity, publication, lease, or cleanup changes
- selecting a backend, partition, device, route, kernel, representation, transfer, or executable
- projecting the value into `PrepareContext`, adding a synthetic node, assigning a producer or
  consumer partition, or attaching the contribution to a partition finalizer
- deriving byte size, alignment, storage span, carrier, physical layout, creator, or
  representation from a logical descriptor
- invoking a creator, materializing or copying the splat value, initializing a representation,
  allocating storage, binding, executing, publishing, or creating mutable run state
- adding a zero-node execution or making a zero-partition graph preparable
- relaxing the CPU integration's requirement for exactly one non-empty CPU partition or its
  mixed/multi-partition rejection
- changing ordinary partition-connected projection, backend-analysis declaration order,
  first-declaration aggregation, workspace assignment, or finalizer inputs
- changing Compiler graph/source/publication behavior, Planning logical-memory behavior, Model
  descriptors, build structure, dependencies, architecture rules, ADRs, architecture tests,
  backend conformance, integration tests, or performance behavior
- a general resource registry, callback, provider, backend switch, service locator, generic map,
  or speculative multi-backend schedule-contribution framework

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Prepare,
  Runtime, concrete backends, lifecycle, and dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010: Stage backend preparation around shared slot assignment](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md) and [roadmap](../../../roadmap.md)
- [Prepare master plan](../master-plan.md) and completed Prepare tasks
  [0001](0001-backend-partition-analysis-and-resource-declaration.md),
  [0002](0002-backend-partition-finalization-handoff.md),
  [0003](0003-prepare-orchestration-and-validation.md),
  [0003A](0003a-immutable-partition-local-dag-analysis-projection.md), and
  [0004](0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
- [Compiler 0006B5](../../compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md)
- Planning [0005](../../planning/tasks/0005-logical-materialization-and-memory-requirements.md)
  and [0006](../../planning/tasks/0006-planning-contract-closure-audit.md)
- CPU [0010F](../../../backends/cpu/tasks/0010f-supported-cpu-lifecycle-integration-adapter.md)
  and [0010G](../../../backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md)
- [Engine 0006](../../engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md)
- Runtime [0008](../../runtime/tasks/0008-prepared-buffer-transfer-and-materialization-schedule.md),
  [0009](../../runtime/tasks/0009-publication-and-result-schedule-steps.md),
  [0010](../../runtime/tasks/0010-prepared-runner-and-dynamic-execution.md), and
  [0015](../../runtime/tasks/0015-leased-publication-representation-access.md)

## Architecture constraints

- Compiler owns the exact final graph, source roles, publication boundary, and resolved logical
  descriptor. Prepare consumes and validates those exact immutable facts without rebuilding them.
- Planning owns logical producer, consumer, and graph-output obligations. A producerless resource
  contribution must retain the exact Planning requirement and must not invent partition ownership.
- Prepare owns complete-set validation, stable ordering, shared slot assignment, finalization
  coordination, and schedule validation. The new contribution is therefore a Prepare root-package
  composition value, not a backend-analysis projection.
- Concrete backends own exact physical declarations and representation creators. The
  contribution's byte size and alignment are supplied from outside Prepare; Prepare only
  validates and transports that geometry into shared assignment.
- Runtime owns the existing physical-plan, initialized-representation, execution, publication,
  lease, and cleanup vocabulary. No new Runtime concept is needed.
- Engine remains the explicit composition root. Prepare must not discover a backend or infer which
  backend should supply this resource.
- The current non-empty-partition execution model remains unchanged. A source-only constant may
  accompany ordinary prepared work, but this task cannot turn a zero-node graph into a runnable
  schedule.
- No module edge, authoritative architecture decision, or dependency direction changes. Stop if
  implementation requires one.

## Package impact and exact type semantics

Changed existing package:

- `io.github.pho001.synaptik.prepare` gains one immutable complete-graph resource contribution and
  one `GraphPreparation` overload. No subpackage is added.

Consumed unchanged packages:

- `io.github.pho001.synaptik.prepare.analysis` continues to own partition-analysis requirements
  unchanged. Its `PreparationResourceRequirement.Buffer` remains limited to projected values and
  is not reused for this unprojected contribution.
- Compiler, Model, and Planning facts remain visible only at shared Prepare orchestration.
- Runtime contracts are consumed unchanged.

`ProducerlessPublishedConstantResource` has exactly four record components in this order:
`GraphValue value`, `LogicalMemoryRequirement logicalRequirement`, `long byteSize`, and
`long byteAlignment`. Its canonical constructor validates reference components in declaration
order and retains each exact immutable reference. It then requires:

1. `logicalRequirement.valueId().equals(value.id())`;
2. `logicalRequirement.descriptor().equals(value.descriptor())`;
3. `logicalRequirement.producerPartition().isEmpty()`;
4. `logicalRequirement.consumerPartitions().isEmpty()`;
5. `logicalRequirement.graphOutput()`;
6. `value.descriptor().shape().isFullyStatic()`;
7. `value.descriptor().layout().isPresent()`;
8. `byteSize >= 0`; and
9. `byteAlignment` is a positive power of two.

The record alone cannot prove exact artifact membership, graph-input/output membership, constant
source classification, node non-consumption, publication order, complete coverage, or non-empty
partition context. `GraphPreparation` proves those facts against one exact `CompileArtifacts`.
Physical byte size may include backend-required geometry and is not equated to logical element
count or referenced span. The caller owns deriving both physical values; Prepare validates only
their generic Runtime-compatible domain and never computes them.

## Exact orchestration and validation semantics

The four-argument overload validates top-level inputs in parameter order: `artifacts`,
`preparations`, `producerlessResources`, then `scheduleAssembler`. It validates and snapshots
preparation and contribution elements before any projection or backend call. Existing preparation
coverage validation retains its current meaning.

Derive the required source-only published-constant IDs from final artifacts in graph-value
encounter order. A required value has all of these facts:

1. its ID is in `graph.inputs()`;
2. its ID is in `graph.outputs()` and therefore in the validated publication boundary;
3. its ID occurs in `constants.constantSources()`;
4. no `CompiledNode.inputs()` occurrence consumes it;
5. no node output produces it, as already implied by valid graph-input membership;
6. its exact logical-memory requirement has no producer partition, no consumer partition, and
   `graphOutput == true`;
7. its Shape is fully static; and
8. its descriptor layout is resolved.

For every supplied contribution, require the exact `GraphValue` reference from
`artifacts.graph().values()` and the exact `LogicalMemoryRequirement` reference from
`artifacts.memory().requirements()`, not merely equal replacements. Require its ID to have the
complete role above. Reject duplicate contributed IDs, missing required IDs, and contributed IDs
outside the required set before constructing any `PrepareContext` or invoking any preparer.

Canonicalize the validated contributions by required graph-value encounter order. Caller list
order is not assignment policy. If the canonical list is non-empty, require at least one planned
non-empty partition. This explicit guard prevents the contribution seam from making a pure
zero-node graph preparable.

After all ordinary analyses succeed and existing cross-partition declaration checks pass, give
the canonical contribution list to the package-private complete-set handoff. The handoff must
validate all ordinary entry sources and all contributed exact sources before assignment or
finalization. It traverses all ordinary partition analyses and their requirements exactly as
today, then traverses canonical producerless contributions. Consequently:

- every existing ordinary buffer and workspace keeps the same slot, plan index, geometry,
  finalizer assignment position, and aggregation behavior;
- each distinct producerless contribution receives the next dense `BufferSlot` and plan index;
- multiple contributions follow final graph-value encounter order;
- a contributed ID must not overlap any ordinary buffer declaration or another contribution;
- contributed geometry is copied unchanged into its `PreparedMemoryPlan.BufferEntry`; it is not
  merged with or reinterpreted from a logical descriptor;
- no `PreparationResourceAssignment` is created for a producerless contribution because no
  partition finalizer owns it; and
- the existing result builds one `PreparedBufferAssignment` per distinct buffer in complete plan
  order, including the contributed value.

The three-argument overload delegates with `List.of()`. Ordinary artifacts with no required
producerless source-only publication preserve exact current behavior. Artifacts requiring such a
resource continue to fail closed until a concrete composition uses the four-argument overload and
supplies complete declarations; CPU 0010H and Engine 0006 own that later wiring.

### Failure types, messages, and precedence

The record uses `NullPointerException("value")` then
`NullPointerException("logicalRequirement")` for null references. Its remaining deliberate
failures are `IllegalArgumentException` with these stable messages, evaluated in the listed
invariant order:

- `logicalRequirement.valueId must match value.id`;
- `logicalRequirement.descriptor must match value.descriptor`;
- `logicalRequirement must be producerless`;
- `logicalRequirement must be consumerless`;
- `logicalRequirement must require graph output`;
- `value descriptor shape must be fully static`;
- `value descriptor layout must be resolved`;
- `byteSize must be non-negative`; and
- `byteAlignment must be a positive power of two`.

The four-argument operation checks top-level references in declared parameter order, producing
`NullPointerException("artifacts")`, `NullPointerException("preparations")`,
`NullPointerException("producerlessResources")`, or
`NullPointerException("scheduleAssembler")`. It then preserves existing indexed preparation and
positional-coverage failures before checking indexed contributions. A null contribution produces
`NullPointerException("producerlessResources[i]")`; the remaining deterministic
`IllegalArgumentException` messages follow below. Supplied contributions are checked in caller
order; missing coverage is checked in canonical required-value order.

- `producerlessResources[i] value reference does not match artifacts graph value: <ValueId>`;
- `producerlessResources[i] logical requirement reference does not match artifacts memory requirement: <ValueId>`;
- `producerlessResources[i] duplicates <ValueId>`;
- `producerlessResources[i] is not a required producerless published constant: <ValueId>`;
- `producerlessResources has no contribution for required <ValueId>`; and
- `producerless resources require at least one non-empty planned partition`.

Complete contribution validation finishes before projection. Existing projection, analysis,
cross-partition declaration, finalization, assembler, and schedule failures retain their current
relative order afterward. The package-private handoff additionally rejects direct malformed use
with `producerless resource overlaps ordinary buffer declaration: <ValueId>` before constructing
the memory plan or invoking a finalizer. Valid orchestration makes duplicate producerless IDs
unreachable at that boundary, but the handoff must still fail closed when tested directly.

## Affected files

Exact future implementation and documentation allowlist (thirteen paths):

1. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/ProducerlessPublishedConstantResource.java` (new)
2. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java`
3. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java`
4. `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/package-info.java`
5. `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationPublicShapeTest.java`
6. `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationTest.java`
7. `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoffTest.java`
8. `docs/api/runtime-api.md`
9. `docs/api/public-api.md`
10. `docs/glossary.md`
11. This task
12. `docs/planning/modules/prepare/master-plan.md`
13. `docs/planning/roadmap.md`

Review without editing Compiler, Planning, Runtime, CPU, Engine, Model, Config, Training, Tensor,
build files, architecture documents, ADRs, architecture tests, backend conformance, integration
tests, and all other paths. If implementation or documentation proves that a fourteenth path is
required, stop and replan rather than widening implicitly.

## Maximum scope

At most the thirteen paths above. The executable change is one immutable contribution, one
source-compatible overload, complete-role validation/canonicalization, and one append-only branch
in existing shared assignment. It adds no module dependency, generated code, benchmark,
repository-wide capability checkpoint, CPU/Engine behavior, or Runtime mechanism.

## Acceptance criteria

- The new record has exactly the four specified components and no additional public/protected
  member beyond documented record accessors and canonical construction.
- `GraphPreparation` exposes exactly the existing three-argument method and the new four-argument
  overload; both remain stateless and synchronous.
- Nulls, indexed null elements, duplicate IDs, equal-but-not-identical artifact references,
  mismatched IDs/descriptors, wrong graph/source/publication/logical roles, dynamic or unresolved
  descriptors, incomplete coverage, extra contributions, and the zero-partition case fail before
  any backend analysis, finalization, or assembler call.
- A valid fully static source-only published splat accompanying at least one ordinary partition is
  absent from every `PrepareContext` yet present exactly once in the complete memory plan and
  `PreparedScheduleContext.bufferAssignments()`.
- Ordinary buffer/workspace slots, plan indices, geometry, aggregation, per-finalizer assignments,
  backend invocation order, and partition projection are unchanged.
- Contributed slots are appended after ordinary buffers and ordered by final graph-value encounter
  order independent of caller contribution order.
- Contributed physical byte size and alignment are accepted exactly from the supplied values;
  Prepare performs no descriptor-to-byte inference.
- The new assignment is sufficient for an unchanged fake assembler to describe one initialized
  constant representation and requested publication while retaining exactly the existing
  execution occurrence for the accompanying partition.
- No pure zero-node schedule succeeds, no synthetic executable/partition/node is created, and no
  CPU or Engine acceptance boundary changes.
- Public/package shape, Javadocs, Runtime/Public API explanations, glossary, task/master/roadmap
  status, and exact scope are synchronized before completion.
- Prepare 0005 is `Complete` only after executable implementation, focused/full Prepare
  validation, and the separate documentation pass all completed. CPU 0010H and Engine 0006 remain
  Draft.

## Tests and validation

Focused Prepare tests must prove:

- exact record components, method overload descriptors, public/final/stateless shapes, and absence
  of extra public surface;
- record null order, ID/descriptor association, producer/consumer/output role, static Shape,
  resolved-layout, non-negative byte-size, and power-of-two alignment validation;
- orchestration null/index validation, immutable list snapshot, duplicate/missing/extra coverage,
  exact-reference rejection, and failure before backend-visible work;
- success with one non-empty fake partition plus one disconnected graph-input/output splat,
  including exact descriptor/logical references, unprojected value, unchanged finalizer
  assignments, appended slot, exact supplied geometry, initialized representation, and publication;
- multiple contributed constants supplied in reverse order but assigned in final graph-value order;
- rejection of a contributed pure zero-node graph without finalization or assembly;
- unchanged three-argument ordinary orchestration behavior;
- direct handoff first-declaration aggregation and workspace behavior with appended contributions;
  and
- immutable result/assignment containers and stable repeated ordinary declarations.

Future implementation validation after executable Java stabilizes:

```bash
./gradlew :modules:prepare:test --tests io.github.pho001.synaptik.prepare.GraphPreparationPublicShapeTest --tests io.github.pho001.synaptik.prepare.GraphPreparationTest --tests io.github.pho001.synaptik.prepare.BackendPartitionFinalizationHandoffTest
./gradlew :modules:prepare:test
./gradlew :modules:prepare:javadoc
git diff --check
```

The documentation-focused pass also validates all changed Markdown links and anchors, unique
headings, balanced fences, LF/final newlines, trailing whitespace, exact thirteen-path scope,
public/package inventory, task/master/roadmap order and status, only one detailed next unfinished
Prepare task, absence of a CPU 0010H task file, and preservation of excluded Java/build/
architecture paths. Do not repeat successful Java tests unless documentation changes executable
Java behavior or exposes a concrete stale-evidence risk.

Validation tier: affected Prepare module plus its Javadoc and focused explanatory/planning
documentation. Repository-wide, CPU, Engine, Runtime, architecture, conformance, integration, and
performance validation are deferred to their owning tasks or CI because this task changes no
module edge or implemented concrete-backend behavior.

## Dependencies

- Compiler 0006B5 — Complete; supplies the resolved static logical descriptor.
- Planning 0005–0006 — Complete; preserves the producerless/consumerless graph-output obligation.
- Prepare 0001–0004 — Complete; supply analysis declarations, complete-set assignment,
  orchestration/schedule validation, immutable partition DAG projection, and the unrelated opaque
  tuning handoff.
- Runtime memory, initialized representation, schedule, publication, runner, and lease contracts
  — Complete and unchanged.

This task is Complete. No later Prepare task is detailed, Ready, or In progress.

## Downstream handoff

CPU 0010H follows this task. In the current sole non-empty CPU composition it will identify the
eligible source-only published splat, derive and supply exact CPU byte size/alignment, create the
initialized CPU representation recipe from the existing Compiler scalar, and expose the
contribution to Engine composition. It must retain zero-node and mixed/multi-partition rejection
and reuse current initialization/publication machinery. This task does not create the CPU 0010H
specification.

Engine 0006 follows CPU 0010H and will wire the CPU contribution through the four-argument Prepare
overload for the selected scalar-objective backward convenience. Runtime needs no task because its
existing initialized-buffer, validity, ordered publication/alias, result lease, and cleanup
contracts already cover the lifecycle.

## Documentation-focused clean-context pass

After Java and focused Prepare tests stabilize, hand the exact diff, baseline status, validation
evidence, public-shape evidence, source-only eligibility/canonical-order decisions, and downstream
ownership boundary to a distinct clean documentation-focused context. Apply the General,
API/Javadoc, and Planning profiles. Finalize every changed Java Javadoc, Prepare package Javadoc,
the Prepare sections of Runtime/Public API documentation, glossary terminology, and planning
status/evidence. Review the architecture contract, focused architecture pages, ADRs, Compiler,
Planning, Runtime, CPU, Engine, Tensor, and Training documentation for impact and record reasoned
no-change conclusions. Do not change executable Java or repeat successful Java tests unless a
documentation correction requires it.

## Architecture impact

Expected impact: None.

The task realizes the existing staged Prepare ownership: complete shared resource validation and
slot assignment occur between backend analysis and finalization, while a concrete backend supplies
physical geometry. The new contribution is required only because a semantically valid published
source has no partition node through which ordinary analysis could declare it. No ownership rule,
dependency direction, execution model, or Runtime lifecycle changes. Stop rather than edit
architecture or an ADR if implementation evidence contradicts this conclusion.

## Implementation prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik on Prepare task 0005. Do not use GSD, commit, or push.
Use a separate clean implementation context. Read AGENTS.md, ARCHITECTURE.md, the planning guide,
documentation rules, this task specification, and every source it identifies as required reading.
Inspect and preserve the dirty worktree. Implement and validate exactly this specification and its
thirteen-path allowlist. Stop and report any architecture conflict, scope conflict, or required
fourteenth path.

After executable Java and the recorded Prepare test run stabilize, hand the exact diff and test
evidence to a distinct clean documentation-focused context. That context must apply the General,
API/Javadoc, and Planning profiles and complete the documentation, Javadoc, glossary, planning,
and final validation defined here without repeating successful Java tests unless executable code
changes or a concrete risk requires it. Mark Complete only after that pass and every acceptance
gate finish.
```

## Stop conditions

Stop and report instead of implementing or widening if any of these occurs:

- the source-only value cannot be identified entirely from existing immutable Compiler and
  Planning artifacts;
- assignment requires choosing or synthesizing a partition/backend owner;
- Prepare must derive physical byte geometry or inspect/materialize the scalar value;
- the contribution must enter `PrepareContext` or a partition finalizer to work;
- success requires a synthetic node, partition, executable, zero-node schedule, or relaxed CPU
  composition boundary;
- Runtime requires a new representation, validity, schedule, publication, lease, or cleanup type;
- a new dependency, architecture rule, ADR, build edit, CPU/Engine implementation, or fourteenth
  path is required; or
- ordinary partition-connected slot/index/finalization behavior cannot remain unchanged.

## Local decisions

- A root-package contribution is narrower and more truthful than projecting a node-disconnected
  value into an arbitrary partition or weakening `PrepareContext`'s partition-local meaning.
- The contribution retains exact graph and logical requirement references so shared Prepare can
  validate Compiler/Planning continuity before accepting backend-supplied physical geometry.
- Canonical graph-value order, not caller order or numeric `ValueId`, defines deterministic order
  among producerless contributions.
- Appending after ordinary declarations preserves every established ordinary slot/index and
  finalizer assignment.
- The contribution has no finalizer assignment because no partition owns it. The schedule
  assembler already receives complete artifacts plus the plan-ordered logical assignment and is
  the existing place where CPU 0010H can later add initialized representation creation.
- An explicit non-empty-partition guard keeps the seam from silently broadening current execution
  composition.
- Implementation retained the four-component record and source-compatible overload exactly as
  planned. Complete-role validation occurs before projection or backend-visible work, and the
  handoff's append-only assignment branch leaves ordinary declaration aggregation and finalizer
  inputs unchanged.

## Known limitations

This task alone does not make the value runnable in CPU or callable through Engine. Until CPU
0010H supplies physical geometry and initialized materialization and Engine 0006 wires that
contribution, the existing composition remains unable to prepare the source-only publication.
Dynamic Shapes, unresolved descriptors, zero-node graphs, and mixed/multi-partition CPU
composition remain unsupported.

## Validation evidence

Implementation context: `01a0a5ae-357d-7463-838d-a84d556e1214` (2026-09-15).

- Implemented the exact four-component public contribution, the four-argument orchestration
  overload with three-argument delegation, exact-reference and complete-role validation,
  graph-value-order canonicalization, the non-empty-partition guard, and append-only complete-set
  slot assignment. No producerless contribution enters a partition context or finalizer
  assignment.
- The focused command
  `./gradlew :modules:prepare:test --tests io.github.pho001.synaptik.prepare.GraphPreparationPublicShapeTest --tests io.github.pho001.synaptik.prepare.GraphPreparationTest --tests io.github.pho001.synaptik.prepare.BackendPartitionFinalizationHandoffTest`
  passed 23 tests with zero failures, errors, or skips.
- The final `./gradlew :modules:prepare:test` passed 52 tests with zero failures, errors, or skips.
  No executable Java changed after that run.

Documentation context: `01a0a5b6-e42c-7211-a8df-4a02d857314d` (2026-09-15).

- Applied the General, API/Javadoc, and Planning profiles after reading the complete architecture
  contract, documentation rules, planning guide/roadmap, Prepare master/task, focused
  architecture and ADR contracts, implementation source, tests, API guides, and glossary.
- Finalized Javadoc/comments only in the three affected production types plus Prepare package
  documentation. Constructor, component-accessor, overload, and package contracts now state
  nullability, exact identity, ordering, snapshots, ownership, side effects, results, failures,
  and the non-allocation/non-materialization boundary without changing executable Java tokens.
- Finalized the Runtime and Public API mental model, added the reusable glossary term, and kept the
  current-vs-planned boundary explicit: Prepare assigns only; CPU 0010H supplies physical
  declaration/initialization and Engine 0006 supplies wiring later; zero-node execution remains
  unsupported.
- Architecture contract, focused architecture pages, and ADRs require no change because the
  implementation realizes the existing staged complete-set assignment boundary without changing
  ownership or dependency direction. CPU and Engine masters remain unchanged because their Draft
  rows already own the later materialization and wiring. Runtime production and API beyond the
  allowed Runtime guide remain unchanged because existing memory, initialized-representation,
  schedule, publication, lease, and cleanup contracts already consume the assignment. Compiler,
  Planning, Model, Config, Tensor, Training, other guides, tests, build files, architecture tests,
  backend conformance, and integration tests require no change because this task changes none of
  their contracts, dependencies, or executable behavior.
- Reused the stable Java test evidence above and did not rerun tests because this documentation
  pass changed no executable Java. Final Prepare Javadoc and the Markdown, exact-scope,
  executable-token, test-preservation, public/package inventory, whitespace, newline, and
  status/order gates passed.
- `./gradlew :modules:prepare:javadoc` passed after final Javadoc edits: 9 actionable tasks, 2
  executed and 7 up-to-date. Inspection of the generated record, facade, and package pages
  confirmed the component/accessor descriptions, both overloads, and package boundary.
- The corrected read-only Ruby validator over the six changed Markdown files passed local-link,
  generated-heading-anchor, explicit-anchor uniqueness, fence-balance, trailing-whitespace, CRLF,
  and LF final-newline checks. Two earlier invocations failed in the checker itself—first because
  the installed Ruby lacks `Array#filter_map`, then because the initial GitHub-heading slug model
  collapsed spaces around punctuation—and did not identify a documentation defect.
- `cmp` against the documentation-context baseline passed for all three allowlisted test files.
  Comment-stripped and whitespace-normalized SHA-256 comparison passed for all three affected Java
  production files: `1daa93e546f29de356c1c32cc370233708c22085e613cfe8c21513f3a727cae3`
  for `ProducerlessPublishedConstantResource.java`,
  `efd37e90e46eef6eacef747534fdd5e1d03176b50bbe4a38fa4c999205af84cd`
  for `GraphPreparation.java`, and
  `6fc1922d974cf85b445d5c5bbd1819ba255d5a2d455d0b5091785035bd480c87`
  for `BackendPartitionFinalizationHandoff.java`; executable Java tokens are unchanged.
- A sorted `git status --porcelain=v1` scope comparison passed for exactly the thirteen task paths.
  This documentation context changed only allowlist paths 1, 2, 3, 4, 8, 9, 10, 11, 12, and 13;
  implementation-owned test paths 5, 6, and 7 remained byte-for-byte unchanged.
- Root-package source inventory plus `javap -public` confirmed the new record's exact constructor,
  four accessors and inherited record-object methods, and exactly the two public static
  `GraphPreparation.prepare(...)` overloads. No CPU 0010H task file exists.
- Status/order inspection confirmed Prepare 0001–0005 Complete, no later detailed Ready or
  In-progress Prepare task, Complete Compiler 0006B5, and Draft CPU 0010H and Engine 0006.
- Final `git diff --check` passed. The full `git status --short` review confirmed that every path
  outside the thirteen-path Prepare 0005 scope remains part of the preserved stacked worktree.

Planning context: `01a0a59b-dccb-7d53-b9d4-f7cf0b53a4c6` (2026-09-15).

- Read the required architecture contract, focused architecture pages, ADRs, documentation rules
  and General/Planning profiles, planning guide/roadmap, complete Prepare lineage, Compiler 0006B5
  and master, Planning 0005–0006, relevant Runtime/CPU/Engine plans and contracts, directly
  relevant source/tests, and the starting dirty diff.
- Source inspection confirmed that `GraphPreparation.Projection` is intentionally node-connected,
  `PrepareContext` constants must be projected node inputs, the complete-set handoff assigns only
  analysis declarations in first-declaration order, and schedule validation already requires
  assignments plus initialized representations for all constant sources and assignments for all
  publications.
- Compiler 0006B5 evidence confirms the exact fully static graph-input + splat + graph-output +
  no-consumer role and canonical resolved descriptor. Planning continues to retain the matching
  producerless/consumerless graph-output logical requirement.
- CPU 0010F evidence confirms the sole non-empty CPU composition rejection boundary; CPU 0010G,
  Runtime memory/schedule/initialized/publication/lease contracts, and Engine 0006 confirm that
  only later CPU physical declaration/materialization and Engine wiring remain.
- The dirty baseline was inspected before edits and contains user-owned Java, test, API,
  glossary, and planning changes from the completed Compiler/Runtime/CPU/Engine frontier. This
  planning pass edits only the five authorized planning paths and preserves all other changes.
- Read-only local-link validation over the five planning paths reported no missing targets; the
  changed content adds no heading-anchor reference. Duplicate-heading and balanced-fence checks
  reported no failure.
- `rg` checks reported no carriage return or trailing whitespace in the five planning paths;
  byte inspection reported `0a` as the final byte of each file.
- Status/order inspection confirmed Prepare 0001–0004 Complete followed by the sole detailed
  Ready Prepare 0005, Complete Compiler 0006B5 before it, and Draft CPU 0010H plus Draft detailed
  Engine 0006 after it. No CPU 0010H task file exists.
- `git diff --check` passed. No Gradle test or Javadoc task was run because this context changes
  planning Markdown only and implements no Java, build, API guide, architecture, or behavior.

## Implementation notes

- Added `ProducerlessPublishedConstantResource` as the exact immutable complete-graph
  contribution and added the four-argument `GraphPreparation.prepare(...)` overload while the
  original overload delegates with `List.of()`.
- `GraphPreparation` validates exact graph/logical references, required role and coverage, caller
  order failures, final graph-value ordering, and the non-empty-partition condition before
  projection or backend work.
- `BackendPartitionFinalizationHandoff` validates contribution overlap and appends each supplied
  geometry after ordinary buffers without adding any per-finalizer assignment.
- Focused tests lock public shape, validation precedence, exact identity, immutable snapshots,
  unchanged ordinary behavior, append ordering/geometry, schedule visibility, and zero-node
  rejection.
- The independent documentation pass changed documentation and comments only; executable Java and
  all tests remain byte-for-byte identical to its starting snapshot.

## Completion summary

- Completed changes: added the exact producerless published-constant handoff, orchestration
  validation/canonicalization, and append-only shared buffer assignment while preserving ordinary
  partition analysis and finalization.
- Files changed or created: exactly the thirteen allowlisted production/Javadoc, Prepare package,
  focused test, API/glossary, task, Prepare master-plan, and roadmap paths.
- Tests and validation: implementation context passed the focused 23-test command and full
  52-test Prepare command with zero failures, errors, or skips; documentation context passed
  Prepare Javadoc and all required documentation/scope/status checks without rerunning tests.
- Documentation-agent review: clean context `01a0a5b6-e42c-7211-a8df-4a02d857314d` finalized
  affected Javadoc/comments, API explanations, glossary impact, and planning evidence.
- Documentation impact: current shared assignment and later CPU/Engine ownership are documented
  without implying execution or materialization.
- Javadoc review: affected public construction, accessors, and overload contracts document inputs,
  results, nullability, identity, ordering, ownership, side effects, and failures.
- Glossary impact: added the reusable producerless published-constant resource term and linked it
  from graph preparation.
- Unresolved issues: None within Prepare 0005.
- Follow-up required: CPU 0010H and Engine 0006 remain separate Draft downstream work.

Status: Complete
