# Task 0008E1: Shared [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) adoption and reconstruction removal

## Reader context

This task records removal of duplicate partition-topology reconstruction, not a change to fusion or
representation policy. PrepareContext.partitionDag() supplies validated stable nodes,
producer/output-port occurrences, consumer/input-port occurrences, and edges. Repeated ports remain
separate occurrences. `CpuPartitionDagDecomposer` maps those facts into CPU-owned unit membership
and dependencies rather than retaining another complete adjacency index.

`CpuPartitionLowering`, `CpuAffineLayoutLowering`, and `CpuSpecializedSubgraphRecognizer` consume
shared occurrences only where equivalent. `CpuFusionProfitabilitySelector` still derives CPU
boundary roles, while logical-memory facts determine publication. `CpuRepresentationPlanner` retains
lowered instruction-use counts because they differ from graph occurrences. Structural oracles
compare ordering, identities, resources, and class bytes; existing semantic and performance evidence
remains controlling because generated and executable work is recorded as unchanged.

Current-topic navigation (not a replacement for the historical record):

- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Make [CPU](../../../../glossary.md#term-cpu) cold analysis consume [Prepare](../../../../glossary.md#prepare)'s existing [immutable](../../../../glossary.md#term-immutable) [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) wherever that
projection already supplies the exact [partition](../../../../glossary.md#partition)-local [producer](../../../../glossary.md#term-producer), consumer, edge, [occurrence](../../../../glossary.md#term-occurrence), and
stable-order fact. Remove the equivalent CPU scans without changing any CPU 0008B–0008E
[decomposition](../../../../glossary.md#term-decomposition), recognition, profitability, representation, resource, [artifact](../../../../glossary.md#term-artifact), execution, or
[Runtime](../../../../glossary.md#term-runtime) result.

The completed path remains:

```text
Prepare validates one partition-local DAG
  -> CPU reads exact node/port occurrences during cold analysis
  -> unchanged CPU units, candidates, decisions, and declarations
  -> unchanged schema-53 artifacts and prepared execution
```

This is an [ownership](../../../../glossary.md#term-ownership-meanings)-[alignment](../../../../glossary.md#term-alignment) and reconstruction-removal task. It is not a [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model),
[fusion](../../../../glossary.md#term-fusion)-policy, representation-policy, generated-code, or Runtime task.

## [Scope](../../../../glossary.md#term-scope)

- Use `PrepareContext.partitionDag()` as the authoritative partition-local topology in the CPU
  analysis sites where the current source independently rediscovers the same facts from<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [CPU glossary](../../../../glossary.md#term-cpu))</span>
  `context.nodes()`:
  - [node](../../../../glossary.md#node) position and exact retained node [identity](../../../../glossary.md#term-identity-meanings);
  - the unique local [producer](../../../../glossary.md#term-producer) [occurrence](../../../../glossary.md#term-occurrence) of a [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot);
  - all local consumer occurrences, including repeated input ports;
  - local producer-to-consumer edges with exact output and [input port](../../../../glossary.md#capture-graph-ports-and-reconstruction) positions;
  - external input occurrences; and
  - stable [topological](../../../../glossary.md#term-topological-order) node order.
- Refactor `CpuPartitionDagDecomposer` so validation, ordinals, vertical-edge/fan-out decisions,
  dependency construction, topological ordering, unit projection, and outside-unit consumer
  detection use shared DAG facts where they are exactly equivalent.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [fan-out glossary](../../../../glossary.md#term-fan-out); [dependency glossary](../../../../glossary.md#term-dependency); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Keep unit-local and candidate-local topology explicit. A contracted unit or enumerated candidate
  is still CPU-owned state; derive its membership and unit dependencies by filtering or mapping
  shared occurrences, not by introducing a second general graph abstraction.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Refactor ordinary pointwise lowering to use shared producer/consumer occurrences when deciding
  external inputs and whether an output is consumed inside the projected unit. Preserve semantic
  input-port order, repeated input occurrences, output order, first-use external-boundary order,
  virtual/materialized output order, instruction order, and store order.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering))</span>
- Refactor affine-chain use validation to use exact shared consumer occurrences while preserving
  the existing connected-chain, one-use intermediate, final-result, publication, descriptor, and
  address-composition rules.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Refactor specialized-subgraph recognition to use shared producer and consumer queries for
  suffix privacy, single-use checks, and transpose-producer recognition. Preserve member-node
  positions, attempt/fact ceilings, exact suffix vocabulary, baseline associations, execution
  dispositions, and fail-closed behavior.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [fail-closed behavior glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Refactor fusion profitability boundary-role calculation to map shared producer and consumer
  occurrences through the already selected CPU unit membership. Preserve <span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>`EXTERNAL_READ`,
  `CROSS_UNIT`, `PARTITION_WRITE`, and [`PUBLICATION`](../../../../glossary.md#term-publication) results and every candidate identity, score,
  tie, and fallback.
- Preserve `CpuPartitionPreparer` as the owner of orchestration, publication projection, selected
  declarations, recognition/profitability/representation sequencing, and plan assembly. It may
  receive Javadoc-only clarification if required by the implementation, but it must not acquire a
  new topology model.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc); [model glossary](../../../../glossary.md#term-model))</span>
- Preserve `CpuRepresentationPlanner`'s IR-level consumer positions and instruction-use counts.
  Those are representation-adjusted candidate facts, not substitutes for graph consumer
  occurrences, and must not be rewritten to use raw DAG counts.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Add focused regressions that prove shared-DAG adoption does not collapse repeated ports,
  multi-output producer positions, stable order, fan-out, unit dependencies, or boundary roles.
  Existing CPU 0008B–0008E decomposition, recognition, profitability, materialization-candidate,
  resource, generated-evidence, and prepared-execution tests remain controlling.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [materialization glossary](../../../../glossary.md#materialization))</span>
- Keep every existing externally callable signature source-compatible. Prefer no signature change.
  A package-internal overload or parameter change is permitted only when the existing signature
  cannot consume the already available <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [parameter glossary](../../../../glossary.md#term-parameter))</span>[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext)/[`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) without duplicating facts;
  it must retain the old source call form or the implementation must stop and record the proven
  reason before proceeding.
- After implementation and CPU validation, run the mandatory clean documentation-focused pass.
  It must finalize affected Javadocs, explain the shared-DAG consumption boundary in the CPU
  backend guide if the current guide is incomplete, and record a reasoned glossary no-change
  conclusion unless an existing term is inaccurate.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [backend glossary](../../../../glossary.md#backend))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- A full Model or compiled-model DAG, graph regions, cross-partition topology, partition creation,
  scheduling, transfers, publication ownership, or a general graph utility/abstraction.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [graph glossary](../../../../glossary.md#graph); [partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Changes to [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag), [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), Prepare validation, or another
  module. The shared projection is already complete for this task.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module))</span>
- New fusion legality or profitability policy, contraction grammar, seed recognition, topology
  ceiling, score, tie rule, barrier rule, materialization candidate, representation selection, or
  promotion policy.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [contraction glossary](../../../../glossary.md#contraction-dimension); [seed glossary](../../../../glossary.md#term-seed); [materialization glossary](../../../../glossary.md#materialization))</span>
- MATMUL or linear epilogues, pooling, attention, losses, native routes, packing, tuning, Config,
  Prepare 0004, CPU 0016, or any later CPU 0008F–0017 capability.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [linear glossary](../../../../glossary.md#term-linear-meanings); [pooling glossary](../../../../glossary.md#term-pooling); [losses glossary](../../../../glossary.md#loss))</span>
- New operations, public APIs, backend capability, schema bump, cache compatibility, artifact
  identity, generated class shape, emitter/generator behavior, executable/finalizer behavior,
  resources, execution order, Runtime behavior, build configuration, architecture rules,
  conformance, or integration behavior.<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [shape glossary](../../../../glossary.md#shape); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- New generated or hot-loop optimization and new performance claims. Do not rerun or invent a
  benchmark when inspection and structural tests prove no generated/hot-path change.
- Creating a detailed specification for [CPU](../../../../glossary.md#term-cpu) 0008F or any later task.
- [Commit](../../../../glossary.md#term-commit), push, staging, or unrelated cleanup in either implementation context.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md).
- [`Current architecture documentation`](../../../../architecture/current-architecture-plan.md).
- [`Module boundaries`](../../../../architecture/module-boundaries.md).
- [`Dependency rules`](../../../../architecture/dependency-rules.md).
- [`Lifecycle and ownership`](../../../../architecture/lifecycle.md).
- [`Runtime, Prepare, and Backend boundary`](../../../../architecture/runtime-prepare-backend-boundary.md).
- [`Performance evidence and model autotuning`](../../../../architecture/performance-evidence-and-tuning.md).
- [`Planning guide`](../../../planning-guide.md).
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md).
- [`Partition-preparer guide`](../../../../backend-guide/partition-preparer.md).
- [`CPU master plan`](../master-plan.md).
- [`Prepare master plan`](../../../modules/prepare/master-plan.md).
- Complete [`Prepare 0003A`](../../../modules/prepare/tasks/0003a-immutable-partition-local-dag-analysis-projection.md).
- Complete [`CPU 0008B`](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md),
  [`CPU 0008C`](0008c-typed-specialized-subgraph-and-epilogue-recognition.md),
  [`CPU 0008D`](0008d-bounded-fusion-profitability-and-typed-decision-facts.md), and
  [`CPU 0008E`](0008e-bounded-multi-input-materialization-and-representation-reuse.md).

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative. Shared Prepare owns validated partition projection;
  CPU owns lowering, fusion, route/representation choice, exact declarations, finalization, and
  executable construction; Runtime executes only immutable prepared work.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- [CPU](../../../../glossary.md#term-cpu) must consume the exact [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) carried by [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext). It must not construct a
  replacement `PartitionDag` for the complete partition or retain a second complete adjacency
  index.
- Unit projection may construct the existing unit-scoped `PrepareContext` required by current
  lowering. Its `PartitionDag` must contain only the exact selected nodes in original stable order,
  and projected logical-memory/publication facts must remain semantically identical.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node))</span>
- Node and port occurrence identity is part of correctness. A repeated input has multiple consumer
  occurrences; a multi-output node has one producer occurrence per output port; value equality
  must not erase those distinctions.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [output port glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- Stable ordering comes from the shared DAG and existing CPU deterministic rules, never hash-map
  iteration, textual IDs, graph identity in candidate keys, or a new sorting policy.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [hash glossary](../../../../glossary.md#term-hash))</span>
- Logical-memory facts remain the authority for [graph](../../../../glossary.md#graph) [publication](../../../../glossary.md#term-publication) and cross-[partition](../../../../glossary.md#partition) obligations.
  `PartitionDag` is topology-only and must not be used to infer publication, materialization,
  transfer, or execution policy.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [materialization glossary](../../../../glossary.md#materialization))</span>
- Candidate-local IR uses and representation-adjusted consumer positions remain CPU-owned. Shared
  graph occurrence counts must not replace them where representation planning needs instruction
  occurrence counts after lowering.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [planning glossary](../../../../glossary.md#term-planning))</span>
- Analysis remains deterministic, cold, immutable, and measurement-free. Finalization cannot
  rerank, add resources, inspect graph semantics, or change the selected plan.
- No public/shared API, dependency, architecture, build, schema, resource, generated-code, or
  Runtime change is authorized. Discovering such a need is a stop condition.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [schema glossary](../../../../glossary.md#term-schema))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.prepare.analysis` — existing [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) and [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext)
  contracts, consumed without modification.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — CPU-private cold decomposition,
  lowering, recognition, profitability, and representation analysis.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — unchanged orchestration and plan
  assembly exercised by integration regressions.

Packages added or changed:

- No package is added.
- Production changes are confined to existing [CPU](../../../../glossary.md#term-cpu)-private [lowering](../../../../glossary.md#lowering) classes.

Type placement:

- `CpuPartitionDagDecomposer` remains the CPU owner of computation-unit membership,
  contraction/enumeration, and candidate-local unit dependencies while consuming shared graph
  occurrences.<span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [graph glossary](../../../../glossary.md#graph))</span>
- `CpuPartitionLowering` and `CpuAffineLayoutLowering` remain the owners of their semantic lowering
  and unit-local virtual/materialized decisions.
- `CpuSpecializedSubgraphRecognizer` remains the [owner](../../../../glossary.md#term-owner) of recognition-only typed facts.
- `CpuFusionProfitabilitySelector` remains the owner of CPU-private boundary roles, scores, and
  selection facts.
- No new production type or general [topology](../../../../glossary.md#term-topology) helper is expected.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAffineLayoutLowering.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuSpecializedSubgraphRecognizer.java`; and
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFusionProfitabilitySelector.java`.

Expected focused test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposerTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAffineLayoutLoweringTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuSpecializedSubgraphRecognizerTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFusionProfitabilitySelectorTest.java`; and
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`.

Expected documentation/[planning](../../../../glossary.md#term-planning) paths across implementation and the mandatory documentation pass:

- `docs/backend-guide/cpu-backend.md`;
- this task specification;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

Conditional documentation path:

- `docs/glossary.md` only if the documentation-focused audit proves an existing [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag),
  CPU decomposition, or CPU preparation term inaccurate. Otherwise record the reasoned no-change
  conclusion in this task's validation evidence and completion summary.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [validation glossary](../../../../glossary.md#term-validation))</span>

Reviewed and frozen unless a stop condition is reported:

- all [Prepare](../../../../glossary.md#prepare) Java and tests, including `PartitionDag`, [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), and [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation);
- `CpuPartitionPreparer.java`, `CpuRepresentationPlanner.java`, and
  `CpuPartitionPreparationPlan.java`;
- code generation, emitters, cache/schema, finalizer, executable, Runtime, resources, Gradle,
  architecture, conformance, and integration paths; and<span data-glossary-navigation="true"> ([code generation glossary](../../../../glossary.md#term-code-generation); [cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema); [Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008F and every later task path.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 16 paths:

- 5 existing [CPU](../../../../glossary.md#term-cpu)-private production/[Javadoc](../../../../glossary.md#term-javadoc) paths;
- 6 existing focused CPU test paths;
- 1 existing CPU [backend](../../../../glossary.md#backend) guide;
- 1 conditional existing glossary path; and
- exactly 3 [planning](../../../../glossary.md#term-planning) paths: this task, CPU master plan, and roadmap.

It may create no Java type, package, [test fixture](../../../../glossary.md#adr-ci-and-qualification) file, resource, evidence bundle, benchmark,
[schema](../../../../glossary.md#term-schema), or later task specification. The normal implementation should stay below the ceiling when
the glossary remains accurate. A path may not be substituted merely to consume unused capacity.
If another path is required, stop and propose a follow-up task with the exact reason.

The established CPU ceilings remain unchanged: 8 [partition](../../../../glossary.md#partition) [nodes](../../../../glossary.md#node) and final units, 28 compatibility
pair attempts, 64 legal complete topologies, 256 enumeration attempts, 16 materialized boundaries,
16 simultaneously live [IR](../../../../glossary.md#term-intermediate-representation-ir) values, 32 indexing-complexity units, 64 generated-code units, and all
CPU 0008D/0008E candidate, representation, resource, and decision ceilings.

## Acceptance criteria

- Every complete-[partition](../../../../glossary.md#partition) [CPU](../../../../glossary.md#term-cpu) [producer](../../../../glossary.md#term-producer)/consumer/edge fact that is exactly represented by
  `context.partitionDag()` is read from that shared projection rather than reconstructed through
  a complete `context.nodes()` producer/consumer [scan](../../../../glossary.md#term-cumulative-scan).
- Remaining node iteration is semantic processing, stable membership traversal, CPU candidate/unit
  mapping, or IR use accounting; each remaining topology-looking scan has a recorded exact reason
  why shared DAG facts are not equivalent.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [topology glossary](../../../../glossary.md#term-topology); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Repeated inputs retain distinct input-port consumer and edge occurrences. Fan-out and single-use
  gates count occurrences exactly as before, including two ports of one consumer node.<span data-glossary-navigation="true"> ([Fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Multi-output nodes retain exact output-port producer identity. Unsupported multi-output forms
  still fail at their established CPU boundary, and supported multi-output specialized families
  retain their existing unit/dependency behavior.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Node order, unit membership, member-node ordinals, dependencies, canonical split, compatibility
  baseline, candidate discovery order, attempt order, and every hard rejection remain unchanged.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Specialized-recognition family/form/attributes/members/boundaries/suffix/disposition and exact
  baseline-unit associations remain unchanged. MATMUL remains recognition-only and unsupported.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Profitability boundary roles, candidate identities, checked scores, comparable/profitable facts,
  tie behavior, incomplete-enumeration fallback, and selected topology remain unchanged.
- Representation candidates retain direct-first order, external-source eligibility, IR instruction
  use counts, single/disjoint-pair identities, `CO_CONSUMED_PAIR`, copy reuse, candidate-only
  selection, resource geometry, and explicit-candidate execution. Ordinary preparation remains
  exact CPU 0008D direct with zero representation materializations.
- Pointwise and affine lowering retain exact boundary order, access bindings, virtual and
  materialized values, IR, structural keys, specialization, declarations, and failure timing.<span data-glossary-navigation="true"> ([Pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering); [bindings glossary](../../../../glossary.md#term-binding); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Selected resources, unit-local workspaces, graph-split buffers, publication positions,
  materialization candidates, prepared plans, finalization, execution order, and Runtime validity
  behavior remain unchanged.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [buffers glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Generator schema remains 53. No generator, emitter, cache envelope, prepared executable, or hot
  loop changes. Existing representative generated class bytes and Class-File/member-reference
  controls remain exact.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- No new performance fork is required or claimed because generated forms and hot work do not
  change. The implementation records source-diff inspection plus structural/class-byte regression
  results and reuses the controlling CPU 0008B–0008E performance evidence. If generated bytes,
  executable hot work, or schema changes, stop and replan with an explicit oracle and performance
  evidence contract.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [regression glossary](../../../../glossary.md#term-regression); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- All existing callable signatures remain source-compatible unless the task records a proven
  impossibility and stops before making the incompatible change.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- A separate clean documentation-focused agent pass finalizes affected Javadocs and explanatory
  documentation, records the glossary impact, and does not rerun successful Java tests unless it
  changes executable Java behavior or identifies a concrete stale-evidence risk.
- CPU 0008E1 alone is [`Ready`](../../../../glossary.md#term-ready) before implementation and becomes `Complete` only after all evidence
  and documentation review pass. CPU 0008F remains `Draft` without a detailed specification.
- Exact paths, links, anchors, headings, fenced blocks, final newlines, whitespace, source/status
  synchronization, absence of a CPU 0008F specification, and an empty staged index pass.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused tests:

```bash
./gradlew :backends:cpu:test --tests '*CpuPartitionDagDecomposerTest' --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuAffineLayoutLoweringTest' --tests '*CpuSpecializedSubgraphRecognizerTest' --tests '*CpuFusionProfitabilitySelectorTest' --tests '*CpuPartitionPreparerTest'
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) must include or preserve direct assertions for repeated input ports, [fan-out](../../../../glossary.md#term-fan-out),
multi-output [producer](../../../../glossary.md#term-producer) positions or [fail-closed](../../../../glossary.md#term-fail-closed) boundaries, stable [node](../../../../glossary.md#node)/unit order, direct
dependencies, recognition single-use/[transpose](../../../../glossary.md#term-transpose) [lookup](../../../../glossary.md#term-lookup), boundary roles, and selected plan/resource
[identity](../../../../glossary.md#term-identity-meanings).

Authoritative affected-[module](../../../../glossary.md#term-module) gate, [run](../../../../glossary.md#term-run) once after the focused matrix passes:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Retain the exact test/suite/skip/failure counts from the generated XML instead of assuming the
[CPU](../../../../glossary.md#term-cpu) 0008E count remains current. Existing resource, finalization, executable, representation, and
generated-evidence suites are part of this gate and must remain passing.

Structural no-change checks:

```bash
git diff -- backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java
rg -n 'CURRENT_VERSION|schema' backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache
```

The first command must be empty. Confirm [schema](../../../../glossary.md#term-schema) 53 through the existing production [constant](../../../../glossary.md#term-constant) and
tests. Use existing generated-evidence and class-byte assertions; do not create a timing probe or
claim a new benchmark result.

Mandatory clean documentation pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

The documentation agent receives the successful Java evidence. It checks all changed Javadocs,
the CPU guide, glossary impact, [planning](../../../../glossary.md#term-planning) status, links/anchors/fences, terminology, examples,
final newlines, and exact path ceiling. It does not repeat the Java suites unless it changes
executable behavior or records a concrete reason.

Final planning/[scope](../../../../glossary.md#term-scope) checks:

```bash
test -f docs/planning/backends/cpu/tasks/0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md
test ! -e docs/planning/backends/cpu/tasks/0008f-portable-matmul-execution-and-bounded-linear-epilogues.md
rg -n '0008E1|0008F' docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md
git diff --name-only
git diff --check
git diff --cached --name-only
git status --short
```

Before completion, verify at most 16 changed paths, exactly the authorized [owners](../../../../glossary.md#term-owner), CPU 0008E1
status synchronized, CPU 0008F still Draft without a spec, no staged path, and no unrelated diff.

Repository-wide validation is deferred to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [backend](../../../../glossary.md#backend)'s
cold analysis only. Architecture tests, backend [conformance](../../../../glossary.md#term-conformance), integration tests, and other modules
are unchanged and need no new run unless the implementation violates the no-boundary-change
assumption; that discovery is a stop condition rather than permission to broaden the task.

## Dependencies

- Clean `main` baseline `ac91ea6e`.
- Complete [Prepare](../../../../glossary.md#prepare) 0003A [immutable](../../../../glossary.md#term-immutable) [partition](../../../../glossary.md#partition)-local [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) projection.
- Complete CPU 0008B decomposition, CPU 0008C recognition, CPU 0008D profitability, and CPU 0008E
  representation-candidate work.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Current generator [schema](../../../../glossary.md#term-schema) 53 and retained [CPU](../../../../glossary.md#term-cpu) 0008B–0008E generated/class/performance evidence.

## Follow-up tasks

- CPU 0008F remains Draft and owns portable MATMUL execution plus bounded linear epilogues after
  this task completes. Do not create or implement its specification here.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008G–0008I and CPU 0009 remain later ordered work.
- No follow-up is expected for shared [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) adoption if every acceptance criterion passes.

## Architecture impact

Expected impact: None.

This task aligns [CPU](../../../../glossary.md#term-cpu) implementation with the existing [Prepare](../../../../glossary.md#prepare) [ownership](../../../../glossary.md#term-ownership-meanings) boundary. It changes no
authoritative rule, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), public/shared contract, or [architecture test](../../../../glossary.md#term-architecture-test). If a shared
[API](../../../../glossary.md#term-application-programming-interface-api) or architecture change becomes necessary, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository from clean main baseline ac91ea6e.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md
in full. Implement CPU task 0008E1 exactly as specified. Do not use a GSD workflow, implement
later tasks, commit, push, or stage. Stop on any architecture conflict, shared/public API need,
generated/hot-path change, schema/resource change, or scope-ceiling breach.

After implementation and the specified CPU validation, hand the diff and exact test evidence to a
separate clean documentation-focused agent/thread. That pass must follow
docs/developer-guide/documentation-rules.md, finalize affected Javadocs, CPU-guide and glossary
impact, planning evidence/status, and documentation validation in the same overall change without
repeating successful Java tests unless it changes executable behavior or records a concrete risk.

Only after both passes succeed, update this task's evidence, implementation notes, completion
summary, and status. Leave CPU 0008F Draft without a specification.
```

## Local decisions

- Shared [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) occurrences replace only equivalent complete-partition topology scans.
  CPU-owned computation-unit membership, contraction enumeration, candidate identities,
  dependencies, semantic input positions, IR instruction-use counts, representation-adjusted
  consumer positions, boundary roles, and logical-memory/publication interpretation remain in
  their existing CPU owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [CPU glossary](../../../../glossary.md#term-cpu); [contraction glossary](../../../../glossary.md#contraction-dimension); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [publication glossary](../../../../glossary.md#term-publication))</span>
- Unit projection filters producer, consumer, and outside-consumer facts from the source DAG by
  exact <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>`CompiledNode` [identity](../../../../glossary.md#term-identity-meanings) and constructs the existing unit-scoped [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext). It adds
  no second complete adjacency index or general graph abstraction.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Existing callable signatures were sufficient. No public or package-internal compatibility
  overload was required.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- The CPU guide now states the implemented shared-DAG consumption boundary because its prior
  decomposition section did not identify the topology source or distinguish shared occurrences
  from CPU-owned unit/candidate/IR accounting.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition))</span>

## Known limitations

- [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) remains partition-local topology only. It does not represent publication,
  cross-partition consumers, materialization, representation, transfer, scheduling, or execution
  policy; the existing logical-memory and CPU-private plan facts remain authoritative for those
  concerns.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- MATMUL remains recognition-only and unsupported. CPU 0008F remains Draft without a detailed
  specification.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation-focused context supplied the authoritative stabilized Java evidence; the clean
  documentation context did not rerun Java tests because it changed Javadoc and Markdown only:<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
  - the six-suite focused [matrix](../../../../glossary.md#term-matrix-meanings) passed 63 tests with zero failures, errors, or skips;
  - exactly one final `./gradlew :backends:cpu:test --rerun-tasks` passed 547 tests across 104
    suites with 3 expected skips and zero failures or errors; and
  - no executable Java behavior changed after those runs.
- The five production changes consume `PrepareContext.partitionDag()` for stable nodes, exact
  producer/output-port and consumer/input-port occurrences, edges, and external inputs. The five
  focused test changes directly retain repeated-port, multi-output producer-position, stable
  unit/dependency, affine-use, recognition single-use, and profitability boundary-role evidence.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [producer glossary](../../../../glossary.md#term-producer); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Source and structural inspection confirms generator schema 53. The production diff beneath
  CPU generated code, cache, executable, and <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [CPU glossary](../../../../glossary.md#term-cpu); [cache glossary](../../../../glossary.md#term-cache))</span>`CpuPartitionFinalizer` is empty. No benchmark was
  required because generated forms, executable/finalizer behavior, and hot work did not change;
  the controlling CPU 0008B–0008E performance evidence remains applicable.
- Clean documentation context `01a04416-f91a-75a0-8ab6-60464ef6a64a` applied the General,
  API/Javadoc, Backend-guide, Planning, and Example profiles. It reviewed the complete final diff,
  all five changed production classes, all five changed tests, the CPU guide, glossary, task,
  CPU master plan, and roadmap. It finalized all five affected class and method Javadocs and the
  CPU guide's shared-DAG/CPU-accounting distinction.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- `./gradlew :backends:cpu:javadoc` passed after the final Javadoc edits with 11 actionable tasks
  (2 executed and 9 up-to-date) and only the 2 expected incubating Vector API warnings.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- Glossary no-change conclusion: the existing [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) and [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) entries already
  define the shared immutable partition-local projection, exact producer/output-port and
  consumer/input-port occurrences, repeated inputs, multi-output producers, and the topology-only
  policy boundary. CPU computation units, lowering, and materialization terms already keep their
  CPU-private ownership. This task introduces no new public reusable term and changes none of
  those definitions.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- A targeted Ruby checker over the four changed Markdown files passed all local file links and
  heading anchors, fenced blocks, CRLF checks, and final-newline checks. The final combined shell
  gate using `git status --porcelain=v1`, `git diff --cached --name-only`, focused `git diff`,
  `rg`, `test`, and `git diff --check` passed task/master/roadmap status coherence, schema 53,
  frozen production path no-diff, the exact 14 authorized changed paths under the 16-path ceiling,
  absence of a CPU 0008F task specification, an empty staged index, and whitespace validation.
- Repository-wide tests, architecture tests, backend conformance, and integration tests were not
  rerun. The task changes one backend's cold analysis without changing a dependency, shared or
  public contract, architecture rule, generated/executable behavior, or end-to-end capability;
  repository-wide validation remains deferred to CPU 0009 or CI as specified.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- `CpuPartitionDagDecomposer` now derives ordinals, edges, fan-out, dependencies, stable
  topological order, and unit-projection outside-consumer facts from the shared DAG while keeping
  contraction and unit membership CPU-owned.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [topological order glossary](../../../../glossary.md#term-topological-order); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [contraction glossary](../../../../glossary.md#contraction-dimension); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `CpuPartitionLowering` uses shared external-input, producer, and consumer occurrences while
  preserving first-use boundary order and repeated semantic input positions in CPU IR.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `CpuAffineLayoutLowering` validates exact intermediate/final consumer occurrences;
  `CpuSpecializedSubgraphRecognizer` uses exact producer and consumer occurrences for suffix
  privacy, single-use, and transpose-producer recognition; and<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose))</span>
  `CpuFusionProfitabilitySelector` maps shared occurrences through selected CPU units while using
  logical-memory facts for publication.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- No Prepare, Runtime, generated/cache/executable/finalizer, resource, build, architecture,
  conformance, integration, or CPU 0008F path changed.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [cache glossary](../../../../glossary.md#term-cache); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Completion summary

- Completed changes: adopted the shared partition-local DAG in the five authorized CPU cold
  analysis owners without changing CPU 0008B–0008E policy, accounting, resources, artifacts, or
  execution.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [CPU glossary](../../../../glossary.md#term-cpu); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Files changed or created: five CPU production/Javadoc paths, five focused CPU test paths, the
  CPU backend guide, this task specification, the CPU master plan, and the roadmap; 14 total
  paths, within the ceiling of 16.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend))</span>
- Tests and validation: reused the implementation context's passing 63-test focused matrix and
  passing final 104-suite/547-test CPU rerun with 3 expected skips; the clean documentation pass
  completed CPU Javadoc and all requested documentation, status, schema, frozen-path, scope,
  staging, and whitespace checks.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean documentation context
  `01a04416-f91a-75a0-8ab6-60464ef6a64a` independently finalized the five affected Javadocs,
  added the necessary CPU-guide boundary explanation, and synchronized planning evidence.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: one focused CPU-guide paragraph explains current shared-DAG consumption
  and the remaining CPU-owned unit/candidate/IR responsibilities. Architecture documentation,
  ADRs, and unrelated guides remain unchanged because no architecture rule changed.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Javadoc review: all five changed production classes now distinguish shared topology facts from
  CPU-owned lowering, unit, candidate, and boundary accounting and retain complete parameter,
  return, and expected-failure documentation.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [lowering glossary](../../../../glossary.md#lowering); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: no change; the existing [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag), CPU computation-unit,
  lowering, and materialization entries already define the exact terms and ownership boundaries.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Unresolved issues: None.
- Follow-up required: None for CPU 0008E1. CPU 0008F remains the planned Draft successor and has
  no detailed specification.

Status: Complete
