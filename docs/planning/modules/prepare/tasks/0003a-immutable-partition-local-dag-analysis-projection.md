# Task 0003A: [Immutable](../../../../glossary.md#term-immutable) [Partition](../../../../glossary.md#partition)-Local [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) Analysis Projection

## Status

Complete

## Goal

Introduce one public [immutable](../../../../glossary.md#term-immutable) [Prepare](../../../../glossary.md#prepare)-owned [directed acyclic graph](../../../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../../../glossary.md#intermediate-representation-dag-and-ssa)) projection for exactly
one `PlannedPartition`. The projection gives [concrete backend](../../../../glossary.md#concrete-backend) analysis stable [partition](../../../../glossary.md#partition)-local
[nodes](../../../../glossary.md#node) and precomputed structural [producer](../../../../glossary.md#term-producer), consumer, edge, external-input-[occurrence](../../../../glossary.md#term-occurrence), and local-
sink facts without exposing the complete cross-[backend](../../../../glossary.md#backend) [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) or making each
backend reconstruct those facts independently.

[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) must retain this projection as its sole node/[topology](../../../../glossary.md#term-topology) source of truth.
`PrepareContext.nodes()` remains a compatibility [view](../../../../glossary.md#term-view-meanings) that delegates to the projection rather
than retaining a second independent node list. [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) constructs the projection once
per planned partition through the authoritative complete-[graph](../../../../glossary.md#graph) projection path and passes that
exact instance into the backend context.

## [Scope](../../../../glossary.md#term-scope)

### Public [immutable](../../../../glossary.md#term-immutable) [partition](../../../../glossary.md#partition) projection

- Add `io.github.pho001.synaptik.prepare.analysis.PartitionDag`, or a more precise domain name if
  implementation evidence demonstrates that [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) is misleading. The final name and
  API must remain Prepare-owned, backend-neutral, public, immutable, and specific to one exact
  planned partition.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- Retain the partition's exact `CompiledNode` references in validated stable topological order.
  The projection contains no node from another partition and no <span data-glossary-navigation="true"> ([topological order glossary](../../../../glossary.md#term-topological-order))</span>[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) reference.
- Precompute immutable structural facts from each `CompiledNode`'s ordered input and output
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) lists. Facts must retain exact producer and consumer node positions plus output/input
  port positions.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Preserve repeated input occurrences independently. If one [node](../../../../glossary.md#node) consumes the same `ValueId` in
  two input ports, both consumer occurrences are present in port order.
- Preserve every [output port](../../../../glossary.md#capture-graph-ports-and-reconstruction) of a multi-output node independently and associate each produced
  `ValueId` with its exact node and output-port position.
- Identify every partition-external input [occurrence](../../../../glossary.md#term-occurrence): an [input port](../../../../glossary.md#capture-graph-ports-and-reconstruction) whose `ValueId` has no
  producer inside this partition. This is a structural occurrence, not an ownership or transfer
  decision.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Identify local sink nodes deterministically. A local sink has no output occurrence consumed by
  another node in this partition; graph publication and cross-partition use do not change that
  topology-only definition.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [topology glossary](../../../../glossary.md#term-topology))</span>
- Expose only narrow indexed or value-keyed queries required by current and near-term analysis
  consumers. Returned lists and occurrence facts are immutable deterministic snapshots. Do not
  introduce a mutable generic graph library, visitor/callback framework, or rewrite surface.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [callback glossary](../../../../glossary.md#term-callback))</span>

### [Fail-closed](../../../../glossary.md#term-fail-closed) construction and [validation](../../../../glossary.md#term-validation)

- Require the projection's [node](../../../../glossary.md#node) IDs to equal `PlannedPartition.nodeIds()` exactly in size,
  identity, and order.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Reject duplicate node IDs and duplicate produced [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) identities, including duplicates
  across output ports of the same node or different nodes.
- Reject a node input produced by that same node or by a later partition node. A missing local
  producer is an external-input occurrence and remains valid.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Reject every disagreement between the supplied `PlannedPartition` and stored nodes before the
  projection can reach backend analysis.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Derive adjacency and edge facts once from the validated ordered ports. Do not accept caller-
  supplied adjacency that could disagree with the nodes.
- Preserve deterministic order based on partition node order and port order, never hash-map
  iteration, object identity, or numeric <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>`ValueId` magnitude.

### [PrepareContext](../../../../glossary.md#prepare-context-preparecontext) and orchestration integration

- Replace the independent stored [node](../../../../glossary.md#node)-list component of `PrepareContext` with the exact
  partition-local DAG projection. <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>`nodes()` delegates directly to `partitionDag().nodes()` (or
  the chosen equivalent accessor) and remains immutable.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Preserve the current values, logical memory requirements, constants, backend inputs, fully
  static descriptor validation, exact-reference retention, deterministic collection order, and
  staged backend analysis/finalization behavior.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [backend glossary](../../../../glossary.md#backend); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation))</span>
- Preserve a source-compatible public constructor accepting the current
  `(PlannedPartition, List<CompiledNode>, List<GraphValue>,
  List<LogicalMemoryRequirement>, Map<ValueId, ScalarValue>, I)` arguments if Java record/class
  rules and the final truthful API shape permit it. That constructor must construct exactly one
  validated projection and delegate to the authoritative representation; it must not store the
  list separately.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>
- Make [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) construct each partition projection once while it holds the complete
  compile graph, then pass only that projection and the existing partition-local value/memory/
  constant facts into <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph))</span>`PrepareContext`. A concrete backend must never receive the complete
  cross-backend <span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend))</span>[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel).
- Keep all contexts constructed and validated before the first backend preparer invocation, as
  required by completed Prepare 0003.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>

### Compatibility evidence

- Record the exact final API shape from implementation evidence. In particular, test and document
  the canonical constructor, any list-taking compatibility constructor, record components or
  class properties, <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`nodes()` delegation, reflection surface, equality, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and
  `toString` behavior.
- Source compatibility for the current constructor call shape is the planned goal where
  technically valid. Do not claim binary compatibility: changing a public record component or
  canonical constructor can change descriptors, record reflection, equality, and textual form.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Prefer the narrowest truthful API. Do not preserve obsolete record-component or textual shape
  by retaining duplicate topology state or inventing a compatibility wrapper with ambiguous
  ownership.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

### Documentation and [validation](../../../../glossary.md#term-validation)

- Add detailed Javadoc for the projection, every public occurrence/edge fact and query, all
  inputs/results/failures, immutability, ordering, repeated occurrences, multi-output semantics,
  and topology-only limitations.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [occurrence glossary](../../../../glossary.md#term-occurrence); [topology glossary](../../../../glossary.md#term-topology))</span>
- Update [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), and `prepare.analysis` package Javadocs where the
  source of node/topology facts changes.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- Add focused Prepare tests for structural construction, validation, stable order, immutable
  queries, delegation, compatibility, reflection, equality, and orchestration identity.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Run downstream CPU source-compatibility and behavior checks without migrating CPU DAG
  reconstruction. CPU 0008E1 owns consumption and deletion of duplicated CPU scans later.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [CPU glossary](../../../../glossary.md#term-cpu); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Inspect architecture tests for the new public Prepare package surface and dependency rules.
  Update them only if their existing inventories or public-surface rules require the new type;
  no dependency edge is expected to change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [dependency edge glossary](../../../../glossary.md#term-dependency-edge))</span>
- Use a separate clean documentation-focused agent after executable Java stabilizes. That pass
  finalizes affected Javadocs, public/Prepare API explanations, backend-contributor guidance,
  glossary impact, and planning evidence without repeating successful Java tests unless it
  changes executable behavior or records a concrete stale-evidence risk.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>
- Run repository-wide validation after implementation and documentation stabilize because this
  task changes a shared public backend-analysis contract consumed by concrete backends.

## Out of [scope](../../../../glossary.md#term-scope)

- [CPU](../../../../glossary.md#term-cpu) adoption, changes to `CpuPartitionDagDecomposer`, or deletion of CPU [producer](../../../../glossary.md#term-producer)/consumer scans
- CPU MATMUL, linear epilogues, fusion policy, profitability, materialization selection, generated
  code, artifact schema, performance claims, capability reporting, or Runtime behavior<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [linear glossary](../../../../glossary.md#term-linear-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- cross-partition producer/consumer ownership inference, publication policy, logical/physical
  memory policy, materialization policy, fusion legality, route choice, cost, or scheduling<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [inference glossary](../../../../glossary.md#term-inference); [publication glossary](../../../../glossary.md#term-publication))</span>
- a complete-model DAG, global graph access, graph regions, callbacks, visitor APIs, rewrites,
  mutable adjacency, execution units, backend-specific facts, or a generic graph framework<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [graph glossary](../../../../glossary.md#graph); [mutable glossary](../../../../glossary.md#term-mutable); [backend glossary](../../../../glossary.md#backend))</span>
- changes to [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), `CompiledNode`, `PlannedPartition`, logical-memory semantics,
  Compiler artifacts, Runtime recipes, Engine, Config, Trace, or another backend<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [Trace glossary](../../../../glossary.md#trace))</span>
- [Prepare](../../../../glossary.md#prepare) 0004 [opaque](../../../../glossary.md#term-opaque) candidate/tuning [handoff](../../../../glossary.md#term-handoff) or any tuning/[cache](../../../../glossary.md#term-cache) behavior
- an architecture-contract, ADR, dependency, Gradle, backend-conformance, or integration change
  unless implementation exposes a concrete contradiction; in that case stop and report it<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): Prepare lifecycle, Prepare ownership,
  backend-facing Compiler isolation, staged backend analysis/finalization, and Runtime boundary.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md).
- [`lifecycle`](../../../../architecture/lifecycle.md).
- [`module boundaries`](../../../../architecture/module-boundaries.md).
- [`dependency rules`](../../../../architecture/dependency-rules.md).
- [`Runtime, Prepare, and Backend Boundary`](../../../../architecture/runtime-prepare-backend-boundary.md).
- [`ADR 0010`](../../../../design/decisions/0010-staged-backend-preparation.md).
- [`planning guide`](../../../planning-guide.md).
- [`Prepare master plan`](../master-plan.md).
- Complete [Prepare](../../../../glossary.md#prepare) tasks [0001](0001-backend-partition-analysis-and-resource-declaration.md),
  [0002](0002-backend-partition-finalization-handoff.md), and
  [0003](0003-prepare-orchestration-and-validation.md).
- [`CPU master plan`](../../../backends/cpu/master-plan.md) and completed CPU 0008B–0008E as
  downstream behavior authorities.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture constraints

- `modules/prepare` owns the backend-neutral projection and may depend only on its existing
  authorized inward modules. It must not depend on a concrete backend.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [modules glossary](../../../../glossary.md#term-module); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Concrete backends receive only exact partition-scoped semantic/planning facts and backend-owned
  inputs, never <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [planning glossary](../../../../glossary.md#term-planning))</span>`CompileArtifacts`, [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), or another [Compiler](../../../../glossary.md#term-compiler)-owned aggregate.
- Planning still selects ownership only. The projection must not derive cross-partition owner,
  route, kernel, fusion, memory, publication, transfer, or performance policy from topology.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [owner glossary](../../../../glossary.md#term-owner); [kernel glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion); [publication glossary](../../../../glossary.md#term-publication); [topology glossary](../../../../glossary.md#term-topology))</span>
- Backend analysis remains the owner of lowering, specialization, fusion, route choice, and exact
  shared-resource declarations. Finalization and Runtime contracts remain unchanged.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- The [Runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path) remains free of [`Operation`](../../../../glossary.md#operation), `CompiledNode`, graph inspection, selection,
  and scheduling decisions.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- The graph remains flat. This task introduces no region, callback, nested graph, control-flow
  representation, or execution DAG.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.prepare.analysis` — public [partition](../../../../glossary.md#partition)-scoped [backend](../../../../glossary.md#backend)-analysis facts.
- `io.github.pho001.synaptik.prepare` — shared complete-[graph](../../../../glossary.md#graph) projection orchestration.

Packages added or changed:

- No new package. The existing public `prepare.analysis` package gains one cohesive topology
  projection and only the narrowly required immutable nested or sibling fact types.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [immutable glossary](../../../../glossary.md#term-immutable))</span>

Type placement:

- `io.github.pho001.synaptik.prepare.analysis.PartitionDag` (preferred name) — Prepare-owned
  immutable structural projection for exactly one planned partition.
- Any public [producer](../../../../glossary.md#term-producer)/consumer/edge [occurrence](../../../../glossary.md#term-occurrence) type must be nested in or colocated with
  [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) and justified as part of the same narrow query contract. Do not add a generic
  `graph`, `util`, or backend-specific package.

Tests mirror the production packages. Black-box reflection tests remain in the existing [Prepare](../../../../glossary.md#prepare)
test packages.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- new `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PartitionDag.java`;
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java`;
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/package-info.java`;
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java`.

Expected focused [Prepare](../../../../glossary.md#prepare) test paths:

- new `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/analysis/PartitionDagTest.java`;
- existing `PrepareContextTest`, `AnalysisPublicShapeTest`, `GraphPreparationTest`, and
  `GraphPreparationPublicShapeTest` only as required by the final truthful [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape).

Expected downstream compatibility-test path:

- existing
  `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java`
  solely to move its malformed duplicate-producer rejection expectation from CPU lowering to<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [lowering glossary](../../../../glossary.md#lowering))</span>
  [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag)/[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) construction. No other assertion, fixture, CPU behavior, or CPU
  test owner is reopened.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [owner glossary](../../../../glossary.md#term-owner))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning) completion paths:

- `docs/api/public-api.md` and the focused Prepare/backend API or guide page that currently
  explains `PrepareContext`;
- `docs/architecture/runtime-prepare-backend-boundary.md` only for implemented-status
  clarification, not authority changes;
- `docs/glossary.md` only if the final public term requires a reusable definition;
- this task, the Prepare master plan, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap.

Review only unless a concrete accepted rule requires a change: `ARCHITECTURE.md`, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010,
[module](../../../../glossary.md#term-module)/[lifecycle](../../../../glossary.md#lifecycle)/[dependency](../../../../glossary.md#term-dependency) explanations, [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), `PlannedPartition`, [Runtime](../../../../glossary.md#term-runtime) and
[Compiler](../../../../glossary.md#term-compiler) APIs, CPU 0008B–0008E source/tests, build files, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance),
and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 20 paths:

- 4 [Prepare](../../../../glossary.md#prepare) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, including exactly 1 new top-level production type;
- 5 Prepare test paths, including exactly 1 new top-level test type;
- at most 3 explanatory [API](../../../../glossary.md#term-application-programming-interface-api)/architecture/guide paths;
- at most 1 glossary path;
- 4 [planning](../../../../glossary.md#term-planning) paths: this task, Prepare master plan, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap; and
- exactly 1 existing CPU test path, solely
  `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java`,
  with no CPU production change and no broader CPU behavior or test refactor; and
- at most 2 architecture-test paths only if an existing exhaustive package/public-surface
  inventory requires them.

No CPU production or other CPU test, [Model](../../../../glossary.md#term-model), Planning, [Compiler](../../../../glossary.md#term-compiler), [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine), Config, [Trace](../../../../glossary.md#trace),
other-[backend](../../../../glossary.md#backend), [Gradle](../../../../glossary.md#term-gradle), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), backend-[conformance](../../../../glossary.md#term-conformance), or integration path may change. The one authorized
CPU test edit may only make the existing malformed duplicate-[producer](../../../../glossary.md#term-producer) case assert rejection while
constructing [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) or [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), before CPU [lowering](../../../../glossary.md#lowering). If another production type,
package, [module](../../../../glossary.md#term-module) edge, behavior [owner](../../../../glossary.md#term-owner), CPU assertion, or CPU test path is required, stop and replan.

## Acceptance criteria

- One public immutable Prepare-owned partition DAG contains exactly the planned partition's
  stable topological nodes and no complete-model reference or out-of-partition node.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition); [topological glossary](../../../../glossary.md#term-topological-order); [model glossary](../../../../glossary.md#term-model))</span>
- Producer/output-port, consumer/input-port, edge, repeated-input, multi-output, external-input-
  occurrence, and local-sink facts are exact, immutable, deterministic, and queryable without
  backend reconstruction.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [backend glossary](../../../../glossary.md#backend))</span>
- Duplicate producers, duplicate node IDs, self/later-produced dependencies, and partition/order
  disagreement fail closed before backend analysis.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- The existing malformed duplicate-[producer](../../../../glossary.md#term-producer) case in `CpuNonAffineMovementLoweringTest` asserts
  this rejection at [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag)/[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) construction rather than expecting CPU
  lowering to own it; no other CPU expectation or behavior changes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering))</span>
- `PrepareContext` retains the [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) as the sole [node](../../../../glossary.md#node)/[topology](../../../../glossary.md#term-topology) state; `nodes()` delegates to it and
  no second list is stored or copied independently.
- The current list-taking construction form remains source-compatible if technically valid and
  delegates through exactly one DAG construction. The task records any unavoidable source-shape
  limitation truthfully.
- Reflection, record-component/class-property [shape](../../../../glossary.md#shape), constructors, equality, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests),
  `toString`, and compatibility behavior are pinned by automated tests and documented without a
  binary-compatibility promise.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) constructs each DAG once through the authoritative full-graph projection
  phase, retains all-contexts-before-analysis ordering, and passes no <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) or
  complete cross-backend node list to a backend.
- Values, logical memory requirements, constants, backend inputs, fully-static validation,
  staged analysis/finalization, slot assignment, schedule assembly, and prepared result behavior
  remain unchanged.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [validation glossary](../../../../glossary.md#term-validation))</span>
- The public query surface contains no generic mutation/rewrite/callback/region/scheduling API and
  no CPU, route, memory, publication, fusion, or performance policy.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [publication glossary](../../../../glossary.md#term-publication); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Focused Prepare tests, downstream CPU source/behavior checks, Javadoc, documentation validation,
  repository tests, applicable architecture tests, exact scope, and whitespace gates pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean documentation-focused pass finalizes all affected Javadocs and documentation,
  records glossary impact, and reuses stable implementation-test evidence appropriately.
- CPU production source and all CPU 0008B–0008E topology identities, stable order, legality,
  candidate counts, generated code, schema 53, performance evidence, and Runtime behavior remain
  unchanged. CPU test source changes only at the one explicitly authorized malformed-construction
  expectation.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- CPU 0008E1 remains Draft without a detailed specification; CPU 0008F remains Draft without a
  detailed specification; Prepare 0004 remains Draft and follows 0003A without renumbering.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation tests must cover:

- empty, one-node, linear, branching/diamond, disconnected, repeated-input, and multi-output
  partitions;<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- exact producer/output-port and consumer/input-port positions, external-input occurrence order,
  edge order, local sinks, value lookup, node lookup, and immutable snapshots;<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence); [lookup glossary](../../../../glossary.md#term-lookup); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- duplicate node IDs, duplicate producers within/across nodes, partition size/order/identity
  disagreement, self-dependency, later-producer dependency, nulls, and malformed ports;<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `PrepareContext.nodes()` exact [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) delegation and exact reference retention;
- current values/memory/[constants](../../../../glossary.md#term-constant)/[backend](../../../../glossary.md#backend)-input validation and dynamic-[Shape](../../../../glossary.md#shape) failure;
- [canonical](../../../../glossary.md#term-canonical-form) and compatibility constructor behavior, reflection/record components, equality,
  [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`;
- [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) constructs all partition DAGs before analysis, constructs each once, passes
  only partition-local nodes, and preserves analysis/finalization/schedule ordering.

[Run](../../../../glossary.md#term-run) focused tests while implementation stabilizes. After the last executable [Prepare](../../../../glossary.md#prepare) change run
one final affected-[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :modules:prepare:test
```

[Compile](../../../../glossary.md#compile) and test the downstream [CPU](../../../../glossary.md#term-cpu) consumer, proving that the compatibility surface remains
sufficient before CPU 0008E1. The sole permitted CPU edit is the expectation-only relocation in
`CpuNonAffineMovementLoweringTest` described above:

```bash
./gradlew :backends:cpu:test
```

The clean documentation-focused pass receives those exact results, changes no executable
behavior, and runs:

```bash
./gradlew :modules:prepare:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also inspects rendered Javadocs and validates all changed Markdown links/anchors, canonical
task headings, fences, final newlines, terminology, exact path [scope](../../../../glossary.md#term-scope), package placement, public
surface, no full-[graph](../../../../glossary.md#graph) exposure, status/[dependency](../../../../glossary.md#term-dependency) order, absence of CPU 0008E1/0008F detailed
specifications, and empty staging.

After implementation and documentation stabilize, run the shared-contract [checkpoint](../../../../glossary.md#term-checkpoint) once:

```bash
./gradlew test :testing:architecture-tests:test
```

Architecture-test source changes are conditional on an existing exhaustive inventory. Backend
[conformance](../../../../glossary.md#term-conformance) and integration source remain unchanged because this task adds no concrete capability
or end-to-end behavior; their existing suites participate through the repository checkpoint.

## Dependencies

- [Prepare](../../../../glossary.md#prepare) 0001–0003 — Complete.
- Planning 0004–0006 and Compiler 0005–0006 — Complete producers of the current immutable graph,
  partitions, memory, constants, and publication facts.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [constants glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication))</span>
- [Runtime](../../../../glossary.md#term-runtime) 0002–0014 — Complete consumers of unchanged prepared memory/schedule contracts.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0010 — Accepted/current.
- CPU 0008B–0008E — Complete downstream reconstruction and behavior evidence that motivates this
  user-authorized interleave.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

No unfinished architecture or shared-contract [dependency](../../../../glossary.md#term-dependency) blocks implementation.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0008E1, `Shared partition-DAG adoption and reconstruction removal`, is a separate Draft CPU
  task. It consumes the shared projection in `CpuPartitionDagDecomposer` and directly related
  cold analysis, removes duplicated producer/consumer scans where the shared facts are exact, and
  preserves every CPU 0008B–0008E behavior/evidence invariant. No detailed CPU 0008E1 task
  specification exists yet.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- CPU 0008F, portable MATMUL execution and bounded linear epilogues, remains Draft, depends on CPU
  0008E1, and has no detailed task specification.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Prepare 0004 remains the existing Draft opaque backend-candidate/tuning-artifact handoff. It
  follows 0003A without renumbering and is not implemented or specified here.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend); [artifact glossary](../../../../glossary.md#term-artifact); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## Architecture impact

Expected impact: None.

This task makes the existing [Prepare](../../../../glossary.md#prepare) projection more explicit and prevents complete-[model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph)
leakage to concrete [backends](../../../../glossary.md#backend). It stays within Prepare's current [ownership](../../../../glossary.md#term-ownership-meanings) of exact [partition](../../../../glossary.md#partition)-
scoped semantic/[planning](../../../../glossary.md#term-planning) facts and the existing [dependency direction](../../../../glossary.md#java-platform-and-build-tools). It changes no authoritative
architecture rule, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) decision, [module](../../../../glossary.md#term-module) edge, [Runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path), or [concrete backend](../../../../glossary.md#concrete-backend) policy.

If implementation requires changing `ARCHITECTURE.md`, exposing [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) to a backend,
moving backend-specific analysis into Prepare, adding a shared execution/scheduling graph, or
changing dependency direction, stop and report the exact uncertainty instead of implementing it.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik Prepare task 0003A. Do not stage, commit,
or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the focused
lifecycle/module-boundary/dependency/runtime-prepare-backend documents and ADR 0010,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the Prepare master plan and tasks
0001–0003, this task, the CPU master plan and completed CPU 0008B–0008E tasks, documentation
rules/profiles, and final affected source/tests/Javadocs.

Implement exactly this Ready task inside its authorized scope. Add one immutable public
partition-local DAG projection in Prepare, make it PrepareContext's sole topology source, retain a
truthful source-compatible list-taking constructor where technically valid, and make
GraphPreparation construct/pass the projection once. Preserve existing values, memory,
constants, backend inputs, static validation, staged lifecycle, and downstream CPU behavior. Do
not migrate CPU reconstruction, expose the complete model DAG, add policy or graph-rewrite APIs,
or implement Prepare 0004, CPU 0008E1, CPU 0008F, or later work. Stop on architecture,
compatibility, package, or maximum-scope conflict.

After stable implementation and recorded Prepare/CPU validation, hand the same diff and evidence
to a separate clean documentation-focused agent. That pass must independently finalize affected
Javadocs, API/guide/boundary documentation, glossary impact, and planning evidence, then run the
recorded documentation checks without repeating successful Java tests unless executable behavior
changes or a concrete stale-evidence risk is recorded. Run the repository/architecture checkpoint
once only after both passes stabilize. Update this task's decisions, limitations, evidence,
implementation notes, completion summary, and status only from actual results.
```

## Local decisions

- [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) is the preferred domain name because the value represents one exact planned
  partition, not the complete compiled graph or an execution schedule. Implementation may select
  a more precise name only with recorded source/API evidence and synchronized package Javadoc.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Structural [occurrence](../../../../glossary.md#term-occurrence) facts preserve positions rather than deduplicating by [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).
  Value-keyed queries may group immutable occurrence lists, but repeated consumer ports remain
  distinct.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- A local sink is topology-only: none of the node's output ports has a local consumer. It does not
  infer publication, partition boundary, memory, or backend ownership.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [node glossary](../../../../glossary.md#node); [publication glossary](../../../../glossary.md#term-publication); [backend ownership glossary](../../../../glossary.md#backend-ownership))</span>
- The authoritative [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) derives adjacency from ordered `CompiledNode` ports. No constructor takes
  caller-authored edge or adjacency collections.
- `PrepareContext.nodes()` remains for source compatibility but is a derived compatibility view.
  The final stored component/property must be the DAG, not both DAG and nodes.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>
- Record reflection/equality/textual compatibility changes are accepted only when they are the
  truthful consequence of one-source-of-truth state and are pinned explicitly. Binary
  compatibility is not promised.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- CPU adoption is deliberately separate so this Prepare capability can be validated independently
  while the unchanged list compatibility view keeps the existing CPU source compiling.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare))</span>
- The malformed duplicate-[producer](../../../../glossary.md#term-producer) [fixture](../../../../glossary.md#term-fixture) in `CpuNonAffineMovementLoweringTest` already crosses
  the new Prepare-owned fail-closed boundary. Its rejection expectation therefore belongs at<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
  `PartitionDag`/[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) construction, not at `CpuPartitionLowering.lower`; moving that
  one expectation is compatibility maintenance and does not authorize CPU production or policy
  work.
- No glossary change is assumed. The documentation pass must update it only if the final public
  term establishes a reusable project-wide distinction not already covered.
- Open questions: None. A need for global ownership facts, policy, execution scheduling, another
  public framework, or architecture changes is a stop-and-replan condition.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Known limitations

- The projection is partition-local and cannot answer cross-partition ownership, publication,
  transfer, memory, fusion, route, or performance questions from topology alone.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [fusion glossary](../../../../glossary.md#term-fusion); [topology glossary](../../../../glossary.md#term-topology))</span>
- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) remains fully static; this task adds no dynamic [binding](../../../../glossary.md#term-binding) or [run](../../../../glossary.md#term-run)-dynamic geometry.
- Existing [CPU](../../../../glossary.md#term-cpu) code may continue reconstructing equivalent facts through `nodes()` until CPU
  0008E1. This task intentionally does not remove that duplication.
- The one authorized CPU test correction changes only where one malformed duplicate-producer
  fixture expects failure. It does not broaden CPU adoption, lowering behavior, or test cleanup.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [fixture glossary](../../../../glossary.md#term-fixture); [lowering glossary](../../../../glossary.md#lowering))</span>
- The compatibility constructor, if retained, provides source-level convenience only. Record
  component, reflection, equality, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), `toString`, serialized form, and binary linkage are
  not promised to remain identical.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- No public generic graph framework, rewrite facility, execution scheduler, or full-model view is
  introduced.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [model glossary](../../../../glossary.md#term-model); [view glossary](../../../../glossary.md#term-view-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)/documentation correction context: `/root/prepare_0003a_doc_correction`.

- `./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest.rejectsExcludedWindowSignaturesTypesLayoutsAndOverflow`
  failed with one test failure at `CpuNonAffineMovementLoweringTest.java:177`. The failure occurs
  while constructing [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) from two [nodes](../../../../glossary.md#node) that produce the same [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), before the
  line-179 assertion can invoke CPU lowering. This is the demonstrated maximum-scope contradiction
  corrected by authorizing only that existing CPU test's expectation relocation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [scope glossary](../../../../glossary.md#term-scope))</span>
- The correction context inspected the complete current [Prepare](../../../../glossary.md#prepare) Java/test diff, including the new
  [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) and `PartitionDagTest`, and found no need to authorize CPU production behavior or
  any second CPU test path.
- The first local Ruby heading validator stopped before document evaluation because this Ruby
  lacks <span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>`Array#filter_map`; the compatible `map`/`compact` retry passed all 20 canonical headings,
  relative Markdown links, backtick fences, and the final newline.
- The current worktree scope validator passed with 12 authorized paths and no CPU production or
  second CPU test path. `git diff --check` and `git diff --cached --check` passed, and
  `git diff --cached --name-only` was empty.
- `git diff --no-index --check /dev/null <this-task>` produced no whitespace diagnostic; its exit
  status was the expected `1` because the task is an untracked non-empty file.

Planning/documentation context: `01a04334-7f08-78f3-9bf0-6efb1684b68e`.

- The context read the required architecture, lifecycle, module-boundary, dependency,
  Runtime/Prepare/backend, ADR 0010, planning, Prepare, CPU 0008B–0008E, documentation-profile,
  source, and focused-test material. No architecture-contract change or uncertainty was found.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- A local Ruby validator passed all four changed Markdown files. It confirmed that relative link
  targets exist, backtick fences are balanced, and every file has a final newline.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- [Canonical](../../../../glossary.md#term-canonical-form) heading inspection found all required task headings in order from `Status` through
  `Completion summary`. The first Ruby heading helper failed before evaluating the file because
  this environment lacks `Array#filter_map`; the compatible `map`/`compact` retry passed all 20
  headings. This was a validator compatibility failure, not a document failure.
- At that pre-implementation planning [checkpoint](../../../../glossary.md#term-checkpoint), status/[dependency](../../../../glossary.md#term-dependency) scans found exactly one
  [`Ready`](../../../../glossary.md#term-ready) master-plan row, Prepare 0003A. CPU 0008E1 and CPU 0008F were Draft; CPU 0008F depended
  on 0008E1; Prepare 0004 remained Draft after 0003A without renumbering.
- `find docs/planning/backends/cpu/tasks -maxdepth 1 -type f` found no CPU 0008E1 or CPU 0008F
  task specification.
- The combined tracked/untracked path [scan](../../../../glossary.md#term-cumulative-scan) found exactly the four authorized planning paths.
  `git status --short -uall` confirmed all four are unstaged and the staged index is empty.
- `git diff --check` and `git diff --cached --check` passed.
- No Java, test, Javadoc, Gradle, repository, performance, generated-code, or schema command ran
  because this pass changes planning Markdown only. No implementation evidence is claimed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [schema glossary](../../../../glossary.md#term-schema))</span>

Implementation context evidence reused by the final documentation pass:

- The final implementation-owned `./gradlew :modules:prepare:test` passed 41 tests with no
  failures, errors, or skips. No executable Java or test changed afterward in the documentation
  context.
- The final implementation-owned `./gradlew :backends:cpu:test` passed 104 suites and 544 tests
  with 3 expected skips and no failures or errors. The sole CPU edit is the authorized malformed
  duplicate-producer assertion relocation; CPU production is unchanged.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Focused implementation tests passed before those final module commands. The implementation
  reported exactly 13 authorized paths and an empty staged index at handoff.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>

Final clean documentation context: `01a043d7-113c-7ee2-8257-42678c1a7be4`.

- Reviewed the authoritative architecture contract, focused lifecycle/module/Prepare/backend
  explanations, public API status, backend preparer guide, glossary, General/API-Javadoc/
  Planning/Example profiles, final Java/Javadoc/test diff, task, Prepare and CPU master plans, and
  roadmap. No architecture, dependency, Gradle, Runtime, Compiler, Model, region, policy, or
  complete-model exposure conflict was found.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model))</span>
- Finalized `PartitionDag`, `PrepareContext`, [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), and package Javadocs plus the
  public API status, focused Runtime/Prepare/backend boundary, backend preparer guide, and
  glossary. The glossary change is warranted because `PartitionDag` is a reusable public project
  term whose topology-only boundary must be distinguished from policy.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- Documented the exact public [shape](../../../../glossary.md#shape): public final [immutable](../../../../glossary.md#term-immutable) `PartitionDag`; exact occurrence/port
  order including repeated inputs and multi-output nodes; external-input occurrences; local
  sinks; topology-only limitations; and the five-component canonical <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`PrepareContext` record plus
  its six-argument source-compatible constructor. No binary compatibility is promised for the
  changed record descriptor, reflection, equality, hash code, or text form.<span data-glossary-navigation="true"> ([hash code glossary](../../../../glossary.md#term-hash-code))</span><span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `./gradlew :modules:prepare:javadoc` first passed with two no-main-description warnings on
  `externalInputs()` and `localSinks()`. After correcting those Javadocs, the final identical
  command passed cleanly. Rendered HTML and `javap -public` inspection confirmed the intended
  type, constructor, method, nested-record, and documentation shape.
- The first local Ruby Markdown validator stopped on an interpolated heading-regex syntax error;
  the second stopped because this Ruby lacks `Array#filter_map`. The compatible
  `map`/`compact` validator then passed all eight changed Markdown files for relative link targets,
  anchors, backtick/tilde fences, and final newlines.
- The exact-scope validator passed with 17 authorized paths, within the maximum 20. It found
  exactly the one authorized CPU test path, no CPU production, no second CPU test, and no
  forbidden path. CPU 0008E1 and CPU 0008F remain Draft and have no detailed task specification;
  Prepare 0004 remains Draft after 0003A.
- `git diff --check` and `git diff --cached --check` passed. `git diff --cached --name-only` was
  empty, and final status inspection confirmed all 17 paths remain unstaged.
- The task-required checkpoint ran exactly once as
  `./gradlew test :testing:architecture-tests:test` and passed. Current XML reports total 364
  suites and 2,491 tests, with 3 expected skips and no failures or errors; architecture tests
  contributed 4 suites and 6 tests. The checkpoint's Prepare report contains 11 suites and 42
  tests with no skips/failures/errors, and its CPU report contains 104 suites and 544 tests with
  the same 3 expected skips and no failures/errors.

## Implementation notes

- Added one [Prepare](../../../../glossary.md#prepare)-owned [immutable](../../../../glossary.md#term-immutable) [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) that validates exact planned membership and
  topological order, derives deterministic occurrence and adjacency facts once, and exposes no
  complete graph, region, policy, or execution model.<span data-glossary-navigation="true"> ([topological order glossary](../../../../glossary.md#term-topological-order); [occurrence glossary](../../../../glossary.md#term-occurrence); [graph glossary](../../../../glossary.md#graph); [model glossary](../../../../glossary.md#term-model))</span>
- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) now stores the DAG as its only topology component. Its canonical record has five
  components; the six-argument partition-and-node-list constructor constructs exactly one DAG and
  preserves source calls without claiming binary compatibility.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [topology glossary](../../../../glossary.md#term-topology); [canonical glossary](../../../../glossary.md#term-canonical-form); [argument glossary](../../../../glossary.md#term-argument); [partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) constructs every partition DAG once while projecting the complete compile
  graph and completes all contexts before invoking a backend preparer.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend))</span>
- The existing CPU malformed duplicate-producer fixture now expects rejection at the shared
  Prepare construction boundary. No CPU production behavior, reconstruction, generated code,
  schema, fusion policy, or Runtime behavior changed.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [producer glossary](../../../../glossary.md#term-producer); [fixture glossary](../../../../glossary.md#term-fixture); [schema glossary](../../../../glossary.md#term-schema); [fusion glossary](../../../../glossary.md#term-fusion); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- CPU adoption and reconstruction deletion remain Draft CPU 0008E1. CPU 0008F remains Draft after
  it, and neither task has a detailed specification.

## Completion summary

- Completed changes: Added the [immutable](../../../../glossary.md#term-immutable) [partition](../../../../glossary.md#partition)-local [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) projection, made it
  [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext)'s sole topology source, integrated one-time graph preparation, and relocated
  the one stale CPU malformed-input expectation to the shared validation boundary.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [CPU glossary](../../../../glossary.md#term-cpu); [validation glossary](../../../../glossary.md#term-validation))</span>
- Files changed or created: 17 authorized paths: 4 Prepare production/Javadoc, 4 Prepare tests,
  4 explanatory documentation/glossary, 4 planning, and exactly 1 CPU test.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Reused the stable 41-test Prepare and 544-test CPU implementation runs;
  final Prepare Javadoc, rendered/public-shape inspection, Markdown, exact-scope, staging, and
  whitespace gates passed; the one required repository/architecture checkpoint passed 2,491
  tests with 3 expected skips and no failures/errors.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [scope glossary](../../../../glossary.md#term-scope); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Documentation-agent review: Clean documentation context
  `01a043d7-113c-7ee2-8257-42678c1a7be4` independently finalized the affected Javadocs,
  explanatory documentation, glossary, planning status, and evidence without changing executable
  Java or tests.
- Documentation impact: Updated public API status, the focused Runtime/Prepare/backend boundary,
  backend contributor guidance, glossary, and synchronized task/master/roadmap state.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Final [Prepare](../../../../glossary.md#prepare) Javadoc passed cleanly; rendered HTML and `javap -public` confirmed
  the intended immutable DAG and five-component/six-argument context surface.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Glossary impact: Added [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) and updated `PrepareContext` because the new public term and
  topology-versus-policy distinction are reusable across backend contributors.
- Unresolved issues: None.
- Follow-up required: None for Prepare 0003A. CPU 0008E1 and CPU 0008F remain separate Draft work
  without detailed specifications.

Status: Complete
