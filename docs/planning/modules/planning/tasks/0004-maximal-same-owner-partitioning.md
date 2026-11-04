# Task 0004: Maximal Same-[Owner](../../../../glossary.md#term-owner) Partitioning

## Status

Complete

## Goal

Add the smallest [backend](../../../../glossary.md#backend)-neutral partitioning capability that turns one complete backend-[owner](../../../../glossary.md#term-owner)
assignment for the [nodes](../../../../glossary.md#node) of an [immutable](../../../../glossary.md#term-immutable) [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) into maximal consecutive same-owner
partitions.

[Mental model](../../../../glossary.md#term-mental-model):

```text
CompiledGraphModel.nodes() in validated topological order
  + exactly one selected BackendId for every NodeId
  -> validate complete node-to-owner coverage
  -> split only when the next node has a different BackendId by equality
  -> immutable ordered PlannedPartition values
```

The result records [backend ownership](../../../../glossary.md#backend-ownership) and ordered [graph](../../../../glossary.md#graph)-node [identity](../../../../glossary.md#term-identity-meanings) only. It is an immutable
[compile](../../../../glossary.md#compile)-time recipe, not [lowering](../../../../glossary.md#lowering), an executable, a physical schedule, a transfer plan, or a
memory allocation.

## Motivation

[Planning](../../../../glossary.md#term-planning) task 0003 can select one exact [`BackendId`](../../../../glossary.md#backend-identity--backendid) [owner](../../../../glossary.md#term-owner) for one [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) after hard
eligibility. The architecture places maximal same-owner partitioning immediately after those
per-occurrence [ownership](../../../../glossary.md#term-ownership-meanings) decisions. The current [model](../../../../glossary.md#term-model) already supplies a structurally closed
[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) whose `nodes()` list is validated [topological order](../../../../glossary.md#term-topological-order), so no [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings),
[topological](../../../../glossary.md#intermediate-representation-dag-and-ssa) sort, [producer](../../../../glossary.md#term-producer) index, or new [graph](../../../../glossary.md#graph) representation is needed for this step.

This task deliberately uses a complete `Map<NodeId, BackendId>` as the [handoff](../../../../glossary.md#term-handoff) from repeated
one-occurrence selection. A later planning/compiler orchestration consumer will build that map;
this task neither invokes nor widens the package-private capability selector. The map avoids a
speculative ownership-row type while making association independent of parallel list position.

## [Scope](../../../../glossary.md#term-scope)

- Add public record `PlannedPartition` in
  `io.github.pho001.synaptik.planning.partition` with exactly these ordered components:

  ```java
  BackendId owner
  List<NodeId> nodeIds
  ```

- Make `PlannedPartition` the immutable backend-neutral partition DTO named by the architecture.
  It is public because later planning memory work, compiler compile artifacts, prepare contracts,
  and concrete backend preparation must be able to consume the recipe without depending on a
  planning implementation class.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [DTO glossary](../../../../glossary.md#data-transfer-object--dto); [planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Give its public [canonical](../../../../glossary.md#term-canonical-form) constructor this exact [validation](../../../../glossary.md#term-validation) and [snapshot](../../../../glossary.md#term-snapshot) behavior:
  1. null [`owner`](../../../../glossary.md#term-owner) -> `NullPointerException("owner")`;
  2. null `nodeIds` -> `NullPointerException("nodeIds")`;
  3. empty `nodeIds` -> `IllegalArgumentException("nodeIds must not be empty")`;
  4. [scan](../../../../glossary.md#term-cumulative-scan) [node](../../../../glossary.md#node) IDs in encounter order and reject the first null at index `i` with
     `NullPointerException("nodeIds[i]")`;
  5. reject the first later equal duplicate at index `i` with
     `IllegalArgumentException("nodeIds[i] duplicates <NodeId diagnostic text>")`; and
  6. [snapshot membership](../../../../glossary.md#occurrence-identity-and-canonical-form) with `List.copyOf`, retaining the exact [`NodeId`](../../../../glossary.md#nodeid) [element](../../../../glossary.md#term-element) references.
- Explicitly declare and document both public record-component accessors. Preserve ordinary
  record equality, hashing, and diagnostic `toString()` behavior.
- Add package-private final stateless class `MaximalSameOwnerPartitioning` in the same package.
  Give it exactly one private no-argument constructor, no fields, no implemented interfaces, and
  exactly one package-private static method:<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>

  ```java
  static List<PlannedPartition> partition(
          CompiledGraphModel graph,
          Map<NodeId, BackendId> ownershipByNodeId)
  ```

- Treat `graph.nodes()` as the sole ordering and adjacency source. Two nodes are adjacent for this
  task only when they occupy consecutive positions in that validated topological list. Graph-edge
  reachability, numeric <span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#term-topological-order))</span>`NodeId` order, map iteration order, phase, operation family, and value
  list order do not redefine adjacency.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Treat `ownershipByNodeId` as one complete owner assignment produced after per-occurrence owner
  selection. Associate keys with graph nodes by <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`NodeId.equals`, never by object identity, numeric
  list position, or map iteration order.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>
- Validate the partitioning inputs in this exact order before creating a [partition](../../../../glossary.md#partition):
  1. null [`graph`](../../../../glossary.md#graph) -> `NullPointerException("graph")`;
  2. null `ownershipByNodeId` -> `NullPointerException("ownershipByNodeId")`;
  3. a null map key -> `NullPointerException("ownershipByNodeId contains null key")`;
  4. inspect non-null map keys in ascending numeric `NodeId.value()` order and reject the first
     key not present in `graph.nodes()` with
     `IllegalArgumentException("ownershipByNodeId contains unknown <NodeId diagnostic text>")`;
  5. walk graph nodes in stored topological order and reject the first node with no equal key
     using<span data-glossary-navigation="true"> ([topological order glossary](../../../../glossary.md#term-topological-order))</span>
     `IllegalArgumentException("ownershipByNodeId missing <NodeId diagnostic text>")`; and
  6. for each covered graph node in that same order, reject a null owner with
     `NullPointerException("ownershipByNodeId[<NodeId diagnostic text>]")`.
- Validate all keys, coverage, and owners before constructing the first output partition. Do not
  expose a partial result or retain the supplied map.
- For a valid non-empty graph, scan nodes once in stored topological order. Start a new partition
  at the first node and whenever the next node's owner is not equal to the current partition owner
  under `BackendId.equals`. Otherwise append the node to the current partition.
- Make every result maximal under that exact consecutive-order rule: no two consecutive result
  partitions have equal owners, and joining either neighboring partition would mix unequal
  owners.
- Store in each partition the exact owner reference obtained for its first node. Later nodes with
  equal but non-identical owner references join that partition without replacing the retained
  first-node owner reference.
- Store the exact `NodeId` references from `graph.nodes()`, not equal key references from the
  ownership map. Preserve graph order inside each partition and partition order by each
  partition's first graph node.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Return a non-null [immutable](../../../../glossary.md#term-immutable) list. Concatenating all `nodeIds()` lists must reproduce the exact
  graph-node identity references in stored order, with every graph node covered exactly once.
- Accept a valid zero-node pass-through graph only with an empty owner map and return an immutable
  empty partition list. Graph inputs and outputs are values, not synthetic nodes or partitions.
- Treat one `CompiledNode` as indivisible. A multi-output producer belongs to exactly one
  partition because ownership attaches to its node occurrence; all its output <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values
  therefore originate at that owner. Fan-out, repeated inputs, merges, graph-output publication,
  or a value crossing into a differently owned consumer does not split the producer node.<span data-glossary-navigation="true"> ([Fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication))</span>
- Do not store or derive partition input/output value lists, producer/use indexes, boundary edges,
  transfers, or materialization requirements. The closed graph plus partition node IDs retains
  enough logical structure for task 0005 to derive those facts without duplicating them here.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Do not split on `GraphPhase`. Forward/backward classification remains graph metadata; this
  ownership-only task has no architecture basis for inventing a phase partition boundary.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Add package [Javadoc](../../../../glossary.md#term-javadoc) explaining the current public partition recipe and internal generation step.
- Add focused tests for the exact DTO and generator shapes, validation order/messages, equality
  association, exact-reference retention, immutable snapshots, zero-node graphs, maximal runs,
  owner transitions, nonconsecutive equal owners, independent adjacent nodes, graph phase
  changes, graph inputs/outputs, fan-out, merges, and multi-output producers.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [graph phase glossary](../../../../glossary.md#graph-phase))</span>
- Finalize Javadocs and affected explanatory/planning documentation through the required separate
  clean-context documentation pass in the same overall implementation change.

## Out of [scope](../../../../glossary.md#term-scope)

- invoking, moving, [widening](../../../../glossary.md#term-widening), or changing `BackendEligibility`, `BackendOwnerSelection`,
  [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery), [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider), or their packages and behavior
- a public partitioning service, planner facade, registry, manager, callback, policy hierarchy,
  plugin, service locator, or compiler entry point<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [callback glossary](../../../../glossary.md#term-callback); [plugin glossary](../../../../glossary.md#term-plugin); [service locator glossary](../../../../glossary.md#term-service-locator); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- a new ownership row, node-owner record, owner list aligned by position, candidate, score,
  partition ID, trace ID, diagnostics result, or failure taxonomy<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [node glossary](../../../../glossary.md#node); [owner glossary](../../../../glossary.md#term-owner); [partition glossary](../../../../glossary.md#partition); [trace glossary](../../../../glossary.md#trace))</span>
- capability evaluation, hard-requirement evaluation, availability evaluation, owner scoring,
  cost/workload classification, transfer/boundary penalties, tuning, or candidate search<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- graph capture, topological sorting, producer/use indexing, graph transformation, validation
  beyond consuming the already closed graph, compiler orchestration, <span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [topological glossary](../../../../glossary.md#term-topological-order); [producer glossary](../../../../glossary.md#term-producer); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation))</span>`CompileArtifacts`, or
  publication planning<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning))</span>
- graph-edge-connected-component partitioning, reordering nodes, joining nonconsecutive nodes,
  splitting by phase, splitting a multi-output node, fusion, specialization, or decomposition<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [specialization glossary](../../../../glossary.md#term-specialization); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- partition boundary input/output lists, boundary-edge records, transfer records, logical
  materialization or memory requirements, liveness, slots, allocation, layout decisions, or
  physical buffers<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [liveness glossary](../../../../glossary.md#term-liveness); [layout glossary](../../../../glossary.md#layout); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- selected devices, routes, kernels, OpenBLAS, Vector API, MPSGraph, CUDA, lowering, backend DAGs,
  prepare behavior, prepared partitions, executables, schedules, runtime residency, or execution<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Vector API glossary](../../../../glossary.md#term-vector-api); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [CUDA glossary](../../../../glossary.md#term-cuda); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>
- trace events or payload schemas, architecture-contract or ADR changes, architecture-test
  changes, dependency or Gradle changes, backend-conformance or integration behavior, root-build
  changes, unrelated refactoring, or another detailed future task specification<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [schemas glossary](../../../../glossary.md#term-schema); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0004](../../../../design/decisions/0004-partition-scoring.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Planning master plan](../master-plan.md)
- [Config master plan](../../config/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Compiler master plan](../../compiler/master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Planning task 0001](0001-operation-capability-query-foundation.md)
- [Planning task 0002](0002-per-query-backend-hard-eligibility.md)
- [Planning task 0003](0003-ownership-candidates-and-baseline-scoring.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral node ownership and maximal same-owner partitioning. The output may
  contain only compile-time graph identity and <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [node glossary](../../../../glossary.md#node); [owner glossary](../../../../glossary.md#term-owner); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid) [ownership](../../../../glossary.md#term-ownership-meanings).
- `CompiledGraphModel.nodes()` is already immutable validated topological order. Planning consumes
  that ordering and must not capture, reorder, transform, or repair the graph.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [topological order glossary](../../../../glossary.md#term-topological-order); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [Compile](../../../../glossary.md#compile)-time partitions hold `BackendId`, not providers, concrete backend objects, preparers,
  devices, executables, routes, kernels, storage, or mutable services.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [kernels glossary](../../../../glossary.md#kernel); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- One node occurrence is the ownership unit supplied by current selection. Multi-output values do
  not create separate ownership units or allow one producer to be split across backends.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [producer glossary](../../../../glossary.md#term-producer))</span>
- Maximal adjacent same-owner work forms partitions after ownership. For this smallest current
  implementation, adjacency is consecutive occurrence order in the graph's validated topological
  node list; only an owner transition creates a partition boundary.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#intermediate-representation-dag-and-ssa))</span><span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- A planned partition remains an immutable recipe. Backend prepare later chooses fusion,
  specialization, routes, kernels, executable units, and physical storage for its assigned region.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [fusion glossary](../../../../glossary.md#term-fusion); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Logical boundary, materialization, and memory facts remain task 0005 work. Omitting duplicated
  boundary lists here does not discard information because the immutable graph and ordered node
  IDs retain producer/consumer structure.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Planning remains independent of runtime, prepare, engine, and concrete backends. Existing public
  model/backend-contract dependencies already cover the record's public <span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model))</span>[`NodeId`](../../../../glossary.md#nodeid) and `BackendId`
  signature, so no dependency change is needed.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Stop if implementation requires a public orchestration facade, compiler behavior, graph
  mutation, selected device, boundary/materialization model, cost/workload classification,
  partition diagnostics schema, dependency edit, or architecture decision.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [compiler glossary](../../../../glossary.md#term-compiler); [device glossary](../../../../glossary.md#term-device); [schema glossary](../../../../glossary.md#term-schema))</span>

## Current contract inventory and [handoff](../../../../glossary.md#term-handoff)

| Contract | Current role in this task | Deliberate boundary |
|---|---|---|
| `BackendOwnerSelection.select(...)` | Produces one exact [`BackendId`](../../../../glossary.md#backend-identity--backendid) for one already eligible [occurrence](../../../../glossary.md#term-occurrence) | Remains package-private and is not called or widened here |
| `CompiledGraphModel.nodes()` | Supplies the complete validated [topological](../../../../glossary.md#term-topological-order) occurrence order | No [capture](../../../../glossary.md#term-capture-meanings), reordering, [producer](../../../../glossary.md#term-producer) index, or [compiler](../../../../glossary.md#term-compiler) orchestration |
| `CompiledNode.id()` | Supplies the [graph](../../../../glossary.md#graph)-local [ownership](../../../../glossary.md#term-ownership-meanings) and [partition](../../../../glossary.md#partition) membership key | [Operation](../../../../glossary.md#operation) and output count do not create extra ownership units |
| `Map<NodeId, BackendId>` | Exact internal handoff for one complete set of per-[node](../../../../glossary.md#node) [owner](../../../../glossary.md#term-owner) results | Not retained, exposed as a new public ownership [model](../../../../glossary.md#term-model), or interpreted by map order |
| `PlannedPartition` | New public [immutable](../../../../glossary.md#term-immutable) owner plus ordered node-ID recipe | No executable, route, boundary-value, transfer, memory, or [runtime](../../../../glossary.md#term-runtime) state |

The current selector and the new generator are intentionally separate package-private operations.
There is still no public workflow connecting provider [evaluation](../../../../glossary.md#term-evaluation), owner selection, and graph-wide
partitioning. [Planning](../../../../glossary.md#term-planning) task 0006 or the first concrete compiler-planning consumer must select that
narrow orchestration surface; this task must not preempt it.

## Package impact

Package added:

- `io.github.pho001.synaptik.planning.partition` — public immutable planned-partition recipe and
  package-private deterministic maximal same-owner generation<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition); [owner glossary](../../../../glossary.md#term-owner))</span>

Type placement:

- `io.github.pho001.synaptik.planning.partition.PlannedPartition` — the architecture-named public
  compile-time DTO shared by later planning, compiler, prepare, and backend consumers<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [DTO glossary](../../../../glossary.md#data-transfer-object--dto); [planning glossary](../../../../glossary.md#term-planning))</span>
- `io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioning` — package-private
  stateless generation operation colocated with the result it creates<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

No root [facade](../../../../glossary.md#term-facade), [ownership](../../../../glossary.md#term-ownership-meanings) package, memory package, [compiler](../../../../glossary.md#term-compiler) package, diagnostics package, generic
`util`, [registry](../../../../glossary.md#term-registry), service, route, [prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend)-specific package is added.

## Proposed production surface

```java
package io.github.pho001.synaptik.planning.partition;

public record PlannedPartition(BackendId owner, List<NodeId> nodeIds) {
    public PlannedPartition {
        // Exact validation and snapshot behavior are specified in Scope.
    }

    @Override
    public BackendId owner() { /* exact retained reference */ }

    @Override
    public List<NodeId> nodeIds() { /* immutable membership snapshot */ }
}

final class MaximalSameOwnerPartitioning {
    private MaximalSameOwnerPartitioning() {}

    static List<PlannedPartition> partition(
            CompiledGraphModel graph,
            Map<NodeId, BackendId> ownershipByNodeId) {
        // Exact behavior is specified in Scope; implementation strategy is not a contract.
    }
}
```

The [shape](../../../../glossary.md#shape) is exact for names, packages, visibility, components, generic [arguments](../../../../glossary.md#term-argument), constructor and
method counts, and absence of state. Method bodies above are conceptual and do not authorize
additional conveniences or overloads.

## [Partition](../../../../glossary.md#partition) semantics

Given [graph](../../../../glossary.md#graph) [node](../../../../glossary.md#node) order `[n0, n1, n2, n3, n4]` and [owners](../../../../glossary.md#term-owner)
`[cpuA, cpuB, metal, metal, cpuC]`, where all three [CPU](../../../../glossary.md#term-cpu) values are equal [`BackendId`](../../../../glossary.md#backend-identity--backendid) values but may
be different references, the result is:

```text
partition 0: owner = exact cpuA reference, nodeIds = [n0, n1]
partition 1: owner = exact metal reference for n2, nodeIds = [n2, n3]
partition 2: owner = exact cpuC reference, nodeIds = [n4]
```

The two CPU regions do not join because they are not consecutive after [ownership](../../../../glossary.md#term-ownership-meanings) assignment.
Conversely, two consecutive independent nodes with equal owners do join: adjacency in this task is
the stored [topological](../../../../glossary.md#term-topological-order) sequence, not the existence of a [producer](../../../../glossary.md#term-producer)-consumer edge.

## [Graph](../../../../glossary.md#graph) boundaries and [producer](../../../../glossary.md#term-producer) rules

- Graph inputs have no producing node and never become partitions. A zero-node pass-through graph
  produces no partition.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition))</span>
- Graph outputs are [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) boundaries and do not split an otherwise consecutive same-owner
  node run.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [run glossary](../../../../glossary.md#term-run))</span>
- A node with multiple outputs remains one partition member. Each output is produced by that
  node's owner, whether it is a graph output, consumed inside the partition, or consumed by one or
  more later partitions.
- A fan-out or merge does not change the node-order scan. Consumers join a producer's partition
  only when they are consecutive in graph order and have an equal selected owner.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- A value consumed across an owner transition identifies future logical boundary/materialization
  work. This task neither creates a physical transfer nor records a boundary DTO.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [DTO glossary](../../../../glossary.md#data-transfer-object--dto))</span>
- Repeated input positions, unused graph inputs, and forward/backward phase changes do not add
  synthetic nodes or force a partition boundary.

## [Validation](../../../../glossary.md#term-validation) and failure contract

| Stage | Condition | Failure |
|---|---|---|
| 1 | null [graph](../../../../glossary.md#graph) | `NullPointerException("graph")` |
| 2 | null [ownership](../../../../glossary.md#term-ownership-meanings) map | `NullPointerException("ownershipByNodeId")` |
| 3 | null map key | `NullPointerException("ownershipByNodeId contains null key")` |
| 4 | first numeric-order key not naming a graph [node](../../../../glossary.md#node) | `IllegalArgumentException("ownershipByNodeId contains unknown <NodeId>")` |
| 5 | first graph-order node without an equal map key | `IllegalArgumentException("ownershipByNodeId missing <NodeId>")` |
| 6 | first graph-order node with a null [owner](../../../../glossary.md#term-owner) | `NullPointerException("ownershipByNodeId[<NodeId>]")` |

All validation succeeds before [partition](../../../../glossary.md#partition) construction begins. A map key equal to a graph node ID
is valid even when it is a different object reference. The output always retains the graph's node
ID reference and the map's first-node owner reference.

## Affected files

The intended implementation [scope](../../../../glossary.md#term-scope) is exactly fifteen paths.

Production — exactly three paths:

- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/PlannedPartition.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/MaximalSameOwnerPartitioning.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/package-info.java`

Tests — exactly two paths:

- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/partition/PlannedPartitionTest.java`
- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/partition/MaximalSameOwnerPartitioningTest.java`

Architecture-status and explanatory documentation — exactly five paths:

- current-versus-planned wording only in `docs/architecture/partition-scoring.md`
- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/user-guide/backend-selection.md`
- `docs/glossary.md`

[Planning](../../../../glossary.md#term-planning) — exactly five paths:

- add and finalize this task
- `docs/planning/modules/planning/master-plan.md`
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `ARCHITECTURE.md`; ADRs; all completed planning source, tests, and
task history; `modules/planning/build.gradle.kts`; architecture tests; [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph) source/tests;
config/[backend](../../../../glossary.md#backend)-contract/[trace](../../../../glossary.md#trace)/[compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[engine](../../../../glossary.md#term-engine) Java and builds; concrete backends;
backend-[conformance](../../../../glossary.md#term-conformance) and integration tests; every other planning master plan and future task row.

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly the fifteen paths listed above. Stop if implementation requires another production or
test type, another document, a public orchestration [API](../../../../glossary.md#term-application-programming-interface-api), an [ownership](../../../../glossary.md#term-ownership-meanings) row, [graph](../../../../glossary.md#graph)-boundary [DTO](../../../../glossary.md#data-transfer-object--dto),
cost/workload classification, diagnostics [schema](../../../../glossary.md#term-schema), [dependency](../../../../glossary.md#term-dependency)/build change, [architecture test](../../../../glossary.md#term-architecture-test),
architecture change, [compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior, or detailed follow-up
specification. Do not hide an incomplete acceptance criterion in a later task.

## Acceptance criteria

- `PlannedPartition` is the exact two-component public record specified above, with a public
  canonical constructor, explicitly documented accessors, ordinary record object methods, and no
  other field, constructor, method, interface, factory, builder, or nested type.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Its constructor enforces the exact owner/list/empty/element/duplicate validation order and
  messages, snapshots list membership, and retains exact owner and node-ID references.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [element glossary](../../../../glossary.md#term-element); [node glossary](../../../../glossary.md#node))</span>
- `MaximalSameOwnerPartitioning` has the exact package-private final stateless [shape](../../../../glossary.md#shape) and sole
  `partition(CompiledGraphModel, Map<NodeId, BackendId>)` method specified above.
- The generator enforces the exact top-level, null-key, numeric unknown-key, graph-order missing,
  and graph-order null-owner validation order and messages before creating any result.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Equal non-identical [`NodeId`](../../../../glossary.md#nodeid) map keys associate correctly. Output node IDs are the exact graph
  references; partition owner is the exact map value for the partition's first node.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- Every valid non-empty graph node occurs exactly once. Partition and node order reproduce the
  stored graph order, every partition is non-empty, and the returned outer and inner lists are
  immutable.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Consecutive equal owners form one maximal run. An owner transition splits; nonconsecutive equal
  owners remain separate; no consecutive result partitions have equal owners.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- A zero-node graph plus empty map returns an immutable empty list. Extra owner keys are rejected,
  including for a zero-node graph.
- Independent consecutive nodes, graph inputs/outputs, fan-out, merges, repeated inputs,
  forward/backward phase changes, and a multi-output node behave exactly as specified without
  edge-connected grouping, phase splitting, or node splitting.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Production code introduces no operation/kind inspection, provider call, availability/config
  interpretation, cost/score, device, route, kernel, boundary value/edge, transfer,
  materialization, allocation, trace, prepare, runtime, or execution state.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel); [materialization glossary](../../../../glossary.md#materialization); [trace glossary](../../../../glossary.md#trace); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Existing capability classes, model graph contracts, dependencies, Gradle files, architecture
  rules, ADRs, and completed-task history remain unchanged.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- The public and compile API pages, backend-selection guide, glossary, partition-scoring status,
  task, planning master plan, trace deferral status, and roadmap accurately distinguish the
  current recipe/generator from planned orchestration, boundary/memory work, prepare, and
  execution.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>
- A separate clean-context documentation-focused agent pass finalizes affected Javadocs,
  explanatory documentation, examples, terminology, links, glossary impact, planning evidence,
  and status in the same overall change.
- Focused tests, one final planning module suite, final planning Javadoc, repository Markdown,
  generated-page, exact-scope, status, later-spec, dependency, forbidden-surface, final-newline,
  trailing-whitespace, and <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check` [validation](../../../../glossary.md#term-validation) all pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation while executable Java changes:

```bash
./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.partition.PlannedPartitionTest --tests io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioningTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final affected-[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:planning:test
```

Record XML suite/test/failure/error/skip counts. Then hand the task, actual diff, and exact Java
evidence to the separate clean-context documentation pass. That pass reuses the successful Java
evidence unless it changes executable Java behavior or records a concrete cross-check risk.

After final [Javadoc](../../../../glossary.md#term-javadoc) and Markdown edits, the documentation pass runs:

```bash
./gradlew :modules:planning:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect generated `package-summary.html` and `PlannedPartition.html`. Confirm that the public page
contains only the [owner](../../../../glossary.md#term-owner)/[node](../../../../glossary.md#node)-ID recipe and that no generated public page or all-classes entry is
created for the package-private generator.

Manual final checks are limited to risks not better locked by tests:

- exact fifteen-path tracked/untracked [scope](../../../../glossary.md#term-scope);
- unchanged [planning](../../../../glossary.md#term-planning) [Gradle](../../../../glossary.md#term-gradle) dependencies and no forbidden import or production surface;
- exactly one [Ready](../../../../glossary.md#term-ready) task during implementation, then synchronized Complete/Draft status;
- no detailed Planning 0005, Planning 0006, Config 0004, [Trace](../../../../glossary.md#trace) 0003, or [Compiler](../../../../glossary.md#term-compiler) task spec;
- no partition ID, boundary-value/edge, transfer, materialization, cost/workload/profile,
  diagnostics, route, kernel, prepare, runtime, or execution type;<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [materialization glossary](../../../../glossary.md#materialization); [kernel glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- generated public/internal Javadoc boundary, links/anchors, fences, final newlines, and whitespace.

No [architecture test](../../../../glossary.md#term-architecture-test) or repository-wide suite is run by habit because this task adds one public
[DTO](../../../../glossary.md#data-transfer-object--dto) over already-public [dependency](../../../../glossary.md#term-dependency) types and one internal planning [operation](../../../../glossary.md#operation) without changing a
module edge, architecture rule, build, [concrete backend](../../../../glossary.md#concrete-backend), or cross-module executable behavior.
[Continuous integration](../../../../glossary.md#term-continuous-integration-ci) or Planning 0006 owns the repository tier unless implementation reveals a
concrete trigger; such a trigger requires stopping before scope expansion.

## Documentation [handoff](../../../../glossary.md#term-handoff)

After the implementation context records the final [planning](../../../../glossary.md#term-planning)-[module](../../../../glossary.md#term-module) result, hand this
specification, the actual fifteen-path diff, and exact focused/final Java evidence to a separate
clean-context documentation-focused agent or thread.

The handoff must identify:

- the exact public `PlannedPartition(owner, nodeIds)` [shape](../../../../glossary.md#shape) and [validation](../../../../glossary.md#term-validation);
- the package-private generator and [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) plus complete [owner](../../../../glossary.md#term-owner)-map input seam;
- [topological](../../../../glossary.md#term-topological-order) consecutive-order adjacency and owner-equality maximality;
- exact [graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node) and first-owner reference retention;
- zero-node, graph-boundary, phase, [fan-out](../../../../glossary.md#term-fan-out), merge, and multi-output rules;
- the absence of boundary values, transfers, memory, lowering, executable, and public
  orchestration behavior; and<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- the five explanatory documents, five planning paths, and validation commands above.

The documentation pass must read the [architecture contract](../../../../glossary.md#architecture-contract), documentation rules, General,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Architecture, User guide, Planning, and Example profiles, final source and tests,
current [model](../../../../glossary.md#term-model) graph and planning contracts, ADRs 0002 and 0004, and all directly affected pages.
It independently finalizes type/constructor/component/method/package Javadocs, current-versus-
planned wording, examples, links, terminology, glossary impact, evidence, and completion status.

It must record reasoned no-change conclusions for capability/config/[backend](../../../../glossary.md#backend)-contract/[trace](../../../../glossary.md#trace) Java,
planning [Gradle](../../../../glossary.md#term-gradle)/dependencies, architecture contract and ADRs, architecture tests, [compiler](../../../../glossary.md#term-compiler)
implementation, [prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[engine](../../../../glossary.md#term-engine), concrete backends, backend-[conformance](../../../../glossary.md#term-conformance) and integration
tests, and every other module. It does not rerun successful Java tests unless executable behavior
changes.

## Dependencies

- Complete [Planning](../../../../glossary.md#term-planning) task 0003 with deterministic per-[occurrence](../../../../glossary.md#term-occurrence) [`BackendId`](../../../../glossary.md#backend-identity--backendid) [owner](../../../../glossary.md#term-owner) selection.
- Complete Planning tasks 0001–0002 and Config task 0003 as the upstream capability, hard-
  eligibility, and optional baseline preference contracts.
- Complete [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph) foundation with [immutable](../../../../glossary.md#term-immutable) [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), validated [topological](../../../../glossary.md#term-topological-order)
  `CompiledNode` order, [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), graph boundaries, phases, and multi-output [nodes](../../../../glossary.md#node).
- Complete [backend](../../../../glossary.md#backend)-contract [identity](../../../../glossary.md#term-identity-meanings) foundation with stable equality-based `BackendId` values.
- Accepted ADR 0004 ordering ownership before maximal adjacent same-owner partitioning and ADR
  0002 keeping backend-specific lowering after planned partitions.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>

## Follow-up tasks

- Planning task 0005 remains Draft for logical materialization and memory requirements derived
  from the immutable graph and planned partitions. Do not create its detailed specification here.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph))</span>
- Planning task 0006 remains Draft for planning contract closure and the concrete public/internal
  orchestration audit before compiler work.
- [Compiler](../../../../glossary.md#term-compiler) [planning](../../../../glossary.md#term-planning) orchestration and `CompileArtifacts` remain with the later compiler frontier.
- Trace partition identifiers and compile payloads remain Draft until their producer-facing
  planning/compiler facts and emission consumer are stable.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [partition glossary](../../../../glossary.md#partition); [compile glossary](../../../../glossary.md#compile); [producer glossary](../../../../glossary.md#term-producer))</span>
- Cost-bearing ownership scoring, Config 0004, device-level selection, concrete backend prepare,
  physical transfers/allocation, and execution remain at their actual future owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [device glossary](../../../../glossary.md#term-device); [concrete backend glossary](../../../../glossary.md#concrete-backend); [prepare glossary](../../../../glossary.md#prepare))</span>

## Architecture impact

Expected impact: None.

This task realizes the architecture's existing maximal same-[owner](../../../../glossary.md#term-owner) partitioning stage with only
[immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) IDs and [`BackendId`](../../../../glossary.md#backend-identity--backendid) [ownership](../../../../glossary.md#term-ownership-meanings). It does not change [module](../../../../glossary.md#term-module) ownership, [dependency
direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) rules, or [backend](../../../../glossary.md#backend) [lowering](../../../../glossary.md#lowering). If implementation reveals a need for graph-edge
component semantics, phase boundaries, public orchestration, boundary-value storage, a selected
[device](../../../../glossary.md#term-device), another [dependency](../../../../glossary.md#term-dependency), or any architecture rule change, stop before editing source or
architecture documentation and report the decision required.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md and the directly relevant profiles,
docs/planning/roadmap.md, the planning/config/backend-contract/trace/compiler master plans,
completed Planning tasks 0001–0003, current model graph and planning source/tests, and
docs/planning/modules/planning/tasks/0004-maximal-same-owner-partitioning.md in full.

Implement task 0004 exactly inside its fifteen authorized paths. Add only the exact public
PlannedPartition recipe, package-private maximal consecutive same-owner generator, two focused
tests, required Javadocs, current-status documentation, and synchronized planning evidence.
Preserve complete owner-map validation, stored topological adjacency, BackendId equality, exact
graph-node and first-owner references, graph-boundary and multi-output rules, and immutable
results. Add no public orchestration facade, ownership row, graph-edge component search, phase
split, boundary/transfer/memory model, cost/workload/profile type, diagnostics schema, compiler,
device/route/kernel, prepare/runtime/backend behavior, dependency/build change, architecture
change, or later detailed task spec. Stop and report any architecture or scope conflict.

After executable Java and the single final planning module suite, hand the actual diff and exact
evidence to a separate clean-context documentation-focused agent or thread. Reuse successful Java
evidence unless executable behavior changes. Do not run architecture or repository-wide tests
without this task's concrete trigger. Mark task 0004 Complete only after every criterion and the
documentation pass succeed.
```

## Local decisions

- Use `CompiledGraphModel.nodes()` as the exact adjacency/order contract. It is already validated
  topological order; a new sequence, graph traversal, or producer index would duplicate compiler/
  model facts and make the smallest partitioning step ambiguous.<span data-glossary-navigation="true"> ([topological order glossary](../../../../glossary.md#term-topological-order); [graph glossary](../../../../glossary.md#graph); [producer glossary](../../../../glossary.md#term-producer); [compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model))</span>
- Use `Map<NodeId, BackendId>` as the internal completed-ownership handoff. It associates by typed
  graph identity and avoids both a position-coupled parallel owner list and a speculative public
  ownership-row contract.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [handoff glossary](../../../../glossary.md#term-handoff); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Make only the immutable partition DTO public. The generator remains internal until planning
  closure or a concrete compiler consumer can justify the orchestration surface.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition); [DTO glossary](../../../../glossary.md#data-transfer-object--dto); [planning glossary](../../../../glossary.md#term-planning))</span>
- Define maximality over consecutive topological occurrences, matching the architecture's
  “maximal adjacent same-owner work” language. Equal owners separated by a different owner form
  distinct partitions; independent equal-owner nodes that are consecutive form one partition.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#intermediate-representation-dag-and-ssa))</span>
- Retain the first [node](../../../../glossary.md#node)'s [owner](../../../../glossary.md#term-owner) reference while comparing by value. Equal [`BackendId`](../../../../glossary.md#backend-identity--backendid) objects name
  the same ownership domain, and replacing the first reference would make result identity depend
  on a later occurrence.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Store only owner plus node IDs. The immutable graph already owns operations, ordered inputs and
  outputs, descriptors, producers, consumers, phases, and graph boundaries; duplicating selected
  boundary values before logical-memory requirements are designed would create drift.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Keep a multi-output node indivisible. Ownership is per operation occurrence, not per produced
  value, and the current model gives each output one shared producer node.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Do not split at `GraphPhase` changes. The architecture names same owner as the partitioning
  criterion and leaves forward/backward schedule shape to later prepare decisions.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [prepare glossary](../../../../glossary.md#prepare))</span>

## Known limitations

- The input owner map is internal and must be assembled by later orchestration; no public planner
  currently runs capability, eligibility, selection, and partitioning end to end.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Partition shape depends on the stored validated topological order. Another legal topological
  ordering may produce different consecutive runs; deterministic graph ordering remains the
  compiler's responsibility.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#intermediate-representation-dag-and-ssa))</span><span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition); [shape glossary](../../../../glossary.md#shape); [topological order glossary](../../../../glossary.md#term-topological-order); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- The recipe contains no partition identifier, phase summary, boundary values, boundary reasons,
  transfers, materialization requirements, memory plan, diagnostic payload, or serialization
  schema.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [serialization glossary](../../../../glossary.md#term-serialization); [schema glossary](../../../../glossary.md#term-schema))</span>
- Consecutive independent nodes with equal owners share a partition, while graph-connected equal-
  owner nodes separated by another owner do not. This is deliberate order adjacency, not a
  connected-component search.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node))</span>
- A partition makes no promise of one fused kernel or one executable. Backend prepare may realize
  it as multiple units and owns every concrete route decision.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [kernel glossary](../../../../glossary.md#kernel); [Backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `/root/plan_planning_0004` read `AGENTS.md`, the complete authoritative
  architecture contract, focused module/dependency/lifecycle/partition-scoring/tracing/prepare-
  backend explanations, ADR 0004, documentation rules and General/Planning profiles, the complete
  planning guide and roadmap, planning/config/backend-contract/trace/compiler master plans,
  completed Planning tasks 0001–0003, and current planning capability source/tests.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [trace glossary](../../../../glossary.md#trace); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- The same context inspected the current [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph) contracts. [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) validates
  and retains topological node order, graph boundaries, producer closure, phase coverage, and
  multi-output nodes, so Planning 0004 needs no graph reconstruction or compiler prerequisite.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#term-topological-order); [node glossary](../../../../glossary.md#node); [producer glossary](../../../../glossary.md#term-producer))</span>
- Frontier reassessment found no architecture conflict: task 0003 supplies the required
  per-occurrence owner result, and the architecture explicitly orders maximal adjacent same-owner
  partitioning next. Config 0004 still lacks a cost-bearing consumer, and Trace/Compiler producer
  contracts remain too incomplete to supersede this bounded planning step.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [owner glossary](../../../../glossary.md#term-owner))</span>
- Planning-stage `python3 /tmp/validate_synaptik_markdown.py` passed for 228 Markdown files, 4,048
  local links, 246 local anchors, 2,862 fence markers, final newlines, and trailing whitespace.
- Planning-stage `git diff --check` passed with no output. The complete tracked/untracked planning
  diff contains exactly five Markdown paths: this task, the planning/config/trace master plans,
  and the roadmap. No Java, test, Gradle, <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`ARCHITECTURE.md`, ADR, architecture-test, API, guide, or
  glossary path changed during planning.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Status checks found exactly one Ready master-plan row and exactly one Ready detailed task:
  Planning 0004. Planning 0001–0003 and Config 0001–0003 remain Complete; Planning 0005–0006,
  Config 0004+, Trace 0003+, and all Compiler work remain Draft. No detailed Planning 0005,
  Planning 0006, Config 0004, Trace 0003, or Compiler task specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- The final exact audit reported `ready_headings=1`, `ready_rows=1`, and `changed_paths=5`; every
  changed path is Markdown, and no Java, test, Gradle, `ARCHITECTURE.md`, or `testing/` path is in
  the diff. The first combined audit used `path` as a zsh loop variable, which shadowed zsh's
  special command-search path and caused only the later `git`/`rg` audit commands to report
  `command not found`; rerunning the audit with `changed_file` passed and changed no repository
  content.
- Manual planning review confirmed the canonical task sections, exact package/type/method shapes,
  deterministic validation order and messages, topological consecutive-order maximality,
  equality/reference rules, zero-node and graph-boundary behavior, multi-output producer
  indivisibility, exact fifteen-path implementation cap, standalone clean-context prompt, and all
  required exclusions. Completed task history is unchanged.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [shapes glossary](../../../../glossary.md#shape))</span>
- Implementation context `/root/implement_planning_0004` added the exact three production paths
  and two focused test paths. It changed no existing capability Java, model graph contract,
  dependency, Gradle file, architecture rule, ADR, architecture test, compiler, prepare, runtime,
  engine, concrete backend, backend-conformance test, integration test, or other module.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- The implementation context ran
  `./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.partition.PlannedPartitionTest --tests io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioningTest`;
  it passed with `BUILD SUCCESSFUL` and 11 tests across the two focused suites.
- The implementation context then ran exactly one final `./gradlew :modules:planning:test`; it
  passed with `BUILD SUCCESSFUL`. Final XML records 49 tests across five suites, with zero
  failures, errors, and skips: 10 capability-contract, 15 eligibility, 13 owner-selection, 4
  planned-partition, and 7 maximal-partitioning tests.
- Documentation context `/root/implement_planning_0004/docs_planning_0004` applied the General,
  API/Javadoc, Architecture, User guide, Planning, and Example profiles. It independently reviewed
  the final partition source and 11 focused tests, current planning capability source/tests,
  current model graph source/tests, architecture contract, ADRs 0002 and 0004, the five affected
  explanatory documents, task, master plans, and roadmap. It changed no executable Java behavior
  or test, so it reused both successful Java-test results and did not rerun them.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- That documentation pass finalized the three production Javadocs, current-versus-planned API and
  architecture wording, task-oriented backend-selection explanation, the [`Partition`](../../../../glossary.md#partition) glossary
  entry, and synchronized planning evidence/status. It preserved the exact public recipe/internal
  generator boundary and added no later promise about orchestration, boundary or memory facts,
  compiler, prepare, runtime, backend, or execution behavior.
- `./gradlew :modules:planning:javadoc` passed with `BUILD SUCCESSFUL`. Inspection of generated
  `partition/package-summary.html`, `partition/PlannedPartition.html`, `allclasses-index.html`, and
  `overview-tree.html` confirmed the public owner-plus-node-ID recipe and documented constructor/
  accessors. No generator page or public-index entry exists for package-private
  `MaximalSameOwnerPartitioning`.
- Final `python3 /tmp/validate_synaptik_markdown.py` passed for 228 Markdown files, 4,047 local
  links, 248 local anchors, 2,870 fence markers, final newlines, and trailing whitespace.
  `git diff --check` passed with no output.
- The final tracked/untracked audit contains exactly the fifteen authorized paths: three
  production, two test, five explanatory-documentation, and five planning paths. `git status
  --short` reports only those paths. Planning 0001–0004 are Complete; Planning 0005–0006, Config
  0004+, Trace 0003+, and compiler work remain Draft. No detailed later specification exists and
  no global task is Ready.
- Final dependency and forbidden-surface review found no planning Gradle/dependency change and no
  public orchestration facade, ownership row, partition ID, graph-edge search, phase split,
  boundary/transfer/materialization/memory type, cost/workload/profile type, diagnostics schema,
  device, route, kernel, prepare, runtime, backend, or execution type.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization); [schema glossary](../../../../glossary.md#term-schema); [device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel))</span>
- Reasoned no-change conclusions: capability Java remains the completed per-query query/provider,
  eligibility, and baseline selection seam because partitioning consumes only a completed owner
  map; config and backend-contract Java remain unchanged because the recipe needs only current
  [`BackendId`](../../../../glossary.md#backend-identity--backendid) and no new declarative input; trace Java remains unchanged because no event,
  correlation identity, or payload schema was introduced. Planning Gradle and dependencies remain
  unchanged because existing public model and backend-contract edges cover <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>[`NodeId`](../../../../glossary.md#nodeid),
  `CompiledGraphModel`, and `BackendId`.
- `ARCHITECTURE.md` and ADRs 0002/0004 remain unchanged because the task implements their existing
  ownership-before-partition and backend-owned-lowering decisions. Architecture tests remain
  unchanged because no dependency or module boundary changed. Compiler remains unchanged because
  no owner-map assembly, orchestration, or compile artifact was added. Prepare, runtime, engine,
  concrete backends, backend conformance, and integration remain unchanged because the result is
  compile-time recipe data only. Every other module remains unchanged because no task acceptance
  criterion requires behavior outside planning and the five explanatory/status documents.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Implementation notes

- Added the exact public two-component `PlannedPartition` record and the exact package-private
  stateless generator specified by the task.
- Validation occurs before output construction. Partition membership follows only stored
  topological order and <span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [Partition glossary](../../../../glossary.md#partition); [topological order glossary](../../../../glossary.md#term-topological-order))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid) equality, retaining exact graph node references and the first
  node's owner reference in immutable results.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [owner glossary](../../../../glossary.md#term-owner); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Tests cover the DTO/generator shapes, exact validation order and messages, equality association,
  immutable snapshots, reference retention, maximal runs, zero-node pass-through, phases, graph
  boundaries, independent nodes, fan-out, merges, repeated inputs, and multi-output producers.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([DTO glossary](../../../../glossary.md#data-transfer-object--dto); [shapes glossary](../../../../glossary.md#shape); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Documentation was finalized in the separate required clean context without changing executable
  behavior or duplicating successful Java validation.

## Completion summary

- Completed changes: added the immutable public planned-partition recipe, internal maximal
  consecutive same-owner generation, focused tests, finalized Javadocs and explanatory docs, and
  synchronized planning status/evidence.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition); [owner glossary](../../../../glossary.md#term-owner))</span>
- Files changed or created: exactly the fifteen paths listed in Affected files.
- Tests and validation: focused 11-test evidence and final 49-test/five-suite planning evidence
  reused from the implementation context; planning Javadoc, generated public/internal boundary,
  repository Markdown, exact scope, status, later-spec, dependency, forbidden-surface, newline,
  whitespace, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check` [validation](../../../../glossary.md#term-validation) passed.
- Documentation-agent review: completed in
  `/root/implement_planning_0004/docs_planning_0004`; executable behavior and tests were unchanged.
- Documentation impact: finalized partition-scoring, public/compile API, backend-selection,
  glossary, task, planning/config/trace master plans, and roadmap current-versus-planned wording.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [trace glossary](../../../../glossary.md#trace))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: finalized and generated successfully for `PlannedPartition`, the internal
  generator contract, and the partition package; the generator remains absent from public pages.
- Glossary impact: replaced the planned generic partition entry with the exact current recipe and
  internal consecutive-order grouping boundary.
- Unresolved issues: None.
- Follow-up required: None for this task; [Planning](../../../../glossary.md#term-planning) 0005–0006 and all later work remain Draft.

Status: Complete
