# Task 0004: Maximal Same-Owner Partitioning

## Status

Complete

## Goal

Add the smallest backend-neutral partitioning capability that turns one complete backend-owner
assignment for the nodes of an immutable compiled graph into maximal consecutive same-owner
partitions.

Mental model:

```text
CompiledGraphModel.nodes() in validated topological order
  + exactly one selected BackendId for every NodeId
  -> validate complete node-to-owner coverage
  -> split only when the next node has a different BackendId by equality
  -> immutable ordered PlannedPartition values
```

The result records backend ownership and ordered graph-node identity only. It is an immutable
compile-time recipe, not lowering, an executable, a physical schedule, a transfer plan, or a
memory allocation.

## Motivation

Planning task 0003 can select one exact `BackendId` owner for one operation occurrence after hard
eligibility. The architecture places maximal same-owner partitioning immediately after those
per-occurrence ownership decisions. The current model already supplies a structurally closed
`CompiledGraphModel` whose `nodes()` list is validated topological order, so no compiler capture,
topological sort, producer index, or new graph representation is needed for this step.

This task deliberately uses a complete `Map<NodeId, BackendId>` as the handoff from repeated
one-occurrence selection. A later planning/compiler orchestration consumer will build that map;
this task neither invokes nor widens the package-private capability selector. The map avoids a
speculative ownership-row type while making association independent of parallel list position.

## Scope

- Add public record `PlannedPartition` in
  `io.github.pho001.synaptik.planning.partition` with exactly these ordered components:

  ```java
  BackendId owner
  List<NodeId> nodeIds
  ```

- Make `PlannedPartition` the immutable backend-neutral partition DTO named by the architecture.
  It is public because later planning memory work, compiler compile artifacts, prepare contracts,
  and concrete backend preparation must be able to consume the recipe without depending on a
  planning implementation class.
- Give its public canonical constructor this exact validation and snapshot behavior:
  1. null `owner` -> `NullPointerException("owner")`;
  2. null `nodeIds` -> `NullPointerException("nodeIds")`;
  3. empty `nodeIds` -> `IllegalArgumentException("nodeIds must not be empty")`;
  4. scan node IDs in encounter order and reject the first null at index `i` with
     `NullPointerException("nodeIds[i]")`;
  5. reject the first later equal duplicate at index `i` with
     `IllegalArgumentException("nodeIds[i] duplicates <NodeId diagnostic text>")`; and
  6. snapshot membership with `List.copyOf`, retaining the exact `NodeId` element references.
- Explicitly declare and document both public record-component accessors. Preserve ordinary
  record equality, hashing, and diagnostic `toString()` behavior.
- Add package-private final stateless class `MaximalSameOwnerPartitioning` in the same package.
  Give it exactly one private no-argument constructor, no fields, no implemented interfaces, and
  exactly one package-private static method:

  ```java
  static List<PlannedPartition> partition(
          CompiledGraphModel graph,
          Map<NodeId, BackendId> ownershipByNodeId)
  ```

- Treat `graph.nodes()` as the sole ordering and adjacency source. Two nodes are adjacent for this
  task only when they occupy consecutive positions in that validated topological list. Graph-edge
  reachability, numeric `NodeId` order, map iteration order, phase, operation family, and value
  list order do not redefine adjacency.
- Treat `ownershipByNodeId` as one complete owner assignment produced after per-occurrence owner
  selection. Associate keys with graph nodes by `NodeId.equals`, never by object identity, numeric
  list position, or map iteration order.
- Validate the partitioning inputs in this exact order before creating a partition:
  1. null `graph` -> `NullPointerException("graph")`;
  2. null `ownershipByNodeId` -> `NullPointerException("ownershipByNodeId")`;
  3. a null map key -> `NullPointerException("ownershipByNodeId contains null key")`;
  4. inspect non-null map keys in ascending numeric `NodeId.value()` order and reject the first
     key not present in `graph.nodes()` with
     `IllegalArgumentException("ownershipByNodeId contains unknown <NodeId diagnostic text>")`;
  5. walk graph nodes in stored topological order and reject the first node with no equal key
     using
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
  partition's first graph node.
- Return a non-null immutable list. Concatenating all `nodeIds()` lists must reproduce the exact
  graph-node identity references in stored order, with every graph node covered exactly once.
- Accept a valid zero-node pass-through graph only with an empty owner map and return an immutable
  empty partition list. Graph inputs and outputs are values, not synthetic nodes or partitions.
- Treat one `CompiledNode` as indivisible. A multi-output producer belongs to exactly one
  partition because ownership attaches to its node occurrence; all its output `ValueId` values
  therefore originate at that owner. Fan-out, repeated inputs, merges, graph-output publication,
  or a value crossing into a differently owned consumer does not split the producer node.
- Do not store or derive partition input/output value lists, producer/use indexes, boundary edges,
  transfers, or materialization requirements. The closed graph plus partition node IDs retains
  enough logical structure for task 0005 to derive those facts without duplicating them here.
- Do not split on `GraphPhase`. Forward/backward classification remains graph metadata; this
  ownership-only task has no architecture basis for inventing a phase partition boundary.
- Add package Javadoc explaining the current public partition recipe and internal generation step.
- Add focused tests for the exact DTO and generator shapes, validation order/messages, equality
  association, exact-reference retention, immutable snapshots, zero-node graphs, maximal runs,
  owner transitions, nonconsecutive equal owners, independent adjacent nodes, graph phase
  changes, graph inputs/outputs, fan-out, merges, and multi-output producers.
- Finalize Javadocs and affected explanatory/planning documentation through the required separate
  clean-context documentation pass in the same overall implementation change.

## Out of scope

- invoking, moving, widening, or changing `BackendEligibility`, `BackendOwnerSelection`,
  `OperationCapabilityQuery`, `BackendCapabilityProvider`, or their packages and behavior
- a public partitioning service, planner facade, registry, manager, callback, policy hierarchy,
  plugin, service locator, or compiler entry point
- a new ownership row, node-owner record, owner list aligned by position, candidate, score,
  partition ID, trace ID, diagnostics result, or failure taxonomy
- capability evaluation, hard-requirement evaluation, availability evaluation, owner scoring,
  cost/workload classification, transfer/boundary penalties, tuning, or candidate search
- graph capture, topological sorting, producer/use indexing, graph transformation, validation
  beyond consuming the already closed graph, compiler orchestration, `CompileArtifacts`, or
  publication planning
- graph-edge-connected-component partitioning, reordering nodes, joining nonconsecutive nodes,
  splitting by phase, splitting a multi-output node, fusion, specialization, or decomposition
- partition boundary input/output lists, boundary-edge records, transfer records, logical
  materialization or memory requirements, liveness, slots, allocation, layout decisions, or
  physical buffers
- selected devices, routes, kernels, OpenBLAS, Vector API, MPSGraph, CUDA, lowering, backend DAGs,
  prepare behavior, prepared partitions, executables, schedules, runtime residency, or execution
- trace events or payload schemas, architecture-contract or ADR changes, architecture-test
  changes, dependency or Gradle changes, backend-conformance or integration behavior, root-build
  changes, unrelated refactoring, or another detailed future task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0004](../../../../design/decisions/0004-partition-scoring.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Planning master plan](../master-plan.md)
- [Config master plan](../../config/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Planning task 0001](0001-operation-capability-query-foundation.md)
- [Planning task 0002](0002-per-query-backend-hard-eligibility.md)
- [Planning task 0003](0003-ownership-candidates-and-baseline-scoring.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral node ownership and maximal same-owner partitioning. The output may
  contain only compile-time graph identity and `BackendId` ownership.
- `CompiledGraphModel.nodes()` is already immutable validated topological order. Planning consumes
  that ordering and must not capture, reorder, transform, or repair the graph.
- Compile-time partitions hold `BackendId`, not providers, concrete backend objects, preparers,
  devices, executables, routes, kernels, storage, or mutable services.
- One node occurrence is the ownership unit supplied by current selection. Multi-output values do
  not create separate ownership units or allow one producer to be split across backends.
- Maximal adjacent same-owner work forms partitions after ownership. For this smallest current
  implementation, adjacency is consecutive occurrence order in the graph's validated topological
  node list; only an owner transition creates a partition boundary.
- A planned partition remains an immutable recipe. Backend prepare later chooses fusion,
  specialization, routes, kernels, executable units, and physical storage for its assigned region.
- Logical boundary, materialization, and memory facts remain task 0005 work. Omitting duplicated
  boundary lists here does not discard information because the immutable graph and ordered node
  IDs retain producer/consumer structure.
- Planning remains independent of runtime, prepare, engine, and concrete backends. Existing public
  model/backend-contract dependencies already cover the record's public `NodeId` and `BackendId`
  signature, so no dependency change is needed.
- Stop if implementation requires a public orchestration facade, compiler behavior, graph
  mutation, selected device, boundary/materialization model, cost/workload classification,
  partition diagnostics schema, dependency edit, or architecture decision.

## Current contract inventory and handoff

| Contract | Current role in this task | Deliberate boundary |
|---|---|---|
| `BackendOwnerSelection.select(...)` | Produces one exact `BackendId` for one already eligible occurrence | Remains package-private and is not called or widened here |
| `CompiledGraphModel.nodes()` | Supplies the complete validated topological occurrence order | No capture, reordering, producer index, or compiler orchestration |
| `CompiledNode.id()` | Supplies the graph-local ownership and partition membership key | Operation and output count do not create extra ownership units |
| `Map<NodeId, BackendId>` | Exact internal handoff for one complete set of per-node owner results | Not retained, exposed as a new public ownership model, or interpreted by map order |
| `PlannedPartition` | New public immutable owner plus ordered node-ID recipe | No executable, route, boundary-value, transfer, memory, or runtime state |

The current selector and the new generator are intentionally separate package-private operations.
There is still no public workflow connecting provider evaluation, owner selection, and graph-wide
partitioning. Planning task 0006 or the first concrete compiler-planning consumer must select that
narrow orchestration surface; this task must not preempt it.

## Package impact

Package added:

- `io.github.pho001.synaptik.planning.partition` — public immutable planned-partition recipe and
  package-private deterministic maximal same-owner generation

Type placement:

- `io.github.pho001.synaptik.planning.partition.PlannedPartition` — the architecture-named public
  compile-time DTO shared by later planning, compiler, prepare, and backend consumers
- `io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioning` — package-private
  stateless generation operation colocated with the result it creates

No root facade, ownership package, memory package, compiler package, diagnostics package, generic
`util`, registry, service, route, prepare, runtime, or backend-specific package is added.

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

The shape is exact for names, packages, visibility, components, generic arguments, constructor and
method counts, and absence of state. Method bodies above are conceptual and do not authorize
additional conveniences or overloads.

## Partition semantics

Given graph node order `[n0, n1, n2, n3, n4]` and owners
`[cpuA, cpuB, metal, metal, cpuC]`, where all three CPU values are equal `BackendId` values but may
be different references, the result is:

```text
partition 0: owner = exact cpuA reference, nodeIds = [n0, n1]
partition 1: owner = exact metal reference for n2, nodeIds = [n2, n3]
partition 2: owner = exact cpuC reference, nodeIds = [n4]
```

The two CPU regions do not join because they are not consecutive after ownership assignment.
Conversely, two consecutive independent nodes with equal owners do join: adjacency in this task is
the stored topological sequence, not the existence of a producer-consumer edge.

## Graph boundaries and producer rules

- Graph inputs have no producing node and never become partitions. A zero-node pass-through graph
  produces no partition.
- Graph outputs are `ValueId` boundaries and do not split an otherwise consecutive same-owner
  node run.
- A node with multiple outputs remains one partition member. Each output is produced by that
  node's owner, whether it is a graph output, consumed inside the partition, or consumed by one or
  more later partitions.
- A fan-out or merge does not change the node-order scan. Consumers join a producer's partition
  only when they are consecutive in graph order and have an equal selected owner.
- A value consumed across an owner transition identifies future logical boundary/materialization
  work. This task neither creates a physical transfer nor records a boundary DTO.
- Repeated input positions, unused graph inputs, and forward/backward phase changes do not add
  synthetic nodes or force a partition boundary.

## Validation and failure contract

| Stage | Condition | Failure |
|---|---|---|
| 1 | null graph | `NullPointerException("graph")` |
| 2 | null ownership map | `NullPointerException("ownershipByNodeId")` |
| 3 | null map key | `NullPointerException("ownershipByNodeId contains null key")` |
| 4 | first numeric-order key not naming a graph node | `IllegalArgumentException("ownershipByNodeId contains unknown <NodeId>")` |
| 5 | first graph-order node without an equal map key | `IllegalArgumentException("ownershipByNodeId missing <NodeId>")` |
| 6 | first graph-order node with a null owner | `NullPointerException("ownershipByNodeId[<NodeId>]")` |

All validation succeeds before partition construction begins. A map key equal to a graph node ID
is valid even when it is a different object reference. The output always retains the graph's node
ID reference and the map's first-node owner reference.

## Affected files

The intended implementation scope is exactly fifteen paths.

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

Planning — exactly five paths:

- add and finalize this task
- `docs/planning/modules/planning/master-plan.md`
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `ARCHITECTURE.md`; ADRs; all completed planning source, tests, and
task history; `modules/planning/build.gradle.kts`; architecture tests; model graph source/tests;
config/backend-contract/trace/compiler/prepare/runtime/engine Java and builds; concrete backends;
backend-conformance and integration tests; every other planning master plan and future task row.

## Maximum scope

Exactly the fifteen paths listed above. Stop if implementation requires another production or
test type, another document, a public orchestration API, an ownership row, graph-boundary DTO,
cost/workload classification, diagnostics schema, dependency/build change, architecture test,
architecture change, compiler/prepare/runtime/backend behavior, or detailed follow-up
specification. Do not hide an incomplete acceptance criterion in a later task.

## Acceptance criteria

- `PlannedPartition` is the exact two-component public record specified above, with a public
  canonical constructor, explicitly documented accessors, ordinary record object methods, and no
  other field, constructor, method, interface, factory, builder, or nested type.
- Its constructor enforces the exact owner/list/empty/element/duplicate validation order and
  messages, snapshots list membership, and retains exact owner and node-ID references.
- `MaximalSameOwnerPartitioning` has the exact package-private final stateless shape and sole
  `partition(CompiledGraphModel, Map<NodeId, BackendId>)` method specified above.
- The generator enforces the exact top-level, null-key, numeric unknown-key, graph-order missing,
  and graph-order null-owner validation order and messages before creating any result.
- Equal non-identical `NodeId` map keys associate correctly. Output node IDs are the exact graph
  references; partition owner is the exact map value for the partition's first node.
- Every valid non-empty graph node occurs exactly once. Partition and node order reproduce the
  stored graph order, every partition is non-empty, and the returned outer and inner lists are
  immutable.
- Consecutive equal owners form one maximal run. An owner transition splits; nonconsecutive equal
  owners remain separate; no consecutive result partitions have equal owners.
- A zero-node graph plus empty map returns an immutable empty list. Extra owner keys are rejected,
  including for a zero-node graph.
- Independent consecutive nodes, graph inputs/outputs, fan-out, merges, repeated inputs,
  forward/backward phase changes, and a multi-output node behave exactly as specified without
  edge-connected grouping, phase splitting, or node splitting.
- Production code introduces no operation/kind inspection, provider call, availability/config
  interpretation, cost/score, device, route, kernel, boundary value/edge, transfer,
  materialization, allocation, trace, prepare, runtime, or execution state.
- Existing capability classes, model graph contracts, dependencies, Gradle files, architecture
  rules, ADRs, and completed-task history remain unchanged.
- The public and compile API pages, backend-selection guide, glossary, partition-scoring status,
  task, planning master plan, trace deferral status, and roadmap accurately distinguish the
  current recipe/generator from planned orchestration, boundary/memory work, prepare, and
  execution.
- A separate clean-context documentation-focused agent pass finalizes affected Javadocs,
  explanatory documentation, examples, terminology, links, glossary impact, planning evidence,
  and status in the same overall change.
- Focused tests, one final planning module suite, final planning Javadoc, repository Markdown,
  generated-page, exact-scope, status, later-spec, dependency, forbidden-surface, final-newline,
  trailing-whitespace, and `git diff --check` validation all pass.

## Tests / validation

Focused development validation while executable Java changes:

```bash
./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.partition.PlannedPartitionTest --tests io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioningTest
```

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :modules:planning:test
```

Record XML suite/test/failure/error/skip counts. Then hand the task, actual diff, and exact Java
evidence to the separate clean-context documentation pass. That pass reuses the successful Java
evidence unless it changes executable Java behavior or records a concrete cross-check risk.

After final Javadoc and Markdown edits, the documentation pass runs:

```bash
./gradlew :modules:planning:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect generated `package-summary.html` and `PlannedPartition.html`. Confirm that the public page
contains only the owner/node-ID recipe and that no generated public page or all-classes entry is
created for the package-private generator.

Manual final checks are limited to risks not better locked by tests:

- exact fifteen-path tracked/untracked scope;
- unchanged planning Gradle dependencies and no forbidden import or production surface;
- exactly one Ready task during implementation, then synchronized Complete/Draft status;
- no detailed Planning 0005, Planning 0006, Config 0004, Trace 0003, or Compiler task spec;
- no partition ID, boundary-value/edge, transfer, materialization, cost/workload/profile,
  diagnostics, route, kernel, prepare, runtime, or execution type;
- generated public/internal Javadoc boundary, links/anchors, fences, final newlines, and whitespace.

No architecture test or repository-wide suite is run by habit because this task adds one public
DTO over already-public dependency types and one internal planning operation without changing a
module edge, architecture rule, build, concrete backend, or cross-module executable behavior.
Continuous integration or Planning 0006 owns the repository tier unless implementation reveals a
concrete trigger; such a trigger requires stopping before scope expansion.

## Documentation handoff

After the implementation context records the final planning-module result, hand this
specification, the actual fifteen-path diff, and exact focused/final Java evidence to a separate
clean-context documentation-focused agent or thread.

The handoff must identify:

- the exact public `PlannedPartition(owner, nodeIds)` shape and validation;
- the package-private generator and `CompiledGraphModel` plus complete owner-map input seam;
- topological consecutive-order adjacency and owner-equality maximality;
- exact graph-node and first-owner reference retention;
- zero-node, graph-boundary, phase, fan-out, merge, and multi-output rules;
- the absence of boundary values, transfers, memory, lowering, executable, and public
  orchestration behavior; and
- the five explanatory documents, five planning paths, and validation commands above.

The documentation pass must read the architecture contract, documentation rules, General,
API/Javadoc, Architecture, User guide, Planning, and Example profiles, final source and tests,
current model graph and planning contracts, ADRs 0002 and 0004, and all directly affected pages.
It independently finalizes type/constructor/component/method/package Javadocs, current-versus-
planned wording, examples, links, terminology, glossary impact, evidence, and completion status.

It must record reasoned no-change conclusions for capability/config/backend-contract/trace Java,
planning Gradle/dependencies, architecture contract and ADRs, architecture tests, compiler
implementation, prepare/runtime/engine, concrete backends, backend-conformance and integration
tests, and every other module. It does not rerun successful Java tests unless executable behavior
changes.

## Dependencies

- Complete Planning task 0003 with deterministic per-occurrence `BackendId` owner selection.
- Complete Planning tasks 0001–0002 and Config task 0003 as the upstream capability, hard-
  eligibility, and optional baseline preference contracts.
- Complete model graph foundation with immutable `CompiledGraphModel`, validated topological
  `CompiledNode` order, `NodeId`, `ValueId`, graph boundaries, phases, and multi-output nodes.
- Complete backend-contract identity foundation with stable equality-based `BackendId` values.
- Accepted ADR 0004 ordering ownership before maximal adjacent same-owner partitioning and ADR
  0002 keeping backend-specific lowering after planned partitions.

## Follow-up tasks

- Planning task 0005 remains Draft for logical materialization and memory requirements derived
  from the immutable graph and planned partitions. Do not create its detailed specification here.
- Planning task 0006 remains Draft for planning contract closure and the concrete public/internal
  orchestration audit before compiler work.
- Compiler planning orchestration and `CompileArtifacts` remain with the later compiler frontier.
- Trace partition identifiers and compile payloads remain Draft until their producer-facing
  planning/compiler facts and emission consumer are stable.
- Cost-bearing ownership scoring, Config 0004, device-level selection, concrete backend prepare,
  physical transfers/allocation, and execution remain at their actual future owners.

## Architecture impact

Expected impact: None.

This task realizes the architecture's existing maximal same-owner partitioning stage with only
immutable graph IDs and `BackendId` ownership. It does not change module ownership, dependency
direction, lifecycle rules, or backend lowering. If implementation reveals a need for graph-edge
component semantics, phase boundaries, public orchestration, boundary-value storage, a selected
device, another dependency, or any architecture rule change, stop before editing source or
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
  model facts and make the smallest partitioning step ambiguous.
- Use `Map<NodeId, BackendId>` as the internal completed-ownership handoff. It associates by typed
  graph identity and avoids both a position-coupled parallel owner list and a speculative public
  ownership-row contract.
- Make only the immutable partition DTO public. The generator remains internal until planning
  closure or a concrete compiler consumer can justify the orchestration surface.
- Define maximality over consecutive topological occurrences, matching the architecture's
  “maximal adjacent same-owner work” language. Equal owners separated by a different owner form
  distinct partitions; independent equal-owner nodes that are consecutive form one partition.
- Retain the first node's owner reference while comparing by value. Equal `BackendId` objects name
  the same ownership domain, and replacing the first reference would make result identity depend
  on a later occurrence.
- Store only owner plus node IDs. The immutable graph already owns operations, ordered inputs and
  outputs, descriptors, producers, consumers, phases, and graph boundaries; duplicating selected
  boundary values before logical-memory requirements are designed would create drift.
- Keep a multi-output node indivisible. Ownership is per operation occurrence, not per produced
  value, and the current model gives each output one shared producer node.
- Do not split at `GraphPhase` changes. The architecture names same owner as the partitioning
  criterion and leaves forward/backward schedule shape to later prepare decisions.

## Known limitations

- The input owner map is internal and must be assembled by later orchestration; no public planner
  currently runs capability, eligibility, selection, and partitioning end to end.
- Partition shape depends on the stored validated topological order. Another legal topological
  ordering may produce different consecutive runs; deterministic graph ordering remains the
  compiler's responsibility.
- The recipe contains no partition identifier, phase summary, boundary values, boundary reasons,
  transfers, materialization requirements, memory plan, diagnostic payload, or serialization
  schema.
- Consecutive independent nodes with equal owners share a partition, while graph-connected equal-
  owner nodes separated by another owner do not. This is deliberate order adjacency, not a
  connected-component search.
- A partition makes no promise of one fused kernel or one executable. Backend prepare may realize
  it as multiple units and owns every concrete route decision.

## Validation evidence

- Planning context `/root/plan_planning_0004` read `AGENTS.md`, the complete authoritative
  architecture contract, focused module/dependency/lifecycle/partition-scoring/tracing/prepare-
  backend explanations, ADR 0004, documentation rules and General/Planning profiles, the complete
  planning guide and roadmap, planning/config/backend-contract/trace/compiler master plans,
  completed Planning tasks 0001–0003, and current planning capability source/tests.
- The same context inspected the current model graph contracts. `CompiledGraphModel` validates
  and retains topological node order, graph boundaries, producer closure, phase coverage, and
  multi-output nodes, so Planning 0004 needs no graph reconstruction or compiler prerequisite.
- Frontier reassessment found no architecture conflict: task 0003 supplies the required
  per-occurrence owner result, and the architecture explicitly orders maximal adjacent same-owner
  partitioning next. Config 0004 still lacks a cost-bearing consumer, and Trace/Compiler producer
  contracts remain too incomplete to supersede this bounded planning step.
- Planning-stage `python3 /tmp/validate_synaptik_markdown.py` passed for 228 Markdown files, 4,048
  local links, 246 local anchors, 2,862 fence markers, final newlines, and trailing whitespace.
- Planning-stage `git diff --check` passed with no output. The complete tracked/untracked planning
  diff contains exactly five Markdown paths: this task, the planning/config/trace master plans,
  and the roadmap. No Java, test, Gradle, `ARCHITECTURE.md`, ADR, architecture-test, API, guide, or
  glossary path changed during planning.
- Status checks found exactly one Ready master-plan row and exactly one Ready detailed task:
  Planning 0004. Planning 0001–0003 and Config 0001–0003 remain Complete; Planning 0005–0006,
  Config 0004+, Trace 0003+, and all Compiler work remain Draft. No detailed Planning 0005,
  Planning 0006, Config 0004, Trace 0003, or Compiler task specification exists.
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
  required exclusions. Completed task history is unchanged.
- Implementation context `/root/implement_planning_0004` added the exact three production paths
  and two focused test paths. It changed no existing capability Java, model graph contract,
  dependency, Gradle file, architecture rule, ADR, architecture test, compiler, prepare, runtime,
  engine, concrete backend, backend-conformance test, integration test, or other module.
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
  or test, so it reused both successful Java-test results and did not rerun them.
- That documentation pass finalized the three production Javadocs, current-versus-planned API and
  architecture wording, task-oriented backend-selection explanation, the `Partition` glossary
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
  device, route, kernel, prepare, runtime, backend, or execution type.
- Reasoned no-change conclusions: capability Java remains the completed per-query query/provider,
  eligibility, and baseline selection seam because partitioning consumes only a completed owner
  map; config and backend-contract Java remain unchanged because the recipe needs only current
  `BackendId` and no new declarative input; trace Java remains unchanged because no event,
  correlation identity, or payload schema was introduced. Planning Gradle and dependencies remain
  unchanged because existing public model and backend-contract edges cover `NodeId`,
  `CompiledGraphModel`, and `BackendId`.
- `ARCHITECTURE.md` and ADRs 0002/0004 remain unchanged because the task implements their existing
  ownership-before-partition and backend-owned-lowering decisions. Architecture tests remain
  unchanged because no dependency or module boundary changed. Compiler remains unchanged because
  no owner-map assembly, orchestration, or compile artifact was added. Prepare, runtime, engine,
  concrete backends, backend conformance, and integration remain unchanged because the result is
  compile-time recipe data only. Every other module remains unchanged because no task acceptance
  criterion requires behavior outside planning and the five explanatory/status documents.

## Implementation notes

- Added the exact public two-component `PlannedPartition` record and the exact package-private
  stateless generator specified by the task.
- Validation occurs before output construction. Partition membership follows only stored
  topological order and `BackendId` equality, retaining exact graph node references and the first
  node's owner reference in immutable results.
- Tests cover the DTO/generator shapes, exact validation order and messages, equality association,
  immutable snapshots, reference retention, maximal runs, zero-node pass-through, phases, graph
  boundaries, independent nodes, fan-out, merges, repeated inputs, and multi-output producers.
- Documentation was finalized in the separate required clean context without changing executable
  behavior or duplicating successful Java validation.

## Completion summary

- Completed changes: added the immutable public planned-partition recipe, internal maximal
  consecutive same-owner generation, focused tests, finalized Javadocs and explanatory docs, and
  synchronized planning status/evidence.
- Files changed or created: exactly the fifteen paths listed in Affected files.
- Tests and validation: focused 11-test evidence and final 49-test/five-suite planning evidence
  reused from the implementation context; planning Javadoc, generated public/internal boundary,
  repository Markdown, exact scope, status, later-spec, dependency, forbidden-surface, newline,
  whitespace, and `git diff --check` validation passed.
- Documentation-agent review: completed in
  `/root/implement_planning_0004/docs_planning_0004`; executable behavior and tests were unchanged.
- Documentation impact: finalized partition-scoring, public/compile API, backend-selection,
  glossary, task, planning/config/trace master plans, and roadmap current-versus-planned wording.
- Javadoc review: finalized and generated successfully for `PlannedPartition`, the internal
  generator contract, and the partition package; the generator remains absent from public pages.
- Glossary impact: replaced the planned generic partition entry with the exact current recipe and
  internal consecutive-order grouping boundary.
- Unresolved issues: None.
- Follow-up required: None for this task; Planning 0005–0006 and all later work remain Draft.

Status: Complete
