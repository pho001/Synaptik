# Task 0005: Logical Materialization and Memory Requirements

## Status

Complete

## Goal

Add the smallest backend-neutral compile-time capability that derives immutable logical value
requirements from one structurally closed `CompiledGraphModel` and its ordered, complete
`PlannedPartition` recipes.

Mental model:

```text
CompiledGraphModel
  + ordered complete PlannedPartition recipes
  -> validate exact partition coverage and maximal owner runs
  -> derive each graph value's producing partition, consuming partitions, descriptor, and
     graph-output obligation
  -> immutable LogicalMemoryPlan
```

The result says which logical value must be available to which planned region and which values
must remain available at the graph-output boundary. It does not choose a transfer, copy, physical
buffer, address, slot, arena, device representation, route, kernel, schedule, or runtime
residency.

## Motivation

Planning task 0004 intentionally stores only one owner and ordered node IDs in each partition.
The immutable graph remains the single source of operations, values, descriptors, producers,
uses, graph boundaries, and phases, so copying boundary lists into `PlannedPartition` would have
created derived state before its consumer was known. Task 0005 is that first consumer: it derives
the logical value relationships needed by the architecture-named `LogicalMemoryPlan` while
leaving physical realization to prepare, concrete backends, and runtime.

The current graph supports static, dynamic, and expression dimensions. A `TensorDescriptor`
therefore preserves more correct logical-size information than an eager byte count: its Shape may
not have a known element count, and a backend's physical allocation may add alignment, padding,
or another representation. This task retains the exact descriptor and does not add element-count
or byte-size fields.

## Scope

- Add public record `LogicalMemoryRequirement` in
  `io.github.pho001.synaptik.planning.memory` with exactly these ordered components:

  ```java
  ValueId valueId
  TensorDescriptor descriptor
  Optional<PlannedPartition> producerPartition
  List<PlannedPartition> consumerPartitions
  boolean graphOutput
  ```

- Define one requirement for every `GraphValue` in `CompiledGraphModel.values()` encounter order.
  It is a logical requirement, not a physical allocation request:
  - `valueId` and `descriptor` identify the value and retain its complete logical tensor facts;
  - an empty `producerPartition` means the value is a graph input with no producing node;
  - a present producer is the exact supplied partition containing the value's producing node;
  - `consumerPartitions` contains each exact supplied partition that consumes the value at least
    once, deduplicated and ordered by the supplied partition list; and
  - `graphOutput` is true exactly when an equal `ValueId` occurs in `graph.outputs()`.
- Give `LogicalMemoryRequirement` this exact public canonical-constructor validation and snapshot
  behavior:
  1. null `valueId` -> `NullPointerException("valueId")`;
  2. null `descriptor` -> `NullPointerException("descriptor")`;
  3. null `producerPartition` -> `NullPointerException("producerPartition")`;
  4. null `consumerPartitions` -> `NullPointerException("consumerPartitions")`;
  5. scan consumer partitions in encounter order and reject the first null at index `i` with
     `NullPointerException("consumerPartitions[i]")`;
  6. reject the first later equal duplicate at index `i` with
     `IllegalArgumentException("consumerPartitions[i] duplicates <PlannedPartition diagnostic text>")`;
     and
  7. snapshot consumer membership with `List.copyOf`.
- Retain the exact `ValueId`, `TensorDescriptor`, present producer-partition, and consumer-
  partition element references. Treat `Optional` as a value-based container and make no promise
  about optional-container identity. Preserve ordinary record equality, hashing, and diagnostic
  `toString()` behavior.
- Explicitly declare and document all five public record-component accessors. Do not add role-
  testing convenience methods, factories, builders, nested types, serialization, or mutation.
- Add public record `LogicalMemoryPlan` in the same package with exactly one component:

  ```java
  List<LogicalMemoryRequirement> requirements
  ```

- Give `LogicalMemoryPlan` this exact public canonical-constructor validation and snapshot
  behavior:
  1. null `requirements` -> `NullPointerException("requirements")`;
  2. scan in encounter order and reject the first null at index `i` with
     `NullPointerException("requirements[i]")`;
  3. reject the first later requirement with an equal `valueId` at index `i` with
     `IllegalArgumentException("requirements[i] duplicates <ValueId diagnostic text>")`; and
  4. snapshot membership with `List.copyOf` while retaining exact requirement references.
- Permit an empty standalone `LogicalMemoryPlan`, even though a valid current graph always has at
  least one output and therefore task-0005 generation returns at least one requirement. The
  public DTO validates its own state rather than claiming graph ownership without a graph.
- Explicitly declare and document the public `requirements()` accessor. Preserve ordinary record
  equality, hashing, and diagnostic `toString()` behavior. Add no other field, method,
  constructor, interface, factory, builder, nested type, or serialization contract.
- Add package-private final stateless class `LogicalMemoryPlanning` in the same package with one
  private no-argument constructor, no fields or implemented interfaces, and exactly one
  package-private static method:

  ```java
  static LogicalMemoryPlan plan(
          CompiledGraphModel graph,
          List<PlannedPartition> partitions)
  ```

- Treat the supplied partition list as the ordered recipes produced by task 0004, while still
  validating it because `PlannedPartition` is publicly constructible. Validate before deriving
  any requirement in this exact order:
  1. null `graph` -> `NullPointerException("graph")`;
  2. null `partitions` -> `NullPointerException("partitions")`;
  3. scan partition elements in encounter order and reject the first null at index `i` with
     `NullPointerException("partitions[i]")`;
  4. scan partition node IDs in partition and membership encounter order; reject the first ID
     not naming a graph node by equality with
     `IllegalArgumentException("partitions[i].nodeIds[j] references unknown <NodeId diagnostic text>")`;
  5. in that same scan, after the current ID is known, reject the first node ID already seen in
     an earlier partition position with
     `IllegalArgumentException("partitions[i].nodeIds[j] duplicates <NodeId diagnostic text>")`;
  6. walk `graph.nodes()` in stored order and reject the first uncovered node with
     `IllegalArgumentException("partitions missing <NodeId diagnostic text>")`;
  7. compare the flattened partition membership with `graph.nodes()` position by position and
     reject the first mismatch with
     `IllegalArgumentException("partitions[i].nodeIds[j] is out of graph order: expected <NodeId diagnostic text>")`;
     and
  8. scan adjacent supplied partitions in encounter order and reject the first partition whose
     owner is equal to its predecessor's owner with
     `IllegalArgumentException("partitions[i].owner equals previous owner <BackendId diagnostic text>")`.
- Complete all partition null, membership, coverage, order, and maximality validation before
  constructing the first `LogicalMemoryRequirement`. Do not repair, reorder, merge, mutate, or
  retain the supplied list container.
- Accept equal but non-identical partition `NodeId` references during validation. Derivation uses
  graph equality for association, retains exact `GraphValue.id()` and descriptor references, and
  retains exact partition element references from the supplied partition list.
- Build producer and use relationships only from the closed graph's stored node input/output
  lists. Do not add them to the model or mutate `CompiledGraphModel`.
- Deduplicate consumer partitions by partition membership, not by input position. Repeated uses
  by one node, several nodes in one partition, or equal repeated input IDs contribute one exact
  consumer-partition reference. Fan-out to several partitions contributes each partition once in
  partition-list order.
- Define the derived classifications exactly, without adding another enum or stored flag:
  - a **graph input** has an empty producer partition;
  - a **partition input** for partition `P` is a value consumed in `P` whose producer is absent or
    is a different partition;
  - a **partition output** is a produced value that is a graph output or has at least one consumer
    partition different from its producer;
  - a **cross-owner boundary** is a produced value with at least one different consumer partition
    whose `owner()` is unequal to the producer owner's value; a graph input has no producer owner
    and therefore is not itself classified as a cross-owner boundary;
  - a **graph output/publication requirement** is exactly a requirement with `graphOutput == true`;
    and
  - a **partition-internal value** has a producer, is not a graph output, and has no consumer
    outside its producer partition. A produced unused value is partition-internal.
- Treat these roles as overlapping where graph structure requires it. A fan-out value can be a
  partition output, several partition inputs, a cross-owner boundary, and a graph output at the
  same time. `LogicalMemoryRequirement` stores the primitive producer/consumer/output facts so
  later consumers do not depend on a prematurely closed role enum.
- Handle graph structure as follows:
  - graph inputs may feed zero, one, or several partitions and have no synthetic producer;
  - an unused graph input has an empty producer and consumer list and remains in the plan;
  - a produced unused value has its producer, no consumers, and remains in the plan;
  - merge inputs remain distinct value requirements;
  - every output of a multi-output node becomes a distinct requirement with the same producer
    partition and its own consumers/output flag;
  - forward/backward phase changes create no synthetic materialization, partition, or schedule
    boundary; cross-phase uses are represented by the same producer/consumer facts; and
  - a valid zero-node pass-through graph requires an empty partition list and produces graph-
    input requirements, with `graphOutput` true for each declared pass-through output.
- Define a **logical materialization requirement** as the compile-time obligation to make a
  logical value available to a consuming partition or preserve it at the graph-output boundary.
  It does not state whether availability is realized by aliasing, copying, transfer, recomputation,
  backend-native representation, or another prepare-time choice.
- Define a **logical memory requirement** as the exact graph-local `ValueId` and retained
  `TensorDescriptor` facts plus their producer, consumer, and graph-output relationships. It is
  not a buffer, slot, address, byte range, arena, pool, allocation, or runtime-residency record.
- Do not calculate or store element counts or byte sizes. A dynamic or expression Shape has no
  current numeric element count, `DataType.byteWidth()` is a logical width rather than backend
  allocation granularity, and resolved layout span still does not choose physical padding or
  representation. Later backend-neutral scoring may estimate sizes from descriptors when its
  cost consumer and units are stable; prepare and concrete backends own physical sizing.
- Do not accept `PublicationBinding` or create `PublicationPlan`. The binding is a standalone
  model DTO and not part of `CompiledGraphModel`; the compiler owns validation and grouping in an
  owning publication plan. For this task, `graph.outputs()` supplies only the logical obligation
  to preserve a value for that later publication stage. No `TensorId`, publication policy,
  gradient publication, or runtime target enters planning memory.
- Add package Javadoc explaining the public plan/requirement DTOs, internal derivation seam,
  classification rules, descriptor-retention rationale, and physical-memory boundary.
- Add focused tests for the exact two DTO and generator shapes, validation order/messages,
  immutable snapshots, equality/reference rules, exact graph-value order, complete partition
  validation, graph inputs and outputs, internal values, same-owner and cross-owner boundaries,
  nonconsecutive repeated owners, fan-out, merge, repeated inputs, unused values, multi-output
  nodes, forward/backward uses, dynamic descriptors, and zero-node pass-through graphs.
- Finalize all affected Javadocs and explanatory/planning documentation through the required
  separate clean-context documentation pass in the same overall implementation change.

## Out of scope

- changing `PlannedPartition`, maximal same-owner generation, capability, eligibility, owner
  selection, scoring configuration, or model graph contracts
- a public planner facade, registry, manager, service locator, compiler entry point, public
  planning orchestration, or invocation of the package-private task-0001–0004 operations
- owner-map assembly, capability matrices, cost/workload classification, numeric scoring,
  planning-cost profiles, transfer estimates, boundary penalties, or tuning candidates
- graph capture, producer/use persistence in model, topological sorting, inference, validation,
  canonicalization, transformation, autograd, backward construction, publication binding,
  `PublicationPlan`, `CompileArtifacts`, or compiler diagnostics
- `PublicationBinding` validation, `TensorId` retention, publication policy, gradient publication,
  or runtime publication targets
- physical transfers, transfer commands, copy commands, routes, source/destination devices,
  current residency, device selection, or device-level capability
- element-count or byte-size fields, concrete byte counts, offsets, alignment, addresses, arenas,
  storage implementations, memory pools, buffer/workspace slots, liveness intervals, allocation,
  coloring, reuse, alias analysis, or lifetime scheduling
- layout resolution, alias-versus-copy choice, contiguous conversion, backend representation,
  fusion, specialization, lowering, route/kernel selection, prepared executables, prepared memory,
  schedules, runtime behavior, or execution
- trace events, payload schemas, rejection diagnostics, serialization, architecture-contract or
  ADR changes, architecture-test changes, dependency/Gradle/build changes, backend-conformance or
  integration behavior, unrelated refactoring, or another detailed future task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0001](../../../../design/decisions/0001-layered-architecture.md)
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0004](../../../../design/decisions/0004-partition-scoring.md)
- [ADR 0005](../../../../design/decisions/0005-training-combined-forward-backward-graph.md)
- [Memory planning strategy](../../../../design/notes/memory-planning-strategy.md)
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
- [Backend-contract master plan](../../backend-contract/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Prepare master plan](../../prepare/master-plan.md)
- [Runtime master plan](../../runtime/master-plan.md)
- [Planning task 0001](0001-operation-capability-query-foundation.md)
- [Planning task 0002](0002-per-query-backend-hard-eligibility.md)
- [Planning task 0003](0003-ownership-candidates-and-baseline-scoring.md)
- [Planning task 0004](0004-maximal-same-owner-partitioning.md)
- [Model graph task 0009](../../model/tasks/0009-compiled-graph-model.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Compiling-graphs guide](../../../../user-guide/compiling-graphs.md)
- [Partition-preparer guide](../../../../backend-guide/partition-preparer.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral logical materialization and memory requirements after maximal
  same-owner partitioning. The result may contain only immutable compile-time graph facts,
  partition recipes, and logical tensor descriptors.
- `CompiledGraphModel` is already structurally closed. Planning may derive producer/use and
  boundary relationships but must not capture, repair, reorder, transform, or persist new model
  indexes.
- `PlannedPartition` remains the owner-plus-node-ID recipe. Task 0005 validates its ordered exact
  graph coverage and derives value relationships without adding boundary state back to that DTO.
- Compile-time planning holds `BackendId` through `PlannedPartition`, never live backends,
  devices, preparers, executable units, routes, kernels, storage, or runtime resources.
- `TensorDescriptor` is the current backend-neutral logical size carrier. Dynamic dimensions,
  unresolved layouts, and backend-specific physical representation make concrete byte size an
  invalid universal field at this boundary.
- Graph-output status is a logical preservation obligation only. The compiler-owned future
  `PublicationPlan` owns `PublicationBinding`, owning-graph validation, publication policy, and
  the later publication handoff.
- Combined forward/backward graphs are one compile-time graph for logical requirements. Phase
  metadata remains in the graph and does not become a physical schedule or a forced partition
  boundary.
- Prepare later maps logical needs to physical slots and schedules valid transfers or
  materializations. Concrete backends own representation, lowering, and executable choices.
  Runtime owns current residency and executes only prepared work.
- Planning remains independent of runtime, prepare, engine, and concrete backends. Existing
  public model and backend-contract dependencies cover every proposed signature, so no dependency
  or architecture-test change is needed.
- Stop if implementation requires a `PublicationBinding` input, compiler aggregate, public
  orchestration facade, physical size or lifetime field, transfer/copy record, selected device,
  slot/allocation model, layout resolution, route/kernel choice, dependency edit, or architecture
  decision.

## Current contract inventory and handoff

| Contract | Current role in this task | Deliberate boundary |
|---|---|---|
| `CompiledGraphModel.values()` | Supplies exact ordered `ValueId` and `TensorDescriptor` facts | No physical storage, derived use index, or publication binding |
| `CompiledGraphModel.nodes()` | Supplies validated topological producer/use relationships and phase-covered computation | No capture, reordering, execution, or persisted index |
| `CompiledGraphModel.inputs()` / `outputs()` | Identify producer-free input and logical output boundaries | No `TensorId`, publication policy, or runtime target |
| `PlannedPartition` | Supplies exact partition membership and `BackendId` ownership | No boundary lists, device, route, transfer, or memory state |
| `PublicationBinding` | Reviewed standalone model association | Not consumed; compiler-owned `PublicationPlan` must add owning context later |
| `TensorDescriptor` | Exact retained logical type, Shape, optional layout, and gradient-eligibility facts | No eager byte count or physical representation |
| `LogicalMemoryRequirement` | New public per-value producer/consumer/output logical recipe | No slot, lifetime, allocation, transfer, or residency |
| `LogicalMemoryPlan` | New public immutable list for later `CompileArtifacts` consumption | No compiler aggregate or prepared memory |

## Package impact

Package added:

- `io.github.pho001.synaptik.planning.memory` — public immutable logical value requirements and
  plan plus package-private deterministic derivation from the closed graph and partitions

Type placement:

- `io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement` — public per-value logical
  memory/materialization relationship needed by later compiler, prepare, and backend consumers
- `io.github.pho001.synaptik.planning.memory.LogicalMemoryPlan` — architecture-named public
  immutable compile-time aggregate intended for later `CompileArtifacts`
- `io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning` — package-private stateless
  derivation operation because no public compiler/planning orchestration consumer exists yet

No root facade, ownership package, transfer package, physical-memory package, slots package,
publication package, compiler package, generic `util`, registry, service, runtime, prepare, or
backend-specific package is added.

## Proposed production surface

```java
package io.github.pho001.synaptik.planning.memory;

public record LogicalMemoryRequirement(
        ValueId valueId,
        TensorDescriptor descriptor,
        Optional<PlannedPartition> producerPartition,
        List<PlannedPartition> consumerPartitions,
        boolean graphOutput) {
    public LogicalMemoryRequirement {
        // Exact validation and snapshots are specified in Scope.
    }

    @Override public ValueId valueId() { /* exact retained reference */ }
    @Override public TensorDescriptor descriptor() { /* exact retained reference */ }
    @Override public Optional<PlannedPartition> producerPartition() { /* value-based optional */ }
    @Override public List<PlannedPartition> consumerPartitions() { /* immutable snapshot */ }
    @Override public boolean graphOutput() { /* exact stored flag */ }
}

public record LogicalMemoryPlan(List<LogicalMemoryRequirement> requirements) {
    public LogicalMemoryPlan {
        // Exact validation and snapshot are specified in Scope.
    }

    @Override public List<LogicalMemoryRequirement> requirements() { /* immutable snapshot */ }
}

final class LogicalMemoryPlanning {
    private LogicalMemoryPlanning() {}

    static LogicalMemoryPlan plan(
            CompiledGraphModel graph,
            List<PlannedPartition> partitions) {
        // Exact behavior is specified in Scope; strategy is not a public contract.
    }
}
```

Names, packages, visibility, components, generic arguments, constructor and method counts, and
absence of state are exact. Method bodies above are conceptual and authorize no additional API.

## Logical classification matrix

| Graph fact | Stored requirement facts | Derived meaning |
|---|---|---|
| Declared input consumed in `p0` and `p2` | no producer; consumers `[p0, p2]` | graph input and partition input of both partitions |
| Value produced and consumed only inside `p0` | producer `p0`; consumers `[p0]`; not graph output | partition-internal |
| Value produced in `p0`, consumed in equal-owner nonconsecutive `p2` | producer `p0`; consumers include `p2` | partition output/input boundary, not cross-owner |
| Value produced in CPU `p0`, consumed in Metal `p1` | producer `p0`; consumers include `p1` | partition output/input and cross-owner boundary |
| Produced value with no use and no graph output | producer present; consumers empty; not graph output | partition-internal unused value |
| Pass-through graph output | no producer; consumers empty; graph output | graph input plus graph-output/publication obligation |

The table explains how to read the primitive facts. It does not introduce a stored role enum,
transfer direction, schedule step, or physical representation.

## Deterministic ordering and identity

- `LogicalMemoryPlan.requirements()` follows `graph.values()` encounter order, not numeric
  `ValueId`, node order, use order, map order, or partition order.
- `consumerPartitions()` follows supplied partition order and contains each consuming partition
  once, regardless of node/use multiplicity.
- Producer and consumer association uses `NodeId` and `ValueId` equality. Generated output retains
  exact graph value/descriptor references and exact supplied partition element references.
- The result does not retain the graph, source partition-list container, nodes, operations,
  phases, input positions, use counts, or temporary indexes.
- Public DTO constructors retain exact immutable element references while snapshotting list
  membership. No list or optional container identity is promised.

## Affected files

The intended implementation scope is exactly eighteen paths. The seven production/test paths are
one cohesive Planning capability; the eleven documentation paths are required because this task
opens a public compile-artifact DTO and changes the current-versus-planned memory boundary.

Production — exactly four paths:

- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryRequirement.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlan.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlanning.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/package-info.java`

Tests — exactly three paths:

- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryRequirementTest.java`
- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlanTest.java`
- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlanningTest.java`

Architecture-status, API, guide, and glossary documentation — exactly six paths:

- current-versus-planned wording only in `docs/architecture/partition-scoring.md`
- `docs/design/notes/memory-planning-strategy.md`
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

Review without modification: `CompiledGraphModel`, `CompiledNode`, `GraphValue`, `GraphPhase`,
`PublicationBinding`, `TensorDescriptor`, `Shape`, `Dimension`, `DataType`, `LayoutDescriptor`,
their tests/Javadocs, all current planning capability/partition source and tests, planning Gradle,
all config/backend-contract/trace Java, architecture contract and ADRs, architecture tests,
`docs/user-guide/compiling-graphs.md`, compiler/prepare/runtime/engine, concrete backends, backend-
conformance and integration tests, and every later task row or specification.

## Maximum scope

Exactly the eighteen paths listed above. Stop if implementation requires another source, test,
document, package, public type, dependency/build edit, architecture test, architecture change,
publication binding/plan, transfer/copy type, physical size/lifetime/slot/allocation type, selected
device, route/kernel, lifecycle behavior, or detailed later task specification. Do not use a
follow-up task to hide incomplete acceptance criteria.

## Acceptance criteria

- `LogicalMemoryRequirement` has exactly the five public record components and constructor/
  accessor surface specified above, with exact validation order/messages, immutable consumer
  membership, exact element/reference retention, ordinary record value behavior, and no added
  API or serialization.
- `LogicalMemoryPlan` has exactly one public `List<LogicalMemoryRequirement>` component and the
  specified validation, duplicate-value rejection, immutable snapshot/reference retention,
  ordinary record behavior, and no added API or serialization.
- `LogicalMemoryPlanning` has exactly one private no-argument constructor and one package-private
  static `plan(CompiledGraphModel, List<PlannedPartition>)` method, with no fields, interfaces,
  state, overloads, nested types, or public exposure.
- The generator validates top-level inputs, partition elements, unknown membership, duplicate
  membership, missing coverage, graph order, and adjacent-owner maximality in the exact specified
  order and with exact messages before constructing a requirement.
- Generated requirements cover every graph value exactly once in graph-value order and retain
  exact graph `ValueId`/descriptor and supplied partition references in immutable results.
- Producer and distinct consumer partitions are derived from the graph and complete partitions
  without persisting or mutating graph indexes. Consumer partitions are deduplicated and returned
  in partition order.
- Graph input, partition input, partition output, cross-owner, graph-output/publication, and
  partition-internal classifications follow the exact matrix and primitive-fact rules above.
- Fan-out, merge, repeated input positions, unused graph inputs, unused produced values, graph
  outputs, multi-output producers, same-owner nonconsecutive partitions, forward/backward uses,
  and zero-node pass-through graphs are covered by focused tests.
- Dynamic and expression-dimension descriptors are retained exactly without element-count or
  byte-size calculation. No physical size, address, slot, lifetime, allocation, transfer, copy,
  device, route, kernel, schedule, residency, or executable field/type is added.
- `PublicationBinding` is not an input or retained fact. `graph.outputs()` supplies only the
  logical graph-output obligation, and compiler-owned `PublicationPlan` remains planned.
- `PlannedPartition` and every completed Planning 0001–0004 source/test contract remain
  unchanged. Task 0004 correctly remains owner-plus-node-ID only because 0005 derives boundaries
  from the graph and partitions without duplicating them there.
- Package Javadocs, public type/constructor/component/method Javadocs, architecture-status/API/
  guide/glossary wording, and planning records state the exact current versus planned boundary.
- A separate clean-context documentation-focused pass finalizes affected Javadocs, documentation,
  examples, links, terminology, glossary impact, planning evidence, and status in this same
  overall change without rerunning successful Java tests unless executable behavior changes or a
  concrete risk is recorded.
- Focused tests, one final planning module suite, planning Javadoc, repository Markdown,
  generated-page inspection, exact eighteen-path scope, dependency/status/later-spec checks,
  final newlines, trailing whitespace, and `git diff --check` pass.
- During implementation Planning 0005 is the sole Ready task. After completion Planning 0001–0005
  are Complete, Planning 0006 and every later task remain Draft without another detailed future
  specification, and no global task is Ready pending a separate frontier reassessment.

## Tests / validation

Focused development validation while Java changes:

```bash
./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirementTest --tests io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanTest --tests io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanningTest
```

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :modules:planning:test
```

Record XML suite/test/failure/error/skip counts. Then hand this specification, the actual diff,
and exact Java evidence to the separate clean-context documentation pass. That pass reuses the
successful Java evidence unless it changes executable Java behavior or records a concrete risk.

After final Javadoc and Markdown edits, the documentation pass runs:

```bash
./gradlew :modules:planning:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect generated `memory/package-summary.html`, `LogicalMemoryRequirement.html`,
`LogicalMemoryPlan.html`, `allclasses-index.html`, and `overview-tree.html`. Confirm that the two
public records are present and the package-private `LogicalMemoryPlanning` has no public page or
index entry.

Manual final checks are limited to risks not better locked by tests:

- exact eighteen-path tracked/untracked scope;
- unchanged planning Gradle dependencies and no forbidden import/surface;
- exactly one Ready task during implementation, then synchronized Complete/Draft status;
- no detailed Planning 0006, Config 0004, Trace 0003, Compiler, Prepare, or Runtime task spec;
- no `PublicationBinding` input, `TensorId`, physical size/lifetime/slot/allocation, transfer/copy,
  device, route/kernel, prepared, runtime, or execution type;
- generated public/internal Javadoc boundary, links/anchors, fences, final newlines, and
  whitespace.

No architecture test or repository-wide suite is run by habit because the task adds public DTOs
over types already exposed through planning's existing `api` dependencies and one internal
planning derivation without changing a module edge, architecture rule, shared build, concrete
backend, or cross-module executable behavior. Planning 0006, continuous integration, or another
recorded capability checkpoint owns the repository tier unless implementation reveals a concrete
trigger. Such a trigger requires stopping before scope expansion.

## Documentation handoff

After implementation records the focused and single final planning-module results, hand this
task, the actual eighteen-path diff, and exact test evidence to a separate documentation-focused
agent or thread with clean context.

The handoff must identify:

- the exact two public record shapes and package-private generator;
- the `CompiledGraphModel` plus ordered `PlannedPartition` input seam;
- exact partition-coverage/maximality validation and generated ordering/reference rules;
- producer, distinct consumer, graph-output facts and the six derived classifications;
- fan-out, merge, repeated/unused input, multi-output, phase, and zero-node behavior;
- retention of `TensorDescriptor` instead of element/byte calculation;
- exclusion of `PublicationBinding`, physical memory, transfers, prepare/runtime/backend behavior,
  and public orchestration; and
- the six explanatory documents, five planning paths, and validation commands above.

The documentation pass must read the architecture contract, documentation rules, General,
API/Javadoc, Architecture, User guide, Planning, and Example profiles, final source/tests, current
model graph/descriptor/layout/publication source/tests, current planning source/tests, focused
architecture/ADR/strategy documents, directly affected pages, and generated Javadocs. It
independently finalizes all public/package-private Javadocs, current-versus-planned wording,
examples, links, terminology, glossary impact, evidence, and completion status.

It must record reasoned no-change conclusions for `PublicationBinding` and model Java, planning
capability/partition Java, config/backend-contract/trace Java, planning Gradle/dependencies,
architecture contract and ADRs, architecture tests, compiler implementation, prepare/runtime/
engine, concrete backends, backend-conformance and integration tests, and every other module. It
does not rerun successful Java tests unless executable behavior changes.

## Dependencies

- Complete Planning task 0004 with public immutable `PlannedPartition` and internal maximal
  consecutive same-owner generation.
- Complete Planning tasks 0001–0003 and Config task 0003 as upstream capability, hard-
  eligibility, owner-selection, and preference context; this task invokes none of those internal
  operations.
- Complete model graph foundation with immutable `CompiledGraphModel`, `GraphValue`,
  `CompiledNode`, `NodeId`, `ValueId`, `GraphPhase`, exact graph boundaries, topological closure,
  repeated inputs, unused inputs, and multi-output nodes.
- Complete model descriptor foundation with `TensorDescriptor`, static/dynamic/expression Shape,
  optional layout, and logical `DataType` width semantics.
- Accepted architecture ordering maximal same-owner partitioning before logical memory and keeping
  physical allocation, preparation, backend representation, and runtime residency later.

## Follow-up tasks

- Planning task 0006 remains Draft for planning contract closure, package/public-internal audit,
  dependency validation, and the concrete orchestration handoff before compiler work. Do not
  create its detailed specification here.
- Compiler planning orchestration, publication planning, `CompileArtifacts`, public compile
  failures, and the connection among current package-private planning steps remain with the later
  compiler frontier.
- Prepare owns physical plan/coverage/schedule validation; runtime owns slots, transfers,
  residency, and execution; concrete backends own representation, materialization mechanisms,
  lowering, and kernels.
- Trace logical-memory payloads remain Draft until the compiler producer and emission schema are
  stable.
- Cost-bearing ownership scoring and Config 0004 remain Draft until a concrete backend-neutral
  cost consumer establishes exact classification and units. This task adds no numeric cost.

## Architecture impact

Expected impact: None.

This task realizes the architecture's existing logical materialization/memory stage with only
immutable graph values, descriptors, graph boundaries, and partition ownership. It does not
change module ownership, dependency direction, compile/prepare/run boundaries, or backend-owned
lowering. If implementation reveals a need for publication bindings, physical size/lifetimes,
transfers, slots/allocation, selected devices, public orchestration, another dependency, or any
architecture-rule change, stop before editing source or architecture documentation and report the
decision required.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md and directly relevant profiles,
docs/planning/roadmap.md, planning/config/backend-contract/trace/compiler/prepare/runtime master
plans, completed Planning tasks 0001–0004, current model graph/descriptor/layout/publication
source/tests, current planning source/tests, and
docs/planning/modules/planning/tasks/0005-logical-materialization-and-memory-requirements.md in full.

Implement task 0005 exactly inside its eighteen authorized paths. Add only the exact public
LogicalMemoryRequirement and LogicalMemoryPlan records, package-private LogicalMemoryPlanning
derivation, three focused tests, required Javadocs, current-status documentation, and synchronized
planning evidence. Preserve exact partition validation, graph-value/partition ordering and
reference rules, producer/consumer/output facts, classification semantics, dynamic descriptors,
and immutable results. Add no PublicationBinding input, TensorId/publication plan, public
orchestration, physical bytes/lifetimes/slots/allocation, transfers/copies, device/route/kernel,
compiler/prepare/runtime/backend behavior, dependency/build/architecture change, or later detailed
task specification. Stop and report any architecture or scope conflict.

After executable Java and the single final planning module suite, hand the actual diff and exact
evidence to a separate clean-context documentation-focused agent or thread. Reuse successful Java
evidence unless executable behavior changes. Do not run architecture or repository-wide tests
without this task's concrete trigger. Mark task 0005 Complete only after every criterion and the
documentation pass succeed.
```

## Local decisions

- Use `CompiledGraphModel` plus ordered `List<PlannedPartition>` as the complete input seam. The
  graph already owns every value, node relationship, descriptor, boundary, and phase needed for
  this derivation; another compiler or model input would duplicate or preempt its owner.
- Keep derivation package-private. The next current need is the immutable result contract;
  Planning 0006 or a concrete compiler consumer must select the narrow end-to-end orchestration
  surface rather than exposing another isolated public operation now.
- Use one per-value `LogicalMemoryRequirement` rather than stored per-partition boundary lists.
  Producer, distinct consumers, and graph-output status encode all requested classifications once
  and handle fan-out without duplicating one value into several public rows.
- Store primitive relationship facts instead of a closed role enum. Partition input/output and
  cross-owner roles overlap and are partition-relative; a role set would either duplicate facts
  or prematurely freeze a taxonomy before prepare consumers exist.
- Retain exact `PlannedPartition` references instead of creating a partition ID or numeric index.
  The current partition recipe is the stable typed identity-bearing compile-time object, and no
  separate partition-identity domain exists.
- Preserve all graph values, including unused graph inputs and unused produced values. Logical
  memory planning cannot silently perform dead-code elimination, and the closed graph is the
  authoritative compile-time state supplied to planning.
- Retain the full exact `TensorDescriptor` rather than `knownElementCount` or bytes. This works for
  dynamic/expression shapes, preserves layout resolution state, avoids overflow/physical-padding
  promises, and gives later owners the actual logical facts.
- Treat `graph.outputs()` as the only current graph-output/publication obligation. Do not accept
  `PublicationBinding`: it is standalone model data for a later compiler-owned `PublicationPlan`,
  and planning cannot establish that owning context here.
- Validate complete ordered partition coverage and adjacent-owner maximality even though task
  0004's internal generator produces valid recipes. The DTO is public and directly constructible,
  so the task-0005 entry point must reject an invalid recipe list before deriving a partial plan.
- Do not split or summarize by `GraphPhase`. A combined forward/backward graph is one compile-time
  planning input; phase stays in the graph for later schedule and lifetime consumers.

## Known limitations

- No public workflow currently invokes capability, eligibility, owner selection, partitioning,
  and logical memory planning end to end. The generator is internal.
- A requirement identifies producer/consumer partitions and graph-output preservation, but it
  does not choose aliasing, copying, transfer direction, recomputation, concrete representation,
  or publication target.
- Cross-owner classification follows backend identity equality only. It does not select devices
  or estimate transfer cost.
- The plan contains no element count, byte count, lifetime interval, use count, partition ID,
  slot, alignment, offset, allocation, reuse/coloring, workspace, or physical storage.
- `graphOutput` does not identify a public tensor or publication policy. A compiler-owned future
  `PublicationPlan` must validate and own `PublicationBinding` values separately.
- Phase metadata is not repeated. Consumers that need forward/backward classification must use
  the associated immutable graph; this task makes no runtime schedule promise.
- No serialization or external compatibility guarantee is established.

## Validation evidence

- Planning context `/root/plan_planning_0005` read `AGENTS.md`, the authoritative architecture
  contract, focused overview/lifecycle/module/dependency/partition/memory/prepare/backend/
  performance/training documents, applicable ADRs, documentation rules and General/Planning
  profiles, planning guide and roadmap, planning/config/backend-contract/trace/compiler/prepare/
  runtime master plans, completed Planning tasks 0001–0004, current model graph/descriptor/
  layout/publication source and tests, current planning source/tests/Javadocs, and directly
  relevant public/compile APIs, glossary, user guides, and backend guides.
- Frontier reassessment found no architecture conflict. Planning owns logical materialization and
  memory immediately after task-0004 partitioning, and the closed graph plus ordered partition
  recipes provide all required immutable facts without compiler, prepare, runtime, or backend
  behavior.
- `PublicationBinding` was deliberately excluded: it is not part of `CompiledGraphModel` and
  cannot prove graph membership by itself; the compiler owns its future `PublicationPlan` context.
  Graph outputs are sufficient for the current logical preservation obligation.
- Descriptor evidence supports retaining `TensorDescriptor`: `Shape.knownElementCount()` is empty
  for dynamic/expression shapes, `DataType.byteWidth()` is logical width only, and
  `LayoutDescriptor` geometry does not select physical representation or materialization.
- Planning-stage `python3 /tmp/validate_synaptik_markdown.py` passed for 229 Markdown files, 4,092
  local links, 248 local anchors, 2,888 fence markers, final newlines, and trailing whitespace.
- Planning-stage `git diff --check` passed with no output. The complete tracked/untracked diff
  contains exactly five Markdown paths: this task, the planning/config/trace master plans, and the
  roadmap. No Java, test, Gradle, `ARCHITECTURE.md`, ADR, architecture-test, API, guide, glossary,
  compiler, prepare, runtime, backend, or other module path changed during planning.
- Status checks found exactly one Ready detailed task heading and exactly one Ready master-plan
  row: Planning 0005. Planning 0001–0004 and Config 0001–0003 remain Complete; Planning 0006,
  Config 0004+, Trace 0003+, and all Compiler/Prepare/Runtime work remain Draft. No detailed
  Planning 0006, Config 0004, Trace 0003, Compiler, Prepare, or Runtime specification exists.
- The final exact audit reported `changed_paths=5`, `ready_headings=1`, `ready_rows=1`, and
  `forbidden_changed_paths=0`. Every changed path is Markdown, and no Java, test, Gradle,
  `ARCHITECTURE.md`, ADR, or `testing/` path is present.
- Manual review confirmed all canonical task sections, exact package/type/method shapes, public
  DTO justification, partition validation/failure ordering and messages, graph-value and
  partition ordering/reference rules, classification matrix, descriptor-retention rationale,
  `PublicationBinding` exclusion, exact eighteen-path implementation cap, standalone clean-
  context prompt, documentation handoff, and required exclusions. Completed task history remains
  unchanged.
- Implementation context `/root/implement_planning_0005` ran the focused command for
  `LogicalMemoryRequirementTest`, `LogicalMemoryPlanTest`, and `LogicalMemoryPlanningTest`: 3
  suites, 14 tests, 0 failures, 0 errors, and 0 skips (4 + 4 + 6). It then ran the single final
  `./gradlew :modules:planning:test`: 8 suites, 63 tests, 0 failures, 0 errors, and 0 skips.
- Documentation context `/root/implement_planning_0005/planning_0005_docs` reused those successful
  Java results because it changed Javadoc and Markdown only; executable Java bodies and tests did
  not change after the final module suite.
- Documentation-context `./gradlew :modules:planning:javadoc` passed with `BUILD SUCCESSFUL`; two
  tasks executed and four were up to date. Generated `memory/package-summary.html`,
  `LogicalMemoryRequirement.html`, and `LogicalMemoryPlan.html` render the logical/physical and
  standalone/generated boundaries. `allclasses-index.html` and `overview-tree.html` contain both
  public records, while package-private `LogicalMemoryPlanning` has no generated page or index
  entry.
- Documentation-context `python3 /tmp/validate_synaptik_markdown.py` passed for 229 Markdown files,
  4,094 local links, 251 local anchors, 2,892 fence markers, final newlines, and trailing
  whitespace.
- The final tracked/untracked audit reports exactly the authorized 18 paths: 4 production, 3 test,
  6 explanatory, and 5 planning paths. `modules/planning/build.gradle.kts` is unchanged; source
  scans found no runtime, prepare, engine, concrete-backend, publication, physical-memory, device,
  route, kernel, or executable import/surface.
- Final status and later-spec checks report zero Ready headings or rows. Planning 0001–0005 are
  Complete, Planning 0006 and later work remain Draft, and no detailed Planning 0006, Config 0004,
  Trace 0003, Compiler, Prepare, or Runtime task specification exists.
- Final `git diff --check`, exact-scope, generated-page, final-newline, trailing-whitespace, and
  `git status --short` checks passed. No architecture or repository-wide Java suite was run because
  the task changed no dependency, architecture rule, shared build, backend behavior, or cross-
  module executable behavior.

## Implementation notes

- Implementation context `/root/implement_planning_0005` added exactly the two public records,
  one package-private stateless generator, package documentation, and three focused tests under
  `io.github.pho001.synaptik.planning.memory`.
- `LogicalMemoryPlanning.plan` validates all partition null, membership, duplicate, coverage,
  graph-order, and adjacent-owner conditions before it derives the first requirement. It uses the
  closed graph's node inputs and outputs without changing or persisting model indexes.
- Generated requirements follow graph-value order, retain exact graph identity/descriptor and
  supplied partition-element references, deduplicate consumers by supplied partition membership,
  and order consumers by the partition list. Graph outputs come only from `graph.outputs()`.
- Tests cover exact public/internal shapes, validation order and messages, snapshots and reference
  retention, graph inputs/outputs, same-owner and cross-owner boundaries, fan-out, merge, repeated
  inputs, unused values, multi-output nodes, forward/backward uses, dynamic/expression
  descriptors, and zero-node pass-through graphs.
- Documentation context `/root/implement_planning_0005/planning_0005_docs` independently reviewed
  the final implementation and tests, finalized all four production Javadocs and the six
  explanatory documents, updated glossary terminology, and synchronized this task, the planning/
  config/trace master plans, and the roadmap. It changed no executable Java behavior or tests.
- No model Java or `PublicationBinding` change was needed: `CompiledGraphModel` already supplies
  the closed values, nodes, and output boundary, while standalone publication bindings lack the
  owning compiler context required for a `PublicationPlan`.
- No completed planning capability/partition Java, config/backend-contract/trace Java, or
  planning Gradle/dependency change was needed. The new signatures use the planning module's
  existing public model and backend-contract edges, and the task neither invokes nor widens the
  earlier internal planning steps.
- No architecture contract, ADR, architecture-test, compiler, prepare, runtime, engine, concrete
  backend, backend-conformance, integration-test, or other-module change was needed. The task
  realizes the existing backend-neutral logical-memory stage without changing dependencies or
  executable lifecycle behavior.

## Completion summary

- Completed changes: Added immutable per-value logical memory requirements, the ordered logical
  memory plan, and internal closed-graph/partition derivation with complete validation and focused
  coverage.
- Files changed or created: Exactly the authorized eighteen paths—four planning-memory production
  files, three planning-memory tests, six explanatory documentation paths, this task, the planning/
  config/trace master plans, and the roadmap.
- Tests and validation: The implementation context passed the three focused suites with 14 tests
  and the single final planning suite with 63 tests across eight suites; the documentation context
  passed planning Javadoc, repository Markdown, generated-page, exact-scope, status, later-spec,
  dependency/import/surface, newline, whitespace, and `git diff --check` validation without
  rerunning Java tests.
- Documentation-agent review: Clean-context documentation review completed in
  `/root/implement_planning_0005/planning_0005_docs`; it changed no executable behavior.
- Documentation impact: Finalized current-versus-planned logical-memory boundaries in partition
  scoring, memory strategy, public/compile API, backend selection, glossary, and planning records.
- Javadoc review: Finalized the two public record contracts, package-private generator contract,
  and package mental model, including standalone-versus-generated output semantics and the
  physical-memory boundary.
- Glossary impact: Updated the planning status convention, logical memory plan, partition,
  partition-scoring configuration, and planning definitions for the current recipes and internal
  derivation.
- Unresolved issues: None.
- Follow-up required: None within this task; Planning 0006 remains Draft and requires a separate
  frontier reassessment and planning step before becoming actionable.

Status: Complete
