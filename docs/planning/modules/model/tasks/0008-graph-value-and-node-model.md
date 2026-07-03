# Task 0008: Graph Value and Node Model

## Status

Complete

## Goal

Define the minimal immutable, backend-independent `GraphValue` and `CompiledNode` records that
separate logical data from computation occurrences. A graph value combines a `ValueId` with its
`TensorDescriptor`; a compiled node combines a `NodeId`, an `Operation`, and ordered immutable
input and output `ValueId` lists. This task validates only facts owned by each record and leaves
whole-graph relationships to task 0009.

## Scope

- Add one public `GraphValue` record in the existing `model.graph` package.
- Store exactly `ValueId id` and `TensorDescriptor descriptor`, in that order.
- Add one public `CompiledNode` record in the same package.
- Store exactly `NodeId id`, `Operation operation`, `List<ValueId> inputs`, and
  `List<ValueId> outputs`, in that order.
- Require every component reference and every list element to be non-null.
- Snapshot input and output lists with `List.copyOf` while preserving their order.
- Permit empty input lists so a node can represent a zero-input semantic source operation.
- Permit repeated input IDs because two input positions may intentionally read the same value.
- Require at least one output and reject a repeated output ID within one node.
- Use record-generated structural equality, hashing, and diagnostic text over the complete state.
- Add one focused unit-test class per production record.
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  implementation roadmap through the required clean-context documentation pass.

## Out of scope

- `CompiledGraphModel`, a graph container, graph builder, graph validator, graph indexes, or ID
  allocation
- producer or consumer fields on `GraphValue`, producer maps, use lists, graph inputs, or graph
  outputs as stored record roles
- graph-wide existence, uniqueness, producer, coverage, topology, cycle, or ordering validation
- descriptor matching between values and operations, operation arity, result count, family
  compatibility, shape inference, data type inference, layout inference, or symbolic binding
- `GraphPhase`, `PublicationBinding`, publication state, public `Tensor`, `TensorId`, labels,
  provenance, host storage, or physical memory facts
- `OperationId`, operation registries, concrete operation kinds, or family-specific attributes
- nullable operations, input-node sentinels, leaf nodes, or a node subtype for graph inputs
- compiler capture or transformations, planning ownership, logical memory planning, prepare,
  runtime, tracing, backend support, lowering, kernels, execution, or device state
- factories, builders, helper types, mutable collection views, parsing, persistence, or
  serialization contracts
- changes to completed identifier, operation, or tensor-descriptor contracts
- Gradle, dependency, capability-baseline, architecture-contract, focused-architecture, other
  module, or unrelated documentation changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the immutable graph model,
  `modules/model` ownership, and runtime hot-path exclusions
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md), especially identifier and operation foundations
- [Model master plan](../master-plan.md), especially `model.graph` ownership and package direction
- [Task 0004](0004-typed-identifiers.md), which defines `NodeId` and `ValueId`
- [Task 0006](0006-operation-model.md), which defines `Operation`
- [Task 0007](0007-tensor-descriptor-model.md), which defines `TensorDescriptor`
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md), which explain
  the current node-versus-value vocabulary

## Legacy evidence

The read-only `legacy/pre-rewrite` branch provides capability evidence through
`tensor.TensorNode`, `graph.model.CompiledNode`, `graph.compile.CompiledNodeSnapshotter`,
`planning.value.GraphValueRef`, `planning.value.GraphValueKind`, and tests that consume their
topology.

Legacy `TensorNode` was mutable and retained an `Operation`, public `Tensor` inputs, a gradient
rule, and a backward flag. Legacy `CompiledNode` made an immutable snapshot, but its nineteen
fields mixed a raw node ID and ordered raw input IDs with an `Operation`, backend ownership,
storage ownership, shape and stride arrays, data type, autograd and leaf flags, label, flat storage
size, and a static-data snapshot. `CompiledNodeSnapshotter` assigned IDs from topological position,
represented leaf tensors as nodes with a null operation, and copied tensor, storage, backend-intent,
and training facts into the same object.

Legacy `GraphValueRef` had only a `NODE` kind and identified a value by its producer's raw node ID.
Planning and memory tests used these references to carry data between partitions, while descriptor
tests relied on ordered node input IDs and empty input lists for leaf tensors. This establishes the
useful capability that immutable compile-time nodes apply operation semantics to ordered logical
inputs and expose logical results. It does not establish that node identity and value identity are
the same, that a node has exactly one output, or that planning and storage facts belong in the
graph model.

The new design preserves only the dataflow capability. It does not copy mutable `Tensor`
references, nullable-operation leaf sentinels, gradient rules, backend ownership, storage owners,
layout or descriptor copies on nodes, labels, static data, raw integer identity, topological ID
allocation, or the one-node/one-value assumption. Graph inputs are standalone `GraphValue`
instances without a producing node; producer relationships are derived later from node outputs.

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- Both records live in `io.github.pho001.synaptik.model.graph`, which already owns `NodeId` and
  `ValueId`.
- Package direction is `model.graph -> model.operation` and `model.graph -> model.tensor`.
  `model.operation` and `model.tensor` must not depend on compiled graph state.
- Production code uses only the completed model contracts and JDK collection utilities. It adds no
  project-module dependency.
- A graph value represents logical data and owns only its graph-local identity and tensor
  description. It does not store a producer because graph inputs have no producer and node outputs
  already provide the producer relationship.
- A compiled node represents computation only. Its operation is always non-null; a graph input is
  a value without a producer, not a node with a null operation or special input operation.
- Node input and output positions are semantically ordered. Defensive snapshots preserve order.
- Empty inputs are valid because backend-independent constant or compiler-generated semantic
  source operations may have no input values. This task does not introduce such an operation kind.
- Repeated input IDs are valid because separate input positions may intentionally reference the
  same logical value, as in `x op x`.
- Every node has at least one output, matching the graph vocabulary that a computation occurrence
  produces one or more values. A repeated output ID within the same node does not describe two
  logical values and is rejected locally.
- The records do not determine whether referenced values exist, whether another node produces an
  output ID, or whether input and output relationships form a valid graph. Task 0009 owns those
  graph-wide checks.
- `NodeId` and `ValueId` remain graph-local and type-separated. The same numeric value is permitted
  in the two domains and in different owning graphs. No `OperationId` is introduced.
- Runtime hot paths must not consume `CompiledNode`; this type remains immutable compile-time
  model state.
- If implementation requires a producer field, graph container, new identity, operation family,
  descriptor copy, helper type, dependency, or architecture change, stop and report the issue.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.graph` — owns graph-local identities and gains the two immutable
  graph element records.
- `io.github.pho001.synaptik.model.operation` — owns the immutable backend-independent `Operation`
  stored by a node.
- `io.github.pho001.synaptik.model.tensor` — owns the immutable `TensorDescriptor` stored by a
  graph value.

Packages added or changed:

- No package is added. The existing `model.graph` package gains two public records.

Type placement:

- `io.github.pho001.synaptik.model.graph.GraphValue` — identifies and describes one logical value
  independently of whether a node produces it.
- `io.github.pho001.synaptik.model.graph.CompiledNode` — identifies one computation occurrence and
  records its operation plus ordered logical inputs and outputs.

Test placement:

- `io.github.pho001.synaptik.model.graph.GraphValueTest` — validates the value record's exact
  state, ownership, failures, and value semantics.
- `io.github.pho001.synaptik.model.graph.CompiledNodeTest` — validates the node record's exact
  state, list invariants, ownership, failures, and value semantics with test-local operation kinds.

## Required contracts

### `GraphValue`

Implement exactly this public record shape:

```java
public record GraphValue(ValueId id, TensorDescriptor descriptor) { ... }
```

The canonical constructor uses `Objects.requireNonNull` with the component-specific messages
`id` and `descriptor`. It stores the exact immutable component references. Explicit `id()` and
`descriptor()` accessors return those same non-null objects.

Do not add a producer `NodeId`, optional producer, consumer IDs, publication role, storage role,
label, tensor identity, or graph-input flag. A graph input and a produced value have the same local
record shape. Task 0009 can derive a producer index from node output lists and can identify inputs
as values absent from that index without duplicating the relationship in `GraphValue`.

Do not override `equals`, `hashCode`, or `toString`. Record-generated equality and hashing use the
typed ID and descriptor values. Diagnostic text exposes both component names and representative
values, but it is not a serialization, parsing, allocation, or global identity contract.

### `CompiledNode`

Implement exactly this public record shape:

```java
public record CompiledNode(
        NodeId id,
        Operation operation,
        List<ValueId> inputs,
        List<ValueId> outputs) { ... }
```

The canonical constructor must:

1. reject null `id`, `operation`, `inputs`, or `outputs` references with
   `NullPointerException` and the exact component name as the message;
2. inspect input positions in encounter order and reject a null input with
   `NullPointerException` and the message `inputs[<index>]`;
3. snapshot inputs with `List.copyOf`, preserving order, an empty list, and repeated IDs;
4. reject an empty output list with `IllegalArgumentException` and the message
   `outputs must not be empty`;
5. inspect output positions in encounter order and reject a null output with
   `NullPointerException` and the message `outputs[<index>]`;
6. reject the first repeated output ID with `IllegalArgumentException` and the exact message
   `outputs[<index>] duplicates <ValueId>`, where `<index>` is the later position and `<ValueId>` is
   the duplicate ID's diagnostic text; and
7. snapshot outputs with `List.copyOf`, preserving their validated order.

The record performs no cross-list or graph lookup. In particular, it does not reject an input ID
that also appears as an output, an ID absent from a future graph value collection, or an output ID
that another future node also claims. Those conditions require the owning graph context and belong
to task 0009.

Explicit accessors document and return the stored non-null ID and operation references and the
immutable list snapshots. The list containers have value semantics: implementation and tests must
not promise or assert identity with caller-supplied lists or with another equal list. Their
contained `ValueId` elements are immutable typed values, so no element copying or conversion is
needed.

Do not override `equals`, `hashCode`, or `toString`. Record-generated equality and hashing include
all four components structurally, including list order and duplicate input positions. Diagnostic
text exposes the component names and representative values but is not a serialization, parser,
execution-dispatch, or graph-validation contract.

## Local versus graph-wide validation

Task 0008 validates only facts visible inside one record:

| Local record validation | Deferred to task 0009 or later |
|---|---|
| non-null record components | existence of every referenced input and output value |
| non-null list elements with indexed messages | uniqueness of `NodeId` values within an owning graph |
| immutable ordered list snapshots | uniqueness of `ValueId` entries within an owning graph |
| empty inputs allowed | exactly one producer for every produced value across all nodes |
| at least one output | graph-input classification from absence of a producer |
| repeated inputs preserved | graph-output declaration and coverage |
| output IDs unique within one node | topological order, self-dependencies, and cycles |
| typed `NodeId`/`ValueId` components | descriptor agreement with operation results |
| record structural value semantics | operation arity, result count, and kind/attribute family compatibility |

The deferred column is not permission to omit those checks from the future graph-container task.
It identifies why this task cannot validate them without inventing a container, indexes, operation
families, or compiler behavior.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/GraphValueTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/CompiledNodeTest.java`

Expected documentation and planning updates during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

`docs/api/compile-api.md` remains conceptual and is not expected to change: this task adds current
model values but no compiler entry point, compile artifact, graph container, or compiler behavior.
The documentation-focused pass must review that conclusion and stop if the implementation creates
an unexpected compile-API impact instead of editing both API pages without a demonstrated need.

## Maximum scope

This task may create or modify at most:

- two production Java files;
- two focused test Java files; and
- the five documentation and planning files listed above.

The four Java source/test files are within the planning guide's normal three-to-eight-source-file
task size.

Do not add production helpers, factories, builders, validators, indexes, or another source file.
Do not modify existing Java source or tests, Gradle files, `ARCHITECTURE.md`, focused architecture
documentation, the capability baseline, `docs/api/compile-api.md`, another module, or unrelated
documentation. If another file or type is required, stop and propose a separately reviewed
follow-up instead of expanding this task.

## Javadoc requirements

- Every public record, canonical constructor, record component, and explicit accessor must have
  meaningful detailed Javadoc; generated component documentation alone is insufficient.
- `GraphValue` Javadoc must define logical graph data, distinguish values from nodes, public
  `Tensor`, and physical storage, and explain why producer/consumer relationships are absent.
- `GraphValue` constructor and accessors must document non-nullness, exact immutable-reference
  retention, graph-local identity scope, descriptor meaning, results, and null failures.
- `CompiledNode` Javadoc must define one computation occurrence, distinguish it from an
  `Operation` and from the data it consumes and produces, and state compile-time and runtime
  boundaries.
- `CompiledNode` constructor Javadoc must document all inputs, list order, defensive snapshots,
  list-container value semantics, permitted empty and repeated inputs, required non-empty unique
  outputs, every null failure, and output validation failures.
- List accessors must document non-null immutable snapshots, order, allowed duplicate inputs,
  unique outputs, and `@return` semantics without promising list-container identity.
- Type Javadocs must explain record-generated equality and hashing and classify `toString()` as
  diagnostic non-serialization text.
- Javadocs must identify graph-wide validations as deferred and must not promise graph existence,
  global uniqueness, topology, descriptor matching, arity compatibility, execution, or backend
  behavior.

## Acceptance criteria

- Exactly `GraphValue` and `CompiledNode` are introduced as public records in `model.graph`.
- `GraphValue` has exactly `ValueId id` and `TensorDescriptor descriptor` components, in that
  order, with no additional instance state.
- `CompiledNode` has exactly `NodeId id`, `Operation operation`, `List<ValueId> inputs`, and
  `List<ValueId> outputs` components, in that order, with no additional instance state.
- `GraphValue` rejects each null component with a component-specific `NullPointerException` and
  retains both immutable references unchanged.
- `GraphValue` contains no producer, consumers, role, label, tensor, storage, publication, or
  runtime fact. A value can be constructed without any node.
- `CompiledNode` rejects null scalar components, list references, and list elements with the
  specified component or indexed messages.
- Caller mutation after construction cannot change either stored list, and mutation through either
  accessor fails with `UnsupportedOperationException`.
- Input and output order is preserved exactly. List equality, not list-container identity, is the
  public ownership contract.
- Empty inputs construct successfully; empty outputs fail with the specified
  `IllegalArgumentException`.
- Repeated input IDs construct successfully and remain in their original positions.
- Single and multiple distinct outputs construct successfully. A repeated output ID in one node
  fails and the exception identifies its later index and duplicate ID.
- Node construction performs no value existence, cross-node producer, topology, graph-output,
  descriptor, arity, result-count, or operation-family validation.
- Equal component values produce equal records and equal hash codes. Changed components or changed
  list order produce unequal records where applicable.
- Diagnostic text identifies each record and all component names with representative values
  without becoming an exact wire-format assertion.
- Reflection tests verify both exact record shapes, component order and types, and absence of
  additional instance fields.
- Tests use only test-local operation kinds and attributes; no production operation family is
  introduced.
- No `OperationId`, graph container, `GraphPhase`, `PublicationBinding`, `CompiledGraphModel`,
  graph builder, graph validator, helper, factory, or index is introduced.
- Production imports are limited to the completed model contracts and JDK `Objects`, `List`, and a
  collection used only for local duplicate-output detection.
- All Javadoc requirements are satisfied and generated model Javadoc includes both records,
  canonical constructors, components, and explicit accessors.
- A separate documentation-focused agent or thread with clean context independently reviews the
  final implementation and tests, finalizes Javadocs, updates the Tensor API and glossary, and
  records terminology, link, example, status, and formatting evidence in the same overall change.
- Task, master-plan row/current status/notes, and roadmap frontier/table have matching final status.
- No existing Java/test contract, Gradle file, architecture document, capability baseline,
  compile-API page, other module, or unrelated documentation is changed.

## Tests / validation

Run after implementation and again after the documentation-focused pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.graph.GraphValueTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.graph.CompiledNodeTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the final diff contains only two new production files, two new test files, and the five allowed
  documentation/planning files;
- reflection reports the exact record components, order, types, and no additional instance fields;
- constructor inspection shows only component null checks, indexed element checks, list snapshots,
  non-empty outputs, and within-node duplicate-output detection;
- input and output accessors expose immutable ordered snapshots and tests never assert list
  container identity;
- tests cover a producer-free `GraphValue`, empty and repeated inputs, single and multiple outputs,
  empty outputs, duplicate outputs, indexed nulls, caller mutation, accessor mutation, equality,
  hashing, list-order sensitivity, and diagnostics;
- `javap -p -c` confirms exact fields, defensive `List.copyOf` snapshots, the specified local
  validations, direct scalar accessors, and record-generated equality, hashing, and text;
- a production import scan finds no public `Tensor`, `TensorId`, storage, compiler, planning,
  runtime, prepare, backend, trace, or concrete operation-family dependency;
- no producer or consumer field, nullable operation sentinel, ID allocator, global uniqueness
  claim, serialization contract, graph-wide lookup, graph phase, publication binding, or graph
  container appears;
- generated Javadoc documents every public record, constructor, component/accessor, ownership
  rule, failure, value semantic, and cross-layer boundary;
- `docs/api/tensor-api.md` presents graph values and nodes as current model contracts, defines their
  relationship to identifiers, descriptors, and operations, and does not claim a graph container,
  compiler behavior, or execution support;
- `docs/glossary.md` marks graph value and node records implemented and preserves the distinctions
  among Tensor, graph value, node, operation, and physical memory slot;
- `docs/api/compile-api.md` remains unchanged unless the task is stopped and replanned with a
  demonstrated compile-API impact;
- local Markdown links and anchors in all five changed documentation/planning files resolve, code
  fences are balanced, and no changed file has trailing whitespace;
- the separate documentation context follows
  `docs/developer-guide/documentation-rules.md`, applies General style plus API and Javadoc style to
  API/Javadoc work and Planning style to planning updates, and records its identity, inspected
  source/tests/diff, commands, results, limitations, Javadoc review, and glossary impact; and
- package direction remains `model.graph -> operation/tensor` with no forbidden dependency.

## Dependencies

- Task 0004 is complete and provides `NodeId` and `ValueId`.
- Task 0006 is complete and provides `Operation`.
- Task 0007 is complete and provides `TensorDescriptor`.
- The `model.graph` package ownership and dependency direction are defined by the model master
  plan.

## Follow-up tasks

- Task 0009 will define the immutable graph container and owns graph-wide value/node existence,
  uniqueness, producer, input/output, coverage, topology, cycle, and ordering validation.
- `GraphPhase`, `PublicationBinding`, and other container-adjacent contracts remain for task 0009
  or later according to their future focused evidence; this task does not decide their record
  shapes or validation.
- Later operation-family tasks own arity, result-count, and kind-to-attributes compatibility
  contracts.
- Later compiler tasks own graph capture, ID allocation, inference, transformations, and
  compilation validation.

Do not create a detailed task-0009 specification as part of this task.

## Architecture impact

Expected impact: None.

The architecture already assigns `GraphValue`, `CompiledNode`, immutable graph state, `NodeId`,
`ValueId`, `Operation`, and `TensorDescriptor` to `modules/model`. This task adds two values in the
planned package and follows the existing package direction. It changes no module boundary,
dependency rule, lifecycle contract, or backend contract, so architecture documentation and
architecture tests require no update. If implementation reveals otherwise, stop and report the
conflicting rule and required decision before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read first and in full:
- AGENTS.md
- ARCHITECTURE.md
- docs/developer-guide/documentation-rules.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0004-typed-identifiers.md
- docs/planning/modules/model/tasks/0006-operation-model.md
- docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md
- docs/planning/modules/model/tasks/0008-graph-value-and-node-model.md
- docs/api/tensor-api.md
- docs/api/compile-api.md
- docs/glossary.md
- the relevant completed graph-identifier, operation, and TensorDescriptor production/test files

Implement task 0008 exactly as specified. Create only GraphValue.java, CompiledNode.java,
GraphValueTest.java, and CompiledNodeTest.java for code and tests.

GraphValue must be a public record with exactly ValueId id and TensorDescriptor descriptor. It
must reject null components, retain the exact immutable references, and contain no producer,
consumer, role, Tensor, storage, publication, or runtime fact.

CompiledNode must be a public record with exactly NodeId id, Operation operation,
List<ValueId> inputs, and List<ValueId> outputs. Reject null components and indexed null list
elements with the task's exact messages. Snapshot both lists with List.copyOf and preserve order.
Allow empty and repeated inputs. Require at least one output and reject a repeated output ID within
the same node. Do not perform graph-wide or operation-family validation. Use only test-local
operation kinds and attributes.

Do not add helpers, factories, builders, validators, indexes, ID allocation, OperationId,
GraphPhase, PublicationBinding, CompiledGraphModel, a graph container, public Tensor or storage
facts, labels, backend/planning/compiler/runtime facts, concrete operation families, dependencies,
or unrelated refactors. Do not modify existing Java contracts/tests, Gradle, ARCHITECTURE.md,
focused architecture docs, capabilities.md, docs/api/compile-api.md, another module, or unrelated
documentation. Stop and report if a required change exceeds the affected-file or maximum-scope
list, if a graph-wide check appears necessary to construct either local record, or if architecture
uncertainty appears.

Add every Javadoc contract required by the task. Run every validation command and manual check.

After code implementation and initial validation, hand the resulting diff to a separate
documentation-focused agent or thread with a clean context in the same overall change. The handoff
must include this task, the implementation diff, affected graph-model API behavior, architecture
constraints, required Tensor API and glossary updates, Javadoc requirements, the reason the
conceptual compile API is expected to remain unchanged, and all validation commands. That agent
must read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md, General style,
API and Javadoc style, Planning style, this task, final source/tests, docs/api/tensor-api.md,
docs/api/compile-api.md, and docs/glossary.md. It must inspect source and tests rather than rely on
the handoff summary, independently finalize both new records' Javadocs, the Tensor API explanation,
glossary/status terminology, planning evidence/status, links, anchors, examples, and formatting,
and record a reasoned no-change conclusion for the compile API or stop on unexpected impact.

At the end, update only this task file, the model master plan, and the roadmap for planning status.
Record local decisions, known limitations, exact validation evidence including the documentation
agent identity and results, implementation notes, and the canonical completion summary. Do not
mark task 0008 Complete until all acceptance criteria, both validation passes, documentation
changes, and status synchronization are complete. Do not create a task-0009 specification.
```

## Local decisions

- `GraphValue` stores only `id` and `descriptor`. A producer component would duplicate the
  relationship already expressed by node output lists and would need an absence state for graph
  inputs; task 0009 can derive that relationship once per graph.
- `CompiledNode` stores output IDs explicitly instead of assuming that node and value identities
  coincide. This preserves typed separation and supports multiple results without copying
  descriptors onto the node.
- Graph inputs are values without producers. Nodes always represent computation and therefore
  always carry a non-null `Operation`; no leaf/input sentinel is added.
- Empty inputs are accepted for future zero-input semantic source operations. Empty outputs are
  rejected because a computation node with no logical result has no established side-effect model
  in the current architecture.
- Repeated inputs are positional and valid. Repeated outputs within one node are locally invalid
  because they do not identify distinct logical results; duplicate producers across different
  nodes require a graph context and remain deferred.
- `List.copyOf` provides immutable ordered snapshots. List-container identity is deliberately not
  part of the API, while contained `ValueId` values need no copying because they are immutable.
- Record-generated equality, hashing, and text cover complete declared state. No custom identity,
  serialization, global uniqueness, or `OperationId` contract is added.

## Known limitations

- The records do not allocate IDs or prove uniqueness within a graph. Same-typed equal IDs can be
  reused in another owning graph, and equal numeric values across `NodeId` and `ValueId` remain
  separate typed domains.
- A `GraphValue` alone does not report whether it is an input, intermediate, or graph output and
  does not expose its producer or consumers.
- A `CompiledNode` alone cannot prove that referenced values exist, that outputs have exactly one
  producer across nodes, or that the relationships are acyclic and topologically ordered.
- The node does not validate operation arity, result count, descriptor compatibility, or
  kind-to-attributes compatibility because no production operation-family contracts exist yet.
- The records provide compile-time model state only. They contain no compiler behavior, planning
  decisions, storage, prepared execution, runtime state, or backend support.

## Validation evidence

- Planning context read the complete architecture contract, current architecture index,
  documentation workflow, profile index, General style, Planning style, API and Javadoc style,
  planning guide, roadmap, model capability baseline, model master plan, tasks 0004–0007, Tensor
  API, compile API, glossary, and relevant current graph-identifier, operation, and
  tensor-descriptor production/tests before defining the contract.
- Read-only legacy inspection used `git ls-tree`, `git grep`, and
  `git show legacy/pre-rewrite:<path>` for `TensorNode`, `CompiledNode`,
  `CompiledNodeSnapshotter`, `GraphValueRef`, `GraphValueKind`, descriptor/topology consumers, and
  source-boundary tests. The branch was not checked out or modified, and no legacy source or
  package structure was copied.
- `git status --short` and manual scope review confirmed exactly three planning-file changes: this
  new task, the model master plan, and the roadmap. No Java, test, Gradle, `ARCHITECTURE.md`, focused
  architecture, capability-baseline, API, glossary, other-module, or unrelated documentation file
  changed during planning.
- A targeted Ruby path-and-heading check resolved all 50 local Markdown links and anchors in the
  three changed planning documents.
- Fence inspection reported even counts in every changed document: eight backtick-fence markers in
  this task and two each in the master plan and roadmap. `rg -n '[[:blank:]]+$'` returned no
  trailing-whitespace matches.
- `git diff --check` passed for tracked changes. `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0008-graph-value-and-node-model.md` produced no whitespace
  diagnostic; its exit status was the expected `1` because the new file differs from `/dev/null`.
- Status review confirmed task 0007 remains `Complete` and task 0008 is `Ready` in this task, the
  master-plan row/current status/notes, and the roadmap frontier/table. Task order and dependencies
  are unchanged, task 0009 remains `Draft`, and no task-0009 specification exists.
- Java tests and Javadoc were not run during this planning-only change because no Java, test,
  dependency, or build file changed. The implementation task requires both focused tests, all model
  tests, model Javadoc, and the complete repository test lifecycle before and after the separate
  documentation pass.
- The implementation phase added only `GraphValue.java`, `CompiledNode.java`,
  `GraphValueTest.java`, and `CompiledNodeTest.java`. Independent source and test inspection
  confirmed the exact two- and four-component record shapes, direct immutable scalar-reference
  retention, the required local validation and messages, ordered `List.copyOf` snapshots, permitted
  empty/repeated inputs, non-empty within-node-unique outputs, and record-generated equality,
  hashing, and diagnostic text. No producer field, graph container, compiler behavior, execution
  state, backend fact, or production operation family was introduced.
- Clean documentation-focused context `/root/review_model_0008_docs` applied General style, API and
  Javadoc style, Planning style, and Example format. It read the complete architecture contract,
  current architecture index, documentation rules and profile index, applicable profiles, planning
  guide, roadmap, capability baseline, model master plan, tasks 0004/0006/0007/0008, Tensor API,
  compile API, glossary, relevant production/tests, generated Javadoc, test reports, and actual
  workspace diff before finalizing documentation.
- The documentation pass changed only Javadoc in the two new production records. It added exact
  caller-visible failure messages, explicit `List.copyOf` and unmodifiable-list behavior, producer
  absence, graph-local scope, local-versus-graph-wide validation, record value/diagnostic semantics,
  and model/compiler/storage/prepare/runtime/backend boundaries. Imports, record declarations,
  constructor logic, accessors, and both new tests remained unchanged.
- Existing `NodeId`, `ValueId`, `Operation`, and `TensorDescriptor` Javadocs were reviewed without
  modification. `NodeId` already distinguishes graph-local computation occurrence from operation
  semantics and value identity; `ValueId` already covers producer-free and multi-output data
  identity plus the memory-slot boundary; `Operation` already separates semantics from node
  occurrence and executable layers; and `TensorDescriptor` already defines immutable logical
  description, ownership, and storage/compiler/runtime/backend exclusions. Task 0008 composes but
  does not change those contracts.
- The Tensor API now presents `GraphValue` and `CompiledNode` as current model contracts, gives a
  newcomer-oriented value-versus-node mental model, documents list ownership and local invariants,
  and includes a complete test-local example connecting a descriptor, two values, an operation,
  and one node. `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-graph-doc-example /tmp/GraphElementExample.java` passed, and `java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-graph-doc-example GraphElementExample`
  printed `IDENTITY`, `[0]`, `[1]`, and `true` on separate lines.
- The glossary now marks both graph-element records implemented, identifies their exact local
  roles, and keeps public `Tensor`, logical graph value, computation node, operation semantics, and
  physical memory slot distinct. Producer derivation and whole-graph validation remain explicitly
  planned for the future graph container.
- `docs/api/compile-api.md` was reviewed and remains unchanged. Task 0008 adds model values only;
  it adds no compiler entry point, graph container, compile artifact, lifecycle behavior, or
  callable compile contract. Its SHA-1 remained
  `9b57bc27d30184199eda0f79bdd7c1015326fa36` throughout the documentation pass.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.graph.GraphValueTest` — `BUILD SUCCESSFUL`; 5 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.graph.CompiledNodeTest` — `BUILD SUCCESSFUL`; 10 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 105 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated output includes both records,
  canonical constructors, explicit accessors, exact failure contracts, immutable list ownership,
  record semantics, and cross-layer exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL` for the complete repository; 36 actionable tasks were
  reported in the final run.
- `javap -classpath modules/model/build/classes/java/main -p -c` for both records confirmed the
  exact fields and accessors, only the specified local constructor checks, direct scalar
  assignments, both `List.copyOf` calls, duplicate-output detection, and generated record methods.
  Production imports remain limited to `TensorDescriptor` or `Operation` plus `Objects`, `List`,
  and `HashSet`; package direction remains `model.graph -> model.tensor/model.operation`.
- A targeted local Markdown path-and-heading check resolved all 105 links and anchors in the five
  changed documentation/planning files. Code fences are balanced, trailing-whitespace checks found
  no matches, and current-versus-planned terminology agrees across Javadoc, API, glossary, task,
  master plan, and roadmap.
- Hash comparison confirmed `GraphValueTest.java`, `CompiledNodeTest.java`, the capability
  baseline, and compile API were unchanged by the documentation pass. Final scope review confirmed
  exactly nine changed or new files: the four new production/test files plus Tensor API, glossary,
  this task, model master plan, and roadmap. No task-0009 specification was created.
- `git diff --check` passed after documentation and planning synchronization.
- The final coordinating context independently reran both focused suites, all model tests, model
  Javadoc, and the complete repository test lifecycle — every command reported `BUILD SUCCESSFUL`.
  Final XML inspection recorded 105 tests with zero failures, errors, or skips; `javap -p`
  reconfirmed the exact two- and four-component records; `git diff --check` passed; and the final
  workspace still contained exactly the nine allowed files, no compile-API change, and no
  task-0009 specification.

## Implementation notes

- Added the exact producer-free `GraphValue` record over `ValueId` and `TensorDescriptor` and the
  exact `CompiledNode` record over `NodeId`, `Operation`, and ordered immutable input/output ID
  snapshots.
- Added focused 5-test and 10-test classes covering record shape, reference/list ownership, all
  local validations and messages, permitted input forms, output invariants, deferred graph-wide
  checks, equality, hashing, ordering, and diagnostics.
- Finalized both records' Javadocs, documented the graph-element mental model and compiled
  test-local example in the Tensor API, and synchronized glossary terminology and implementation
  status.
- Synchronized this task, the model master plan, and the implementation roadmap only after the
  required validation passed. Task 0009 is the next `Draft` planning frontier and has no detailed
  specification.

## Completion summary

- Completed changes: Implemented and documented immutable graph values and computation nodes with
  ordered local dataflow relationships and no graph-wide or executable behavior.
- Files changed or created: Two production records, two focused test classes, Tensor API, glossary,
  this task specification, model master plan, and implementation roadmap.
- Tests and validation: Focused 5-test and 10-test suites, all 105 model tests, model Javadoc, full
  repository tests, compiled documentation example, bytecode/import checks, link/anchor, fence,
  whitespace, terminology, scope, and diff checks passed.
- Documentation-agent review: Clean context `/root/review_model_0008_docs` independently reviewed
  source, tests, generated output, reports, API, glossary, compile API impact, and planning status.
- Documentation impact: Graph values and nodes are current model contracts; the compiled graph
  container, graph-wide validation, compiler integration, storage, preparation, runtime, and
  backend execution remain planned in their owning work.
- Javadoc review: New `GraphValue` and `CompiledNode` Javadocs are complete. Existing `NodeId`,
  `ValueId`, `Operation`, and `TensorDescriptor` Javadocs remain accurate for the recorded reasons
  and required no out-of-scope changes.
- Glossary impact: `GraphValue` and `CompiledNode` are marked implemented while Tensor, value,
  node, operation, and memory-slot boundaries remain distinct.
- Compile API impact: None; no compiler entry point, graph container, compile artifact, or compiler
  behavior was added, so `docs/api/compile-api.md` remains unchanged.
- Unresolved issues: None.
- Follow-up required: None. Plan task 0009 separately before implementation.

Status: Complete
