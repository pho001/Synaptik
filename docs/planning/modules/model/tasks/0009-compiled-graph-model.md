# Task 0009: [Compiled Graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) [Model](../../../../glossary.md#term-model)

**Editorial reading key (later context, not a status update).**

Compiler constructs this container from ordered value/node descriptions; its constructor checks structural closure and producer-before-consumer order rather than compiling or executing anything. PublicationBinding describes a public-Tensor-to-graph-value association; it is not a copy into that Tensor's storage. Graph-local values can be inputs without a producer and still acquire values from a later invocation. Model defines the record contracts; Compiler owns capture and publication planning, while invocation buffers and cleanup remain downstream resource concerns.

Focused current reading: [Current graph and publication records](../../../../reference/tensor/compiled-graph.md#graph-values-and-compiled-nodes). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the minimal [immutable](../../../../glossary.md#term-immutable) compiled-[graph](../../../../glossary.md#graph) container, its forward/backward phase classification,
and a standalone [publication](../../../../glossary.md#term-publication)-[binding](../../../../glossary.md#term-binding) [data-transfer object](../../../../glossary.md#data-transfer-object--dto) ([DTO](../../../../glossary.md#data-transfer-object--dto)). [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) stores
ordered graph values, ordered [topological](../../../../glossary.md#term-topological-order) [nodes](../../../../glossary.md#node), declared inputs and outputs, and one phase for
each node. Its constructor proves structural closure without performing [compiler](../../../../glossary.md#term-compiler) transformations,
[operation](../../../../glossary.md#operation)-family [validation](../../../../glossary.md#term-validation), [planning](../../../../glossary.md#term-planning), preparation, storage allocation, publication execution, or
[runtime](../../../../glossary.md#term-runtime) work.

## [Scope](../../../../glossary.md#term-scope)

- Add public enum `GraphPhase` in the existing `model.graph` package with exactly `FORWARD` and
  `BACKWARD`.
- Add public record `PublicationBinding` in the same package with exactly `TensorId tensorId` and
  `ValueId valueId`, in that order.
- Keep `PublicationBinding` standalone. It is [model](../../../../glossary.md#term-model) data for the [compiler](../../../../glossary.md#term-compiler)'s later
  `PublicationPlan`; it is not a component of [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel).
- Add public record `CompiledGraphModel` in the same package with exactly ordered
  `List<GraphValue> values`, `List<CompiledNode> nodes`, `List<ValueId> inputs`,
  `List<ValueId> outputs`, and `Map<NodeId, GraphPhase> nodePhases`, in that order.
- [Snapshot](../../../../glossary.md#term-snapshot) all lists with `List.copyOf`, preserving caller encounter order, and snapshot the phase
  map as an immutable value map without promising iteration order.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Validate non-null containers and elements, unique value and node IDs, unique declared input and
  output IDs, value-reference closure, producer rules, node topology, and exact node-phase
  coverage in the canonical constructor.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [node glossary](../../../../glossary.md#node); [producer glossary](../../../../glossary.md#term-producer); [topology glossary](../../../../glossary.md#term-topology); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Permit repeated node inputs, zero-input nodes, unused declared inputs, and a zero-node
  pass-through graph whose declared input is also an output.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Require at least one declared output, which also makes an empty value list invalid through
  output-reference validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Use record-generated structural equality, hashing, and diagnostic text over complete declared
  state.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Add one focused unit-test class for each new production type.
- During implementation, run a separate clean-context documentation-focused pass that finalizes
  affected Javadocs, API explanations, glossary terminology, task evidence, master-plan status,
  and roadmap status in the same overall change.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- graph builders, factories, mutable validators, stored producer/consumer indexes, lookup
  conveniences, or additional public methods beyond the exact enum and record contracts<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [mutable glossary](../../../../glossary.md#term-mutable); [producer glossary](../../../../glossary.md#term-producer); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- changes to the completed component [shapes](../../../../glossary.md#shape) or behavior of `GraphValue`, `CompiledNode`,
  [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [`TensorId`](../../../../glossary.md#tensorid), `TensorDescriptor`, or [`Operation`](../../../../glossary.md#operation)
- producer or consumer fields on values, graph-role fields on values, phase fields on nodes, or a
  new node subtype<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node))</span>
- optimizer graph phase, optimizer-update representation, compile modes, or training-step
  scheduling<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [graph phase glossary](../../../../glossary.md#graph-phase); [compile glossary](../../../../glossary.md#compile))</span>
- graph capture, ID allocation, inference, canonicalization, optimization, autograd, backward
  graph construction, dead-code elimination, or any other compiler pass<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [backward graph glossary](../../../../glossary.md#backward-graph); [dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- operation arity, result-count, kind-to-attributes, descriptor, shape, data type, layout, or
  symbolic-constraint validation<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [descriptor glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [validation glossary](../../../../glossary.md#term-validation))</span>
- backend ownership, partitioning, logical memory planning, capability analysis, cost scoring,
  lowering, kernels, or backend state<span data-glossary-navigation="true"> ([backend ownership glossary](../../../../glossary.md#backend-ownership); [planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend))</span>
- physical or host storage, device residency, prepared schedules, runtime targets, execution,
  publication execution, or mutable run state<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [runtime glossary](../../../../glossary.md#term-runtime); [targets glossary](../../../../glossary.md#term-target); [publication glossary](../../../../glossary.md#term-publication); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- `PublicationPlan`, `CompileArtifacts`, public [`Tensor`](../../../../glossary.md#tensor), tensor factories, tensor provenance,
  gradient roles, publication policies, or runtime publication targets<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- labels, serialization, parsing, persistence, registries, globally unique IDs, or cross-graph ID
  comparison<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- new packages, project dependencies, Gradle changes, other modules, architecture-contract
  changes, focused-architecture changes, or architecture-test changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- task 0010 implementation or a detailed task-0010 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the `modules/model`, core
  lifecycle, compile-artifacts, training-graph, runtime-hot-path, and dependency rules<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [compile glossary](../../../../glossary.md#compile); [artifacts glossary](../../../../glossary.md#term-artifact); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>, especially the current
  forward/backward graph and future optimizer-graph boundary<span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially `model.graph` [ownership](../../../../glossary.md#term-ownership-meanings) and package direction
- [Task 0004](0004-typed-identifiers.md), which defines [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot)
- [Task 0008](0008-graph-value-and-node-model.md), which defines the immutable graph elements and
  defers graph-wide validation here<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [elements glossary](../../../../glossary.md#term-element); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, [compile API](../../../../api/compile-api.md), and
  [glossary](../../../../glossary.md), which explain current and planned [graph](../../../../glossary.md#graph) terminology

## Prior evidence and rejected legacy coupling

No additional read-only legacy inspection was needed to plan this task. Task 0008 already records
the relevant legacy evidence: the old compiled-[node](../../../../glossary.md#node) [snapshot](../../../../glossary.md#term-snapshot) mixed [operation](../../../../glossary.md#operation) semantics, [graph](../../../../glossary.md#graph)
[identity](../../../../glossary.md#term-identity-meanings), [backend](../../../../glossary.md#backend) and storage [ownership](../../../../glossary.md#term-ownership-meanings), [tensor](../../../../glossary.md#tensor) facts, [autograd](../../../../glossary.md#autograd) flags, and static data, while old
[planning](../../../../glossary.md#term-planning) values derived identity from [producer](../../../../glossary.md#term-producer)-node IDs.

This task continues to reject that coupling. The new graph container stores only graph values,
computation nodes, explicit graph boundaries, and [graph phase](../../../../glossary.md#graph-phase). `PublicationBinding` connects the
already separate public-tensor and graph-value identity domains without placing a `Tensor`,
storage, backend, policy, [gradient](../../../../glossary.md#term-gradient) role, [runtime](../../../../glossary.md#term-runtime) [target](../../../../glossary.md#term-target), or [publication](../../../../glossary.md#term-publication) execution state in the
[model](../../../../glossary.md#term-model). No legacy source, package structure, or implementation shortcut is to be copied.

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- All three types live in `io.github.pho001.synaptik.model.graph`, which owns immutable compiled
  graph state and graph-local identities.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- `PublicationBinding` may import [`TensorId`](../../../../glossary.md#tensorid) from `model.tensor`. [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) composes only
  contracts already owned by `model.graph`. No reverse [dependency](../../../../glossary.md#term-dependency) from `model.tensor` to compiled
  graph state is permitted.
- Production code uses only completed model contracts and JDK collection utilities. It adds no
  project-module dependency.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [module glossary](../../../../glossary.md#term-module))</span>
- `GraphPhase` classifies node work as forward or backward. It is not a compile mode or runtime
  schedule. Do not add an optimizer phase: a compiled optimizer-update graph is a future direction
  that requires the explicit architecture updates described by the current contract.<span data-glossary-navigation="true"> ([compile mode glossary](../../../../glossary.md#compile-mode--compilemode); [runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `PublicationBinding` is an immutable model DTO for later compiler publication planning. It is
  not part of <span data-glossary-navigation="true"> ([DTO glossary](../../../../glossary.md#data-transfer-object--dto); [compiler glossary](../../../../glossary.md#term-compiler); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning))</span>`CompiledGraphModel`; the architecture keeps `CompileArtifacts.graph` and
  `CompileArtifacts.publication` separate.
- A publication binding maps public tensor identity to one logical graph value identity. It does
  not retain a public <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph value glossary](../../../../glossary.md#graph-value))</span>[`Tensor`](../../../../glossary.md#tensor) object and does not describe whether the publication is an output,
  gradient, host copy, runtime target, policy, storage location, or backend operation.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [host glossary](../../../../glossary.md#term-host); [target glossary](../../../../glossary.md#term-target); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- `CompiledGraphModel` is immutable compile-time model state. Runtime hot paths must not consume
  it or its <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>`CompiledNode` values.
- The order of `values`, [`nodes`](../../../../glossary.md#node), `inputs`, and `outputs` is part of the record value. Node order is
  topological. Input and output list order is caller-defined deterministic graph-boundary order.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#term-topological-order))</span>
- `nodePhases` is a structural value map. It must contain exactly one non-null phase for every
  graph node and no key for an absent node. Map equality is structural, but iteration order is not
  part of the API and must not be promised by implementation, tests, or documentation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Every declared input and output and every node input and output must resolve to a listed
  `GraphValue`.
- A declared graph input is producer-free. Every listed value that is not a declared input is
  produced exactly once. A value may not have multiple producers.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- For each node in stored order, every input must be either a declared graph input or an output of
  an earlier node. This rejects self-dependencies, later-node dependencies, and cycles without
  adding a separately stored index or compiler sorting behavior.
- Repeated node input positions remain valid. A node may have no inputs, as already allowed by
  `CompiledNode`. Existing node-local output [validation](../../../../glossary.md#term-validation) remains unchanged.
- The graph must declare at least one output. A graph with no nodes is valid when every listed
  value is a declared input; in particular, one value may appear in both graph `inputs` and [graph](../../../../glossary.md#graph)
  `outputs` as a pass-through result.
- Constructor-only temporary sets and maps may support validation. No producer map, consumer map,
  ID lookup table, phase index, or other derived mutable or immutable index is stored in the
  record.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- The model constructor validates structural closure only. It does not transform, infer, optimize,
  allocate, schedule, plan, prepare, lower, publish, or execute the graph.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
- If implementation requires another graph component, public method, stored index, package,
  dependency, compiler behavior, optimizer phase, architecture change, or scope expansion, stop
  and report the conflict.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.graph` — owns graph identities, graph elements, graph phase,
  publication binding, and immutable compiled graph state.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [elements glossary](../../../../glossary.md#term-element); [graph phase glossary](../../../../glossary.md#graph-phase); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [immutable glossary](../../../../glossary.md#term-immutable); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- `io.github.pho001.synaptik.model.tensor` — owns the public-state [`TensorId`](../../../../glossary.md#tensorid) referenced by a
  publication binding.

Packages added or changed:

- No package is added. The existing `model.graph` package gains three public types.

Type placement:

- `io.github.pho001.synaptik.model.graph.GraphPhase` — classifies a compiled node's forward or
  backward compile-time role next to the graph state it describes.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [compile glossary](../../../../glossary.md#compile))</span>
- `io.github.pho001.synaptik.model.graph.PublicationBinding` — carries the model-level identity
  association used by a later compiler-owned publication plan.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [publication plan glossary](../../../../glossary.md#publication-plan))</span>
- `io.github.pho001.synaptik.model.graph.CompiledGraphModel` — owns the immutable ordered graph
  elements, declared boundaries, phase association, and local structural-closure validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

Test placement:

- `io.github.pho001.synaptik.model.graph.GraphPhaseTest` — verifies the exact closed enum set and
  ordinary enum value behavior.
- `io.github.pho001.synaptik.model.graph.PublicationBindingTest` — verifies exact record state,
  reference ownership, failures, value semantics, and boundary exclusions.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `io.github.pho001.synaptik.model.graph.CompiledGraphModelTest` — verifies exact record state,
  snapshots, every graph-wide structural invariant and message, valid boundary cases, phase
  coverage, ordering, and record value semantics.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>

## Required contracts

### `GraphPhase`

Implement exactly this public enum [shape](../../../../glossary.md#shape):

```java
public enum GraphPhase {
    FORWARD,
    BACKWARD
}
```

`FORWARD` classifies work from the captured forward computation. `BACKWARD` classifies [gradient](../../../../glossary.md#term-gradient)
work introduced for backward computation. Do not add [`OPTIMIZER`](../../../../glossary.md#term-optimizer), `UPDATE`, `UNKNOWN`, a sentinel,
an integer code, parsing, [aliases](../../../../glossary.md#aliasing-and-arena), [mutable](../../../../glossary.md#term-mutable) [metadata](../../../../glossary.md#term-metadata), or methods. Declaration order is `FORWARD`
then `BACKWARD`; it is tested as the complete current vocabulary, not as a serialized ordinal
contract.

### `PublicationBinding`

Implement exactly this public record shape:

```java
public record PublicationBinding(TensorId tensorId, ValueId valueId) { ... }
```

The [canonical](../../../../glossary.md#term-canonical-form) constructor validates components in declaration order with
`Objects.requireNonNull`. A null component throws `NullPointerException` with the exact message
`tensorId` or `valueId`. The record retains both exact [immutable](../../../../glossary.md#term-immutable) references. Explicit accessors
return those references and document the distinct [identity](../../../../glossary.md#term-identity-meanings) scopes.

Do not add a [`Tensor`](../../../../glossary.md#tensor), [graph](../../../../glossary.md#graph) container, [graph phase](../../../../glossary.md#graph-phase), [publication](../../../../glossary.md#term-publication) kind, [gradient](../../../../glossary.md#term-gradient) flag, policy,
[runtime](../../../../glossary.md#term-runtime) [target](../../../../glossary.md#term-target), [host](../../../../glossary.md#term-host)/[device](../../../../glossary.md#term-device) location, storage, [backend](../../../../glossary.md#backend), or execution state. Do not add
cross-container [validation](../../../../glossary.md#term-validation): a standalone [binding](../../../../glossary.md#term-binding) cannot prove that its [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) belongs to a
particular graph. A future [compiler](../../../../glossary.md#term-compiler)-owned `PublicationPlan` performs owning-context validation and
remains separate from [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel).

Do not override `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), or `toString`. Record-generated equality and hashing include
both typed identities. Generated text is diagnostic and not a [serialization](../../../../glossary.md#term-serialization), parsing, global
identity, or runtime publication contract.

### [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel)

Implement exactly this public record shape:

```java
public record CompiledGraphModel(
        List<GraphValue> values,
        List<CompiledNode> nodes,
        List<ValueId> inputs,
        List<ValueId> outputs,
        Map<NodeId, GraphPhase> nodePhases) { ... }
```

The [canonical](../../../../glossary.md#term-canonical-form) constructor validates and snapshots the supplied state in the exact encounter order
defined below. It may build temporary local collections, but the five declared record components
are its only instance fields.

#### [Validation](../../../../glossary.md#term-validation) and [snapshot](../../../../glossary.md#term-snapshot) order

Apply these checks in order so focused tests can prove stable failures when more than one supplied
fact is invalid:

1. Reject null component references in declaration order with `Objects.requireNonNull` and the
   exact messages `values`, [`nodes`](../../../../glossary.md#node), `inputs`, `outputs`, and `nodePhases`.
2. Inspect `values` in list order. Reject a null [element](../../../../glossary.md#term-element) with `NullPointerException` and message
   `values[<index>]`. Reject the first later repeated `GraphValue.id()` with
   `IllegalArgumentException` and message
   `values[<index>] duplicates <ValueId>`. Snapshot the validated list with `List.copyOf`.
3. Inspect `nodes` in list order. Reject a null element with `NullPointerException` and message
   `nodes[<index>]`. Reject the first later repeated `CompiledNode.id()` with
   `IllegalArgumentException` and message
   `nodes[<index>] duplicates <NodeId>`. Snapshot the validated list with `List.copyOf`.
4. Inspect `inputs` in list order. Reject a null element with `NullPointerException` and message
   `inputs[<index>]`. Reject the first later repeated ID with `IllegalArgumentException` and
   message `inputs[<index>] duplicates <ValueId>`. Snapshot the validated list with
   `List.copyOf`.
5. Reject an empty `outputs` list with `IllegalArgumentException` and exact message
   `outputs must not be empty`. Then inspect outputs in list order. Reject a null element with
   `NullPointerException` and message `outputs[<index>]`. Reject the first later repeated ID with
   `IllegalArgumentException` and message
   `outputs[<index>] duplicates <ValueId>`. Snapshot the validated list with `List.copyOf`.
6. Validate phase-map elements before structural phase coverage. First reject a null key with
   `NullPointerException` and exact message `nodePhases contains null key`. Then visit the
   non-null keys in ascending `NodeId.value()` order so the result does not depend on the caller's
   map iteration order; reject a null phase with `NullPointerException` and exact message
   `nodePhases[<NodeId>]`. Snapshot the validated map with `Map.copyOf`; neither the stored map nor
   its accessor promises iteration order.
7. Resolve declared `inputs` in list order. Reject an ID absent from `values` with
   `IllegalArgumentException` and message
   `inputs[<index>] references unknown <ValueId>`.
8. Resolve declared `outputs` in list order. Reject an ID absent from `values` with
   `IllegalArgumentException` and message
   `outputs[<index>] references unknown <ValueId>`.
9. Inspect nodes in stored order. For each node, inspect inputs and then outputs in their stored
   order:
   - reject an input ID absent from `values` with `IllegalArgumentException` and message
     `nodes[<nodeIndex>].inputs[<inputIndex>] references unknown <ValueId>`;
   - reject an input that is neither a declared [graph](../../../../glossary.md#graph) input nor an output of an earlier node with
     `IllegalArgumentException` and message
     `nodes[<nodeIndex>].inputs[<inputIndex>] is not available before <NodeId>: <ValueId>`;
   - reject an output ID absent from `values` with `IllegalArgumentException` and message
     `nodes[<nodeIndex>].outputs[<outputIndex>] references unknown <ValueId>`;
   - reject an output that is also a declared graph input with `IllegalArgumentException` and
     message
     `nodes[<nodeIndex>].outputs[<outputIndex>] produces graph input <ValueId>`; and
   - reject an output already produced by an earlier node with `IllegalArgumentException` and
     message
     `nodes[<nodeIndex>].outputs[<outputIndex>] gives <ValueId> a second producer; first producer is <NodeId>`.

   After all inputs of a node pass, outputs from that node become available to later nodes. They
   are never available to another input position of the same node.
10. Inspect `values` in stored order. Reject the first value that is neither a declared input nor
    produced by a node with `IllegalArgumentException` and message
    `values[<index>] is neither a graph input nor a node output: <ValueId>`.
11. Check phase coverage in stored node order. Reject the first node without a phase entry with
    `IllegalArgumentException` and message `nodePhases missing <NodeId>`.
12. Check for absent-node phase keys in ascending `NodeId.value()` order. Reject the first unknown
    key with `IllegalArgumentException` and message
    `nodePhases contains unknown <NodeId>`.

In every message above, `<ValueId>` and `<NodeId>` [mean](../../../../glossary.md#term-mean) the identifier's record-generated
diagnostic text, for example `ValueId[value=7]` and `NodeId[value=3]`. Indices are zero-based.
Map keys are sorted only for deterministic validation; sorted or insertion iteration is not a
stored-map contract.

#### Stored-state and value semantics

Explicit accessors document and return the [immutable](../../../../glossary.md#term-immutable) snapshots. The four lists preserve encounter
order. Their containers have value semantics: tests must not promise [identity](../../../../glossary.md#term-identity-meanings) with caller-supplied
lists or another equal list. The phase map is unmodifiable and has structural map semantics, but
its iteration order is unspecified. Caller mutation after construction cannot change the record,
and mutation through any collection accessor throws `UnsupportedOperationException`.

Do not store [validation](../../../../glossary.md#term-validation) collections or expose [lookup](../../../../glossary.md#term-lookup) methods such as `value(ValueId)`,
`node(NodeId)`, `producer(ValueId)`, or `consumers(ValueId)`. Later owning [compiler](../../../../glossary.md#term-compiler) and [planning](../../../../glossary.md#term-planning)
work can build indexes appropriate to their [lifecycle](../../../../glossary.md#lifecycle) without changing this immutable [model](../../../../glossary.md#term-model).

Do not override `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), or `toString`. Record-generated equality and hashing include
the ordered lists and structural phase map. Reordering a list changes equality; supplying an equal
map with a different iteration order does not. Generated text is diagnostic and not a
[serialization](../../../../glossary.md#term-serialization), parsing, validated-execution, scheduling, or [backend](../../../../glossary.md#backend)-[dispatch](../../../../glossary.md#term-dispatch) format.

## Valid and invalid [graph](../../../../glossary.md#graph) forms

The following table summarizes the structural boundary without adding [compiler](../../../../glossary.md#term-compiler) semantics:

| Form | Result |
|---|---|
| One declared input is also the only output; no [nodes](../../../../glossary.md#node) or phases | Valid pass-through graph |
| No declared inputs; one zero-input node produces the output | Valid semantic-source [shape](../../../../glossary.md#shape) |
| A node reads the same earlier value in multiple input positions | Valid positional reuse |
| A declared input is unused | Valid; [liveness](../../../../glossary.md#term-liveness) and dead-input cleanup are compiler concerns |
| A listed non-input value has no [producer](../../../../glossary.md#term-producer) | Invalid incomplete graph |
| A declared input is produced by any node | Invalid input role |
| Two nodes produce the same value | Invalid multiple-producer graph |
| A node reads its own output or a later node's output | Invalid stored [topology](../../../../glossary.md#term-topology) |
| A phase is missing or names an absent node | Invalid phase coverage |
| A declared output list is empty | Invalid graph boundary |

This task does not require every produced value to be consumed, every input to be used, or every
listed value to lead to a declared output. Those are compiler/liveness policy decisions rather
than structural closure rules.

## Affected files

Expected new production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphPhase.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/PublicationBinding.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledGraphModel.java`

Expected new test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/GraphPhaseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/PublicationBindingTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/CompiledGraphModelTest.java`

Existing Java files that the documentation-focused pass may change only in [Javadoc](../../../../glossary.md#term-javadoc):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

`docs/architecture/training-graph.md` must be reviewed but is not expected to change. The current
focused architecture already limits the [graph phase](../../../../glossary.md#graph-phase) vocabulary to current forward/backward work
and labels an [optimizer](../../../../glossary.md#term-optimizer) [graph](../../../../glossary.md#graph) as future architecture work. No architecture change is expected.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- three new production Java files;
- three new focused test Java files;
- [Javadoc](../../../../glossary.md#term-javadoc) only in the two existing [graph](../../../../glossary.md#graph)-[element](../../../../glossary.md#term-element) production files listed above; and
- the six documentation and [planning](../../../../glossary.md#term-planning) files listed above.

Do not modify existing Java behavior or tests, [Gradle](../../../../glossary.md#term-gradle) files, `ARCHITECTURE.md`, focused architecture
documentation, architecture tests, the capability baseline, another [module](../../../../glossary.md#term-module), or unrelated
documentation. Do not create a task-0010 specification. If another file, type, public method,
[dependency](../../../../glossary.md#term-dependency), or architectural clarification is required, stop and report the issue instead of
expanding this task.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Every public type, enum constant, canonical constructor, record component, and explicit accessor
  must have meaningful detailed Javadoc. Generated component documentation alone is insufficient.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `GraphPhase` Javadoc must define compile-time node classification, distinguish phase from compile
  mode and runtime schedule, and explain why the current vocabulary contains only forward and
  backward work. It must not promise ordinal serialization.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [node glossary](../../../../glossary.md#node); [compile mode glossary](../../../../glossary.md#compile-mode--compilemode); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- `PublicationBinding` Javadoc must distinguish [`TensorId`](../../../../glossary.md#tensorid) from [graph](../../../../glossary.md#graph)-local [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), explain its
  later `PublicationPlan` role, state that it is separate from [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), and exclude
  [`Tensor`](../../../../glossary.md#tensor), [gradient](../../../../glossary.md#term-gradient)-role, policy, [target](../../../../glossary.md#term-target), storage, [backend](../../../../glossary.md#backend), and [runtime](../../../../glossary.md#term-runtime) [publication](../../../../glossary.md#term-publication) state.
- `PublicationBinding` constructor and accessors must document exact immutable-reference retention,
  identity scope, non-nullness, result semantics, and exact null failures.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [scope glossary](../../../../glossary.md#term-scope))</span>
- `CompiledGraphModel` type Javadoc must give a newcomer a values/nodes/boundaries/phases mental
  model, define immutable compile-time ownership, and distinguish the record from compiler
  behavior, compile artifacts, public <span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>`Tensor`, publication plans, physical storage, prepared
  execution, and runtime state.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- `CompiledGraphModel` constructor Javadoc must document all five inputs, deterministic list order,
  phase-map order non-contract, snapshots, every structural invariant, validation order, all exact
  failure types/messages, and the permitted pass-through and zero-input-node forms.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [validation glossary](../../../../glossary.md#term-validation))</span>
- Collection accessors must document non-null immutable snapshots, list ordering, map iteration
  non-contract, allowed repeated node inputs through contained nodes, and
  `UnsupportedOperationException` on attempted mutation without promising container [identity](../../../../glossary.md#term-identity-meanings).
- Type Javadocs must explain record-generated equality and hashing and classify generated text as
  diagnostic rather than serialization or execution data.
- Existing `GraphValue` and `CompiledNode` Javadocs must be independently reviewed. Their behavior
  and declarations must not change; only stale wording that calls the graph container “future” may
  be updated to describe the now-current owning `CompiledGraphModel` contract.
- Javadocs must not promise operation arity or descriptor validation, compiler transformations,
  stored indexes, lookup complexity, map iteration order, backend ownership, planning, storage,
  preparation, publication execution, or runtime behavior.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [arity glossary](../../../../glossary.md#term-arity); [descriptor glossary](../../../../glossary.md#term-descriptor); [lookup glossary](../../../../glossary.md#term-lookup); [backend ownership glossary](../../../../glossary.md#backend-ownership); [planning glossary](../../../../glossary.md#term-planning))</span>

## Acceptance criteria

- Exactly `GraphPhase`, `PublicationBinding`, and [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) are added as public types in
  `model.graph`; no production helper type or additional package is introduced.
- `GraphPhase.values()` contains exactly `FORWARD` then `BACKWARD`, with no optimizer or sentinel
  phase and no added behavior.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `PublicationBinding` has exactly `TensorId tensorId` and `ValueId valueId` components in that
  order, no additional instance state, and no [`Tensor`](../../../../glossary.md#tensor), gradient, policy, storage, backend, target,
  or runtime component.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [target glossary](../../../../glossary.md#term-target); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `PublicationBinding` rejects null components with the exact messages, retains exact immutable
  references, and remains standalone rather than becoming part of <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`CompiledGraphModel`.
- `CompiledGraphModel` has exactly the five specified components in the specified order and no
  additional instance fields, stored indexes, or lookup conveniences.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- All component containers and elements are non-null. Every failure uses the specified exception
  type, exact message, zero-based index, and encounter order.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- [Graph](../../../../glossary.md#graph) values and [node](../../../../glossary.md#node) IDs are unique; declared input and output IDs are independently unique.
- The graph declares at least one output. Every declared input and output and every node reference
  resolves to a listed graph value.<span data-glossary-navigation="true"> ([graph value glossary](../../../../glossary.md#graph-value))</span>
- An empty value list therefore cannot construct: with the required non-empty output list, its
  first output fails as an unknown value reference.
- Every declared input is producer-free. Every non-input listed value has exactly one producer,
  and no value has multiple producers.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Stored node order is topological: each node input is a declared input or an output of an earlier
  node. Self-dependencies, later-node dependencies, and cycles are rejected.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#term-topological-order))</span>
- Repeated node inputs and zero-input nodes remain valid. No operation-family, arity, result-count,
  or descriptor matching is added.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [arity glossary](../../../../glossary.md#term-arity); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- A zero-node pass-through graph whose one listed value is both input and output constructs
  successfully with an empty phase map.
- Every stored node has exactly one non-null phase, and no map key refers to an absent node.
- Caller mutation after construction cannot affect any component. Mutation through each list or
  map accessor fails with `UnsupportedOperationException`.
- List order is preserved and participates in record equality. Map iteration order is not tested
  or documented; equal mappings have structural equality regardless of caller map order.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Equal component values produce equal records and hash codes. Representative changed components
  produce unequal records. Diagnostic text names all components without becoming an exact wire
  format assertion.<span data-glossary-navigation="true"> ([hash codes glossary](../../../../glossary.md#hash-codes-and-content-digests))</span><span data-glossary-navigation="true"> ([wire format glossary](../../../../glossary.md#term-wire-format))</span>
- Reflection verifies exact public record/enum shapes, component order and erased types, and no
  additional record instance fields. Manual bytecode/API inspection confirms no stored derived
  indexes or unexpected public methods.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Production imports remain limited to completed model contracts and JDK collection/validation
  utilities. No compiler, planning, runtime, prepare, backend, storage, config, trace, or concrete
  operation-family dependency is introduced.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [validation glossary](../../../../glossary.md#term-validation); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- All Javadoc requirements are satisfied, and generated model Javadoc includes all three types,
  enum constants, record constructors, components, accessors, ownership rules, failures, value
  semantics, and cross-layer boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [constants glossary](../../../../glossary.md#term-constant); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- A separate clean-context documentation-focused agent independently inspects the implementation,
  tests, generated Javadoc, and final diff; finalizes affected new and existing Javadocs, Tensor
  API, compile API, glossary, planning evidence, links, terminology, examples, and status; and
  records exact validation evidence in this same overall change.
- The Tensor API marks the graph container, phases, and publication binding as current model
  contracts while keeping compiler entry points, <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding))</span>`PublicationPlan`, public `Tensor`, planning,
  prepare, runtime, and execution planned.
- The [compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) distinguishes the now-current [model](../../../../glossary.md#term-model) DTOs from the still-planned [compiler](../../../../glossary.md#term-compiler)-owned
  `PublicationPlan`, `CompileArtifacts`, and [engine](../../../../glossary.md#term-engine) `CompiledGraph` facade. It does not claim a
  runnable compiler.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- The glossary marks the three new contracts implemented and preserves distinctions among tensor,
  publication binding, graph model, graph phase, compile artifacts, prepared execution, and
  runtime state.<span data-glossary-navigation="true"> ([graph phase glossary](../../../../glossary.md#graph-phase); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Task, master-plan row/current status/notes, and roadmap frontier/table have matching final status.
  When task 0009 is complete, task 0010 becomes the next `Draft` planning frontier, but no detailed
  task-0010 specification is created in this change.
- No existing Java behavior/test, Gradle file, architecture contract, focused architecture file,
  architecture test, capability baseline, other module, or unrelated documentation is changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [architecture test glossary](../../../../glossary.md#term-architecture-test); [module glossary](../../../../glossary.md#term-module))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) after implementation and again after the documentation-focused pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.graph.GraphPhaseTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.graph.PublicationBindingTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.graph.CompiledGraphModelTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the final diff stays within the three new production files, three new tests, at most two
  existing Javadoc-only Java edits, and six allowed documentation/planning files;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- reflection reports the exact record components, order, erased types, enum vocabulary, and no
  additional record instance fields;
- constructor-focused tests cover every exact exception type/message and validation precedence,
  including indexed outer-list nulls, duplicates, unknown references, topology, producers, phase
  nulls, missing phases, and unknown phase keys;<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- valid-form tests cover a multi-node forward/backward graph, an empty-node pass-through graph, a
  zero-input source node, repeated node inputs, unused graph inputs, and immutable collection
  ownership;<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [backward graph glossary](../../../../glossary.md#backward-graph); [graph glossary](../../../../glossary.md#graph); [immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- invalid-form tests cover empty outputs, duplicate IDs, unresolved declared and node references,
  an empty value list paired with the required output, a produced graph input, multiple producers,
  an unproduced non-input value, self/later-node dependency, phase under-coverage, and phase
  over-coverage;
- `javap -p -c` confirms exact fields, `List.copyOf` and `Map.copyOf` snapshots, only temporary
  validation collections, direct scalar accessors, no stored indexes or lookup methods, and
  record-generated equality, hashing, and text;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- a manual or reflection-based public API check confirms the records expose only their canonical
  constructor, five or two component accessors, and generated record methods, and <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`GraphPhase`
  exposes only ordinary enum API plus its two constants;<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
- production import scans find no compiler, planning, runtime, prepare, backend, storage, config,
  trace, public <span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [trace glossary](../../../../glossary.md#trace))</span>[`Tensor`](../../../../glossary.md#tensor), [publication](../../../../glossary.md#term-publication)-plan, or concrete [operation](../../../../glossary.md#operation)-family [dependency](../../../../glossary.md#term-dependency);
- the phase-map implementation and tests make no insertion, sorted, or hash iteration-order
  promise; sorting is used only for deterministic invalid-map validation;<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- `PublicationBinding` is not a [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) component, and no `PublicationPlan`,
  `CompileArtifacts`, public `Tensor`, optimizer phase, runtime target, policy, storage, or backend
  state appears;<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [target glossary](../../../../glossary.md#term-target))</span>
- generated Javadoc documents every new public type, constant, constructor, component/accessor,
  failure, snapshot, ordering, value-semantic, and layer-boundary contract;<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- existing `GraphValue` and `CompiledNode` source changes, if any, are Javadoc-only and replace
  stale temporal wording without changing their completed behavior;
- `docs/api/tensor-api.md`, `docs/api/compile-api.md`, and `docs/glossary.md` distinguish current
  model contracts from planned compiler/runtime contracts, use novice-readable terminology, and
  include or update examples only when they can be validated or clearly labeled conceptual;<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- `docs/architecture/training-graph.md` remains unchanged because no optimizer graph phase or
  architecture behavior is added;<span data-glossary-navigation="true"> ([graph phase glossary](../../../../glossary.md#graph-phase))</span>
- all local Markdown links and heading anchors in changed documentation resolve, code fences are
  balanced, changed files have no trailing whitespace, and terminology agrees with the glossary;
- the documentation-focused context follows
  `docs/developer-guide/documentation-rules.md`, applies General style plus API and Javadoc style to
  API/Javadoc work and Planning style to planning updates, and records its identity, source/test
  inspection, selected profiles, commands, results, limitations, Javadoc review, and glossary
  impact;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- task 0009 status matches the master-plan row/current status/notes and roadmap frontier/table;
  task 0010 remains only a master-plan/roadmap row with no detailed specification; and
- package direction remains `model.graph -> model.tensor`, with no forbidden [module](../../../../glossary.md#term-module) dependency.

## Dependencies

- Task 0004 is complete and provides [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).
- Task 0008 is complete and provides `GraphValue` and `CompiledNode` with the local invariants on
  which graph-wide validation relies.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation))</span>
- The [architecture contract](../../../../glossary.md#architecture-contract) and [model](../../../../glossary.md#term-model) master plan already assign `GraphPhase`,
  `PublicationBinding`, and [immutable](../../../../glossary.md#term-immutable) [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) state to `modules/model`.

## Follow-up tasks

- Task 0010 remains the next ordered model task after task 0009 completes. It will define host
  storage separately from graph values and runtime device storage.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [host storage glossary](../../../../glossary.md#host-storage); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [device glossary](../../../../glossary.md#term-device))</span>
- Later public-[tensor](../../../../glossary.md#tensor) tasks own `Tensor`, factories, [identity](../../../../glossary.md#term-identity-meanings) allocation, and [provenance](../../../../glossary.md#provenance).
- Later compiler tasks own graph capture, ID allocation, inference, transformations, autograd,
  publication-plan construction, binding validation in its owning context, compiler-side indexes,
  and compile artifacts.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- Later planning, prepare, runtime, and backend tasks own graph consumers, backend ownership,
  logical and physical memory, schedules, executable state, publication execution, and runtime
  residency.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [backend ownership glossary](../../../../glossary.md#backend-ownership); [residency glossary](../../../../glossary.md#residency))</span>

Do not create a detailed task-0010 specification as part of task 0009.

## Architecture impact

Expected impact: None.

The architecture already assigns [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), `GraphPhase`, `PublicationBinding`,
[immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) values and [nodes](../../../../glossary.md#node), and typed IDs to `modules/model`. The graph record contains only
[compile](../../../../glossary.md#compile)-time [model](../../../../glossary.md#term-model) state; [publication](../../../../glossary.md#term-publication) remains a separate association for later [compiler](../../../../glossary.md#term-compiler) [planning](../../../../glossary.md#term-planning);
and the enum represents only current forward/backward work. No [module](../../../../glossary.md#term-module) boundary, [dependency](../../../../glossary.md#term-dependency) rule,
[lifecycle](../../../../glossary.md#lifecycle) contract, [optimizer](../../../../glossary.md#term-optimizer)-graph policy, or [backend](../../../../glossary.md#backend) contract changes. Therefore
`ARCHITECTURE.md`, focused architecture documentation, ADRs, and architecture tests require no
update. If implementation reveals otherwise, stop and report the conflicting rule and required
decision before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read first and in full:
- AGENTS.md
- ARCHITECTURE.md
- docs/architecture/current-architecture-plan.md
- docs/architecture/training-graph.md
- docs/developer-guide/documentation-rules.md
- docs/planning/planning-guide.md
- docs/planning/roadmap.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0004-typed-identifiers.md
- docs/planning/modules/model/tasks/0008-graph-value-and-node-model.md
- docs/planning/modules/model/tasks/0009-compiled-graph-model.md
- docs/api/tensor-api.md
- docs/api/compile-api.md
- docs/glossary.md
- all current production and test files in io.github.pho001.synaptik.model.graph
- the current TensorId production/test contract

Implement task 0009 exactly as specified. Create only GraphPhase.java,
PublicationBinding.java, CompiledGraphModel.java, GraphPhaseTest.java,
PublicationBindingTest.java, and CompiledGraphModelTest.java for new code and tests.

GraphPhase must contain exactly FORWARD and BACKWARD in that order. Do not add an optimizer or
sentinel phase.

PublicationBinding must be a public record with exactly TensorId tensorId and ValueId valueId. It
is a standalone model DTO for a later compiler PublicationPlan and must not be a
CompiledGraphModel component. Reject null components with the task's exact messages and add no
Tensor, gradient role, policy, runtime target, storage, backend, or execution state.

CompiledGraphModel must be a public record with exactly ordered List<GraphValue> values,
List<CompiledNode> nodes, List<ValueId> inputs, List<ValueId> outputs, and
Map<NodeId, GraphPhase> nodePhases. Follow the task's exact validation and snapshot order,
exception types, messages, and zero-based indices. Preserve list order with List.copyOf. Store an
immutable structural phase map with Map.copyOf and do not promise iteration order. Validate unique
IDs, explicit-boundary and node-reference closure, producer-free inputs, exactly one producer for
each non-input, topological node order, and exact phase coverage. Allow repeated node inputs,
zero-input nodes, unused inputs, and a zero-node pass-through graph. Require at least one output.
Use temporary constructor-local validation collections only; add no stored indexes or lookup
conveniences.

Do not change completed GraphValue or CompiledNode behavior or record shape. Do not add builders,
factories, helpers, compiler passes, ID allocation, operation-family/descriptor validation,
PublicationPlan, CompileArtifacts, public Tensor, provenance, planning, backend ownership,
storage, prepare, runtime, execution, dependencies, or unrelated refactors. Do not modify Gradle,
ARCHITECTURE.md, focused architecture docs, architecture tests, capabilities.md, another module,
or unrelated documentation. Do not create or implement task 0010. Stop and report if a required
change exceeds the affected-file or maximum-scope list or if architecture uncertainty appears.

Add every Javadoc contract required by the task. Run every focused and aggregate validation
command and perform every manual API, bytecode, import, ownership, documentation, scope, and
status check.

After code implementation and initial validation, hand the resulting diff to a separate
documentation-focused agent or thread with a clean context in the same overall change. The
handoff must include this task specification, the actual implementation/test diff, affected graph
model and publication-binding behavior, architecture constraints, expected Tensor API/compile
API/glossary/planning updates, potentially stale temporal Javadoc in GraphValue and CompiledNode,
the no-change expectation for focused architecture, and every validation command.

That documentation agent must read AGENTS.md, ARCHITECTURE.md, the current architecture index,
training-graph explanation, documentation rules, profile index, General style, API and Javadoc
style, Planning style, Example format when an example changes, this task, final source/tests,
Tensor API, compile API, glossary, model master plan, and roadmap. It must inspect the actual
source, tests, generated Javadoc, reports, and diff rather than rely on the handoff summary. It
must independently finalize every new/affected Javadoc, current-versus-planned API status,
newcomer-readable examples and terminology, glossary impact, links, anchors, fences, whitespace,
and planning status/evidence. It may make Javadoc-only wording updates to GraphValue.java and
CompiledNode.java. It must not change their declarations or behavior. It must record a reasoned
no-change conclusion for docs/architecture/training-graph.md or stop on unexpected architecture
impact.

At the end, update only this task file, the model master plan, and the roadmap for planning status.
Record local decisions, known limitations, exact validation evidence including the documentation
agent identity and results, implementation notes, and the canonical completion summary. Do not
mark task 0009 Complete until all acceptance criteria, both validation passes, documentation
changes, and status synchronization are complete. After completion, task 0010 may be named as the
next Draft frontier, but do not create its detailed specification.
```

## Local decisions

- `GraphPhase` is deliberately closed to forward and backward work. Adding an optimizer phase now
  would turn a future architecture direction into a current model contract without the required
  architecture updates.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [model glossary](../../../../glossary.md#term-model))</span>
- [Publication](../../../../glossary.md#term-publication) association is a standalone [`TensorId`](../../../../glossary.md#tensorid)-to-[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) DTO. Keeping it outside the
  graph record preserves the architecture's separation between <span data-glossary-navigation="true"> ([DTO glossary](../../../../glossary.md#data-transfer-object--dto))</span>`CompileArtifacts.graph` and
  `CompileArtifacts.publication`.
- Explicit [graph](../../../../glossary.md#graph) inputs and outputs are ordered boundary declarations rather than flags on
  `GraphValue`. This keeps one value [shape](../../../../glossary.md#shape) for inputs, intermediates, and outputs.
- Producer relationships are derived once during construction for validation and are not stored.
  Node output lists remain the single stored source of producer truth.<span data-glossary-navigation="true"> ([Producer glossary](../../../../glossary.md#term-producer); [validation glossary](../../../../glossary.md#term-validation); [Node glossary](../../../../glossary.md#node))</span>
- Topological order is part of the node-list contract. Validating against the available-value set
  avoids adding a separate cycle representation or silently sorting compiler output in the model.<span data-glossary-navigation="true"> ([Topological order glossary](../../../../glossary.md#term-topological-order); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- A pass-through graph is valid because an input may also be a requested output without any
  computation. At least one output still gives every graph a usable declared result boundary.
- Zero-input nodes remain valid for semantic source operations. This task validates graph
  structure, not whether a particular operation kind is allowed to have zero inputs.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Unused inputs and unconsumed produced values are structurally valid. Liveness, reachability, and
  dead-code cleanup belong to compiler policy.<span data-glossary-navigation="true"> ([Liveness glossary](../../../../glossary.md#term-liveness))</span>
- `List.copyOf` and `Map.copyOf` provide immutable snapshots. List order is semantic; phase-map
  iteration order is not. Temporary key sorting exists only to make invalid-map failures stable.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Record-generated equality, hashing, and text cover complete stored state. No serialization,
  lookup-complexity, allocation, execution, or globally unique identity contract is added.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [lookup glossary](../../../../glossary.md#term-lookup); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Known limitations

- The graph container does not allocate IDs, capture expressions, infer descriptors, validate
  operation families, or transform graph order.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [capture glossary](../../../../glossary.md#term-capture-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- The graph can contain unused declared inputs and produced values that do not reach a declared
  output. Compiler liveness and dead-code elimination remain later work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [liveness glossary](../../../../glossary.md#term-liveness); [dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- `PublicationBinding` alone cannot prove that its value belongs to a particular graph or that the
  tensor was requested under a particular publication policy. The later compiler-owned
  publication plan owns that context.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [publication glossary](../../../../glossary.md#term-publication); [publication plan glossary](../../../../glossary.md#publication-plan))</span>
- `GraphPhase` cannot represent optimizer-update work. That remains intentionally unavailable
  until an explicit future architecture change authorizes a compiled optimizer graph.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- The record exposes no producer, consumer, value, or node lookup convenience. Consumers may
  construct lifecycle-appropriate indexes outside the immutable model.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [node glossary](../../../../glossary.md#node); [lookup glossary](../../../../glossary.md#term-lookup); [lifecycle glossary](../../../../glossary.md#lifecycle); [immutable glossary](../../../../glossary.md#term-immutable); [model glossary](../../../../glossary.md#term-model))</span>
- The types provide compile-time model state only. They contain no backend ownership, memory plan,
  physical storage, prepared execution, runtime publication, or residency state.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [backend ownership glossary](../../../../glossary.md#backend-ownership); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Planning context read the complete architecture contract, current architecture index,
  documentation workflow, profile index, General and Planning styles, planning guide, roadmap,
  model master plan, task 0008, Tensor API, compile API, glossary, training-graph explanation, and
  all current <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [architecture contract glossary](../../../../glossary.md#architecture-contract); [compile glossary](../../../../glossary.md#compile))</span>`model.graph` production/tests before defining this contract.
- No legacy branch inspection was needed. The rejected legacy coupling is based on the recorded
  read-only evidence in completed task 0008; no legacy source or package structure was copied.
- `git status --short --untracked-files=all` and an exact path comparison confirmed exactly three
  planning-file changes: this new task, the model master plan, and the roadmap. No Java, test,
  build, architecture, API, glossary, other-module, or unrelated documentation file changed.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- A targeted Ruby Markdown-link check resolved all 50 local links in the three changed planning
  documents. No changed link contains a heading fragment, so there was no changed anchor to
  resolve.
- Fence inspection reported even counts in every changed document: ten backtick-fence markers in
  this task and two each in the master plan and roadmap. `rg -n '[[:blank:]]+$'` returned no
  trailing-whitespace matches.
- `git diff --check` passed for tracked changes. `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0009-compiled-graph-model.md` produced no whitespace
  diagnostic; its exit status was the expected `1` because the new file differs from `/dev/null`.
- Status review confirmed task 0008 remains `Complete` and task 0009 is linked and [`Ready`](../../../../glossary.md#term-ready) in this
  task, the master-plan row/current status/notes, and the roadmap frontier/table. Task order and
  dependencies are unchanged, task 0010 remains `Draft`, and no task-0010 specification exists.
- The implementation phase added only `GraphPhase.java`, `PublicationBinding.java`,
  `CompiledGraphModel.java`, and their three focused tests. Independent source and test inspection
  confirmed the exact enum/record shapes, validation order and messages, list and map snapshots,
  structural closure, supported boundary cases, standalone publication association, and absence of
  stored indexes or cross-layer state.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- Clean documentation-focused context `/root/implement_model_0009/review_model_0009_docs` applied
  General style plus API and Javadoc style to Java/API work, Planning style to planning updates,
  and Example format to the revised graph-model example. It read the architecture contract,
  current architecture index, training-graph explanation, documentation workflow and profiles,
  planning guide, roadmap, model master plan, tasks 0004/0008/0009, Tensor API, compile API,
  glossary, all <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`model.graph` production/tests, [`TensorId`](../../../../glossary.md#tensorid) production/test, generated Javadoc,
  model test XML/HTML reports, and the complete tracked/untracked workspace diff.
- The documentation pass finalized all three new public types' Javadocs and made Javadoc-only
  temporal-wording updates to `GraphValue.java` and `CompiledNode.java`. Zero-context diff review
  confirmed that their declarations, imports, constructors, accessors, and behavior are unchanged.
  Their Javadocs now identify [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) as the current owner of graph-wide validation;
  public <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>[`Tensor`](../../../../glossary.md#tensor) and [operation](../../../../glossary.md#operation)-family compatibility remain planned.
- The Tensor [API](../../../../glossary.md#term-application-programming-interface-api) now presents `GraphPhase`, `CompiledGraphModel`, and `PublicationBinding` as
  current model contracts; documents immutable collection ownership, structural invariants,
  valid pass-through/source/repeated-input forms, and model/compiler/runtime boundaries; and keeps
  public <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime))</span>`Tensor`, [compiler](../../../../glossary.md#term-compiler) entry points, `PublicationPlan`, `CompileArtifacts`, planning, prepare,
  runtime, and execution planned.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
- The revised complete test-local [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model) example compiles with
  `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-graph-model-doc-example /tmp/GraphElementExample.java`. Running it with
  `java -cp modules/model/build/classes/java/main:/tmp/synaptik-graph-model-doc-example
  GraphElementExample` prints [`IDENTITY`](../../../../glossary.md#term-identity-meanings), `[0]`, `[1]`, `FORWARD`, and `TensorId[value=9]` on
  separate lines, matching the documented result.
- The compile API now distinguishes the current compiler-neutral model DTOs from the planned
  compiler-owned `PublicationPlan` and `CompileArtifacts` and planned [engine](../../../../glossary.md#term-engine) `CompiledGraph`
  facade. It explicitly states that no runnable compiler exists.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- The glossary marks `GraphPhase`, `CompiledGraphModel`, and `PublicationBinding` implemented and
  preserves distinctions among planned public `Tensor`, standalone publication binding,
  compile-time graph model, planned compile artifacts, planned prepared execution, and runtime
  state.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.graph.GraphPhaseTest` — `BUILD SUCCESSFUL`; 2 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.graph.PublicationBindingTest` — `BUILD SUCCESSFUL`; 4 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.graph.CompiledGraphModelTest` — `BUILD SUCCESSFUL`; 18 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML totals are 129 tests, zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated output includes every new
  type, enum constant, canonical constructor, accessor, immutable-ownership rule, validation
  failure, ordering contract, value semantic, and lifecycle boundary. Rendered Javadoc also
  reflects the two allowed existing Javadoc-only updates.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `./gradlew test` — `BUILD SUCCESSFUL` for the complete repository; the final run reported 36
  actionable tasks.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `javap -p` and `javap -p -c` confirmed exactly two fields for `PublicationBinding`, exactly five
  fields for `CompiledGraphModel`, only `FORWARD` then `BACKWARD` for `GraphPhase`, direct
  accessors, four `List.copyOf` calls, one `Map.copyOf` call, constructor-local validation
  collections, generated record methods, and no stored index or lookup convenience.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Production import review found only `TensorId`, `Objects`, and JDK collection/validation types.
  No compiler, planning, runtime, prepare, backend, storage, config, trace, public <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [backend glossary](../../../../glossary.md#backend); [trace glossary](../../../../glossary.md#trace))</span>`Tensor`,
  publication-plan, or concrete operation-family dependency appears.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- A targeted local Markdown path-and-heading check resolved all 115 links and anchors in the six
  changed documentation/planning files. Fence counts are balanced, trailing-whitespace inspection
  found no matches, and current-versus-planned terminology agrees across Javadoc, API pages,
  glossary, task, master plan, and roadmap.
- Final scope review confirmed exactly fourteen allowed changed/new repository files: three new
  production files, three new tests, Javadoc-only edits to <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`GraphValue.java` and
  `CompiledNode.java`, and the six authorized documentation/planning files. No detailed task-0010
  specification exists.
- `docs/architecture/training-graph.md` remains unchanged. `GraphPhase` only classifies current
  forward/backward compile-time nodes and adds no optimizer phase, compile mode, runtime schedule,
  or architecture behavior; therefore <span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [optimizer glossary](../../../../glossary.md#term-optimizer); [compile mode glossary](../../../../glossary.md#compile-mode--compilemode))</span>`ARCHITECTURE.md`, focused architecture documentation,
  ADRs, and architecture tests require no update.
- `git diff --check` passed after implementation, documentation finalization, and status
  synchronization.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- The final coordinating context independently reran all three focused suites, all model tests,
  model Javadoc, and the complete repository test lifecycle after the documentation pass; every
  required command reported `BUILD SUCCESSFUL`. Final XML inspection again recorded 129 tests
  with zero failures, errors, or skips. It also reran the compiled documentation example, repeated
  `javap -p`/`javap -p -c`, import, exact 14-path scope, 115-link/anchor, fence, trailing-whitespace,
  status, task-0010-absence, and diff checks with the same passing results. One non-required
  chained XML-summary/Javadoc invocation could not open Gradle's wrapper lock under the sandbox;
  the exact standalone <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`./gradlew :modules:model:javadoc` command immediately succeeded.

## Implementation notes

- Added the exact two-value `GraphPhase` enum, exact two-component standalone
  `PublicationBinding`, and exact five-component [immutable](../../../../glossary.md#term-immutable) [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel).
- Added focused tests for closed phase vocabulary, publication identity/value semantics, exact
  graph record shape, validation precedence and messages, valid graph forms, immutable ownership,
  structural equality, and diagnostics.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Finalized new and affected Javadocs, updated the Tensor and compile API references, synchronized
  glossary terminology, and recorded the reason architecture and training-graph documentation did
  not change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Synchronized task, model master plan, and roadmap only after implementation, documentation, and
  validation passed. Task 0010 is the next <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`Draft` frontier without a detailed specification.

## Completion summary

- Completed changes: Implemented and documented the immutable compiled graph container, exact
  forward/backward node classification, and standalone tensor-to-value publication binding.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [node glossary](../../../../glossary.md#node); [tensor glossary](../../../../glossary.md#tensor); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding))</span>
- Files changed or created: Three production types, three focused test classes, Javadoc-only
  updates to two existing graph-element records, Tensor API, compile API, glossary, this task,
  model master plan, and implementation roadmap.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [element glossary](../../../../glossary.md#term-element); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 2-test, 4-test, and 18-test suites, all 129 model tests, generated
  model Javadoc, full repository tests, compiled documentation example, bytecode/API/import,
  link/anchor, fence, whitespace, terminology, scope, and diff checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Clean context
  `/root/implement_model_0009/review_model_0009_docs` independently reviewed final source, tests,
  generated docs, reports, APIs, glossary, planning state, and actual workspace diff.
- Documentation impact: Current model graph DTOs and their structural contracts are documented;
  compiler entry points, publication plan, compile artifacts, public Tensor, planning, prepare,
  runtime, and execution remain explicitly planned.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [publication plan glossary](../../../../glossary.md#publication-plan); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Every new public type and member is fully documented. Existing `GraphValue` and
  `CompiledNode` received Javadoc-only temporal updates; their completed behavior remains intact.
- Glossary impact: The three new contracts are marked implemented, with compile-time model,
  publication, prepared-execution, and runtime-state boundaries kept distinct.
- Architecture impact: None. The architecture already authorizes all three model types, and no
  optimizer graph phase, module boundary, lifecycle rule, or dependency rule changed.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [graph phase glossary](../../../../glossary.md#graph-phase); [module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Unresolved issues: None.
- Follow-up required: None. Task 0010 is the next `Draft` [planning](../../../../glossary.md#term-planning) frontier.

Status: Complete
