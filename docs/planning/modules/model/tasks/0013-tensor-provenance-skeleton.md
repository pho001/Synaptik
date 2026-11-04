# Task 0013: [Tensor](../../../../glossary.md#tensor) [Provenance](../../../../glossary.md#provenance) Skeleton

**Editorial reading key (later context, not a status update).**

Expression helpers, rather than the application through a public provenance setter, attach an Operation and ordered exact input Tensor references here. Those references describe how to produce a result later; they do not copy input values or attach result storage. A factory leaf has no producing expression, including when it is a storage-free input placeholder. This record precedes shared TensorProducer/output-slot identity; the linked current explanation is intentionally a contrast with the original single-result representation, not a rewrite of its deferred multi-output decision. Compiler remains responsible for capturing the expression into graph-local records.

Focused current reading: [Current producer-based provenance](../../../../reference/tensor/identity.md#expression-producers-and-indexed-provenance). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add the smallest [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model) contract that records how a public expression [tensor](../../../../glossary.md#tensor) was
produced: one [backend](../../../../glossary.md#backend)-independent [`Operation`](../../../../glossary.md#operation) and an ordered [snapshot](../../../../glossary.md#term-snapshot) of its input `Tensor`
objects. Attach that optional [provenance](../../../../glossary.md#provenance) to `Tensor` as stable origin [metadata](../../../../glossary.md#term-metadata) and add one
package-private factory path that future operation-family tasks can use without bypassing the
existing [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-wide [`TensorId`](../../../../glossary.md#tensorid) allocator.

The contract prepares later [compiler](../../../../glossary.md#term-compiler)-owned [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) while preserving the architecture rule
that a public `Tensor` is not an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node). It does not create a [graph](../../../../glossary.md#graph), assign graph-local [identity](../../../../glossary.md#term-identity-meanings),
infer [descriptors](../../../../glossary.md#term-descriptor), or implement any mathematical operation.

## [Scope](../../../../glossary.md#term-scope)

- Add one public [immutable](../../../../glossary.md#term-immutable) `TensorProvenance` record with exactly `Operation operation` and
  ordered `List<Tensor> inputs` components.
- Validate non-null operation/list/elements, snapshot the list, preserve order, and allow empty or
  repeated inputs.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [elements glossary](../../../../glossary.md#term-element); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Add one final `Optional<TensorProvenance>` field and one public `provenance()` accessor to
  [`Tensor`](../../../../glossary.md#tensor).
- Extend the sole package-private Tensor constructor with one provenance optional while preserving
  all existing identity, descriptor, label, host-storage, synchronization, equality, hashing, and
  diagnostic behavior.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [host glossary](../../../../glossary.md#term-host); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Keep every existing public `TensorFactory` creation/population method leaf-producing with empty
  provenance.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- Add exactly one package-private `TensorFactory.createDerived(...)` method for future
  model-owned expression construction. It allocates identity through the existing allocator,
  attaches one already validated provenance object, and creates no storage.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Add focused provenance tests and update only the existing exact-shape Tensor and TensorFactory
  tests required by the new contract.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Finalize affected Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through
  the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- any concrete [`OperationKind`](../../../../glossary.md#operationkind), family-specific attributes, arithmetic or other expression method,
  descriptor/data-type/shape/layout inference, broadcasting, arity validation, or result checking<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [inference glossary](../../../../glossary.md#term-inference); [broadcasting glossary](../../../../glossary.md#broadcasting); [arity glossary](../../../../glossary.md#term-arity); [validation glossary](../../../../glossary.md#term-validation))</span>
- graph capture, graph traversal, topological sorting, cycle detection, common-subexpression
  elimination, <span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [graph glossary](../../../../glossary.md#graph); [topological glossary](../../../../glossary.md#term-topological-order); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>`CompiledNode`, `GraphValue`, [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), or
  `OperationId` creation or attachment
- producer/consumer indexes, graph membership, output index or multi-output producer grouping,
  publication bindings/plans, compile artifacts, or compiler entry points<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [publication glossary](../../../../glossary.md#term-publication); [bindings glossary](../../../../glossary.md#term-binding); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- gradient rules, backward flags, gradient Tensor state, trainable role, autograd, optimizer, or
  training behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Tensor glossary](../../../../glossary.md#tensor); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- provenance mutation, replacement, clearing, builder, registry, global lookup, weak references,
  serialization, reflection-based discovery, or a service locator<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup); [serialization glossary](../../../../glossary.md#term-serialization); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- storage allocation or attachment for derived tensors, typed access, eager evaluation, execution,
  runtime residency, prepared state, backend ownership/support, device state, kernels, or routes<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [backend ownership glossary](../../../../glossary.md#backend-ownership); [device glossary](../../../../glossary.md#term-device); [kernels glossary](../../../../glossary.md#kernel))</span>
- changing public factory overloads, eager leaf initializer semantics, ID allocation policy,
  existing operation contracts, graph contracts, dependencies, Gradle, architecture, packages, or
  another module<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#masks-padding-and-eager-evaluation))</span><span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [operation glossary](../../../../glossary.md#operation); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- implementing or creating the detailed specification for task 0014

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0004](0004-typed-identifiers.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0008](0008-graph-value-and-node-model.md)
- [Task 0009](0009-compiled-graph-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0012I](0012i-bernoulli-random-tensor-creation.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Capability origin

The legacy implementation stored an optional [operation](../../../../glossary.md#operation) and [mutable](../../../../glossary.md#term-mutable) parent-[tensor](../../../../glossary.md#tensor) list inside an
internal mutable Tensor [node](../../../../glossary.md#node), together with [gradient](../../../../glossary.md#term-gradient), backward, [runtime](../../../../glossary.md#term-runtime), and storage behavior. That
is evidence that public expression results need discoverable operation/input origin for [graph
capture](../../../../glossary.md#capture-graph-ports-and-reconstruction). It is not a design to copy.

The new contract retains only the selected capability as a new [immutable](../../../../glossary.md#term-immutable) value. It does not copy
legacy source, `TensorNode`, mutable [graph](../../../../glossary.md#graph) setters, gradient rules, backward flags, runtime [aliases](../../../../glossary.md#aliasing-and-arena),
package structure, or layer coupling.

## Architecture constraints

- `modules/model` owns public [Tensor](../../../../glossary.md#tensor) state, [operation](../../../../glossary.md#operation) semantics, and the minimal [provenance](../../../../glossary.md#provenance) value.
- `Tensor` remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and must not become a `CompiledNode`, `GraphValue`, or
  other IR object.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- [`TensorId`](../../../../glossary.md#tensorid) remains stable public Tensor [identity](../../../../glossary.md#term-identity-meanings). Provenance must contain no [graph](../../../../glossary.md#graph)-local
  [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), `OperationId`, or owning-graph identity.
- `Operation` remains a backend-independent semantic value and exposes no support, ownership,
  route, execution, compiler, runtime, or backend behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Provenance is immutable after Tensor construction. Tensor host-storage association remains the
  only mutable state in this task.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [host glossary](../../../../glossary.md#term-host))</span>
- Graph traversal, validation, snapshotting, ID assignment, compilation, and conversion into
  immutable graph records belong to a later compiler task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Package direction remains acyclic: `model.tensor` may depend on `model.operation`; operation must
  not depend on Tensor, provenance, or graph state.
- Public factory initializers remain provenance-free leaves. Only the new package-private derived
  path may attach provenance.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span>
- Central `TensorFactory` identity allocation remains the only construction path used by future
  expression helpers; no second allocator, caller-supplied ID, registry, or service is introduced.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- Stop if implementation needs graph identity, output grouping, descriptor inference, operation-
  family validation, mutable provenance, another package/module, dependency, or architecture rule.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor state, TensorFactory identity
  construction, provenance, and same-package focused tests.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.model.operation` — supplies the [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent
  [`Operation`](../../../../glossary.md#operation) value referenced by [provenance](../../../../glossary.md#provenance).
- `java.util` — supplies `List`, `Optional`, and defensive immutable snapshots.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorProvenance` — public immutable origin DTO because
  public Tensor inspection and a later compiler module must be able to read the semantic producer
  and ordered input tensors.<span data-glossary-navigation="true"> ([DTO glossary](../../../../glossary.md#data-transfer-object--dto); [compiler glossary](../../../../glossary.md#term-compiler); [module glossary](../../../../glossary.md#term-module); [producer glossary](../../../../glossary.md#term-producer))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — retains optional provenance as stable metadata
  beside ID, descriptor, and label; host storage remains its sole mutable field.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [descriptor glossary](../../../../glossary.md#term-descriptor); [host storage glossary](../../../../glossary.md#host-storage); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- `io.github.pho001.synaptik.model.tensor.TensorFactory` — retains centralized identity allocation
  and gains one package-private derived-construction seam for later same-package expression code.
- `TensorProvenanceTest`, `TensorTest`, and `TensorFactoryTest` remain in the same package so they
  can verify package-private constructor/factory boundaries without widening production API.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Required contract

### TensorProvenance record

Create exactly:

```java
public record TensorProvenance(Operation operation, List<Tensor> inputs)
```

The record has exactly those two components in that order and no additional instance/static state,
nested type, factory, builder, or convenience method. Its [canonical](../../../../glossary.md#term-canonical-form) constructor validates in this
order:

1. null [`operation`](../../../../glossary.md#operation): `NullPointerException`, message `operation`;
2. null `inputs`: `NullPointerException`, message `inputs`;
3. first null input in ascending order: `NullPointerException`, message `inputs[<index>]`;
4. [snapshot](../../../../glossary.md#term-snapshot) the fully validated list with `List.copyOf(inputs)`.

An empty input list is valid for future zero-input semantic operations. Repeated [Tensor](../../../../glossary.md#tensor) references
are valid because an operation may consume the same expression in more than one ordered role.
Input order is semantic and must be preserved. The record retains the exact [immutable](../../../../glossary.md#term-immutable) `Operation`
reference and exact [identity](../../../../glossary.md#term-identity-meanings)-bearing Tensor [elements](../../../../glossary.md#term-element), but it never retains the caller's [mutable](../../../../glossary.md#term-mutable)
list object.

Record-generated `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString` remain in use. Tensor elements have ordinary
object-identity equality, so [provenance](../../../../glossary.md#provenance) equality does not turn equal Tensor IDs into equal input
objects. Diagnostic text is not [serialization](../../../../glossary.md#term-serialization), [dispatch](../../../../glossary.md#term-dispatch), [graph](../../../../glossary.md#graph) identity, or a stable [wire format](../../../../glossary.md#term-wire-format).

Do not validate operation [arity](../../../../glossary.md#term-arity), kind-to-attribute compatibility, input [descriptors](../../../../glossary.md#term-descriptor), output
descriptor, data types, [shapes](../../../../glossary.md#shape), [layouts](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient) intent, cycles, or graph-wide structure. Those
checks need operation-family or [compiler](../../../../glossary.md#term-compiler) context that this local origin [DTO](../../../../glossary.md#data-transfer-object--dto) does not own.

### [Tensor](../../../../glossary.md#tensor) state and constructor

After this task, Tensor has exactly these five fields in this order:

```java
private final TensorId id;
private final TensorDescriptor descriptor;
private final Optional<String> label;
private final Optional<TensorProvenance> provenance;
private HostTensorStorage hostStorage;
```

The first four fields are final. [Host storage](../../../../glossary.md#host-storage) remains the only [mutable](../../../../glossary.md#term-mutable) field. Replace the sole
package-private constructor with exactly:

```java
Tensor(
        TensorId id,
        TensorDescriptor descriptor,
        Optional<String> label,
        Optional<TensorProvenance> provenance,
        Optional<HostTensorStorage> hostStorage)
```

Validate container references in [parameter](../../../../glossary.md#term-parameter) order with `Objects.requireNonNull` and exact messages
`id`, [`descriptor`](../../../../glossary.md#term-descriptor), `label`, [`provenance`](../../../../glossary.md#provenance), and `hostStorage`. Complete those null checks before
label [normalization](../../../../glossary.md#term-normalization) or storage semantic [validation](../../../../glossary.md#term-validation). Preserve the existing label normalization and
then the existing storage type/capacity/[liveness](../../../../glossary.md#term-liveness) behavior and messages unchanged.

Store the supplied non-null provenance optional through ordinary assignment. Treat `Optional` as a
value-based container: no [API](../../../../glossary.md#term-application-programming-interface-api) or test may promise its [identity](../../../../glossary.md#term-identity-meanings). A present result must contain the
exact [immutable](../../../../glossary.md#term-immutable) `TensorProvenance` reference supplied at construction. Provenance and host storage
are independent: package-private construction may represent a derived tensor with no storage or a
derived tensor with compatible host storage without changing either contract.

Add exactly one declared public method:

```java
public Optional<TensorProvenance> provenance()
```

It returns the immutable optional origin value without [synchronization](../../../../glossary.md#term-synchronization). Existing accessors and
storage methods retain their [signatures](../../../../glossary.md#term-signature) and behavior; only the three storage association methods
remain synchronized. `equals` and [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests) remain inherited object [identity](../../../../glossary.md#term-reference-identity). `toString()` remains
exactly [metadata](../../../../glossary.md#term-metadata)-only over Tensor type, ID, descriptor, and normalized label and must not include
provenance, [operation](../../../../glossary.md#operation), inputs, [graph](../../../../glossary.md#graph) expansion, or storage facts.

### [Leaf](../../../../glossary.md#term-leaf) and derived factory construction

Every existing public TensorFactory method must continue to create a [provenance](../../../../glossary.md#provenance)-free leaf. Change
only the existing final constructor call in public `create(...)` so it passes
`Optional.empty()` in the provenance position while passing caller storage unchanged. Do not add a
public provenance [parameter](../../../../glossary.md#term-parameter) or overload and do not change [validation](../../../../glossary.md#term-validation), ID, allocation, import,
population, random-source, label, or storage behavior.

Add exactly one package-private method:

```java
static Tensor createDerived(
        TensorDescriptor descriptor,
        Optional<String> label,
        TensorProvenance provenance)
```

Validate [`descriptor`](../../../../glossary.md#term-descriptor), `label`, and `provenance` for null in that order with exact messages matching
the parameter names before allocating an ID. Then allocate exactly once with the existing private
`nextTensorId()`, invoke the new [Tensor](../../../../glossary.md#tensor) constructor with `Optional.of(provenance)` and
`Optional.empty()` [host storage](../../../../glossary.md#host-storage), and return that exact Tensor.

The method does not inspect the descriptor, [operation](../../../../glossary.md#operation), or inputs; normalize labels; infer [gradient](../../../../glossary.md#term-gradient)
intent; allocate/attach storage; copy tensors; traverse provenance; or validate semantic
compatibility. Existing Tensor construction remains the sole label semantic path. Consequently:

- null factory [arguments](../../../../glossary.md#term-argument) consume no ID;
- a blank label consumes the allocated ID and fails through existing Tensor validation;
- allocator exhaustion occurs before Tensor construction and no ID is rolled back or reused;
- successful construction returns no host storage and the exact provenance reference.

The method is a package-private construction seam, not public [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), a [compiler](../../../../glossary.md#term-compiler) [API](../../../../glossary.md#term-application-programming-interface-api), a
[registry](../../../../glossary.md#term-registry), or a [service locator](../../../../glossary.md#term-service-locator). Do not change `nextTensorId`, its fields, or exhaustion/[concurrency](../../../../glossary.md#term-concurrency)
policy.

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| Empty [provenance](../../../../glossary.md#provenance) input list | Valid zero-input semantic origin |
| Same [Tensor](../../../../glossary.md#tensor) appears twice in inputs | Valid; both ordered roles are preserved |
| Caller mutates source list after construction | Provenance remains unchanged |
| Caller supplies a null input [element](../../../../glossary.md#term-element) | Invalid with indexed message |
| TensorFactory [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf) creation | Tensor provenance is empty |
| Package-private derived creation | Exact provenance is present and [host storage](../../../../glossary.md#host-storage) is empty |
| Derived creation with blank label | Invalid after ID allocation; ID is consumed |
| Tensor storage is attached/replaced/cleared | Provenance remains the same exact value |
| Input Tensor storage later changes or dies | Provenance input [identity](../../../../glossary.md#term-identity-meanings) remains; no storage [snapshot](../../../../glossary.md#term-snapshot) is implied |
| Two provenance values use equal [Operation](../../../../glossary.md#operation) and same input objects | Record value equality may be true; this is not [producer](../../../../glossary.md#term-producer)-[occurrence](../../../../glossary.md#term-occurrence) identity |
| Two operations have equal semantic values | Equality does not perform [common-subexpression elimination](../../../../glossary.md#term-common-subexpression-elimination-cse) |
| Provenance would require [graph](../../../../glossary.md#graph)-local IDs or cycle checks | Deferred to [compiler](../../../../glossary.md#term-compiler)-owned [capture](../../../../glossary.md#term-capture-meanings) |

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProvenanceTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping:

- `docs/api/compile-api.md` — graph capture and compiler entry points remain conceptual and
  unimplemented.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- `docs/api/training-api.md` — no [gradient](../../../../glossary.md#term-gradient), trainable, [optimizer](../../../../glossary.md#term-optimizer), or session behavior changes.
- `docs/planning/modules/model/capabilities.md` — already selects minimal provenance and legacy
  capability evidence without prescribing this local representation.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Existing [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`TensorId`](../../../../glossary.md#tensorid), `TensorDescriptor`,
  `GraphValue`, `CompiledNode`, [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), and `PublicationBinding` Javadocs.
- Focused architecture documentation and architecture tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most three production files, three tests, and five documentation/[planning](../../../../glossary.md#term-planning) files: eleven paths
total. The existing [Tensor](../../../../glossary.md#tensor) exact-[shape](../../../../glossary.md#shape) test and TensorFactory exact-method test must change in the
same [atomic](../../../../glossary.md#resource-transaction-and-rollback) buildable change because the constructor/field/method contracts evolve together.

Do not modify [operation](../../../../glossary.md#operation) or [graph](../../../../glossary.md#graph) production/tests, any other existing factory behavior test,
[Gradle](../../../../glossary.md#term-gradle), `AGENTS.md`, `ARCHITECTURE.md`, focused architecture docs/tests, capabilities, another
[module](../../../../glossary.md#term-module), or unrelated documentation. Do not create a task-0014 specification. Stop beyond eleven
paths or when an architecture, multi-output, [capture](../../../../glossary.md#term-capture-meanings), [inference](../../../../glossary.md#term-inference), or operation-family decision is
required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document `TensorProvenance` as immutable expression-origin metadata, not IR, occurrence identity,
  executable behavior, or graph membership.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- Record/type/constructor/accessor Javadocs must explain exact Operation retention, ordered
  defensive list snapshot, exact Tensor-element identities, empty/repeated input acceptance,
  null/index failures, value equality limits, and deferred validation.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [snapshot glossary](../../../../glossary.md#term-snapshot); [Tensor glossary](../../../../glossary.md#tensor); [element glossary](../../../../glossary.md#term-element); [validation glossary](../../../../glossary.md#term-validation))</span>
- Update Tensor type/constructor/accessor Javadocs for immutable optional provenance, exact
  reference retention, Optional value semantics, stable lifetime, lack of synchronization need,
  independence from storage mutation, and distinction from graph-local nodes/values/IDs.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [lifetime glossary](../../../../glossary.md#term-lifetime); [synchronization glossary](../../../../glossary.md#term-synchronization); [nodes glossary](../../../../glossary.md#node))</span>
- Preserve and explicitly document that host storage remains Tensor's sole mutable state, ordinary
  object equality remains, and diagnostic text omits provenance to avoid graph expansion and is not
  serialization.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [mutable glossary](../../../../glossary.md#term-mutable); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Update TensorFactory type/create Javadocs so public methods are clearly provenance-free eager
  leaves while the package-private derived seam owns no graph capture or semantic validation.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- Document `createDerived` parameters, null order, ID/failure effects, no-storage result, exact
  provenance retention, and all <span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>`@return`/`@throws` behavior.
- Review existing operation, descriptor, identity, graph-model, storage, and publication Javadocs;
  record why they remain accurate or stop on an out-of-scope discrepancy.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [model glossary](../../../../glossary.md#term-model); [publication glossary](../../../../glossary.md#term-publication); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one public `TensorProvenance` record exists with the two required components/order and no
  extra state/API. Constructor validation, list snapshot, ordering, empty/repeated inputs, record
  methods, and indexed failures match this task.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [validation glossary](../../../../glossary.md#term-validation); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Tensor has exactly five fields in the required order, with exactly the first four final and only
  host storage mutable. It retains exactly one package-private five-parameter constructor.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [mutable glossary](../../../../glossary.md#term-mutable); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- [Tensor](../../../../glossary.md#tensor) declares exactly one new public `provenance()` accessor. It is not synchronized and
  returns empty or the exact underlying provenance reference using Optional value semantics.
- All existing Tensor metadata, label, storage compatibility/lifetime/synchronization, identity
  equality/hashing, and diagnostic behavior remain unchanged.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [lifetime glossary](../../../../glossary.md#term-lifetime); [synchronization glossary](../../../../glossary.md#term-synchronization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Storage transitions, read-only storage, and late storage death do not alter provenance. Tensor
  diagnostic text remains unchanged and contains no operation/input/provenance detail.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Existing public TensorFactory API and all eager initializer behavior remain unchanged and return
  provenance-free leaves.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Exactly one package-private `createDerived` method exists with the required signature,
  validation order, centralized ID allocation, no-storage result, exact provenance, and failure
  side effects. No second allocator or public caller-supplied provenance API appears.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- No operation family, expression method, inference, traversal, graph record/ID, gradient,
  compiler, storage allocation, runtime, backend, dependency, package, or architecture behavior is
  introduced.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused [provenance](../../../../glossary.md#provenance)/Tensor/factory tests and all existing [model](../../../../glossary.md#term-model)/repository tests pass.
- Reflection, `javap`, import, bytecode, package-direction, eleven-path, documentation, and status
  checks pass.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- A separate documentation-focused agent/thread finalizes affected Javadocs, Tensor API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for compile/training API, capabilities, architecture docs, and related contracts.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Task 0013 becomes Complete only after both passes. The separately coordinated task 0013A remains
  the next Draft planning frontier without a detailed specification; task 0014 also remains Draft
  without a detailed specification.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must cover exact record/components/[API](../../../../glossary.md#term-application-programming-interface-api), validation order/messages, [mutable](../../../../glossary.md#term-mutable)-source
[snapshot](../../../../glossary.md#term-snapshot), order, empty/repeated inputs, exact references, record equality limits, exact [Tensor](../../../../glossary.md#tensor)
fields/constructor/public methods, absent/present [provenance](../../../../glossary.md#provenance), Optional semantics, storage
independence, unchanged diagnostics/equality, all public factory leaves, `createDerived` null/blank/
exhaustion/ID behavior, and allocator-state restoration after reflective boundary checks. Do not
weaken or remove existing Tensor/storage/factory allocator/[concurrency](../../../../glossary.md#term-concurrency)/exhaustion coverage.

Manually inspect `javap -p -c -s` for all three affected production types; verify exact fields,
components, constructor/method visibility and [signatures](../../../../glossary.md#term-signature), final/synchronized modifiers,
`List.copyOf`, one `nextTensorId` call in `createDerived`, empty storage, empty public-[leaf](../../../../glossary.md#term-leaf)
provenance, and no extra state. Inspect imports/package direction for forbidden [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/
[planning](../../../../glossary.md#term-planning)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend)/training dependencies. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), links, anchors,
code fences, terminology, whitespace, exact eleven paths, synchronized planning status, and absence
of a task-0014 spec.

## Dependencies

- Task 0006 supplies the [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [`Operation`](../../../../glossary.md#operation) value.
- Task 0011 supplies Tensor identity/descriptor/label/storage behavior and its package-private
  construction boundary.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Task 0012 supplies the central [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-wide ID allocator and public [leaf](../../../../glossary.md#term-leaf) construction path.
- Tasks 0008–0009 define the separate immutable [graph](../../../../glossary.md#graph) [model](../../../../glossary.md#term-model) that [provenance](../../../../glossary.md#provenance) must not duplicate.

## Follow-up tasks

- Separately coordinated task 0013A owns full-value and identity-matrix tensor creation and remains
  the next Draft frontier without a detailed specification.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [tensor glossary](../../../../glossary.md#tensor))</span>
- The post-0013A model foundation checkpoint must explicitly decide whether task 0014 remains the
  next sequential frontier or a documented cross-module vertical slice should begin.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module); [slice glossary](../../../../glossary.md#slice))</span>
- Task 0014, if selected next, owns concrete elementwise operation kinds/attributes, descriptor
  semantics, public expression construction, and use of <span data-glossary-navigation="true"> ([elementwise operation glossary](../../../../glossary.md#elementwise-unary-binary-and-interval-vocabulary); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`createDerived`.
- A later compiler task owns provenance traversal, graph-local ID allocation, graph validation,
  immutable graph snapshotting, publication planning, and compile artifacts.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- Multi-output public producer grouping/output-slot semantics must be planned only when a concrete
  operation task requires them; this skeleton does not silently choose that architecture.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [operation glossary](../../../../glossary.md#operation))</span>

Do not create detailed follow-up specifications in this task.

## Architecture impact

Expected impact: None. The architecture already assigns minimal public [Tensor](../../../../glossary.md#tensor) origin [metadata](../../../../glossary.md#term-metadata) and
[backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics to `modules/model`, while [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) belongs to the
[compiler](../../../../glossary.md#term-compiler). This task implements that boundary without changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) or dependencies.

If implementation requires [graph](../../../../glossary.md#graph)-local [identity](../../../../glossary.md#term-identity-meanings), compiler traversal, multi-output [producer](../../../../glossary.md#term-producer)
identity, [mutable](../../../../glossary.md#term-mutable) [provenance](../../../../glossary.md#provenance), or another module/package direction, stop and report the required
decision before editing architecture or expanding [scope](../../../../glossary.md#term-scope).

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0004/0006/0007/0008/0009/0011/0012/0012I/0013, Tensor API,
Compile API, glossary, current Operation/Tensor/TensorFactory/graph contracts and focused tests,
and Java 26 Gradle configuration.

Implement task 0013 exactly. Add only TensorProvenance.java for the new production concept; modify
only Tensor.java and TensorFactory.java otherwise. Add TensorProvenanceTest and update TensorTest
and TensorFactoryTest only for the exact new state/constructor/accessor/package-private factory
surface. Preserve all existing eager initializer, allocator, storage, and operation/graph behavior.

TensorProvenance is exactly a public record of Operation plus an ordered immutable List<Tensor>.
Follow exact null/index validation, snapshot/order/reference, empty/repeated input, and record-value
rules. Tensor gets immutable Optional provenance, one non-synchronized accessor, and the exact new
constructor while host storage remains its sole mutable state. Public factory paths remain empty-
provenance leaves. Add exactly one package-private createDerived that uses the existing allocator,
attaches exact provenance, creates no storage, and performs no inference/traversal/semantic check.

Do not add operation families/expression methods, graph IDs/records/capture/traversal, descriptor
inference, output grouping, gradients/autograd, storage/evaluation, compiler/runtime/backend work,
dependencies/build/architecture changes, or later specs. Stop beyond eleven paths or on
architecture/multi-output uncertainty.

Run every specified focused/aggregate test, Javadoc, bytecode/import/manual, documentation/link/
whitespace/scope/status check. Then hand the actual diff and validation evidence to a separate
clean-context documentation agent/thread in the same change. It must independently inspect source,
tests, generated Javadoc, and evidence; finalize permitted Javadocs/Tensor API/glossary/planning;
record reasoned compile/training API, capability, architecture, and existing-contract no-change
conclusions; and rerun validation.

Update only this task, model master plan, and roadmap for planning status/evidence. Do not mark
0013 Complete until both passes and all validation succeed. Leave task 0014 Draft without a
detailed specification. Do not commit or push.
```

## Local decisions

- One `TensorProvenance` value contains only semantic operation plus ordered inputs; Tensor already
  owns output descriptor and identity, so duplicating them would invite inconsistency.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Provenance is final optional Tensor metadata. A setter would permit origin changes/cycles during
  capture and repeat the legacy mutable-node coupling.<span data-glossary-navigation="true"> ([Provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [mutable glossary](../../../../glossary.md#term-mutable); [node glossary](../../../../glossary.md#node))</span>
- Inputs are strong identity references in an immutable list snapshot. Weak references could lose
  the graph before compilation; copying Tensor objects would destroy expression identity.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Empty and repeated inputs are valid local structures. Arity and graph-wide validity require
  future semantic/compiler context.<span data-glossary-navigation="true"> ([Arity glossary](../../../../glossary.md#term-arity); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Public factory methods remain leaves. One package-private `createDerived` centralizes ID
  allocation without exposing caller-supplied provenance or implementing expressions early.
- No output slot or producer occurrence ID is added. Current operation-family planning can model
  one public result per provenance; a real multi-output need must receive an explicit later design.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence); [planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model))</span>

## Known limitations

- Provenance holds strong references to the reachable input expression DAG until tensors become
  unreachable; no pruning or weak-reference policy exists.<span data-glossary-navigation="true"> ([Provenance glossary](../../../../glossary.md#provenance); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [tensors glossary](../../../../glossary.md#tensor))</span>
- This skeleton does not prove acyclicity, descriptor compatibility, operation arity, or graph
  validity and cannot compile or execute an expression.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [arity glossary](../../../../glossary.md#term-arity); [graph glossary](../../../../glossary.md#graph); [compile glossary](../../../../glossary.md#compile))</span>
- Record value equality is not producer-occurrence identity and must not be used as automatic
  common-subexpression elimination.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- Public factory output remains leaf-only. Concrete public derived tensors arrive with later
  operation-family tasks.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- Multi-output producer grouping/output slots, graph-local IDs, gradient rules, and capture
  snapshots are deferred to owning tasks.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the architecture/[model](../../../../glossary.md#term-model) boundaries, [lifecycle](../../../../glossary.md#lifecycle), current [Operation](../../../../glossary.md#operation)/[Tensor](../../../../glossary.md#tensor)/factory
and [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) contracts, completed task chain, capability baseline, Tensor/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api),
glossary, exact-[shape](../../../../glossary.md#shape) tests, and read-only legacy Tensor/TensorNode/traversal evidence. The proposed
contract stays in `model.tensor`, adds no [dependency](../../../../glossary.md#term-dependency) or architecture rule, preserves central ID
allocation, and fits eleven implementation/documentation paths.

- Implementation context `/root/implement_model_0013` added `TensorProvenance`, extended `Tensor`
  and `TensorFactory`, and updated only the three focused test classes. Independent documentation
  context `/root/implement_model_0013/review_model_0013_docs` then inspected the actual source,
  tests, generated Javadoc, XML reports, bytecode, build configuration, and complete workspace
  diff. It applied General plus API/Javadoc style to Java, Tensor API, and glossary work and
  Planning style to this task, the model master plan, and roadmap. Example format was reviewed but
  no executable example changed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorProvenanceTest` — `BUILD SUCCESSFUL`; 6 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 14 tests, zero failures,
  errors, or skips. The final test also proves that input storage may die and be cleared without
  changing retained provenance, and that all optional-container null checks precede label
  normalization.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — `BUILD SUCCESSFUL`; 8 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 33 suites and 251
  tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` without Javadoc errors. Generated pages
  contain `TensorProvenance`, `Tensor.provenance()`, immutable origin/list ownership, sole mutable
  host storage, provenance-free public leaves, and the graph-capture boundary. Source review covers
  the package-private Tensor constructor and <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [mutable glossary](../../../../glossary.md#term-mutable); [host storage glossary](../../../../glossary.md#host-storage); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>`createDerived`, which public/protected Javadoc does
  not render.
- `./gradlew test` — `BUILD SUCCESSFUL` for the repository; the validation run reported 36
  actionable tasks with no failing task.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for `TensorProvenance`,
  `Tensor`, and `TensorFactory` confirmed exactly two final record fields and one `List.copyOf`,
  exactly five Tensor fields in required order with only storage mutable, the five-argument
  package-private constructor and direct unsynchronized provenance accessor, and only the three
  storage methods synchronized. Targeted <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`createDerived` bytecode contains descriptor/label/
  provenance null checks, one <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>`nextTensorId()` call, `Optional.of(provenance)`, empty storage, and
  one Tensor construction, with no traversal, inference, or semantic validation.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Import and package-direction scans found only the existing model/JDK dependencies plus the new
  permitted <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`model.tensor -> model.operation` edge. No graph, compiler, planning, runtime, prepare,
  backend, engine, trace, or training dependency was introduced; operation and storage packages do
  not depend back on Tensor provenance.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
- The corrected local Markdown checker resolved 173 file targets and heading anchors across the
  five changed documentation/planning files with zero errors. Two preliminary checker invocations
  stopped before completing because one Ruby regular expression interpolated a heading quantifier
  and the installed Ruby lacks <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`Array#filter_map`; the compatible corrected checker produced the
  passing result. Backtick fence counts are even, no tilde fence is present, terminology/status
  checks found no stale planned-provenance claim, targeted trailing-whitespace scans found no
  matches, and `git diff --check` passed.
- Task-specific scope contains exactly the authorized eleven paths: three production files, three
  focused tests, Tensor API, glossary, this task, model master plan, and roadmap. A concurrent
  unrelated edit in <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`capabilities.md` adds only task-0013A factory-capability bullets; it was clean
  at documentation handoff, does not overlap provenance, was preserved without editing, and is
  excluded from task 0013's path count by coordinator direction. Concurrent task-0013A Draft
  planning in the already-authorized master plan and roadmap was also preserved and merged with
  task 0013 status. Task 0014 remains Draft and no task-0014 specification exists.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- `docs/api/compile-api.md` remains unchanged because task 0013 adds no capture entry point,
  compiler behavior, graph-local allocation, compile artifact, or publication plan.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [publication plan glossary](../../../../glossary.md#publication-plan))</span>
  `docs/api/training-api.md` remains unchanged because no gradient, trainable, autograd, optimizer,
  or session behavior changed. <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`capabilities.md` already selected minimal provenance and needed no
  provenance edit; its unrelated concurrent task-0013A addition is described above.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests remain
  unchanged because provenance stays in the model, graph capture remains compiler-owned, and no
  module boundary, dependency rule, lifecycle, backend behavior, or end-to-end contract changed.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [module glossary](../../../../glossary.md#term-module))</span>
- Existing `Operation`, [`OperationKind`](../../../../glossary.md#operationkind), and [`OperationAttrs`](../../../../glossary.md#operationattrs) Javadocs remain accurate because
  provenance retains an Operation without changing semantic-kind/attribute validation or adding
  occurrence identity. <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`TensorId`](../../../../glossary.md#tensorid) and `TensorDescriptor` remain accurate because identity and
  logical descriptor semantics did not change. `GraphValue`, `CompiledNode`,
  [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), and `PublicationBinding` remain accurate because provenance adds no
  graph-local ID, membership, capture, producer index, or publication context. <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [publication glossary](../../../../glossary.md#term-publication))</span>`HostTensorStorage`
  and `MemorySegmentStorage` remain accurate because host storage ownership, sizing, liveness, and
  raw-memory behavior did not change.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>

## Implementation notes

- Added the exact public `TensorProvenance(Operation, List<Tensor>)` record with deterministic null
  checks, ordered immutable snapshot ownership, exact element/reference retention, empty/repeated
  input support, and record value semantics.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [element glossary](../../../../glossary.md#term-element))</span>
- Added final optional [provenance](../../../../glossary.md#provenance) to [`Tensor`](../../../../glossary.md#tensor), a direct unsynchronized accessor, and the exact
  five-argument package-private constructor while retaining host storage as the sole mutable state,
  ordinary object equality, and provenance-free metadata diagnostics.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [host storage glossary](../../../../glossary.md#host-storage); [mutable glossary](../../../../glossary.md#term-mutable); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Kept every public factory construction/population path provenance-free and added one
  package-private `createDerived` seam that reuses central ID allocation, attaches exact provenance,
  creates no storage, and performs no graph or semantic work.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Finalized affected Javadocs, Tensor API, glossary, and planning status. The public reference now
  distinguishes implemented origin metadata from still-planned concrete expression construction
  and compiler capture.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>

## Completion summary

- Completed changes: Implemented and documented immutable Tensor provenance and the bounded
  derived-construction seam without turning Tensor into graph IR.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Files changed or created: `TensorProvenance.java`, `Tensor.java`, `TensorFactory.java`, their
  three focused tests, Tensor API, glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused suites passed 6/6, 14/14, and 8/8; all 251 model tests, model
  Javadoc, root tests, bytecode/API/import/package checks, generated documentation, 173 Markdown
  link/anchor checks, fence/terminology/whitespace checks, scope/status checks, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0013/review_model_0013_docs` completed the independent pass using General,
  API/Javadoc, and Planning profiles; no executable example changed.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe current immutable provenance, public
  factory leaves, the internal derived seam, record equality limits, and compiler/graph boundaries.
  Compile API, Training API, capabilities, architecture documents/tests, and unrelated contracts
  required no provenance-specific edit for the reasons recorded above.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [Compile glossary](../../../../glossary.md#compile))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `TensorProvenance`, affected [`Tensor`](../../../../glossary.md#tensor), and affected `TensorFactory` contracts are
  final; related operation, identity, descriptor, graph, publication, and storage Javadocs remain
  accurate unchanged.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [publication glossary](../../../../glossary.md#term-publication))</span>
- Glossary impact: The implementation-status convention, Provenance, Tensor, Tensor factory, and
  Tensor-versus-graph-value distinction now reflect the implemented contract.<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Unresolved issues: None for task 0013. The preserved unrelated `capabilities.md` and task-0013A
  planning changes belong to their separate coordinated work.
- Follow-up required: None for task 0013. Task 0013A is the next Draft planning frontier without a
  detailed specification; task 0014 remains Draft without a detailed specification.

Status: Complete
