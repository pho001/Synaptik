# Task 0018L: Shared Multi-Output [Tensor](../../../../glossary.md#tensor) [Provenance](../../../../glossary.md#provenance)

**Editorial reading key (current context, not a status update).**

This task records the original shared-producer migration, not today’s full contract. An expression
helper passes an operation, ordered input Tensor references and output descriptors to the
package-private factory; these are descriptions and references, not computed result values. A
producer identifies one operation occurrence; provenance selects its zero-based output slot. For
negation, slot zero describes the result, without evaluating it. Equal-looking calls remain distinct
occurrences, and descriptor agreement requires the same object, not merely equal fields. Model
semantics defines meaning; execution providers realize supported routes rather than redefine it.
Semantic responsibility is not buffer ownership: storage-free expressions neither execute nor own
result buffers.

Read the current [producer
contract](../../../../reference/tensor/identity.md#expression-producers-and-indexed-provenance) for
the later canonical-wrapper lookup and producer/result reference cycle, contrasting with this task’s
prohibition on retaining outputs. Current invalid-index exceptions also differ from those specified
here. The [multi-output identity
section](../../../../reference/tensor/identity.md#multi-output-and-hidden-output-identity) explains
hidden sibling slots in current descriptions; those examples do not expand this task’s historical
scope or imply shared physical storage.

## Status

Complete

## Goal

Represent every derived public [Tensor](../../../../glossary.md#tensor) as one indexed output of one [immutable](../../../../glossary.md#term-immutable) expression [producer](../../../../glossary.md#term-producer),
so a single [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) can honestly own one or several ordered
Tensor results before [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings).

Use one producer [model](../../../../glossary.md#term-model) for both single-output and multi-output expressions. A producer owns the
operation, ordered input Tensor references, and ordered output [descriptors](../../../../glossary.md#term-descriptor). Each result Tensor
owns [provenance](../../../../glossary.md#provenance) containing the exact shared producer reference and its zero-based output index.

This task provides the model foundation required by later top-K, explicit [graph](../../../../glossary.md#graph) [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), [dropout](../../../../glossary.md#dropout--inverted-dropout),
[normalization](../../../../glossary.md#term-normalization)-statistics, and other genuine multi-output operations. It does not implement any of
those operations or compiler capture.

## [Scope](../../../../glossary.md#term-scope)

- Add one public final non-record [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) [identity](../../../../glossary.md#term-identity-meanings) class in the existing [tensor](../../../../glossary.md#tensor) package.
- Give each [producer](../../../../glossary.md#term-producer) exactly:
  - one [immutable](../../../../glossary.md#term-immutable) [`Operation`](../../../../glossary.md#operation) reference;
  - one immutable ordered [snapshot](../../../../glossary.md#term-snapshot) of input Tensor references; and
  - one immutable ordered snapshot of output `TensorDescriptor` references.
- Derive the actual output count from `outputDescriptors().size()`; do not store an independent
  count field.
- Validate the producer's final input and output counts through the operation's selected
  `OperationSignature`.
- Preserve empty and repeated input positions when the [signature](../../../../glossary.md#term-signature) permits them.
- Require at least one output descriptor, permit repeated descriptor references, and retain every
  exact immutable descriptor reference in encounter order.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Give `TensorProducer` ordinary object identity: two separately created producers remain
  different occurrences even when operation, inputs, and output descriptors are structurally
  equal.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>
- Replace the existing two-component `TensorProvenance(Operation, List<Tensor>)` record with a
  two-component `TensorProvenance(TensorProducer, int outputIndex)` record.
- Validate every provenance output index against its producer and provide derived compatibility
  accessors for <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>`operation()`, `inputs()`, and `outputDescriptor()`.
- Require a derived Tensor's descriptor to be the exact descriptor reference selected by its
  provenance output index.
- Replace the existing package-private single-output `TensorFactory.createDerived` boundary with
  a producer-aware single-output boundary that receives the operation and ordered inputs.
- Add one package-private multi-output factory boundary that accepts an operation, ordered inputs,
  and ordered output descriptors, creates exactly one producer, and returns all indexed output
  Tensors in one immutable ordered list.
- Migrate every existing single-output expression helper to the new factory boundary without
  changing its public Tensor API, operation, input order, result descriptor, ID allocation,
  label, storage, or semantic behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Update focused provenance/factory/Tensor tests and direct test-local provenance construction for
  the intentional new model.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Add focused test-local multi-output coverage without adding a production multi-output operation.
- Finalize affected Javadoc, Tensor API, Compile API, glossary, task evidence, master plan, and
  roadmap through the required targeted documentation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `topK`, sort, argsort, dropout, RNG state, normalization statistics, or any other production
  multi-output operation or result record<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- public `TopKResult`, `DropoutResult`, tuple, pair, destructuring, or general public multi-output
  collection API<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- changing current public [Tensor](../../../../glossary.md#tensor) expression method [signatures](../../../../glossary.md#term-signature) or fluent single-output syntax
- adding a public method to retrieve sibling output Tensor objects from a [producer](../../../../glossary.md#term-producer)
- storing output Tensor references in [`TensorProducer`](../../../../glossary.md#producer--tensorproducer)
- storing an independent output-count field in addition to output [descriptors](../../../../glossary.md#term-descriptor)
- storing output labels, host storage, values, gradients, publication state, or mutable result
  state in a producer<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [gradients glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- adding `ProducerId`, [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), graph membership, graph-local identity, or compiler
  state to Tensor, provenance, or producer<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- turning `TensorProducer` into `CompiledNode`, graph IR, a graph builder, a capture session, a
  registry, a service, or a canonicalization key<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [capture glossary](../../../../glossary.md#term-capture-meanings); [registry glossary](../../../../glossary.md#term-registry); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- structural producer equality, producer interning, common-subexpression elimination, or merging
  separately invoked equal expressions<span data-glossary-navigation="true"> ([common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- [compiler](../../../../glossary.md#term-compiler) traversal, [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), `GraphValue` or `CompiledNode` creation, descriptor inference,
  publication binding, optimization, autograd, or backward construction<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [binding glossary](../../../../glossary.md#term-binding); [autograd glossary](../../../../glossary.md#autograd))</span>
- [backend ownership](../../../../glossary.md#backend-ownership), [lowering](../../../../glossary.md#lowering), [kernel](../../../../glossary.md#kernel) selection, [prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime), execution, or [device](../../../../glossary.md#term-device) storage
- changing [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), `OperationSignature`, `CompiledNode`, or any production
  operation-family contract
- changing unstack semantics; task 0018O owns its conversion to repeated [scalar select](../../../../glossary.md#scalar-select)
- dependencies, [Gradle](../../../../glossary.md#term-gradle), architecture tests, another [module](../../../../glossary.md#term-module), `ARCHITECTURE.md`, or focused
  architecture documentation
- creating a detailed task-0018M specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially public Tensor state,
  model-owned operation semantics, immutable graph state, and the rule that Tensor is not IR<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially “Multi-output [provenance](../../../../glossary.md#provenance)”
- [Model master plan](../master-plan.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0008](0008-graph-value-and-node-model.md)
- [Task 0009](0009-compiled-graph-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns public Tensor state, Tensor descriptors, operation semantics, expression
  provenance, and this pre-capture producer description.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state and is not an IR node. Its producer reference describes
  expression origin only and establishes no graph membership.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../../glossary.md#node))</span>
- [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) is not `CompiledNode`. One producer may be captured into different nodes with
  different [`NodeId`](../../../../glossary.md#nodeid) and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values in separately compiled [graphs](../../../../glossary.md#graph).
- Producer identity is occurrence identity in the public expression model only. It is represented
  by the exact Java object reference, not a graph-local or globally allocated identifier.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [model glossary](../../../../glossary.md#term-model))</span>
- [`Operation`](../../../../glossary.md#operation) remains [backend](../../../../glossary.md#backend)-independent semantics. `TensorProducer` may consume its selected
  structural signature but must not add backend support, compiler policy, lowering, execution, or
  kernel metadata.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [compiler glossary](../../../../glossary.md#term-compiler); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Ordered output descriptors are immutable model facts already required by public result Tensors.
  They are not output Tensor objects, graph values, physical buffers, or compiler artifacts.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [buffers glossary](../../../../glossary.md#term-buffer); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Producer construction validates local structural counts only. It must not perform family-
  specific Shape or DataType inference, graph-wide validation, cycle traversal, or executable
  support checks.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- A producer must never retain its result Tensor objects. The direction remains result Tensor to
  provenance to producer, preventing a producer/output/provenance reference cycle.
- Single-output expressions and future multi-output expressions must use the same producer and
  indexed-provenance contracts. Do not retain a second legacy provenance representation or a
  nullable union of old and new state.
- If implementation cannot preserve exact existing single-output public behavior while completing
  this atomic migration, or if compiler/graph identity appears necessary in the producer, stop and
  report the design conflict before changing architecture or another module.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor state, descriptors, factories,
  expression helpers, producer identity, and provenance.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>

Packages added or changed:

- No package is added.
- The existing tensor package gains one cohesive producer contract and migrates its provenance and
  package-private expression-construction boundaries.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorProducer` — immutable expression-occurrence
  identity shared by all outputs because it composes operation, inputs, and ordered output
  descriptors before graph capture.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [occurrence glossary](../../../../glossary.md#term-occurrence); [operation glossary](../../../../glossary.md#operation); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- `io.github.pho001.synaptik.model.tensor.TensorProvenance` — immutable association from one result
  Tensor to one exact producer output position.
- `io.github.pho001.synaptik.model.tensor.TensorFactory` — package-private atomic construction
  boundary for one or all outputs of a producer.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — enforces exact descriptor/provenance-position
  agreement while retaining its existing public state role.

Test placement:

- `io.github.pho001.synaptik.model.tensor.TensorProducerTest` — validates producer ownership,
  identity, counts, snapshots, and absence of graph/runtime state.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Existing `TensorProvenanceTest`, `TensorFactoryTest`, and `TensorTest` validate indexed provenance,
  single- and multi-output construction, descriptor agreement, identity, and failure behavior.
- Existing expression tests remain behavior regression coverage; modify only tests that directly
  construct the replaced provenance contract or assert its exact API shape.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>

## Required contracts

### [`TensorProducer`](../../../../glossary.md#producer--tensorproducer)

Create one public final non-record class with exactly these three instance fields:

```java
private final Operation operation;
private final List<Tensor> inputs;
private final List<TensorDescriptor> outputDescriptors;
```

The class must:

- use a package-private constructor so producer creation remains inside the tensor expression-
  construction boundary;<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [tensor glossary](../../../../glossary.md#tensor))</span>
- null-check [`operation`](../../../../glossary.md#operation), `inputs`, and `outputDescriptors` in that order;
- inspect input [elements](../../../../glossary.md#term-element) in ascending index order and reject a null element with message
  `inputs[index]`;
- [snapshot](../../../../glossary.md#term-snapshot) inputs with `List.copyOf`, preserving order, empty input, and repeated exact Tensor
  references when permitted by the operation signature;<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature))</span>
- reject an empty output-[descriptor](../../../../glossary.md#term-descriptor) list with
  `IllegalArgumentException("outputDescriptors must not be empty")`;
- inspect output descriptor elements in ascending index order and reject a null element with
  message `outputDescriptors[index]`;
- snapshot output descriptors with `List.copyOf`, preserving order and repeated exact immutable
  descriptor references;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- after local collection [validation](../../../../glossary.md#term-validation), call
  `operation.signature().validateOccurrence(inputs.size(), outputDescriptors.size())` exactly once;
- retain the exact operation reference and exact list-element references;
- expose exactly `operation()`, `inputs()`, `outputDescriptors()`, and `outputCount()` as public
  accessors;
- derive `outputCount()` from `outputDescriptors.size()` and store no separate count;
- keep ordinary object [identity](../../../../glossary.md#term-reference-identity) equality and hashing by overriding neither `equals` nor [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests);
- add no public factory, builder, mutator, producer identifier, output Tensor accessor, lookup,
  registry, traversal, or graph API; and<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [registry glossary](../../../../glossary.md#term-registry); [graph glossary](../../../../glossary.md#graph); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- contain no [backend](../../../../glossary.md#backend), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), storage, value, [gradient](../../../../glossary.md#term-gradient), [publication](../../../../glossary.md#term-publication), or execution state.

The constructor may use temporary local validation state but must add no cached collection,
index, or [mutable](../../../../glossary.md#term-mutable) field. `toString()` is not a [serialization](../../../../glossary.md#term-serialization) or [identity](../../../../glossary.md#term-identity-meanings) contract; do not add an
object-method override merely to expose inputs or descriptors recursively.

### `TensorProvenance`

Replace the current record with exactly these components and order:

```java
public record TensorProvenance(
        TensorProducer producer,
        int outputIndex) {
}
```

The [canonical](../../../../glossary.md#term-canonical-form) constructor must:

1. reject a null [producer](../../../../glossary.md#term-producer) with `NullPointerException("producer")`;
2. reject a negative index with
   `IllegalArgumentException("outputIndex must be non-negative: " + outputIndex)`; and
3. reject an index greater than or equal to `producer.outputCount()` with a message identifying
   the index and available output count.

The record must retain the exact producer reference and primitive index. Add these derived public
accessors without adding record components or fields:

```java
Operation operation();
List<Tensor> inputs();
TensorDescriptor outputDescriptor();
```

They return the producer's exact [operation](../../../../glossary.md#operation), [immutable](../../../../glossary.md#term-immutable) inputs list, and exact [descriptor](../../../../glossary.md#term-descriptor) at
`outputIndex`, respectively. Keeping `operation()` and `inputs()` as derived accessors preserves
the useful read surface of the current [provenance](../../../../glossary.md#provenance) contract while changing [ownership](../../../../glossary.md#term-ownership-meanings) to the
shared producer.

Record-generated equality and hashing must use exact producer [identity](../../../../glossary.md#term-identity-meanings) through ordinary
[`TensorProducer`](../../../../glossary.md#producer--tensorproducer) object equality plus the output index. Two positions of one producer are unequal;
two separately invoked structurally equal producers remain unequal.

Do not add a compatibility constructor retaining the old `(Operation, List<Tensor>)` [model](../../../../glossary.md#term-model), a null
producer sentinel, optional output index, output [Tensor](../../../../glossary.md#tensor) list, [graph](../../../../glossary.md#graph) identity, traversal, or [mutable](../../../../glossary.md#term-mutable)
[binding](../../../../glossary.md#term-binding) phase.

### [Tensor descriptor](../../../../glossary.md#tensor-descriptor) agreement

The package-private [`Tensor`](../../../../glossary.md#tensor) constructor must preserve its existing component/null/label/storage
[validation](../../../../glossary.md#term-validation) order and add one local [provenance](../../../../glossary.md#provenance) agreement check after non-null component validation
and before storage attachment:

```text
if provenance is present:
    provenance.outputDescriptor() must be the exact same reference as descriptor
```

Reject a mismatch with an `IllegalArgumentException` whose message clearly identifies [descriptor](../../../../glossary.md#term-descriptor)
and provenance output-descriptor disagreement. Use [reference identity](../../../../glossary.md#term-reference-identity), not [structural equality](../../../../glossary.md#term-structural-equality),
because factory construction deliberately shares the one [immutable](../../../../glossary.md#term-immutable) descriptor object between the
[producer](../../../../glossary.md#term-producer) slot and result Tensor.

Do not inspect producer inputs, traverse provenance, infer descriptors, reject repeated producer
descriptors, or change Tensor equality, hashing, labels, storage [synchronization](../../../../glossary.md#term-synchronization), or diagnostics.

### `TensorFactory` construction boundaries

Replace the existing package-private single-output method accepting a preconstructed
`TensorProvenance` with:

```java
static Tensor createDerived(
        TensorDescriptor descriptor,
        Optional<String> label,
        Operation operation,
        List<Tensor> inputs);
```

It must validate non-null [arguments](../../../../glossary.md#term-argument) in declaration order before allocating a [Tensor](../../../../glossary.md#tensor) ID, create one
`TensorProducer(operation, inputs, List.of(descriptor))`, create [provenance](../../../../glossary.md#provenance) `(producer, 0)`, and
construct exactly one unlabeled-or-supplied-label, storage-free Tensor through the existing
package-private Tensor constructor. It must not copy or replace the [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), or
input Tensor references beyond the [producer](../../../../glossary.md#term-producer)'s [immutable](../../../../glossary.md#term-immutable) list [snapshot](../../../../glossary.md#term-snapshot).

Add exactly one package-private multi-output method:

```java
static List<Tensor> createDerivedOutputs(
        Operation operation,
        List<Tensor> inputs,
        List<TensorDescriptor> outputDescriptors);
```

It must:

1. null-check arguments in declaration order;
2. construct exactly one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), which owns element, snapshot, non-empty, and signature-
   cardinality validation;<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [signature glossary](../../../../glossary.md#term-signature); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
3. allocate one fresh Tensor ID per ordered descriptor only after producer [validation](../../../../glossary.md#term-validation) succeeds;
4. iterate the producer's immutable output-descriptor snapshot rather than the caller's list, and
   construct one storage-free, unlabeled Tensor per output position with the exact descriptor
   reference and `TensorProvenance(producer, index)`;
5. preserve output order; and
6. return one immutable ordered list containing every output Tensor.

Prevalidation failures consume no Tensor IDs. If identifier exhaustion occurs after some output
IDs have been allocated, consumed IDs are not rolled back or reused; no partial result list is
returned. Add no reset hook, caller-supplied ID, label list, output factory [callback](../../../../glossary.md#term-callback), generic tuple,
stream, array, map, or public multi-output method.

### Existing expression migration

Every current package-private expression helper that constructs:

```java
TensorProvenance provenance = new TensorProvenance(operation, inputs);
return TensorFactory.createDerived(descriptor, label, provenance);
```

must instead delegate once to the new [producer](../../../../glossary.md#term-producer)-aware single-output factory with the same exact:

- result [descriptor](../../../../glossary.md#term-descriptor) reference;
- label optional;
- [operation](../../../../glossary.md#operation) reference;
- ordered input [Tensor](../../../../glossary.md#tensor) references; and
- externally observable ID, provenance, storage, and failure behavior, except for the intentional
  new producer and output-index provenance representation.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>

Do not refactor operation-specific [validation](../../../../glossary.md#term-validation), combine expression helpers, alter public methods,
or change operation semantics while performing this mechanical migration.

## Affected files

Expected core production files:

- new `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected mechanical production migration:

- current `Tensor*Expressions.java` files under
  `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/` that directly construct
  `TensorProvenance` or call the replaced package-private `createDerived` overload

Expected tests:

- new `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProducerTest.java`
- existing `TensorProvenanceTest.java`
- existing `TensorFactoryTest.java`
- existing `TensorTest.java`
- existing [model](../../../../glossary.md#term-model) tests that directly construct `TensorProvenance` or assert its exact record/API
  shape; modify no unrelated test behavior<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Expected no-change reviews:

- `docs/planning/modules/model/capabilities.md` already records the selected producer, ordered
  output-descriptor, and indexed-result direction after this planning change;<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `ARCHITECTURE.md` and focused architecture documents already establish Tensor versus graph-node
  identity and require no architecture update.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- one new production Java file and the three existing core [tensor](../../../../glossary.md#tensor) files listed above;
- only current tensor expression helpers that must mechanically migrate from direct provenance
  construction to the new single-output factory boundary;
- one new focused producer test and only existing model tests that directly construct provenance,
  exercise the changed factory/Tensor validation, or assert the changed API shape;<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [shape glossary](../../../../glossary.md#shape))</span>
- the six implementation-time documentation/[planning](../../../../glossary.md#term-planning) files listed above; and
- no file in another module, operation family, graph package, build configuration, architecture
  documentation, or architecture tests.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

This is a documented [atomic](../../../../glossary.md#resource-transaction-and-rollback)-migration exception to the normal 12–18-file guardrail. A partial
migration would leave two incompatible [provenance](../../../../glossary.md#provenance) models, require [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) to branch on
legacy versus shared origins, or allow output [descriptors](../../../../glossary.md#term-descriptor) to disagree with result Tensors. Most
affected files receive one mechanical factory-call replacement. Do not use the larger path
allowance for unrelated cleanup, reformatting, public [API](../../../../glossary.md#term-application-programming-interface-api) expansion, or [operation](../../../../glossary.md#operation)-specific changes.

If another production concept, another package, an operation-family change, [graph](../../../../glossary.md#graph)/compiler code,
or a second provenance representation is required, stop and report the needed design decision.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) as immutable pre-capture expression-occurrence identity, including
  exact-reference ownership, ordered inputs and output descriptors, signature validation,
  identity equality, and its distinction from Tensor, <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [signature glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [Tensor glossary](../../../../glossary.md#tensor))</span>`CompiledNode`, [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).
- Document why producer output descriptors are not output Tensor references and therefore create
  no producer/result/provenance cycle.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Document `TensorProvenance` as one indexed output association and explain every component,
  derived accessor, validation, identity/equality behavior, and single-output index zero.
- Update Tensor Javadoc for exact descriptor/provenance-position agreement without implying graph
  membership or runtime residency.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>
- Update factory Javadocs for single- and multi-output validation, ownership, immutable result
  order, allocation/ID side effects, labels, and absence of storage.
- Correct affected expression-helper Javadocs only where they name the replaced factory signature
  or describe the old operation-and-input provenance shape.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [shape glossary](../../../../glossary.md#shape))</span>
- Explain the model to newcomers in Tensor API and glossary with one single-output example and one
  conceptual two-output example. Do not claim <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`topK` or [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) is implemented.
- Update Compile API only to explain what a future capture pass can observe from the now-current
  producer/output-index model. Keep compiler traversal and graph creation planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Acceptance criteria

- [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) is one public final non-record identity class with exactly operation, immutable
  ordered inputs, and immutable ordered output descriptors as instance fields.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Producer construction validates component references, indexed elements, non-empty outputs, and
  final input/output counts through the exact selected <span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>`OperationSignature`.
- Producer lists preserve order, repeated permitted positions, exact immutable element references,
  and cannot be mutated through accessors.
- [Producer](../../../../glossary.md#term-producer) output count is derived from the output [descriptor](../../../../glossary.md#term-descriptor) list and has no independent field.
- Two separately created structurally equal producers remain unequal and have distinct references;
  ordinary identity hash codes are not treated as unique identifiers. Every output from one
  multi-output construction retains the exact same producer reference.<span data-glossary-navigation="true"> ([hash codes glossary](../../../../glossary.md#hash-codes-and-content-digests))</span>
- `TensorProvenance` has exactly producer and outputIndex record components in order, rejects every
  invalid index, and derives operation, inputs, and output descriptor from the producer.
- A one-output expression has producer output count one and [provenance](../../../../glossary.md#provenance) index zero.
- A test-local two-output expression produces two fresh Tensors with one exact shared producer,
  indices zero and one, the exact ordered descriptor references, independent Tensor IDs, empty
  labels, empty storage, and an immutable result list.
- A derived Tensor rejects a provenance position whose descriptor is merely equal but not the exact
  descriptor reference stored for that producer output.
- Existing public [Tensor](../../../../glossary.md#tensor) expression [signatures](../../../../glossary.md#term-signature) and fluent single-output syntax remain unchanged.
- Every current single-output helper preserves its exact operation, input order, descriptor,
  label, identity-allocation, storage-free result, and behavior while using producer index zero.
- No output Tensor reference, graph-local identity, compiler state, backend metadata, runtime
  state, storage, or executable behavior appears in producer or provenance.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [metadata glossary](../../../../glossary.md#term-metadata); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- No production multi-output operation, public result wrapper, sibling-output lookup, producer
  registry, interning, or structural deduplication is added.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [registry glossary](../../../../glossary.md#term-registry))</span>
- All affected public contracts and methods have complete current [Javadoc](../../../../glossary.md#term-javadoc).
- The targeted clean-context documentation pass finalizes Javadoc, Tensor API, Compile API,
  glossary, planning evidence, links, and terminology without repeating successful Java tests
  unless executable behavior changes afterward.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>
- Task, master-plan row, and roadmap row have matching final status.
- Task 0018M remains Draft without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, use focused [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), factory, and [Tensor](../../../../glossary.md#tensor) tests as needed.
After executable Java stabilizes, record one final [module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test
```

The targeted documentation pass then runs, after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also checks local Markdown links and anchors, balanced fences, terminology,
and final newlines in the affected documentation. It reuses the recorded [model](../../../../glossary.md#term-model)-test result unless
it changes executable Java behavior.

Repository-wide `./gradlew test` is deferred to the foundation-contract capability [checkpoint](../../../../glossary.md#term-checkpoint)
after task 0018N. This task changes one module and no [dependency](../../../../glossary.md#term-dependency) or architecture boundary.

Automated tests must cover producer [identity](../../../../glossary.md#term-identity-meanings), exact [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape), list [ownership](../../../../glossary.md#term-ownership-meanings), exact [descriptor](../../../../glossary.md#term-descriptor)
references, [signature](../../../../glossary.md#term-signature) [cardinality](../../../../glossary.md#term-cardinality), provenance indices, descriptor mismatch, single-output
compatibility, multi-output construction, and ID side effects. Do not require repeated manual
reflection, `javap`, or [bytecode](../../../../glossary.md#term-bytecode) commands when ordinary [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and automated tests prove the
same facts.

Final [scope](../../../../glossary.md#term-scope) review must confirm:

- only allowed `modules/model` tensor-package Java files and documentation/[planning](../../../../glossary.md#term-planning) files changed;
- no output Tensor reference or independent output-count field exists in [`TensorProducer`](../../../../glossary.md#producer--tensorproducer);
- no old `(Operation, List<Tensor>)` provenance representation remains;
- no [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), graph-package, compiler, backend, runtime, or service-locator import
  appears in producer or provenance;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- no [Gradle](../../../../glossary.md#term-gradle), dependency, architecture, another-module, or [operation](../../../../glossary.md#operation)-family file changed; and
- 0018L is `Complete` everywhere only after both passes, while 0018M has no task file.

## Dependencies

- Task 0007: [immutable](../../../../glossary.md#term-immutable) [Tensor](../../../../glossary.md#tensor) [descriptors](../../../../glossary.md#term-descriptor) — Complete.
- Task 0008: [graph](../../../../glossary.md#graph) values and multi-output-capable compiled [nodes](../../../../glossary.md#node) — Complete.
- Task 0009: immutable [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) [model](../../../../glossary.md#term-model) — Complete.
- Task 0011: public Tensor state — Complete.
- Task 0012: Tensor construction and identifier allocation — Complete.
- Task 0013: current single-output Tensor provenance — Complete and intentionally replaced by this
  unified producer/output-position model.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [producer glossary](../../../../glossary.md#term-producer))</span>
- Task 0018K: exact kind/attributes [signatures](../../../../glossary.md#term-signature) and local [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality) — Complete.

## Follow-up tasks

- 0018O — normalize unstack into repeated scalar select rather than treating it as a genuine
  shared multi-output primitive.<span data-glossary-navigation="true"> ([scalar select glossary](../../../../glossary.md#scalar-select))</span>
- 0019B — use shared producer provenance for state-consuming/state-producing graph randomness and
  dropout.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- 0019C — add a typed public top-K result whose values and indices share one [producer](../../../../glossary.md#term-producer).
- 0021 — use shared producer [provenance](../../../../glossary.md#provenance) where [normalization](../../../../glossary.md#term-normalization) returns auxiliary statistics.

Do not create detailed specifications for these follow-ups in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns public [Tensor](../../../../glossary.md#tensor) state, [immutable](../../../../glossary.md#term-immutable) [descriptors](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation) semantics,
and expression [provenance](../../../../glossary.md#provenance) to `modules/model`, distinguishes Tensor from [graph](../../../../glossary.md#graph) [IR](../../../../glossary.md#term-intermediate-representation-ir), and reserves
graph-local [identities](../../../../glossary.md#term-identity-meanings) for [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) contexts. This task makes that existing provenance
[model](../../../../glossary.md#term-model) capable of representing multiple results without changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), dependencies,
[lifecycle](../../../../glossary.md#lifecycle), or [runtime](../../../../glossary.md#term-runtime) visibility.

If implementation requires a graph-local [identity](../../../../glossary.md#term-identity-meanings) on Tensor, a new cross-module [dependency](../../../../glossary.md#term-dependency), or an
architecture rule change, stop and report the exact conflict before editing `ARCHITECTURE.md` or
focused architecture documentation.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, task 0018K, task 0018L, and the affected Tensor/provenance/factory/expression source and
tests in full.

Implement task 0018L exactly as specified. Stay inside modules/model and the explicitly allowed
documentation/planning files. Preserve every current public Tensor expression signature and
single-output semantic behavior. Stop on any scope or architecture conflict. Run the task-level
model validation once after executable code stabilizes.

Then hand the actual diff and recorded Java-test evidence to a separate clean-context
documentation-focused agent in the same change. That agent must inspect final source/tests,
finalize affected Javadocs, Tensor API, Compile API, glossary, task/master/roadmap status and
documentation validation, and must not repeat successful Java tests unless executable behavior
changed or it records a concrete reason.

Do not mark 0018L Complete until both passes succeed. Leave 0018M Draft without a detailed
specification. Do not commit or push.
```

## Local decisions

- [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) is a final non-record identity class with a package-private constructor and
  exactly three final fields. Public accessors expose the exact operation and immutable ordered
  snapshots; no value-based object methods or producer identifier were added.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [producer glossary](../../../../glossary.md#term-producer))</span>
- Both factory seams construct the producer before allocating a Tensor ID. The single-output seam
  shares the supplied descriptor as producer slot zero and uses provenance index zero. The
  multi-output seam iterates the producer snapshot, assigns one ID per position, and returns an
  immutable list.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Tensor descriptor/provenance agreement uses reference identity. This keeps one descriptor object
  authoritative for a producer slot and its result without broadening structural inference.<span data-glossary-navigation="true"> ([reference identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor); [inference glossary](../../../../glossary.md#term-inference))</span>
- Existing single-output helpers delegate operation and ordered inputs to the factory. Their public
  signatures and operation-specific validation remain unchanged. Current unstack continues to
  create independent one-output producers because task 0018O owns its semantic normalization.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Known limitations

- No production operation currently declares multiple outputs, and no public multi-output result
  wrapper or sibling-result lookup exists. Focused package-local tests exercise the representation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Producer/provenance state is expression origin only. Compiler traversal, graph-node/value
  assignment, inference, optimization, gradients, backend lowering, runtime behavior, and
  execution remain unimplemented or separately owned.<span data-glossary-navigation="true"> ([Producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- A producer intentionally retains exact input Tensor references and output descriptor references,
  but never output Tensor objects. It provides no cycle traversal, graph-wide validation,
  canonicalization, interning, or serialization identity.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [serialization glossary](../../../../glossary.md#term-serialization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0018l_implementation` first ran
  `./gradlew :modules:model:compileTestJava`; it failed with 49 expected atomic-migration compile
  errors from remaining test-local calls to the replaced provenance constructor and derived-factory
  signature. The implementation then migrated every directly affected test call.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [compile glossary](../../../../glossary.md#compile); [provenance glossary](../../../../glossary.md#provenance); [signature glossary](../../../../glossary.md#term-signature))</span>
- The same context reran `./gradlew :modules:model:compileTestJava` after that migration:
  `BUILD SUCCESSFUL in 1s`; 2 actionable tasks, with 1 executed and 1 up-to-date.
- The implementation context ran focused validation with
  `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest
  --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest`: `BUILD SUCCESSFUL in 5s`; 3 actionable tasks,
  with 1 executed and 2 up-to-date.
- Implementation context `/root/task_0018l_implementation` ran final
  `./gradlew :modules:model:test` after executable Java stabilized: `BUILD SUCCESSFUL in 1s`. XML
  reports contain 749 tests across 87 suites, with 0 failures, 0 errors, and 0 skipped tests.
- Clean documentation-focused context
  `/root/task_0018l_implementation/task_0018l_docs` independently reviewed `AGENTS.md`,
  `ARCHITECTURE.md`, the current architecture index, documentation rules and General,
  API/Javadoc, Planning, and Example profiles, planning guide, capability baseline, model master
  plan, roadmap, tasks 0018K and 0018L, final source/tests and actual diff, Tensor API, Compile API,
  and glossary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The documentation context changed only Javadocs and documentation after the successful model
  test. It therefore reused the implementation evidence and did not repeat Java tests, as required
  by the planning guide and documentation workflow.
- The documentation context ran `./gradlew :modules:model:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc) edits:
  `BUILD SUCCESSFUL in 1s`; 2 actionable tasks executed and the configuration [cache](../../../../glossary.md#term-cache) was reused.
- The runnable single-output [producer](../../../../glossary.md#term-producer) example was compiled with
  `javac -cp modules/model/build/classes/java/main -d /tmp/synaptik-0018l-doc-example
  /tmp/SingleOutputProducerExample.java` and executed with the model classes. It printed the
  documented five lines: `1`, `0`, `true`, `true`, `true`.
- Targeted local Markdown validation over the six affected documentation/planning files passed
  after final edits: 447 local links, including 137 heading anchors, resolved; fenced code blocks
  were balanced; every file had a final newline; and no trailing whitespace was found.
- Final `git diff --check` passed with no output. Scope/status inspection confirmed the task's
  58-path combined inventory: every Java path is under the model tensor production/test package,
  and no architecture, build, architecture-test, conformance, integration, or other-module path
  changed. It also confirmed exactly three producer fields, no output Tensor or independent
  output-count field, no legacy provenance constructor or representation, no graph/compiler/
  backend/runtime imports in producer or provenance, synchronized 0018L <span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [conformance glossary](../../../../glossary.md#term-conformance); [module path glossary](../../../../glossary.md#term-module-path); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>`Complete` status, and no
  task 0018M specification.
- `docs/planning/modules/model/capabilities.md` was reviewed but not edited by implementation or
  documentation work: its pre-existing authorized planning diff already selects the exact
  producer/output-descriptor/indexed-result design. <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>`ARCHITECTURE.md` and focused architecture
  documents remain unchanged because this task adds model-owned pre-capture provenance without
  changing module ownership, dependency direction, lifecycle, or graph/runtime identity.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Repository-wide tests remain deferred to the foundation-contract checkpoint after task 0018N,
  as recorded before implementation.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Implementation notes

- Added public final [identity](../../../../glossary.md#term-identity-meanings)-based [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) with exact operation, immutable ordered input
  Tensor references, immutable ordered output descriptor references, derived output count, and
  signature-cardinality validation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [signature glossary](../../../../glossary.md#term-signature); [cardinality glossary](../../../../glossary.md#term-cardinality); [validation glossary](../../../../glossary.md#term-validation))</span>
- Replaced legacy operation/input provenance with exact producer plus zero-based output index.
  Compatibility accessors now derive operation, inputs, and selected descriptor from the producer;
  record equality combines producer identity with the index.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [producer glossary](../../../../glossary.md#term-producer))</span>
- Hardened package-private Tensor construction so a derived Tensor's descriptor must be the exact
  descriptor reference selected by provenance.
- Replaced the old derived factory seam with producer-aware single-output construction and added
  package-private atomic multi-output construction with immutable result order and documented ID
  side effects.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Migrated all current expression helpers to the unified factory boundary. Existing operations,
  input order, descriptors, labels, storage-free results, public signatures, and semantic
  validation remain unchanged; current unstack outputs remain independent occurrences.
- Added focused producer, provenance, Tensor, factory, and expression regression coverage for
  exact API shape, snapshots, identity, cardinality, descriptor agreement, multi-output sharing,
  failure order, and ID consumption.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>
- Finalized affected Javadocs, Tensor API, Compile API, glossary, this task, model master plan, and
  roadmap. The references now distinguish producer output slots from unstack semantic coordinates
  and from graph-local <span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [graph glossary](../../../../glossary.md#graph))</span>[`NodeId`](../../../../glossary.md#nodeid)/[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) identities.

## Completion summary

- Completed changes: introduced unified identity-based producer and indexed provenance contracts,
  exact descriptor-slot agreement, single/multi-output package-private factory boundaries, and a
  complete migration of existing single-output expressions.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Files changed or created: one new production and one new focused test file; the three core
  tensor contracts, current expression helpers, directly affected tensor tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: reused the final 749-test/87-suite model run with zero failures, errors, or
  skips; model Javadoc, compiled/running documentation example, 447-link/137-anchor Markdown and
  fence/newline/whitespace checks, final scope/status audit, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: clean context
  `/root/task_0018l_implementation/task_0018l_docs` completed the independent API/Javadoc,
  Planning, General, and Example-profile review.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor and Compile API references now explain shared producer identity,
  indexed output provenance, exact descriptor ownership, future capture observations, and current
  limitations. Architecture documentation requires no change because no architecture boundary
  changed.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Javadoc review: all affected public and contract-relevant Javadocs were reviewed and finalized;
  stale helper wording about preconstructed provenance was corrected. No executable Java changed
  during the documentation pass.
- Glossary impact: added [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), updated provenance, Tensor, factory, and unstack
  distinctions, and kept compiler/runtime terminology aligned.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0018L. Task 0018M remains Draft without a detailed
  specification.

Status: Complete
