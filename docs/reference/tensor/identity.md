# [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) and expression [producers](../../glossary.md#term-producer)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Follow an expression back to its inputs

When application or layer code calls `input.neg()`, the Tensor method creates a result
description and records the exact `input` object in a new producer. It does not copy or negate
the input's stored numbers at that moment. A leaf is an input with no producing expression;
it may have data (as `zeros` does here) or be storage-free.
The producer owns immutable connection lists, not an external memory lifetime.
Use [creation](creation.md) to supply a leaf and [storage association](storage-association.md)
to understand which mutable part of that Tensor can later change.

**[Reader model](../../glossary.md#term-reader-model).** Identity answers “which particular object or [occurrence](../../glossary.md#term-occurrence)?”, not “do these
descriptions have equal fields?” Two tensors may have equal [shapes](../../glossary.md#shape) and types without being the
same tensor. A **[producer](../../glossary.md#term-producer)** represents one use of an [operation](../../glossary.md#operation), while **[provenance](../../glossary.md#provenance)** connects
each result to that producer and its numbered **output slot**. **[Canonical](../../glossary.md#term-canonical-form) wrapper** means the
one Tensor object assigned to that slot, not a freshly recreated equivalent object.
The existing multi-output diagram below shows why sibling results share one producer.
None of these identity relationships proves that the results share physical storage.

<a id="public-tensor-state"></a>
## Public [Tensor](../../glossary.md#tensor) state

The public `Tensor` contract lives in `io.github.pho001.synaptik.model.tensor`. The current final
class is [mutable](../../glossary.md#term-mutable) [API](../../glossary.md#term-application-programming-interface-api) state, not an intermediate-representation [node](../../glossary.md#node). It has five state components:

| Component | Stability and [ownership](../../glossary.md#term-ownership-meanings) |
|---|---|
| `id()` | Returns the exact [immutable](../../glossary.md#term-immutable) [`TensorId`](../../glossary.md#tensorid) reference retained for the tensor's [lifetime](../../glossary.md#term-lifetime). |
| `descriptor()` | Returns the exact immutable `TensorDescriptor` reference retained for the tensor's lifetime. |
| `label()` | Returns an immutable optional diagnostic value; present text is stripped and must remain non-blank. |
| `provenance()` | Returns immutable optional expression-origin [metadata](../../glossary.md#term-metadata); a present value is the exact retained `TensorProvenance`. |
| `hostStorage()` | Returns a synchronized [snapshot](../../glossary.md#term-snapshot) of the optional [borrowed](../../glossary.md#ownership-borrowing-and-lease) `HostTensorStorage` association. |

<a id="expression-producers-and-indexed-provenance"></a>
### Expression [producers](../../glossary.md#term-producer) and indexed [provenance](../../glossary.md#provenance)

A derived [Tensor](../../glossary.md#tensor) is one indexed result of one [`TensorProducer`](../../glossary.md#producer--tensorproducer). The [producer](../../glossary.md#term-producer) is [immutable](../../glossary.md#term-immutable)
pre-[capture](../../glossary.md#term-capture-meanings) expression-[occurrence](../../glossary.md#term-occurrence) [identity](../../glossary.md#term-identity-meanings): it retains one exact [`Operation`](../../glossary.md#operation), an immutable ordered
[snapshot](../../glossary.md#term-snapshot) of exact input Tensor references, and an immutable ordered snapshot of exact output
[descriptor](../../glossary.md#term-descriptor) references. The selected [operation signature](../../glossary.md#operation-signature) validates those final input and output
counts. Separate producer objects remain separate occurrences even when all three parts are
structurally equal.

`TensorProvenance(producer, outputIndex)` associates one result Tensor with a zero-based producer
position. Its `operation()`, `inputs()`, and `outputDescriptor()` methods derive their exact
references from the producer. `producer.output(outputIndex)` returns that factory-created
position's exact [canonical](../../glossary.md#term-canonical-form) Tensor wrapper; it does not allocate, traverse, infer, or reconstruct a
wrapper. A derived Tensor retains the same descriptor object as its selected producer position; a
merely equal descriptor object is rejected. Negative indices and indices at least
`producer.outputCount()` fail with `IndexOutOfBoundsException` that reports the requested index
and available count. Single-output expressions use a one-descriptor producer and index zero.

Factory construction validates the complete occurrence before requesting its first Tensor ID,
creates wrappers in ascending output order, and assigns the producer's final immutable output
snapshot before returning any wrapper. Every wrapper has the exact slot descriptor, current label
policy, no storage, and indexed provenance back to the same producer. Failure after allocation
does not roll back consumed IDs, and a failed multi-output request publishes no partial result.

These references intentionally form one immutable, safely published
`Tensor -> TensorProvenance -> TensorProducer -> outputs -> Tensor` cycle. Retaining one result may
retain its sibling results. The cycle owns no external resource and is ordinarily
garbage-collectable when the complete occurrence is unreachable. Producer identity is not a
[`NodeId`](../../glossary.md#nodeid), [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot), [graph](../../glossary.md#graph) membership, [compiler](../../glossary.md#term-compiler)-capture result, [gradient](../../glossary.md#term-gradient) or backward state,
[runtime](../../glossary.md#term-runtime) state, or execution promise. The producer constructor and derived-construction seams are
package-private; the public [API](../../glossary.md#term-application-programming-interface-api) exposes producers created by Tensor expressions and the public
indexed output accessor.

A [general scan](../../glossary.md#general-differentiable-scan) additionally retains `Optional<ScanBody>` through `producer.scanBody()`.
This is reusable source-expression containment, not a captured graph or executable [callback](../../glossary.md#term-callback).
Ordinary expressions and fixed recurrent scans have empty attachments. The
[general-scan construction API](general-scan.md#general-scan-source-construction)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#term-cumulative-scan))</span> describes its positional
[bindings](../../glossary.md#term-binding), canonical result groups and preallocation checks.

<a id="single-output-example"></a>
#### Single-output example

<a id="goal-and-inputs-2"></a>
##### Goal and inputs

Observe the [producer](../../glossary.md#term-producer) and [provenance](../../glossary.md#provenance) of one current [unary](../../glossary.md#term-unary) expression. The input is a storage-backed
[`FLOAT32`](../../glossary.md#term-float32) [Tensor](../../glossary.md#tensor) with [Shape](../../glossary.md#shape) `[2]`; the result is [metadata](../../glossary.md#term-metadata) for negation, not evaluated values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class SingleOutputProducerExample {
    public static void main(String[] args) {
        Tensor input = TensorFactory.zeros(
                Shape.of(2), DataType.FLOAT32, Optional.empty(), true);
        // Zeros already contain data; negation only records how a later result is produced.
        Tensor result = input.neg();

        TensorProvenance origin = result.provenance().orElseThrow();
        TensorProducer producer = origin.producer();

        System.out.println(producer.outputCount());
        System.out.println(origin.outputIndex());
        System.out.println(producer.inputs().getFirst() == input);
        System.out.println(origin.outputDescriptor() == result.descriptor());
        System.out.println(producer.operation() == origin.operation());
        System.out.println(producer.output(0) == result);
    }
}
```

<a id="result-and-interpretation-2"></a>
##### Result and interpretation

The program prints:

```text
1
0
true
true
true
true
```

The result has one [producer](../../glossary.md#term-producer) output at index zero, retains the exact input reference, and shares
the exact [descriptor](../../glossary.md#term-descriptor) and [operation](../../glossary.md#operation) references exposed through [provenance](../../glossary.md#provenance). Indexed [lookup](../../glossary.md#term-lookup) returns
that same exact result object. It does not prove that negation ran, that a [graph](../../glossary.md#graph) was captured, or
that a [gradient](../../glossary.md#term-gradient) rule exists.

<a id="useful-variation"></a>
##### Useful variation

Calling `input.neg()` twice creates two structurally equivalent but [identity](../../glossary.md#term-identity-meanings)-distinct [producers](../../glossary.md#term-producer).
The [model](../../glossary.md#term-model) does not intern them or perform [common-subexpression elimination](../../glossary.md#term-common-subexpression-elimination-cse).

<a id="multi-output-and-hidden-output-identity"></a>
#### Multi-output and hidden-output [identity](../../glossary.md#term-identity-meanings)

```text
producer P
  operation:          one current multi-output operation
  inputs:             [input]
  outputDescriptors: [primaryDescriptor, auxiliaryDescriptor]
  outputs:            [primaryResult, auxiliaryResult]

primaryResult.provenance   = (P, 0)
auxiliaryResult.provenance = (P, 1)
P.output(0)                = primaryResult   // exact object identity
P.output(1)                = auxiliaryResult // exact object identity
```

Both results share the exact [producer](../../glossary.md#term-producer) reference, while each retains the exact [descriptor](../../glossary.md#term-descriptor) from its
own ordered position. The producer retains both [canonical](../../glossary.md#term-canonical-form) wrappers, including output positions
omitted from an ergonomic result [carrier](../../glossary.md#term-storage-carrier).

For [dropout](../../glossary.md#dropout--inverted-dropout), `result.output().provenance().orElseThrow().producer().output(1)` returns the hidden
same-[occurrence](../../glossary.md#term-occurrence) [`BOOL`](../../glossary.md#data-type--datatype) keep-[mask](../../glossary.md#term-boolean-mask) wrapper. `DropoutResult` remains the two-component public carrier
for the visible output and next [RNG state](../../glossary.md#graph-rng-state--graphrngstate); the mask is not added to it. For [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization)
training, producer outputs three and four are the hidden saved [mean](../../glossary.md#term-mean) and saved inverse [standard
deviation](../../glossary.md#term-standard-deviation). `BatchNormTrainingResult` still exposes only the visible output, next running mean, and
next running [variance](../../glossary.md#term-variance). These [lookups](../../glossary.md#term-lookup) expose current [model](../../glossary.md#term-model) identity and descriptors only; they do
not [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) dropout, compute statistics, construct [gradients](../../glossary.md#term-gradient), [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), [compile](../../glossary.md#compile), or execute
the occurrence.


---

[Previous topic](descriptors.md) · [Next topic](creation.md)
