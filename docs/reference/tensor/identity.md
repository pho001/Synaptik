# Tensor identity and expression producers

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="public-tensor-state"></a>
## Public Tensor state

The public `Tensor` contract lives in `io.github.pho001.synaptik.model.tensor`. The current final
class is mutable API state, not an intermediate-representation node. It has five state components:

| Component | Stability and ownership |
|---|---|
| `id()` | Returns the exact immutable `TensorId` reference retained for the tensor's lifetime. |
| `descriptor()` | Returns the exact immutable `TensorDescriptor` reference retained for the tensor's lifetime. |
| `label()` | Returns an immutable optional diagnostic value; present text is stripped and must remain non-blank. |
| `provenance()` | Returns immutable optional expression-origin metadata; a present value is the exact retained `TensorProvenance`. |
| `hostStorage()` | Returns a synchronized snapshot of the optional borrowed `HostTensorStorage` association. |

<a id="expression-producers-and-indexed-provenance"></a>
### Expression producers and indexed provenance

A derived Tensor is one indexed result of one `TensorProducer`. The producer is immutable
pre-capture expression-occurrence identity: it retains one exact `Operation`, an immutable ordered
snapshot of exact input Tensor references, and an immutable ordered snapshot of exact output
descriptor references. The selected operation signature validates those final input and output
counts. Separate producer objects remain separate occurrences even when all three parts are
structurally equal.

`TensorProvenance(producer, outputIndex)` associates one result Tensor with a zero-based producer
position. Its `operation()`, `inputs()`, and `outputDescriptor()` methods derive their exact
references from the producer. `producer.output(outputIndex)` returns that factory-created
position's exact canonical Tensor wrapper; it does not allocate, traverse, infer, or reconstruct a
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
`NodeId`, `ValueId`, graph membership, compiler-capture result, gradient or backward state,
runtime state, or execution promise. The producer constructor and derived-construction seams are
package-private; the public API exposes producers created by Tensor expressions and the public
indexed output accessor.

A general scan additionally retains `Optional<ScanBody>` through `producer.scanBody()`.
This is reusable source-expression containment, not a captured graph or executable callback.
Ordinary expressions and fixed recurrent scans have empty attachments. The
[general-scan construction API](general-scan.md#general-scan-source-construction) describes its positional
bindings, canonical result groups and preallocation checks.

<a id="single-output-example"></a>
#### Single-output example

<a id="goal-and-inputs-2"></a>
##### Goal and inputs

Observe the producer and provenance of one current unary expression. The input is a storage-backed
`FLOAT32` Tensor with Shape `[2]`; the result is metadata for negation, not evaluated values.

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

The result has one producer output at index zero, retains the exact input reference, and shares
the exact descriptor and operation references exposed through provenance. Indexed lookup returns
that same exact result object. It does not prove that negation ran, that a graph was captured, or
that a gradient rule exists.

<a id="useful-variation"></a>
##### Useful variation

Calling `input.neg()` twice creates two structurally equivalent but identity-distinct producers.
The model does not intern them or perform common-subexpression elimination.

<a id="multi-output-and-hidden-output-identity"></a>
#### Multi-output and hidden-output identity

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

Both results share the exact producer reference, while each retains the exact descriptor from its
own ordered position. The producer retains both canonical wrappers, including output positions
omitted from an ergonomic result carrier.

For dropout, `result.output().provenance().orElseThrow().producer().output(1)` returns the hidden
same-occurrence `BOOL` keep-mask wrapper. `DropoutResult` remains the two-component public carrier
for the visible output and next RNG state; the mask is not added to it. For batch-normalization
training, producer outputs three and four are the hidden saved mean and saved inverse standard
deviation. `BatchNormTrainingResult` still exposes only the visible output, next running mean, and
next running variance. These lookups expose current model identity and descriptors only; they do
not sample dropout, compute statistics, construct gradients, capture a graph, compile, or execute
the occurrence.


---

[Previous topic](descriptors.md) · [Next topic](creation.md)
