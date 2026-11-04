# [Tensor](../../glossary.md#tensor) [descriptors](../../glossary.md#term-descriptor)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## What crosses the constructor boundary?

Application code, or a framework component deriving an operation's output, calls
`new TensorDescriptor(type, shape, layout, requiresGrad)`. It supplies immutable descriptions,
not arrays of numeric values. The record keeps those facts and validates their consistency;
it does not own or close a memory region. A known shape and a known layout are separate facts:
six positions can be known before their storage mapping is chosen.
Continue with [factory creation choices](creation.md) to attach values and
[layout geometry](layouts.md#resolved-layouts) to understand a present layout.

**[Reader model](../../glossary.md#term-reader-model).** A descriptor is a description of a value, not its data. **[Data type](../../glossary.md#data-type--datatype)** says how
one [element](../../glossary.md#term-element) is represented, **[shape](../../glossary.md#shape)** gives the [axis](../../glossary.md#term-axis) sizes, and **[layout](../../glossary.md#layout)** optionally maps
[coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) to storage positions. A **[gradient](../../glossary.md#term-gradient)** describes how changing an input affects a
selected result; `requiresGrad` requests eligibility for that calculation, not proof that the
calculation is supported. A descriptor can therefore describe `[2, 3]` [FLOAT32](../../glossary.md#term-float32) data without
allocating six numbers, and without yet choosing [strides](../../glossary.md#term-stride) or an offset.

<a id="tensor-descriptors"></a>
## [Tensor](../../glossary.md#tensor) [descriptors](../../glossary.md#term-descriptor)

The public `TensorDescriptor` contract lives in
`io.github.pho001.synaptik.model.tensor`. It is an [immutable](../../glossary.md#term-immutable) record with four components:

| Component | Meaning |
|---|---|
| `dataType` | The non-null logical type of every [element](../../glossary.md#term-element). |
| [`shape`](../../glossary.md#shape) | The non-null ordered logical [dimensions](../../glossary.md#dimension). |
| [`layout`](../../glossary.md#layout) | A non-null `Optional`: present for resolved numeric geometry, empty for unresolved geometry. |
| `requiresGrad` | Whether [model](../../glossary.md#term-model)-level [gradient](../../glossary.md#term-gradient) eligibility is requested. This may be true only for a [differentiable](../../glossary.md#derivative-and-chain-rule) [data type](../../glossary.md#data-type--datatype). |

Layout resolution and shape resolution are separate facts. A dynamic shape must use
`Optional.empty()` because its numeric [strides](../../glossary.md#term-stride) and span are not known. A fully static shape may
also use `Optional.empty()` when layout has not been resolved yet; the descriptor does not invent a
[contiguous](../../glossary.md#term-contiguous) default.

When a layout is present, construction rebuilds it against the paired shape through
`LayoutDescriptor.of(...)` and compares the complete value. The check covers [rank](../../glossary.md#term-rank), strides,
element offset, [view](../../glossary.md#term-view-meanings) flag, derived layout kind, and [referenced element span](../../glossary.md#referenced-element-span). It proves that the
public geometry is compatible with the paired shape. It cannot prove which shape originally
created the layout because `LayoutDescriptor` deliberately does not retain source-shape [identity](../../glossary.md#term-identity-meanings).

`TensorDescriptor` uses record equality and hashing across all four components. Its `layout`
component follows `Optional` value semantics: inspect presence or compare values, and do not rely
on the optional container's identity. A present optional contains the exact immutable
`LayoutDescriptor` object supplied to the constructor. The generated text form is useful for
diagnostics but is not a [serialization](../../glossary.md#term-serialization) format.

<a id="complete-descriptor-example"></a>
### Complete [descriptor](../../glossary.md#term-descriptor) example

<a id="goal-and-inputs-1"></a>
#### Goal and inputs

Create three [descriptors](../../glossary.md#term-descriptor) that distinguish a fully static unresolved value, the same static value
with resolved [row-major](../../glossary.md#term-row-major) geometry, and a dynamic unresolved value. The static [shape](../../glossary.md#shape) is `[2, 3]`, the
dynamic shape is `[batch, 3]`, and every value uses [`FLOAT32`](../../glossary.md#term-float32).

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;

public final class TensorDescriptorExample {
    public static void main(String[] args) {
        Shape staticShape = Shape.of(2, 3);
        // Static sizes alone do not choose how the six positions map to memory.
        TensorDescriptor staticUnresolved = new TensorDescriptor(
                DataType.FLOAT32, staticShape, Optional.empty(), false);

        LayoutDescriptor rowMajor = LayoutDescriptor.contiguous(staticShape);
        TensorDescriptor resolved = new TensorDescriptor(
                DataType.FLOAT32, staticShape, Optional.of(rowMajor), true);

        // A symbolic batch has no numeric strides until its size is resolved.
        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        TensorDescriptor dynamicUnresolved = new TensorDescriptor(
                DataType.FLOAT32, dynamicShape, Optional.empty(), false);

        System.out.println(staticUnresolved.layout().isEmpty());
        System.out.println(resolved.layout().orElseThrow().kind());
        System.out.println(resolved.layout().orElseThrow().referencedElementSpan());
        System.out.println(dynamicUnresolved.shape());
    }
}
```

<a id="meaningful-lines"></a>
#### Meaningful lines

- `staticUnresolved` shows that known sizes do not force a [layout](../../glossary.md#layout) decision.
- `LayoutDescriptor.contiguous(staticShape)` resolves [row-major](../../glossary.md#term-row-major) [strides](../../glossary.md#term-stride) `[3, 1]`, offset `0`, and
  referenced span <span data-glossary-navigation="true"> ([referenced span glossary](../../glossary.md#referenced-element-span))</span>`6`; `resolved` pairs that geometry with the same [shape](../../glossary.md#shape).
- `requiresGrad` is true only on `resolved`. [`FLOAT32`](../../glossary.md#term-float32) permits that request because it is
  differentiable.<span data-glossary-navigation="true"> ([differentiable glossary](../../glossary.md#derivative-and-chain-rule))</span>
- `dynamicUnresolved` keeps the symbolic [`batch`](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [dimension](../../glossary.md#dimension) and therefore has no numeric layout.

<a id="result-and-interpretation-1"></a>
#### Result and interpretation

The program prints:

```text
true
DENSE_CONTIGUOUS
6
Shape[batch, 3]
```

These results show the two [layout](../../glossary.md#layout) states and the concrete geometry of the resolved `[2, 3]`
example. They do not allocate [tensor](../../glossary.md#tensor) storage, create a public `Tensor` or [graph](../../glossary.md#graph) [node](../../glossary.md#node), infer the
dynamic [`batch`](../../glossary.md#term-batch-machine-learning-and-hyperparameters) size, choose [materialization](../../glossary.md#materialization), select a [backend](../../glossary.md#backend), or execute an [operation](../../glossary.md#operation).

<a id="failures-and-useful-variations-1"></a>
#### Failures and useful variations

- Supplying `Optional.of(rowMajor)` with `dynamicShape` fails with `IllegalArgumentException`
  because numeric geometry cannot be validated for an unresolved dimension.<span data-glossary-navigation="true"> ([dimension glossary](../../glossary.md#dimension))</span>
- Requesting [gradients](../../glossary.md#term-gradient) for [`INT32`](../../glossary.md#data-type--datatype), [`INT64`](../../glossary.md#data-type--datatype), or [`BOOL`](../../glossary.md#data-type--datatype) fails with `IllegalArgumentException`.
- A present [layout](../../glossary.md#layout) whose [rank](../../glossary.md#term-rank) or reconstructed geometry differs from the paired [shape](../../glossary.md#shape) fails with
  `IllegalArgumentException`; checked reconstruction [overflow](../../glossary.md#term-overflow) remains an `ArithmeticException`.
- Passing `null` for the [data type](../../glossary.md#data-type--datatype), shape, or optional itself fails with `NullPointerException`.


---

[Previous topic](layouts.md) · [Next topic](identity.md)
