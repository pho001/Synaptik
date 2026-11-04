# [Constant](../../glossary.md#term-constant), full-value and [identity](../../glossary.md#term-identity-meanings) examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Creating values now, not an expression for later

The application calls `TensorFactory` with primitive values, a shape and explicit gradient intent.
The factory creates independent heap-backed storage; no producing expression is attached, so
these tensors are leaves. “Leaf” does not mean every leaf must contain data: `create` can make
a storage-free one. Nor does it mean the Tensor must be closed; these heap arrays have automatic
lifetimes. Contrast [borrowed arena memory](host-storage.md#borrowed-lifetime-example) with
[value import](import.md), which also copies into independent storage.

**[Reader model](../../glossary.md#term-reader-model).** A **[scalar](../../glossary.md#term-scalar)** has [rank](../../glossary.md#term-rank) zero: no [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) [axes](../../glossary.md#term-axis), but one value.
`full` repeats one value throughout a [shape](../../glossary.md#shape); `zeros` and `ones` use particular fill values.
An **identity [matrix](../../glossary.md#term-matrix-meanings)** has ones where row and column indices are equal and zeros elsewhere;
the rectangular form uses the same rule even when row and column counts differ.
“Identity” here is a numeric pattern, not [Tensor](../../glossary.md#tensor) object [identity](../../glossary.md#term-reference-identity).
[BFLOAT16](../../glossary.md#term-bfloat16) is a compact [floating-point format](../../glossary.md#floating-point-precision-and-rounding); **[ties-to-even](../../glossary.md#term-ties-to-even) [rounding](../../glossary.md#term-rounding)** resolves an exactly
halfway conversion by choosing the encoding with an even retained low bit.
These factory examples populate storage immediately, unlike [operation](../../glossary.md#operation) expressions that only
describe future arithmetic.

<a id="complete-constant-creation-example"></a>
### Complete [constant](../../glossary.md#term-constant)-creation example

<a id="goal-and-inputs-3"></a>
#### Goal and inputs

Create a [BFLOAT16](../../glossary.md#term-bfloat16) [scalar](../../glossary.md#term-scalar) from a halfway [binary32](../../glossary.md#floating-point-precision-and-rounding) value, create a `2 × 2` [INT32](../../glossary.md#data-type--datatype) one [tensor](../../glossary.md#tensor), and
create an independent zero tensor using only the one's [shape](../../glossary.md#shape) and [data type](../../glossary.md#data-type--datatype). The scalar input
`1.00390625f` is halfway between adjacent BFLOAT16 values and rounds to the even representation
`0x3F80` for `1.0`.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ConstantCreationExample {
    public static void main(String[] args) {
        // Round the supplied float once into BFLOAT16 storage rather than retaining a float array.
        Tensor scalar = TensorFactory.scalarBFloat16(
                1.00390625f, Optional.of("scale"), true);
        Tensor ones = TensorFactory.ones(
                Shape.of(2, 2), DataType.INT32, Optional.of("ones"), false);
        Tensor zeros = TensorFactory.zerosLike(
                ones, Optional.of("zeros"), false);

        short[] scalarData = (short[]) scalar.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();
        int[] oneData = (int[]) ones.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.printf("%04X%n", scalarData[0] & 0xFFFF);
        System.out.println(Arrays.toString(oneData));
        System.out.println(zeros.descriptor().shape());
        System.out.println(zeros.descriptor().layout().orElseThrow().kind());
        System.out.println(ones.hostStorage().orElseThrow()
                != zeros.hostStorage().orElseThrow());
    }
}
```

<a id="meaningful-lines-1"></a>
#### Meaningful lines

- `scalarBFloat16` makes the rounding request explicit and creates a differentiable rank-0
  descriptor before copying the converted raw bits into independent storage.<span data-glossary-navigation="true"> ([rounding glossary](../../glossary.md#term-rounding); [differentiable glossary](../../glossary.md#derivative-and-chain-rule); [rank glossary](../../glossary.md#term-rank); [descriptor glossary](../../glossary.md#term-descriptor); [raw bits glossary](../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- `ones` selects the `int[]` [carrier](../../glossary.md#term-storage-carrier) from [`INT32`](../../glossary.md#data-type--datatype) and fills its four logical positions with exact
  integer one.
- `zerosLike` reuses only `ones`' immutable shape and data type. Its supplied label and gradient
  request are independent, and it creates a new dense descriptor and storage.<span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable); [shape glossary](../../glossary.md#shape); [data type glossary](../../glossary.md#data-type--datatype); [gradient glossary](../../glossary.md#term-gradient))</span>
- Heap-base inspection exposes the already implemented raw storage only so the values are visible
  in this example; it is not a typed Tensor access or export API.<span data-glossary-navigation="true"> ([Heap glossary](../../glossary.md#term-heap); [Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-3"></a>
#### Result and interpretation

The program prints:

```text
3F80
[1, 1, 1, 1]
Shape[2, 2]
DENSE_CONTIGUOUS
true
```

The result demonstrates [BFLOAT16](../../glossary.md#term-bfloat16) [ties-to-even](../../glossary.md#term-ties-to-even) [rounding](../../glossary.md#term-rounding), exact [INT32](../../glossary.md#data-type--datatype) one population, dense
like-shaped creation, and independent storage. It does not provide general fill values, numeric
conversion, random generation, typed access/export, [view](../../glossary.md#term-view-meanings) preservation, or execution; integer
range creation is a separate factory method described below.

<a id="failures-and-useful-variations-2"></a>
#### Failures and useful variations

- Requesting [gradients](../../glossary.md#term-gradient) for [`INT32`](../../glossary.md#data-type--datatype), [`INT64`](../../glossary.md#data-type--datatype), or [`BOOL`](../../glossary.md#data-type--datatype) fails before source or destination
  allocation.
- A dynamic [shape](../../glossary.md#shape) or logical [element](../../glossary.md#term-element) count above `Integer.MAX_VALUE` fails before destination or
  ID allocation.
- A zero-sized static shape is valid and produces an empty backing array for either zeros or ones.

<a id="complete-full-value-and-rectangular-identity-example"></a>
### Complete full-value and rectangular-[identity](../../glossary.md#term-identity-meanings) example

<a id="goal-and-inputs-4"></a>
#### Goal and inputs

Create a nontrivial `2 × 3` [INT64](../../glossary.md#data-type--datatype) [tensor](../../glossary.md#tensor) whose six logical positions all contain `-7`, and create
a wide `2 × 4` [FLOAT32](../../glossary.md#term-float32) [identity](../../glossary.md#term-identity-meanings) [matrix](../../glossary.md#term-matrix-meanings). Both results use explicit labels and [gradient](../../glossary.md#term-gradient) intent. The
identity is [differentiable](../../glossary.md#derivative-and-chain-rule) because FLOAT32 is differentiable; the INT64 full tensor must disable
gradients.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class FullIdentityExample {
    public static void main(String[] args) {
        // These initializers populate independent leaves; they do not build arithmetic producers.
        Tensor full = TensorFactory.full(
                Shape.of(2, 3), -7L, Optional.of("weights"), false);
        Tensor identity = TensorFactory.identityMatrix(
                2, 4, DataType.FLOAT32, Optional.of("projection"), true);

        long[] fullValues = (long[]) full.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();
        float[] identityValues = (float[]) identity.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(fullValues));
        System.out.println(Arrays.toString(identityValues));
        System.out.println(identity.descriptor().shape());
        System.out.println(identity.descriptor().layout().orElseThrow().kind());
        System.out.println(full.provenance().isEmpty() && identity.provenance().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-values"></a>
#### Meaningful lines and intermediate values

- `full(Shape.of(2, 3), -7L, ...)` selects [INT64](../../glossary.md#data-type--datatype) from the Java `long` argument. Row-major
  positions <span data-glossary-navigation="true"> ([argument glossary](../../glossary.md#term-argument))</span>`0` through `5` all become `-7`, which can be read as two logical rows
  `[-7, -7, -7]` and `[-7, -7, -7]`.
- `identityMatrix(2, 4, FLOAT32, ...)` creates two rows and four columns.
  For row $i$ and column $j$, row-major position is $4i+j$, with $0\leq i<2$ and
  $0\leq j<4$. The $\min(2,4)=2$ diagonal positions are $4(0)+0=0$ and $4(1)+1=5$.
  They receive `1.0f`; the other six positions retain `0.0f`.
- `heapBase()` inspection is used only to make the current raw values observable. It is not a
  typed Tensor access or export API.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- The descriptor queries verify that identity creation retained the rectangular shape and
  synthesized canonical dense-contiguous geometry. The final expression verifies that both
  public initializers produced leaves without provenance.<span data-glossary-navigation="true"> ([descriptor glossary](../../glossary.md#term-descriptor); [identity glossary](../../glossary.md#term-identity-meanings); [shape glossary](../../glossary.md#shape); [canonical glossary](../../glossary.md#term-canonical-form); [contiguous glossary](../../glossary.md#term-contiguous); [provenance glossary](../../glossary.md#provenance))</span>

<a id="result-and-interpretation-4"></a>
#### Result and interpretation

The program prints:

```text
[-7, -7, -7, -7, -7, -7]
[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
Shape[2, 4]
DENSE_CONTIGUOUS
true
```

The first line is the complete copied [INT64](../../glossary.md#data-type--datatype) [carrier](../../glossary.md#term-storage-carrier) in [row-major](../../glossary.md#term-row-major) order. The second line represents
the rectangular [matrix](../../glossary.md#term-matrix-meanings) rows `[1, 0, 0, 0]` and `[0, 1, 0, 0]`. The result proves exact full-value
population, main-diagonal placement, dense [metadata](../../glossary.md#term-metadata), and [provenance](../../glossary.md#provenance)-free [leaf](../../glossary.md#term-leaf) construction. It
does not add post-construction fill, typed [Tensor](../../glossary.md#tensor) export, an [identity](../../glossary.md#term-identity-meanings) [operation](../../glossary.md#operation), [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction), or
[backend](../../glossary.md#backend) execution.

<a id="failures-and-useful-variations-3"></a>
#### Failures and useful variations

- Replacing `Shape.of(2, 3)` with a dynamic shape fails before source, destination, or ID
  allocation. A static zero-element shape is valid and creates empty storage.<span data-glossary-navigation="true"> ([shape glossary](../../glossary.md#shape); [element glossary](../../glossary.md#term-element))</span>
- A negative row is rejected before a negative column; zero rows or zero columns create a valid
  empty rank-two identity matrix.<span data-glossary-navigation="true"> ([rank-two glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix); [identity glossary](../../glossary.md#term-identity-meanings); [matrix glossary](../../glossary.md#term-matrix-meanings))</span>
- Requesting gradients for an INT32, INT64, or BOOL full or identity tensor fails before
  allocation. <span data-glossary-navigation="true"> ([gradients glossary](../../glossary.md#term-gradient); [BOOL glossary](../../glossary.md#data-type--datatype))</span>`eye(...)` has the same [validation](../../glossary.md#term-validation) and values as `identityMatrix(...)`, but a
  separate call still returns a fresh Tensor and ID.

Deterministic range creation has exactly two overloads. `range(int, int, int, label)` creates an
[`INT32`](../../glossary.md#data-type--datatype) result, while `range(long, long, long, label)` creates [`INT64`](../../glossary.md#data-type--datatype). Both produce [eager](../../glossary.md#term-eager-evaluation),
[rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix), non-[differentiable](../../glossary.md#derivative-and-chain-rule) [tensors](../../glossary.md#tensor) with inclusive start and exclusive end. A positive step must
advance from a smaller start toward a larger end; a negative step must advance from a larger start
toward a smaller end. Step zero, equal bounds, and the wrong direction are rejected rather than
producing an empty tensor or correcting the step. Exact count arithmetic prevents primitive
[overflow](../../glossary.md#term-overflow), and the final emitted value may be less than one step from the exclusive bound. Counts
above `Integer.MAX_VALUE` are rejected before allocation.

Range label and [argument](../../glossary.md#term-argument) validation precede allocation. A successful call builds one complete
exact [carrier](../../glossary.md#term-storage-carrier) and delegates once to matching flat import. A blank label therefore fails after the
carrier, destination, and ID have been allocated but before the flat copy; it consumes the ID.
Identifier exhaustion is also observed after both arrays exist and before copying. Neither failure
rolls back an identifier.

Strict and cyclic prefix preparation is not a production Tensor [API](../../glossary.md#term-application-programming-interface-api). Repository tests may [prepare](../../glossary.md#prepare)
such arrays in test source and pass the finished carrier to public flat import, but no public
factory method or product inventory exposes that [fixture](../../glossary.md#term-fixture) convenience.


---

[Previous topic](creation.md) · [Next topic](random-initialization.md)
