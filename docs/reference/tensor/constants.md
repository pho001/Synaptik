# Constant, full-value and identity examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="complete-constant-creation-example"></a>
### Complete constant-creation example

<a id="goal-and-inputs-3"></a>
#### Goal and inputs

Create a BFLOAT16 scalar from a halfway binary32 value, create a `2 × 2` INT32 one tensor, and
create an independent zero tensor using only the one's shape and data type. The scalar input
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
  descriptor before copying the converted raw bits into independent storage.
- `ones` selects the `int[]` carrier from `INT32` and fills its four logical positions with exact
  integer one.
- `zerosLike` reuses only `ones`' immutable shape and data type. Its supplied label and gradient
  request are independent, and it creates a new dense descriptor and storage.
- Heap-base inspection exposes the already implemented raw storage only so the values are visible
  in this example; it is not a typed Tensor access or export API.

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

The result demonstrates BFLOAT16 ties-to-even rounding, exact INT32 one population, dense
like-shaped creation, and independent storage. It does not provide general fill values, numeric
conversion, random generation, typed access/export, view preservation, or execution; integer
range creation is a separate factory method described below.

<a id="failures-and-useful-variations-2"></a>
#### Failures and useful variations

- Requesting gradients for `INT32`, `INT64`, or `BOOL` fails before source or destination
  allocation.
- A dynamic shape or logical element count above `Integer.MAX_VALUE` fails before destination or
  ID allocation.
- A zero-sized static shape is valid and produces an empty backing array for either zeros or ones.

<a id="complete-full-value-and-rectangular-identity-example"></a>
### Complete full-value and rectangular-identity example

<a id="goal-and-inputs-4"></a>
#### Goal and inputs

Create a nontrivial `2 × 3` INT64 tensor whose six logical positions all contain `-7`, and create
a wide `2 × 4` FLOAT32 identity matrix. Both results use explicit labels and gradient intent. The
identity is differentiable because FLOAT32 is differentiable; the INT64 full tensor must disable
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

- `full(Shape.of(2, 3), -7L, ...)` selects INT64 from the Java `long` argument. Row-major
  positions `0` through `5` all become `-7`, which can be read as two logical rows
  `[-7, -7, -7]` and `[-7, -7, -7]`.
- `identityMatrix(2, 4, FLOAT32, ...)` creates a wide matrix. Its diagonal length is
  `min(2, 4) = 2`, so row-major indices `0 × 4 + 0 = 0` and `1 × 4 + 1 = 5` receive `1.0f`.
  The other six positions retain `0.0f`.
- `heapBase()` inspection is used only to make the current raw values observable. It is not a
  typed Tensor access or export API.
- The descriptor queries verify that identity creation retained the rectangular shape and
  synthesized canonical dense-contiguous geometry. The final expression verifies that both
  public initializers produced leaves without provenance.

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

The first line is the complete copied INT64 carrier in row-major order. The second line represents
the rectangular matrix rows `[1, 0, 0, 0]` and `[0, 1, 0, 0]`. The result proves exact full-value
population, main-diagonal placement, dense metadata, and provenance-free leaf construction. It
does not add post-construction fill, typed Tensor export, an identity operation, graph capture, or
backend execution.

<a id="failures-and-useful-variations-3"></a>
#### Failures and useful variations

- Replacing `Shape.of(2, 3)` with a dynamic shape fails before source, destination, or ID
  allocation. A static zero-element shape is valid and creates empty storage.
- A negative row is rejected before a negative column; zero rows or zero columns create a valid
  empty rank-two identity matrix.
- Requesting gradients for an INT32, INT64, or BOOL full or identity tensor fails before
  allocation. `eye(...)` has the same validation and values as `identityMatrix(...)`, but a
  separate call still returns a fresh Tensor and ID.

Deterministic range creation has exactly two overloads. `range(int, int, int, label)` creates an
`INT32` result, while `range(long, long, long, label)` creates `INT64`. Both produce eager,
rank-one, non-differentiable tensors with inclusive start and exclusive end. A positive step must
advance from a smaller start toward a larger end; a negative step must advance from a larger start
toward a smaller end. Step zero, equal bounds, and the wrong direction are rejected rather than
producing an empty tensor or correcting the step. Exact count arithmetic prevents primitive
overflow, and the final emitted value may be less than one step from the exclusive bound. Counts
above `Integer.MAX_VALUE` are rejected before allocation.

Range label and argument validation precede allocation. A successful call builds one complete
exact carrier and delegates once to matching flat import. A blank label therefore fails after the
carrier, destination, and ID have been allocated but before the flat copy; it consumes the ID.
Identifier exhaustion is also observed after both arrays exist and before copying. Neither failure
rolls back an identifier.

Strict and cyclic prefix preparation is not a production Tensor API. Repository tests may prepare
such arrays in test source and pass the finished carrier to public flat import, but no public
factory method or product inventory exposes that fixture convenience.


---

[Previous topic](creation.md) · [Next topic](random-initialization.md)
