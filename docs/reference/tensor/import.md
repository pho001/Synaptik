# Flat and nested value import examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## The application's arrays stop being inputs after the copy

Application code calls `TensorFactory.fromFlatArray` or `fromNestedArray` with its arrays.
The factory reads them during the call and returns a Tensor with independent heap storage.
It does not retain the supplied arrays, so the application may modify them after return,
but must not modify them concurrently with import. The returned Tensor is a leaf because
there is no producing expression, not because “leaf” is another name for stored data.
Contrast [retaining an existing region](storage-association.md), where lifetime remains with
the region's owner, and read [creation constraints](creation.md) for exact accepted carriers.

**[Reader model](../../glossary.md#term-reader-model).** Flat import receives one [linear](../../glossary.md#term-linear-meanings) primitive array and an explicit description
of its [shape](../../glossary.md#shape) and type. Nested import receives arrays of arrays and infers the rectangular shape
from their structure. **Rectangular** means corresponding [axes](../../glossary.md#term-axis) have consistent lengths, rather
than rows of varying length. **[Row-major](../../glossary.md#term-row-major)** flattening visits the last axis fastest.
Import creates an independent copy; wrapping [borrowed](../../glossary.md#ownership-borrowing-and-lease) storage would instead retain an existing
region. [BOOL](../../glossary.md#data-type--datatype) **[normalization](../../glossary.md#term-normalization)** here means converting all nonzero source bytes to [canonical](../../glossary.md#term-canonical-form)
true byte `1`, not statistically rescaling numerical data.

<a id="complete-flat-import-example"></a>
### Complete flat-import example

<a id="goal-and-inputs-12"></a>
#### Goal and inputs

Import three logical [BOOL](../../glossary.md#data-type--datatype) values from bytes `{0, -2, 3}`, verify [canonical](../../glossary.md#term-canonical-form) storage, and show that
later source mutation does not affect the [tensor](../../glossary.md#tensor).

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class FlatImportExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(3);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        byte[] source = {0, -2, 3};

        // Import copies and normalizes BOOL bytes; it does not borrow source.
        Tensor tensor = TensorFactory.fromFlatArray(
                descriptor, Optional.of("mask"), source);
        byte[] imported = (byte[]) tensor.hostStorage()
                .orElseThrow()
                .segment()
                .heapBase()
                .orElseThrow();

        source[0] = 9;
        System.out.println(tensor.label().orElseThrow());
        System.out.println(Arrays.toString(imported));
    }
}
```

<a id="meaningful-lines-9"></a>
#### Meaningful lines

- `LayoutDescriptor.contiguous(shape)` supplies the required resolved dense-[contiguous](../../glossary.md#term-contiguous) geometry.
- The `byte[]` [carrier](../../glossary.md#term-storage-carrier) matches [`BOOL`](../../glossary.md#data-type--datatype); import normalizes both non-zero values to [canonical](../../glossary.md#term-canonical-form) byte
  `1` while copying into a distinct factory-allocated array.
- Mutating `source` after import demonstrates that the returned tensor does not retain that array.
  The raw heap-base inspection uses the already exposed storage segment only to make this example's
  result observable; it is not a typed Tensor access API.<span data-glossary-navigation="true"> ([tensor glossary](../../glossary.md#tensor); [heap glossary](../../glossary.md#term-heap); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-12"></a>
#### Result and interpretation

The program prints:

```text
mask
[0, 1, 1]
```

The result proves [BOOL](../../glossary.md#data-type--datatype) [normalization](../../glossary.md#term-normalization), copied [ownership](../../glossary.md#term-ownership-meanings), label retention, and [row-major](../../glossary.md#term-row-major) order for a
dense flat import. It does not demonstrate nested [shape](../../glossary.md#shape) [inference](../../glossary.md#term-inference-meanings) or provide public typed access
or export, numeric conversion, [view](../../glossary.md#term-view-meanings) scattering, native allocation, [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency), or [backend](../../glossary.md#backend)
execution.

<a id="failures-and-useful-variations-9"></a>
#### Failures and useful variations

- Using this `byte[]` with a [descriptor](../../glossary.md#term-descriptor) other than [BOOL](../../glossary.md#data-type--datatype) or either [FP8](../../glossary.md#term-fp8) type fails before destination or ID allocation.
- An offset, strided, broadcast, or unresolved [layout](../../glossary.md#layout) is rejected before source-length [validation](../../glossary.md#term-validation).
- A source of length two or four fails because the [shape](../../glossary.md#shape) has exactly three logical [elements](../../glossary.md#term-element).
- `short[]` imports raw [BFLOAT16](../../glossary.md#term-bfloat16) bits; it does not convert Java `float` values.

<a id="complete-nested-import-example"></a>
### Complete nested-import example

<a id="goal-and-inputs-13"></a>
#### Goal and inputs

Import the rectangular `float[][]` value `{{1, 2}, {3, 4}}`, infer its [descriptor](../../glossary.md#term-descriptor), preserve the
[gradient](../../glossary.md#term-gradient) request, and verify that later source mutation does not affect the [tensor](../../glossary.md#tensor). The source has
declared [rank](../../glossary.md#term-rank) two, exact `float` leaves, and two [elements](../../glossary.md#term-element) along each [axis](../../glossary.md#term-axis).

```java
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class NestedImportExample {
    public static void main(String[] args) {
        float[] firstRow = {1.0f, 2.0f};
        float[][] source = {firstRow, {3.0f, 4.0f}};

        // Both array levels are inspected now; later edits to firstRow are independent.
        Tensor tensor = TensorFactory.fromNestedArray(
                source, Optional.of("weights"), true);
        float[] imported = (float[]) tensor.hostStorage()
                .orElseThrow()
                .segment()
                .heapBase()
                .orElseThrow();

        firstRow[0] = 99.0f;
        System.out.println(tensor.descriptor().dataType());
        System.out.println(tensor.descriptor().shape());
        System.out.println(tensor.descriptor().layout().orElseThrow().kind());
        System.out.println(tensor.descriptor().requiresGrad());
        System.out.println(tensor.label().orElseThrow());
        System.out.println(Arrays.toString(imported));
    }
}
```

<a id="meaningful-lines-10"></a>
#### Meaningful lines

- Passing the nested primitive array as `Object` does not erase its runtime metadata: the factory
  reads declared rank two and ultimate carrier <span data-glossary-navigation="true"> ([metadata glossary](../../glossary.md#term-metadata); [rank glossary](../../glossary.md#term-rank); [carrier glossary](../../glossary.md#term-storage-carrier))</span>`float`, which infer [`FLOAT32`](../../glossary.md#term-float32) and [shape](../../glossary.md#shape) `[2, 2]`.
- `true` requests [gradient](../../glossary.md#term-gradient) eligibility on the inferred [descriptor](../../glossary.md#term-descriptor). `FLOAT32` permits that request;
  an integral or BOOL carrier would reject it before destination or ID allocation.<span data-glossary-navigation="true"> ([BOOL glossary](../../glossary.md#data-type--datatype))</span>
- The factory flattens the two rows in encounter order and delegates the resulting `float[]` to
  flat import, which creates the independent destination array.
- Mutating `firstRow` after return demonstrates source independence. The heap-base inspection is
  only a way to observe this example's raw result; it is not a public typed Tensor access API.<span data-glossary-navigation="true"> ([heap glossary](../../glossary.md#term-heap); [Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-13"></a>
#### Result and interpretation

The program prints:

```text
FLOAT32
Shape[2, 2]
DENSE_CONTIGUOUS
true
weights
[1.0, 2.0, 3.0, 4.0]
```

The result proves exact [carrier](../../glossary.md#term-storage-carrier) [inference](../../glossary.md#term-inference-meanings), fully static dense [shape](../../glossary.md#shape) and [layout](../../glossary.md#layout) inference,
[row-major](../../glossary.md#term-row-major) copying, label retention, [gradient](../../glossary.md#term-gradient) eligibility, and source independence. It does not
promise a [snapshot](../../glossary.md#term-snapshot) under concurrent source mutation, numeric conversion, dynamic-shape inference,
[view](../../glossary.md#term-view-meanings) population, typed export, native allocation, or execution.

<a id="failures-and-useful-variations-10"></a>
#### Failures and useful variations

- `new int[2][0]` is valid and infers [shape](../../glossary.md#shape) `[2, 0]`; `new int[0][2]` is rejected because axis 1 is
  not observable from the empty root.
- `new float[][] {{1}, {2, 3}}` is rejected as ragged at [axis](../../glossary.md#term-axis) `1`, path `[1]`.
- `new float[][] {null, {1}}` is rejected for a null subarray at path `[0]`.
- `Integer[][]`, `boolean[][]`, and [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) `float[]` values are outside this [API](../../glossary.md#term-application-programming-interface-api).


---

[Previous topic](graph-randomness.md) · [Next topic](storage-association.md)
