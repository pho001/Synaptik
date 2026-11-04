# Flat and nested value import examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="complete-flat-import-example"></a>
### Complete flat-import example

<a id="goal-and-inputs-12"></a>
#### Goal and inputs

Import three logical BOOL values from bytes `{0, -2, 3}`, verify canonical storage, and show that
later source mutation does not affect the tensor.

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

- `LayoutDescriptor.contiguous(shape)` supplies the required resolved dense-contiguous geometry.
- The `byte[]` carrier matches `BOOL`; import normalizes both non-zero values to canonical byte
  `1` while copying into a distinct factory-allocated array.
- Mutating `source` after import demonstrates that the returned tensor does not retain that array.
  The raw heap-base inspection uses the already exposed storage segment only to make this example's
  result observable; it is not a typed Tensor access API.

<a id="result-and-interpretation-12"></a>
#### Result and interpretation

The program prints:

```text
mask
[0, 1, 1]
```

The result proves BOOL normalization, copied ownership, label retention, and row-major order for a
dense flat import. It does not demonstrate nested shape inference or provide public typed access
or export, numeric conversion, view scattering, native allocation, runtime residency, or backend
execution.

<a id="failures-and-useful-variations-9"></a>
#### Failures and useful variations

- Using this `byte[]` with a descriptor other than BOOL or either FP8 type fails before destination or ID allocation.
- An offset, strided, broadcast, or unresolved layout is rejected before source-length validation.
- A source of length two or four fails because the shape has exactly three logical elements.
- `short[]` imports raw BFLOAT16 bits; it does not convert Java `float` values.

<a id="complete-nested-import-example"></a>
### Complete nested-import example

<a id="goal-and-inputs-13"></a>
#### Goal and inputs

Import the rectangular `float[][]` value `{{1, 2}, {3, 4}}`, infer its descriptor, preserve the
gradient request, and verify that later source mutation does not affect the tensor. The source has
declared rank two, exact `float` leaves, and two elements along each axis.

```java
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class NestedImportExample {
    public static void main(String[] args) {
        float[] firstRow = {1.0f, 2.0f};
        float[][] source = {firstRow, {3.0f, 4.0f}};

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
  reads declared rank two and ultimate carrier `float`, which infer `FLOAT32` and shape `[2, 2]`.
- `true` requests gradient eligibility on the inferred descriptor. `FLOAT32` permits that request;
  an integral or BOOL carrier would reject it before destination or ID allocation.
- The factory flattens the two rows in encounter order and delegates the resulting `float[]` to
  flat import, which creates the independent destination array.
- Mutating `firstRow` after return demonstrates source independence. The heap-base inspection is
  only a way to observe this example's raw result; it is not a public typed Tensor access API.

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

The result proves exact carrier inference, fully static dense shape and layout inference,
row-major copying, label retention, gradient eligibility, and source independence. It does not
promise a snapshot under concurrent source mutation, numeric conversion, dynamic-shape inference,
view population, typed export, native allocation, or execution.

<a id="failures-and-useful-variations-10"></a>
#### Failures and useful variations

- `new int[2][0]` is valid and infers shape `[2, 0]`; `new int[0][2]` is rejected because axis 1 is
  not observable from the empty root.
- `new float[][] {{1}, {2, 3}}` is rejected as ragged at axis `1`, path `[1]`.
- `new float[][] {null, {1}}` is rejected for a null subarray at path `[0]`.
- `Integer[][]`, `boolean[][]`, and rank-one `float[]` values are outside this API.


---

[Previous topic](graph-randomness.md) · [Next topic](storage-association.md)
