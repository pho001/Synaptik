# Scalar selection, axis gather and one-hot encoding

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="scalar-select-expressions"></a>
### Scalar select expressions

`Tensor.select(int axis, long index)` fixes one scalar coordinate on an existing source axis and
removes that axis from a fresh result. The operation is a single-result scalar selection, not
elementwise conditional `WHERE`, repeated-select unstack, half-open interval `SLICE`,
or a gather whose indices come from a Tensor.

The caller may address an input axis from the front with a non-negative value or from the end with
a negative value in `[-rank, rank - 1]`. Construction normalizes that axis exactly once. When the
selected Dimension is static, one negative index adds the selected extent once, and every
normalized index must be in `[0, extent)`. A selected zero extent therefore rejects every index.
When the selected Dimension is dynamic, a non-negative index is retained unchanged and its upper-
bound check is deferred; a negative index is rejected because no numeric extent exists for local
normalization.

The result Shape omits exactly the selected Dimension. Every unaffected static or dynamic
Dimension remains the exact input reference in its original order. Selecting the only axis of a
rank-one Tensor produces the canonical scalar Shape `[]`.

Unresolved input layout produces unresolved result layout. An empty result also stays unresolved
because it references no storage element and needs no arbitrary offset geometry. Otherwise,
resolved dense, offset, strided, or zero-stride input geometry produces one new view-marked
logical `LayoutDescriptor`: the selected stride is removed, and checked arithmetic advances the
element offset by `normalizedIndex * selectedStride`. This metadata describes a logical view; the
result has no host storage and does not promise a physical alias or zero-copy execution.

Every successful result preserves the exact input data type and gradient eligibility, has no
label or host storage, and records `SelectKind.SELECT`, one normalized
`SelectAttrs(normalizedAxis, normalizedIndex)`, and provenance `[input]`. Repeated, nested, and
same-coordinate requests remain distinct fresh expressions.

<a id="complete-scalar-select-expression-example"></a>
#### Complete scalar-select expression example

<a id="goal-and-inputs-32"></a>
##### Goal and inputs

Select the final coordinate of axis one from a resolved `FLOAT32` Tensor with Shape `[2, 3, 4]`,
then inspect the normalized semantics and logical view geometry. The input has canonical strides
`[12, 4, 1]` and offset zero.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ScalarSelectExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3, 4);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true));

        Tensor selected = input.select(-2, -1);
        SelectAttrs attrs = (SelectAttrs) selected.provenance().orElseThrow()
                .operation().attrs();
        LayoutDescriptor layout = selected.descriptor().layout().orElseThrow();

        System.out.println(selected.descriptor().shape());
        System.out.println(attrs);
        System.out.println(Arrays.toString(layout.strides()));
        System.out.println(layout.storageOffset());
        System.out.println(layout.isView());
        System.out.println(selected.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(selected.descriptor().requiresGrad());
        System.out.println(selected.label().isEmpty() && selected.hostStorage().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-17"></a>
##### Meaningful lines and intermediate results

- Raw axis `-2` adds rank three once and becomes axis `1`. Its static extent is three, so raw index
  `-1` adds three once and becomes index `2`. The equivalent positive request is
  `select(1, 2)`.
- Removing axis one from Shape `[2, 3, 4]` leaves Shape `[2, 4]`. The input strides
  `[12, 4, 1]` lose selected stride `4`, producing `[12, 1]`.
- The checked result offset is `0 + 2 * 4 = 8`. The result retains `FLOAT32` and true gradient
  eligibility, records exact one-input provenance, and remains unlabeled and storage-free.

<a id="result-and-interpretation-32"></a>
##### Result and interpretation

The program prints:

```text
Shape[2, 4]
SelectAttrs[axis=1, index=2]
[12, 1]
8
true
true
true
true
```

The output demonstrates raw axis/index normalization, axis removal, checked logical view geometry,
exact semantics, and fresh provenance. It does not demonstrate selected values, a physical storage
alias, a gradient rule, compiler capture, materialization, backend lowering, or execution.

<a id="failures-and-useful-variations-29"></a>
##### Failures and useful variations

- Every axis is invalid for a scalar input. Other axes outside `[-rank, rank - 1]` fail through
  Shape normalization before result identity allocation.
- A static selected extent rejects an index below `-extent` or at least `extent` after one
  normalization. A dynamic selected extent accepts every non-negative `long` unchanged with its
  upper bound deferred, but rejects a negative value because it cannot normalize it locally.
- Selecting the only axis of Shape `[5]` produces scalar Shape `[]`. Selecting a dynamic axis
  removes it while preserving exact unaffected Dimension references, but current dynamic input
  descriptors cannot provide resolved numeric layout geometry.
- Result-element-count, offset, layout-classification, or referenced-span overflow propagates as
  `ArithmeticException` before Tensor identity allocation. Identifier exhaustion can occur only
  during final derived construction.

<a id="axis-gather-expressions"></a>
### Axis-gather expressions

The public axis-gather methods produce expressions with ordered logical inputs
`[data, indices]`. `data` is the receiver and supplies the result data type and gradient
eligibility. Both require exact `INT32` or `INT64` indices; no floating or BOOL index tensor is
accepted or implicitly cast. `gather` and `gatherElements` normalize one positive or negative axis
exactly once against the data Shape after their null and index-type checks; `embedding` fixes that
axis at zero after validating its rank-two floating receiver.

The methods differ in how they relate the two input Shapes to the result:

| Public method | Semantic kind | Shape rule |
|---|---|---|
| `gather(indices, axis)` | `GATHER` | Replace the selected data Dimension with every indices Dimension in order. |
| `embedding(indices)` | `GATHER` at axis zero | Require rank-two floating weights and append their exact axis-one embedding Dimension to the complete indices Shape. |
| `gatherElements(indices, axis)` | `GATHER_ELEMENTS` | Require equal ranks and equal Dimensions on every non-selected axis, then retain the exact indices Shape as the result. |

Structural equality is conservative for dynamic Dimensions. The same dynamic symbol passes where
equality is required; different symbols fail rather than creating a new symbolic constraint.
`gather` needs no compatibility check between the inserted indices Shape and the omitted data
Dimension, and scalar indices remove that axis without inserting another. `gatherElements`
permits the selected indices extent to differ from the data extent.

Every successful request returns a fresh unlabeled, storage-free result Tensor with unresolved
layout, even when both provenance inputs have resolved geometry. Its provenance contains the exact
semantic kind, one `IndexAxisAttrs` with the normalized axis, and exact input references in
`[data, indices]` order.
Expression construction never interprets, normalizes, clamps, or bounds-checks index values. It
defines no gradient or repeated-index rule, compiler capture or canonicalization, materialization,
backend lowering, or execution behavior.

<a id="embedding-convenience"></a>
#### Embedding convenience

`weights.embedding(indices)` gives the common lookup-table use of Gather a direct public spelling.
The receiver is a rank-two floating table shaped `[vocabulary, embeddingSize]`; indices may have
any rank but must use exact `INT32` or `INT64`. The result Shape is:

```text
indices.shape + [weights.shape[1]]
```

Every indices Dimension is retained by exact reference and in its original order, followed by the
exact weight axis-one Dimension. Scalar indices therefore produce `[embeddingSize]`. Result type
and gradient eligibility come only from weights. Layout remains unresolved, and the result has no
label or storage.

<a id="complete-embedding-metadata-example"></a>
##### Complete embedding metadata example

The goal is to construct lookup metadata for weights shaped `[10, 4]` and indices shaped `[2, 3]`
and inspect the ordinary Gather occurrence. Both inputs are storage-free because construction
uses descriptors, not element values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class EmbeddingExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor weights = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(2, 3), false);

        Tensor embedded = weights.embedding(indices);
        var provenance = embedded.provenance().orElseThrow();

        System.out.println(embedded.descriptor().shape());
        System.out.println(provenance.operation().kind());
        System.out.println(provenance.operation().attrs());
        System.out.println(provenance.inputs().get(0) == weights);
        System.out.println(provenance.inputs().get(1) == indices);
        System.out.println(provenance.outputIndex());
        System.out.println(embedded.descriptor().requiresGrad());
        System.out.println(embedded.descriptor().layout().isEmpty()
                && embedded.label().isEmpty()
                && embedded.hostStorage().isEmpty());
    }
}
```

The indices axes `2` and `3` are retained first, and the weight embedding extent `4` is appended.
The program prints:

```text
Shape[2, 3, 4]
GATHER
IndexAxisAttrs[axis=0]
true
true
0
true
true
```

This proves the derived Shape, direct axis-zero Gather semantics, ordered `[weights, indices]`
provenance, single output at index zero, weight-only gradient eligibility, and storage-free result
metadata. Each valid call creates one producer and one fresh Tensor ID. It does not read the six
index values, select weight rows, construct a gradient, compile a graph, or execute work.

Negative and out-of-range stored index values remain accepted at construction because model code
does not inspect values. They are invalid for later ordinary Gather execution and must not wrap,
clamp, select padding, or select a default row. Compiler constant analysis may reject values it can
prove invalid; after dynamic extents are bound, preparation or prepared execution must enforce the
ordinary Gather bounds contract safely. This convenience adds no `EMBEDDING` kind and no
padding-index, sparse-gradient, maximum-norm, or frequency-scaling option.

<a id="complete-axis-gather-expression-example"></a>
#### Complete axis-gather expression example

<a id="goal-and-inputs-33"></a>
##### Goal and inputs

Compare both Shape relationships for `FLOAT32` data with Shape `[2, 3, 4]`. The example uses
storage-free index tensors because current expression construction needs only their immutable
descriptors.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class AxisGatherExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor insertedIndices = tensor(DataType.INT64, Shape.of(5, 6), false);
        Tensor alignedIndices = tensor(DataType.INT32, Shape.of(2, 7, 4), false);

        Tensor gathered = data.gather(insertedIndices, -2);
        Tensor elements = data.gatherElements(alignedIndices, 1);

        IndexAxisAttrs attrs = (IndexAxisAttrs) gathered.provenance().orElseThrow()
                .operation().attrs();
        System.out.println(gathered.descriptor().shape());
        System.out.println(gathered.provenance().orElseThrow().operation().kind());
        System.out.println(elements.descriptor().shape());
        System.out.println(elements.provenance().orElseThrow().operation().kind());
        System.out.println(attrs);
        System.out.println(gathered.provenance().orElseThrow().inputs().get(0) == data);
        System.out.println(gathered.provenance().orElseThrow().inputs().get(1) == insertedIndices);
        System.out.println(gathered.descriptor().requiresGrad());
        System.out.println(gathered.descriptor().layout().isEmpty()
                && gathered.label().isEmpty()
                && gathered.hostStorage().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-18"></a>
##### Meaningful lines and intermediate results

- Raw axis `-2` normalizes once against rank three to axis `1`.
- `gather` inserts complete Shape `[5, 6]` where extent `3` was, producing `[2, 5, 6, 4]`.
- `gatherElements` verifies the aligned extents `2` and `4`, ignores the selected extents `3` and
  `7`, and retains exact indices Shape `[2, 7, 4]`.
- All results retain `FLOAT32` and true gradient eligibility from `data`; the index tensors do not
  contribute either fact.

<a id="result-and-interpretation-33"></a>
##### Result and interpretation

The program prints:

```text
Shape[2, 5, 6, 4]
GATHER
Shape[2, 7, 4]
GATHER_ELEMENTS
IndexAxisAttrs[axis=1]
true
true
true
true
```

The output demonstrates axis normalization, both Shape rules, metadata retention, unresolved
layout, and ordered provenance. It does not demonstrate
gathered values, valid index-value bounds, gradient behavior, compiler capture, backend lowering,
or execution.

<a id="failures-and-useful-variations-30"></a>
##### Failures and useful variations

- Null data and indices fail in that order. An unsupported indices data type fails before axis
  normalization. An invalid axis then fails through `Shape.normalizeAxis`.
- `gatherElements` checks rank first, then non-axis Dimensions in increasing axis order and reports
  the first mismatch. Its selected-axis extents may differ.
- Scalar indices passed to `gather` remove the selected data axis without inserting another one.
  Rank-one `gather` with scalar indices produces scalar Shape `[]`.
- For both signatures, local validation completes before result identity
  allocation. Identifier exhaustion can occur only during final derived construction.

<a id="one-hot-encoding-expressions"></a>
### One-hot encoding expressions

`indices.oneHot(depth)` creates metadata for one dense trailing-axis indicator encoding. The
receiver must have exact INT32 or INT64 type, and `depth` must be a positive `long`. For receiver
Shape `[d0, ..., dn]`, the result Shape is:

```text
[exact d0 reference, ..., exact dn reference, new StaticDimension(depth)]
```

Every existing Dimension object is retained by reference and in order. The appended Dimension is
fresh. A scalar receiver therefore produces Shape `[depth]`; a receiver containing a zero extent
remains valid and has zero logical rows. `Long.MAX_VALUE` depth is structurally accepted because
construction does not calculate a result element count or materialize storage.

For an input coordinate `p` with eventual index value `i`, the exact logical meaning is:

```text
result[p..., j] = (i == j), for 0 <= j < depth
```

The result has exact BOOL type, `requiresGrad=false`, unresolved layout, no label, and no host
storage. Each successful call creates one `ONE_HOT` operation with `OneHotAttrs(depth)`, one fresh
single-output producer retaining exact input `[indices]`, provenance output index zero, and one
fresh Tensor identity.

<a id="conceptual-vector-and-scalar-examples"></a>
#### Conceptual vector and scalar examples

The goal is to connect index values to the requested logical result while keeping the current
metadata-only boundary explicit. For conceptual INT64 values `[2, 0, 1]` with Shape `[3]` and
depth `3`, the requested values are:

```text
input:  [2, 0, 1]
result: [[false, false, true],
         [true,  false, false],
         [false, true,  false]]
```

Each input position becomes one row on the new trailing class axis. Index `2` selects trailing
position `2`, index `0` selects position `0`, and index `1` selects position `1`. The result Shape
is `[3, 3]`. A conceptual scalar INT32 index `2` with depth `4` instead requests rank-one Shape
`[4]` and values `[false, false, true, false]`.

These value arrays are conceptual expected results, not output produced by current code.
`oneHot` inspects only receiver metadata, so it does not read the example values, allocate the
BOOL result, compile an operation, or execute encoding. A caller can inspect the current
construction contract with storage-free metadata:

```java
Tensor indices = TensorFactory.create(new TensorDescriptor(
        DataType.INT64, Shape.of(3), Optional.empty(), false));
Tensor encoded = indices.oneHot(3);

System.out.println(encoded.descriptor().shape());
System.out.println(encoded.descriptor().dataType());
System.out.println(encoded.provenance().orElseThrow().operation());
```

The observable metadata is Shape `[3, 3]`, BOOL, and an operation whose kind is `ONE_HOT` and
whose attributes are `OneHotAttrs[depth=3]`. This proves Shape, type, and semantic provenance; it
does not prove or calculate the conceptual BOOL values.

<a id="failures-and-execution-boundary"></a>
#### Failures and execution boundary

- A receiver whose type is not INT32 or INT64 fails before depth validation with
  `oneHot indices data type must be INT32 or INT64: <type>`.
- Depth zero or a negative depth fails with `depth must be positive: <depth>`. Null helper input,
  type failure, and depth failure occur before Tensor identity allocation.
- Valid eventual execution requires `0 <= i < depth` for every index value. Negative and
  out-of-range values are invalid; they do not wrap, clamp, select a default, or produce an
  all-false row. Model construction deliberately accepts stored invalid values because it never
  reads storage. A later compiler may reject provably invalid constants, while future preparation
  or prepared execution must safely enforce the bound for dynamically supplied values.
- There is no configurable axis, Tensor-valued depth, on/off value, output type, ignore index,
  sparse form, gradient rule, compiler support, backend lowering, or runtime implementation.


---

[Previous topic](slicing.md) · [Next topic](scatter.md)
