# Matrix semantic identity, sorting and Top-K

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="matrix-multiplication-semantic-kind"></a>
### Matrix-multiplication semantic kind

`MatmulKind` contains exactly `MATMUL`. The kind pairs only with
`NoOperationAttrs.INSTANCE`; its family-owned `OperationSignature` fixes two logical inputs and
one logical output. It records backend-independent vector, matrix, and batched matrix
multiplication meaning, including the accumulation policies above. The kind stores no operands,
Shapes, deferred constraints, result descriptor, provenance, algorithm, gradient, compiler,
backend, or execution state. `Tensor.matmul` separately owns input-aware local validation and
expression construction.

<a id="stable-sort-and-argsort-expressions"></a>
### Stable sort and argsort expressions

`input.sort(axis)` and `input.argsort(axis)` request ascending stable ordering along one logical
axis. Their two-argument overloads select descending non-NaN order when `descending` is true.
Every current data type is eligible: signed integers use signed numerical order, BOOL uses
`false < true`, and floating values use numerical order with the additional rules below.

For each independent slice along the normalized axis:

- equal finite, integral, or BOOL values keep increasing original logical indices;
- negative zero precedes positive zero ascending and follows it descending;
- negative and positive infinity participate in ordinary numerical order;
- all NaNs remain after every non-NaN in both directions; multiple NaNs keep input order; and
- `sort` retains the selected input representation, including signed zero and NaN payload bits.

The following value table shows the requested result for one conceptual FLOAT64 row. It describes
eventual ordering semantics; current model construction does not evaluate the row.

| Request | Values | Logical indices |
|---|---|---|
| Input | `[3.0, NaN, -0.0, +0.0, 3.0]` | `[0, 1, 2, 3, 4]` |
| Ascending | `[-0.0, +0.0, 3.0, 3.0, NaN]` | `[2, 3, 0, 4, 1]` |
| Descending | `[3.0, 3.0, +0.0, -0.0, NaN]` | `[0, 4, 3, 2, 1]` |

<a id="complete-metadata-example"></a>
#### Complete metadata example

The example constructs both requests and inspects facts that the current model actually records.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Optional;

Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT64, Shape.of(5), Optional.empty(), true));
Tensor values = input.sort(-1);
Tensor indices = input.argsort(0, true);

System.out.println(values.descriptor().dataType());
System.out.println(indices.descriptor().dataType());
System.out.println(values.provenance().orElseThrow().operation());
System.out.println(indices.provenance().orElseThrow().operation());
```

The output metadata is FLOAT64, INT64, `Operation[kind=SORT,
attrs=SortAttrs[axis=0, descending=false]]`, and `Operation[kind=ARGSORT,
attrs=SortAttrs[axis=0, descending=true]]`. The negative axis was normalized before attributes
construction. This proves type and semantic provenance; it does not calculate the table's values.

Each successful call creates a fresh, independent one-input, one-output producer at output slot
zero. Both results retain the exact input Shape reference and leave layout unresolved. `sort`
also preserves input type and gradient eligibility; `argsort` always uses INT64 and false gradient
eligibility. Results are unlabeled and storage-free. Static empty and singleton selected extents
are valid, as are dynamic extents, because the Shape does not change. A scalar has no valid axis.

`OrderingKind` contains exactly `SORT` and `ARGSORT`. Each kind requires `SortAttrs`, one input,
and one output. `SortAttrs(axis, descending)` holds an already normalized non-negative axis and
the direction flag; stability and NaN-last placement are fixed semantics, not attributes.
Construction checks only metadata and creates provenance. It neither compares values nor selects
an algorithm, gradient rule, compiler behavior, backend route, runtime behavior, or execution
support.

Current Metal execution covers both kinds for all seven carriers for canonical
dense ranks `1..16` whose positive dimensions, element counts, strides, selected extents, and
one-dimensional grid fit unsigned 32 bits. Its integer-only custom route implements the order above
and copies selected representations exactly. Dynamic, empty, noncanonical, over-limit, and
generated-backward occurrences remain outside that backend domain.

<a id="top-k-values-and-indices"></a>
### Top-K values and indices

`input.topK(k, axis)` selects the largest `k` entries of each logical-axis slice and requests
stable sorted output. The complete overload
`input.topK(k, axis, largest, sorted)` can instead select the smallest entries or return the same
selected pairs in deterministic original logical-index order. Both methods return one
`TopKResult(values, indices)` from a single genuine two-output producer.

Top-K uses the same complete stable ordering as `sort` and `argsort`: signed integers use signed
order, BOOL uses `false < true`, infinities use numerical order, negative zero is below positive
zero before direction reversal, NaNs remain after every non-NaN for both directions, and equal
values or NaNs prefer increasing original logical-axis index. The selected set is the first `k`
entries of that complete largest or smallest order. `sorted == true` retains selection order;
`sorted == false` reorders the selected pairs by increasing original logical-axis index without
changing membership or breaking value/index pairing.

For conceptual input `[3.0, NaN, -0.0, +0.0, 3.0]`, the four requests mean:

| Request | Selected values | Selected logical indices |
|---|---|---|
| largest, sorted | `[3.0, 3.0, +0.0]` | `[0, 4, 3]` |
| largest, unsorted | `[3.0, +0.0, 3.0]` | `[0, 3, 4]` |
| smallest, sorted | `[-0.0, +0.0, 3.0]` | `[2, 3, 0]` |
| smallest, unsorted | `[3.0, -0.0, +0.0]` | `[0, 2, 3]` |

This table fixes eventual semantic results; current construction does not inspect or evaluate the
input values. Selected values preserve exact representations, including signed zero and NaN
payload bits.

<a id="complete-construction-example"></a>
#### Complete construction example

The example constructs top-K metadata and verifies the two output slots. It is runnable without
an execution backend because it inspects descriptors and provenance only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TopKResult;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Optional;

Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT64, Shape.of(2, 5), Optional.empty(), true));
TopKResult result = input.topK(3, -1, true, false);

System.out.println(result.values().descriptor().shape());
System.out.println(result.values().descriptor().dataType());
System.out.println(result.indices().descriptor().dataType());
System.out.println(result.values().provenance().orElseThrow().producer()
        == result.indices().provenance().orElseThrow().producer());
System.out.println(result.values().provenance().orElseThrow().outputIndex());
System.out.println(result.indices().provenance().orElseThrow().outputIndex());
System.out.println(result.values().provenance().orElseThrow().operation().attrs()
        .equals(new TopKAttrs(1, 3, true, false)));
```

The output is `Shape[2, 3]`, `FLOAT64`, `INT64`, `true`, `0`, `1`, and `true`. Axis `-1` was
normalized to axis one. One producer retains exact input `[input]` and ordered descriptors
`[values, indices]`; the two fresh wrappers have separate IDs and provenance slots zero and one.
This proves current Shape, type, attributes, and occurrence identity. It does not calculate any
selected value or index.

All seven current data types are eligible. The values descriptor preserves input type and gradient
eligibility; the indices descriptor is INT64 with false gradient eligibility. Both descriptors
retain the same exact fresh result Shape, leave layout unresolved, and produce unlabeled,
storage-free wrappers. The result Shape preserves rank and every unselected Dimension reference,
but replaces the selected dimension with a fresh `StaticDimension(k)`.

`k` must be non-negative. When the selected input extent is static, construction also requires
`k <= extent`; zero is valid, including for a zero extent. A dynamic or expression-based selected
extent accepts construction without binding. Later compiler or binding validation must enforce
`bound extent >= k` before any conforming execution; it must not clamp, pad, wrap, or return fewer
pairs. Axis normalization happens before K validation, every scalar axis fails, and all local
failures occur before result-ID allocation.

`TopKKind.TOP_K` pairs only with `TopKAttrs(axis, k, largest, sorted)` and fixes one input and two
outputs. This current API constructs backend-neutral metadata and immutable pre-capture
provenance. It does not provide a selection algorithm, value evaluation, gradient rule, compiler
capture or dynamic-bound enforcement, backend support, runtime behavior, or execution.

Current Metal execution covers positive-K TOP_K for all seven carriers over the
same canonical dense unsigned-32-bit-bounded geometry. One custom native step always validates,
materializes, and keeps the paired values and INT64 indices, including when only one role is a
public target. Zero/dynamic K, empty/dynamic geometry, over-limit/noncanonical layouts, and
generated backward graphs remain outside that backend domain.


---

[Previous topic](pooling.md) · [Next topic](host-storage.md)
