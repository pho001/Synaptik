# [Matrix](../../glossary.md#term-matrix-meanings) semantic [identity](../../glossary.md#term-identity-meanings), sorting and Top-K

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** Sorting reorders values along an [axis](../../glossary.md#term-axis) independently for each remaining
[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix). **Stable** means candidates equal under the ordering retain original index order.
`argsort` returns those source positions rather than sorted values. **Top-K** selects only
`k` value/index pairs; its “unsorted” option still returns a defined original-index ordering,
not arbitrary or random order. A [logical index](../../glossary.md#axis-rank-scalar-vector-and-matrix) is an axis coordinate, not a physical memory
offset. NaNs and signed zeros need explicit ordering rules, so ordinary “ascending” intuition
alone is insufficient. The matrix-kind section first distinguishes the name of an [operation](../../glossary.md#operation)
from the input-aware [Tensor](../../glossary.md#tensor) method that constructs an [occurrence](../../glossary.md#term-occurrence).

<a id="matrix-multiplication-semantic-kind"></a>
### [Matrix](../../glossary.md#term-matrix-meanings)-multiplication semantic kind

`MatmulKind` contains exactly [`MATMUL`](../../glossary.md#matrix-multiplication--matmul). The kind pairs only with
`NoOperationAttrs.INSTANCE`; its family-owned `OperationSignature` fixes two logical inputs and
one logical output. It records [backend](../../glossary.md#backend)-independent [vector](../../glossary.md#term-vector), matrix, and batched [matrix
multiplication](../../glossary.md#matrix-multiplication--matmul) meaning, including the accumulation policies above. The kind stores no operands,
[Shapes](../../glossary.md#shape), deferred constraints, result [descriptor](../../glossary.md#term-descriptor), [provenance](../../glossary.md#provenance), algorithm, [gradient](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler),
backend, or execution state. `Tensor.matmul` separately owns input-aware local [validation](../../glossary.md#term-validation) and
expression construction.

<a id="stable-sort-and-argsort-expressions"></a>
### Stable sort and argsort expressions

`input.sort(axis)` and `input.argsort(axis)` request ascending stable ordering along one logical
[axis](../../glossary.md#term-axis). Their two-[argument](../../glossary.md#term-argument) overloads select descending non-[NaN](../../glossary.md#term-nan) order when `descending` is true.
Every current [data type](../../glossary.md#data-type--datatype) is eligible: signed integers use signed numerical order, [BOOL](../../glossary.md#data-type--datatype) uses
`false < true`, and floating values use numerical order with the additional rules below.

For each independent [slice](../../glossary.md#slice) along the [normalized axis](../../glossary.md#normalized-axis):

- equal [finite](../../glossary.md#term-finite), integral, or BOOL values keep increasing original logical indices;
- [negative zero](../../glossary.md#nan-infinity-and-signed-zero) precedes [positive zero](../../glossary.md#nan-infinity-and-signed-zero) ascending and follows it descending;
- negative and [positive infinity](../../glossary.md#nan-infinity-and-signed-zero) participate in ordinary numerical order;
- all NaNs remain after every non-NaN in both directions; multiple NaNs keep input order; and
- `sort` retains the selected input representation, including [signed zero](../../glossary.md#term-signed-zero) and [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) bits.

The following value table shows the requested result for one conceptual [FLOAT64](../../glossary.md#term-float64) row. It describes
eventual ordering semantics; current [model](../../glossary.md#term-model) construction does not evaluate the row.

| Request | Values | Logical indices |
|---|---|---|
| Input | `[3.0, NaN, -0.0, +0.0, 3.0]` | `[0, 1, 2, 3, 4]` |
| Ascending | `[-0.0, +0.0, 3.0, 3.0, NaN]` | `[2, 3, 0, 4, 1]` |
| Descending | `[3.0, 3.0, +0.0, -0.0, NaN]` | `[0, 4, 3, 2, 1]` |

<a id="complete-metadata-example"></a>
#### Complete [metadata](../../glossary.md#term-metadata) example

The example constructs both requests and inspects facts that the current [model](../../glossary.md#term-model) actually records.

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

The output metadata is [FLOAT64](../../glossary.md#term-float64), [INT64](../../glossary.md#data-type--datatype), `Operation[kind=SORT,
attrs=SortAttrs[axis=0, descending=false]]`, and `Operation[kind=ARGSORT,
attrs=SortAttrs[axis=0, descending=true]]`. The negative [axis](../../glossary.md#term-axis) was normalized before attributes
construction. This proves type and semantic [provenance](../../glossary.md#provenance); it does not calculate the table's values.

Each successful call creates a fresh, independent one-input, one-output [producer](../../glossary.md#term-producer) at output slot
zero. Both results retain the exact input [Shape](../../glossary.md#shape) reference and leave [layout](../../glossary.md#layout) unresolved. `sort`
also preserves input type and [gradient](../../glossary.md#term-gradient) eligibility; `argsort` always uses INT64 and false gradient
eligibility. Results are unlabeled and storage-free. Static empty and [singleton](../../glossary.md#term-singleton) selected extents
are valid, as are dynamic extents, because the Shape does not change. A [scalar](../../glossary.md#term-scalar) has no valid axis.

`OrderingKind` contains exactly `SORT` and `ARGSORT`. Each kind requires `SortAttrs`, one input,
and one output. `SortAttrs(axis, descending)` holds an already normalized non-negative axis and
the direction flag; stability and [NaN](../../glossary.md#term-nan)-last placement are fixed semantics, not attributes.
Construction checks only metadata and creates provenance. It neither compares values nor selects
an algorithm, gradient rule, [compiler](../../glossary.md#term-compiler) behavior, [backend route](../../glossary.md#backend-route), [runtime](../../glossary.md#term-runtime) behavior, or execution
support.

Current [Metal](../../glossary.md#term-metal) execution covers both kinds for all seven carriers for [canonical](../../glossary.md#term-canonical-form)
dense ranks `1..16` whose positive [dimensions](../../glossary.md#dimension), [element](../../glossary.md#term-element) counts, [strides](../../glossary.md#term-stride), selected extents, and
one-dimensional grid fit unsigned 32 bits. Its integer-only custom route implements the order above
and copies selected representations exactly. Dynamic, empty, noncanonical, over-limit, and
generated-backward occurrences remain outside that [backend](../../glossary.md#backend) domain.

<a id="top-k-values-and-indices"></a>
### Top-K values and indices

`input.topK(k, axis)` selects the largest `k` entries of each logical-[axis](../../glossary.md#term-axis) [slice](../../glossary.md#slice) and requests
stable sorted output. The complete overload
`input.topK(k, axis, largest, sorted)` can instead select the smallest entries or return the same
selected pairs in deterministic original logical-index order. Both methods return one
`TopKResult(values, indices)` from a single genuine two-output [producer](../../glossary.md#term-producer).

Top-K uses the same complete stable ordering as `sort` and `argsort`: signed integers use signed
order, [BOOL](../../glossary.md#data-type--datatype) uses `false < true`, [infinities](../../glossary.md#nan-infinity-and-signed-zero) use numerical order, [negative zero](../../glossary.md#nan-infinity-and-signed-zero) is below [positive
zero](../../glossary.md#nan-infinity-and-signed-zero) before direction reversal, NaNs remain after every non-[NaN](../../glossary.md#term-nan) for both directions, and equal
values or NaNs prefer increasing original logical-[axis index](../../glossary.md#axis-rank-scalar-vector-and-matrix). The selected set is the first `k`
entries of that complete largest or smallest order. `sorted == true` retains selection order;
`sorted == false` reorders the selected pairs by increasing original logical-axis index without
changing membership or breaking value/index pairing.

For conceptual input `[3.0, NaN, -0.0, +0.0, 3.0]`, the four requests [mean](../../glossary.md#term-mean):

| Request | Selected values | Selected logical indices |
|---|---|---|
| largest, sorted | `[3.0, 3.0, +0.0]` | `[0, 4, 3]` |
| largest, unsorted | `[3.0, +0.0, 3.0]` | `[0, 3, 4]` |
| smallest, sorted | `[-0.0, +0.0, 3.0]` | `[2, 3, 0]` |
| smallest, unsorted | `[3.0, -0.0, +0.0]` | `[0, 2, 3]` |

This table fixes eventual semantic results; current construction does not inspect or evaluate the
input values. Selected values preserve exact representations, including [signed zero](../../glossary.md#term-signed-zero) and [NaN
payload](../../glossary.md#nan-infinity-and-signed-zero) bits.

<a id="complete-construction-example"></a>
#### Complete construction example

The example constructs top-K [metadata](../../glossary.md#term-metadata) and verifies the two output slots. It is runnable without
an execution [backend](../../glossary.md#backend) because it inspects [descriptors](../../glossary.md#term-descriptor) and [provenance](../../glossary.md#provenance) only.

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

The output is `Shape[2, 3]`, [`FLOAT64`](../../glossary.md#term-float64), [`INT64`](../../glossary.md#data-type--datatype), `true`, `0`, `1`, and `true`. [Axis](../../glossary.md#term-axis) `-1` was
normalized to axis one. One [producer](../../glossary.md#term-producer) retains exact input `[input]` and ordered descriptors
`[values, indices]`; the two fresh wrappers have separate IDs and provenance slots zero and one.
This proves current [Shape](../../glossary.md#shape), type, attributes, and [occurrence](../../glossary.md#term-occurrence) [identity](../../glossary.md#term-identity-meanings). It does not calculate any
selected value or index.

All seven current data types are eligible. The values descriptor preserves input type and [gradient](../../glossary.md#term-gradient)
eligibility; the indices descriptor is INT64 with false gradient eligibility. Both descriptors
retain the same exact fresh result Shape, leave [layout](../../glossary.md#layout) unresolved, and produce unlabeled,
storage-free wrappers. The result Shape preserves [rank](../../glossary.md#term-rank) and every unselected [Dimension](../../glossary.md#dimension) reference,
but replaces the selected dimension with a fresh `StaticDimension(k)`.

`k` must be non-negative. When the selected input [extent](../../glossary.md#term-extent) is static, construction also requires
`k <= extent`; zero is valid, including for a zero extent. A dynamic or expression-based selected
extent accepts construction without [binding](../../glossary.md#term-binding). Later [compiler](../../glossary.md#term-compiler) or binding [validation](../../glossary.md#term-validation) must enforce
`bound extent >= k` before any conforming execution; it must not clamp, pad, wrap, or return fewer
pairs. Axis [normalization](../../glossary.md#term-normalization) happens before K validation, every [scalar](../../glossary.md#term-scalar) axis fails, and all local
failures occur before result-ID allocation.

`TopKKind.TOP_K` pairs only with `TopKAttrs(axis, k, largest, sorted)` and fixes one input and two
outputs. This current [API](../../glossary.md#term-application-programming-interface-api) constructs backend-neutral metadata and [immutable](../../glossary.md#term-immutable) pre-[capture](../../glossary.md#term-capture-meanings)
provenance. It does not provide a selection algorithm, value [evaluation](../../glossary.md#term-evaluation), gradient rule, compiler
capture or dynamic-bound enforcement, backend support, [runtime](../../glossary.md#term-runtime) behavior, or execution.

Current [Metal](../../glossary.md#term-metal) execution covers positive-K TOP_K for all seven carriers over the
same [canonical](../../glossary.md#term-canonical-form) dense unsigned-32-bit-bounded geometry. One custom native step always validates,
materializes, and keeps the paired values and INT64 indices, including when only one role is a
public [target](../../glossary.md#term-target). Zero/dynamic K, empty/dynamic geometry, over-limit/noncanonical layouts, and
generated backward [graphs](../../glossary.md#graph) remain outside that backend domain.


---

[Previous topic](pooling.md) · [Next topic](host-storage.md)
