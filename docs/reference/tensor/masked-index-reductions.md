# Masked, [Boolean](../../glossary.md#data-type--datatype) and index reductions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** A **[mask](../../glossary.md#term-boolean-mask)** is a true/false [tensor](../../glossary.md#tensor) marking which positions participate.
Masked reductions exclude false positions before arithmetic; multiplying all data by numeric
zeros and ones is not equivalent when excluded values contain [NaN](../../glossary.md#term-nan) or [infinity](../../glossary.md#term-infinity).
Boolean `all` asks whether every selected truth value is true; `any` asks whether at least one is.
**Arg-min** and **arg-max** return the position of an [extremum](../../glossary.md#term-extremum), not the extremum's value.
A **tie policy** selects which position wins when several candidates are equal.
Mask [shapes](../../glossary.md#shape) align from the last [axis](../../glossary.md#term-axis); they do not automatically align with axes named “[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)”
or “time.”

<a id="masked-sum-and-mean-expressions"></a>
### Masked sum and [mean](../../glossary.md#term-mean) expressions

The two masked aggregate overloads build one [axis](../../glossary.md#term-axis)-removing expression from a floating input and
an exact [BOOL](../../glossary.md#data-type--datatype) [mask](../../glossary.md#term-boolean-mask):

| Public form | [Operation](../../glossary.md#operation) kind | Result axis |
|---|---|---|
| `sum(axis, mask)` | `AggregateReductionKind.SUM` | Removed |
| `mean(axis, mask)` | `AggregateReductionKind.MEAN` | Removed |

The mask uses the same ordinary right-aligned [broadcasting](../../glossary.md#broadcasting) rule as other [model](../../glossary.md#term-model) expressions, but
the relationship is directional: `broadcast(inputShape, maskShape)` must equal `inputShape`.
The mask may use trailing [dimensions](../../glossary.md#dimension) and [singleton](../../glossary.md#term-singleton) expansion, but it may not add a leading axis
or enlarge an input singleton. A [scalar](../../glossary.md#term-scalar) mask broadcasts to the complete input. Equal symbolic or
expression dimensions are accepted when the existing local [Shape](../../glossary.md#shape) rule can prove equality.

Right [alignment](../../glossary.md#term-alignment) does not infer semantic axis intent. For input `[batch, time, features]`, a mask
shaped `[batch, time]` generally compares `time` with [`features`](../../glossary.md#term-feature) and is rejected. A caller that
means the first two axes inserts a trailing singleton explicitly, producing
`[batch, time, 1]`. That [rank](../../glossary.md#term-rank)-edit expression remains visible as the exact mask [producer](../../glossary.md#term-producer) consumed
by the reduction.

Every valid call returns a fresh [Tensor](../../glossary.md#tensor) with the input's exact floating [data type](../../glossary.md#data-type--datatype) and [gradient](../../glossary.md#term-gradient)
eligibility, unresolved [layout](../../glossary.md#layout), no label, and no [host storage](../../glossary.md#host-storage). Its operation carries
`MaskedReductionAttrs(normalizedAxis)`, and [provenance](../../glossary.md#provenance) retains exact ordered references
`[input, alignedMask]`. The mask's gradient eligibility does not affect the result.

The semantic meaning excludes false mask positions. Masked sum produces zero when no value is
selected. Masked mean divides each output by its selected true-count and produces [NaN](../../glossary.md#term-nan) when that
count is zero; no [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) or bit pattern is promised. False positions exclude their input
before aggregation, including NaN and [infinity](../../glossary.md#term-infinity). A static zero-sized reduction axis therefore
produces zero for every masked-sum [slice](../../glossary.md#slice) and NaN for every masked-mean slice. A [runtime](../../glossary.md#term-runtime) zero-sized
or all-false dynamic slice follows the same semantic rule. Expression construction records these
rules but does not align storage, materialize a mask, inspect values, count positions, aggregate,
divide, create a gradient rule, [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), or execute work.

For [FLOAT32](../../glossary.md#term-float32), masking and the true-count determine the exact contributor set before
arithmetic. Every selected value participates exactly once under the [aggregate floor](../../glossary.md#term-aggregate-floor); masked MEAN
still performs the mandatory division by the positive true-count. Existing qualifying final
exact-zero [publication](../../glossary.md#term-publication) freedoms remain local to the final SUM or quotient. False positions stay
excluded even when NaN or infinity, all-false and zero-sized slices remain positive-zero SUM and
NaN MEAN results, and no [tolerance](../../glossary.md#term-tolerance), term change, [identity](../../glossary.md#term-identity-meanings) insertion, or predicate change is
allowed.

<a id="complete-masked-reduction-example"></a>
#### Complete masked-reduction example

<a id="goal-and-inputs-22"></a>
##### Goal and inputs

Build masked sum and [mean](../../glossary.md#term-mean) [metadata](../../glossary.md#term-metadata) for a storage-free [`FLOAT32`](../../glossary.md#term-float32) input shaped
`[batch, time, 4]` and a [BOOL](../../glossary.md#data-type--datatype) [mask](../../glossary.md#term-boolean-mask) initially shaped `[batch, time]`. The example makes the mask's
intended [axes](../../glossary.md#term-axis) explicit with `expandDims(2)`, then observes both [producer](../../glossary.md#term-producer) boundaries and the
masked-reduction result metadata. It does not calculate aggregate values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class MaskedReductionExpressionExample {
    public static void main(String[] args) {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension time = new DynamicDimension("time");
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, time, new StaticDimension(4)),
                Optional.empty(),
                true));
        Tensor mask = TensorFactory.create(new TensorDescriptor(
                DataType.BOOL,
                Shape.ofDimensions(batch, time),
                Optional.empty(),
                false));

        Tensor alignedMask = mask.expandDims(2);
        Tensor sum = input.sum(1, alignedMask);
        Tensor mean = input.mean(-2, alignedMask);
        MaskedReductionAttrs attrs =
                (MaskedReductionAttrs) sum.provenance().orElseThrow().operation().attrs();

        System.out.println("alignedMaskShape=" + alignedMask.descriptor().shape());
        System.out.println("sumShape=" + sum.descriptor().shape());
        System.out.println("meanShape=" + mean.descriptor().shape());
        System.out.println("axis=" + attrs.axis());
        System.out.println("rankEditInput="
                + (alignedMask.provenance().orElseThrow().inputs().get(0) == mask));
        System.out.println("orderedInputs="
                + (sum.provenance().orElseThrow().inputs().get(0) == input
                && sum.provenance().orElseThrow().inputs().get(1) == alignedMask));
        System.out.println("metadata="
                + (sum.descriptor().dataType() == DataType.FLOAT32
                && sum.descriptor().requiresGrad()
                && sum.descriptor().layout().isEmpty()
                && sum.label().isEmpty()
                && sum.hostStorage().isEmpty()));
    }
}
```

<a id="meaningful-lines-and-intermediate-results-7"></a>
##### Meaningful lines and intermediate results

- `expandDims(2)` inserts a trailing [singleton](../../glossary.md#term-singleton), changing the [mask](../../glossary.md#term-boolean-mask) [Shape](../../glossary.md#shape) from `[batch, time]` to
  `[batch, time, 1]`. The new [Tensor](../../glossary.md#tensor) records the original mask as its [rank](../../glossary.md#term-rank)-edit [producer](../../glossary.md#term-producer) input.
- Ordinary right [alignment](../../glossary.md#term-alignment) now compares [`batch`](../../glossary.md#term-batch-machine-learning-and-hyperparameters) with `batch`, `time` with `time`, and singleton
  `1` with [feature](../../glossary.md#term-feature) [extent](../../glossary.md#term-extent) `4`, so the mask broadcasts exactly to the input Shape.
- Both `1` and `-2` normalize to input [axis](../../glossary.md#term-axis) `1`, the `time` axis.
- Removing input axis `1` produces `[batch, 4]`. The exact `batch` [dimension](../../glossary.md#dimension) object is retained.
- The masked [operation](../../glossary.md#operation)'s [provenance](../../glossary.md#provenance) stores the input first and the exact `alignedMask` Tensor
  second. Only the input supplies result type and gradient eligibility.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient))</span>

<a id="result-and-interpretation-22"></a>
##### Result and interpretation

The program prints:

```text
alignedMaskShape=Shape[batch, time, 1]
sumShape=Shape[batch, 4]
meanShape=Shape[batch, 4]
axis=1
rankEditInput=true
orderedInputs=true
metadata=true
```

The output proves the caller-visible [rank](../../glossary.md#term-rank) edit, ordinary broadcast-compatible [Shape](../../glossary.md#shape), local [axis](../../glossary.md#term-axis)
[normalization](../../glossary.md#term-normalization), axis removal, result [metadata](../../glossary.md#term-metadata), and the two [producer](../../glossary.md#term-producer) boundaries. It does not prove
[mask](../../glossary.md#term-boolean-mask) [materialization](../../glossary.md#materialization), selected values, a sum or [mean](../../glossary.md#term-mean) result, [gradient](../../glossary.md#term-gradient) support, [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings),
[backend](../../glossary.md#backend) support, or execution.

<a id="failures-and-useful-variations-19"></a>
##### Failures and useful variations

- A null [mask](../../glossary.md#term-boolean-mask) fails with `NullPointerException`. Integral or BOOL input and every non-BOOL mask
  fail with <span data-glossary-navigation="true"> ([BOOL glossary](../../glossary.md#data-type--datatype))</span>`IllegalArgumentException`; type checks occur before [axis](../../glossary.md#term-axis) [validation](../../glossary.md#term-validation).
- An invalid axis fails with `IndexOutOfBoundsException`, including every axis for a scalar input.
  Axis validation occurs before mask broadcasting.<span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar); [broadcasting glossary](../../glossary.md#broadcasting))</span>
- Incompatible aligned [dimensions](../../glossary.md#dimension) fail with `IllegalArgumentException`. A compatible broadcast
  that would enlarge the input or add a leading result axis also fails. No cast, symbolic
  constraint, reshape, or rank edit is inserted.<span data-glossary-navigation="true"> ([rank glossary](../../glossary.md#term-rank))</span>
- Scalar masks, equal-rank masks, lower-rank trailing masks, zero extents, equal dynamic symbols,
  exact equal expressions, reused unknown dimensions, and mask-side singleton dimensions are
  supported structurally. Distinct unknown dimensions remain locally unprovable. Repeated valid
  calls remain fresh expressions.<span data-glossary-navigation="true"> ([singleton glossary](../../glossary.md#term-singleton))</span>

<a id="boolean-aggregate-expressions"></a>
### [Boolean](../../glossary.md#data-type--datatype) aggregate expressions

The six current boolean aggregate methods build conjunction (`all`) or disjunction (`any`)
reduction [metadata](../../glossary.md#term-metadata) from one exact [BOOL](../../glossary.md#data-type--datatype) input:

| Family | Full form | [Axis](../../glossary.md#term-axis) removed | Axis optionally retained |
|---|---|---|---|
| Boolean conjunction | `all()` | `all(axis)` | `all(axis, keepDimensions)` |
| Boolean disjunction | `any()` | `any(axis)` | `any(axis, keepDimensions)` |

Every full form selects all input axes, records `NoOperationAttrs.INSTANCE`, and returns a fresh
[canonical](../../glossary.md#term-canonical-form) [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) BOOL [scalar](../../glossary.md#term-scalar). An axis form accepts a positive or negative axis and normalizes it
against the input [`Shape`](../../glossary.md#shape). A false `keepDimensions` removes the selected axis; true replaces only
that axis with a new static [extent](../../glossary.md#term-extent) of one. Removing the sole axis of a [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) input also returns
the canonical scalar shape. Every unaffected static or dynamic [`Dimension`](../../glossary.md#dimension) object is retained by
exact reference.

Every result has exact `DataType.BOOL`, false [gradient](../../glossary.md#term-gradient) eligibility, unresolved [layout](../../glossary.md#layout), no label,
and no [host storage](../../glossary.md#host-storage). [Provenance](../../glossary.md#provenance) records `AggregateReductionKind.ALL` or
`AggregateReductionKind.ANY` with exactly the receiver as its one input. This is different from
[elementwise](../../glossary.md#term-elementwise) `logicalAnd(Tensor)` and `logicalOr(Tensor)`: those methods record
`BooleanLogicalKind.AND` or `BooleanLogicalKind.OR`, retain two ordered inputs, and broadcast
corresponding positions instead of reducing a domain. Equal words such as “and” and “all” describe
related truth concepts, but their typed [operation](../../glossary.md#operation) [identities](../../glossary.md#term-identity-meanings) and provenance arities are distinct.

Expression construction accepts scalar, ordinary static, zero-extent, and dynamic shapes without
reading truth bytes or [element](../../glossary.md#term-element) counts. It does not short-circuit, evaluate conjunction or
disjunction, decide whether an empty conjunction is true or an empty disjunction is false, define
gradient behavior, [capture](../../glossary.md#term-capture-meanings) or optimize a [graph](../../glossary.md#graph), report [backend](../../glossary.md#backend) support, or execute work.

Failure behavior is local and deterministic:

- `all` and `any` accept only exact BOOL input. Every floating or integral input fails with
  `IllegalArgumentException`; no numeric truthiness or implicit cast is used. Type validation
  occurs before axis validation and result identity allocation.<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation); [identity glossary](../../glossary.md#term-identity-meanings))</span>
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`. Every axis is invalid
  for a scalar input, while a full scalar reduction is valid and returns a fresh scalar.
- Repeated and nested calls always create fresh expression tensors; no truth simplification,
  canonicalization, or common-subexpression elimination occurs here.<span data-glossary-navigation="true"> ([tensors glossary](../../glossary.md#tensor); [canonicalization glossary](../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../glossary.md#term-common-subexpression-elimination-cse))</span>

<a id="arg-extrema-expressions"></a>
### Arg-[extrema](../../glossary.md#saturation-pole-and-extremum) expressions

The six current `argMin` and `argMax` methods build index-producing aggregate [metadata](../../glossary.md#term-metadata) along
exactly one input [axis](../../glossary.md#term-axis):

| Public forms | Retained [dimension](../../glossary.md#dimension) | Tie policy |
|---|---|---|
| `argMin(axis)`, `argMax(axis)` | Removed | Explicitly supplies `FIRST_INDEX` |
| `argMin(axis, keepDimensions)`, `argMax(axis, keepDimensions)` | Caller choice | Explicitly supplies `FIRST_INDEX` |
| Complete three-[argument](../../glossary.md#term-argument) forms | Caller choice | Exact non-null caller policy |

The convenience forms choose the smallest [logical index](../../glossary.md#axis-rank-scalar-vector-and-matrix) among equal extrema by passing
`ArgExtremaTiePolicy.FIRST_INDEX` directly to the shared construction boundary. This is a
public-overload convenience, not an implicit default in `ArgExtremaAttrs`: the semantic attribute
record requires an explicit non-null policy. A complete overload can pass `LAST_INDEX`, which
requests the largest logical index among equal candidates. Logical indices are [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) along
the normalized selected axis, independent of [storage offset](../../glossary.md#term-storage-offset), [stride](../../glossary.md#term-stride), [layout](../../glossary.md#layout), or traversal.

Every form accepts floating and signed-integral input, including [FP8](../../glossary.md#term-fp8), and rejects [BOOL](../../glossary.md#data-type--datatype);
numeric inputs are neither promoted nor converted. The positive or negative caller axis is
normalized exactly once against the input [`Shape`](../../glossary.md#shape). A false `keepDimensions` removes the selected
axis; true replaces only that axis with a new static [extent](../../glossary.md#term-extent) of one. Removing the sole axis of a
[rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) [tensor](../../glossary.md#tensor) produces the [canonical](../../glossary.md#term-canonical-form) [scalar](../../glossary.md#term-scalar) Shape. Every unaffected static or dynamic
`Dimension` object is retained by exact reference.

Every result is a fresh `DataType.INT64` Tensor with false [gradient](../../glossary.md#term-gradient) eligibility, unresolved
layout, no label, and no [host storage](../../glossary.md#host-storage). [Provenance](../../glossary.md#provenance) records the exact
`AggregateReductionKind.ARG_MIN` or `ARG_MAX`, one
`ArgExtremaAttrs(normalizedAxis, keepDimensions, tiePolicy)`, exactly the receiver as its sole
ordered input, and output index zero. Repeated or nested calls are not simplified or canonicalized.

Integral candidates use signed order. Floating candidates use one shared total selection policy:
[NaN](../../glossary.md#term-nan) is preferred to every non-NaN for both minimum and maximum; multiple NaNs are ties; [negative
zero](../../glossary.md#nan-infinity-and-signed-zero) orders below [positive zero](../../glossary.md#nan-infinity-and-signed-zero); [infinities](../../glossary.md#nan-infinity-and-signed-zero) retain their usual order; and otherwise equal numeric
values are ties. The explicit policy chooses the first or last logical index after this ordering.
This fixes semantic results without choosing an algorithm or traversal.

A statically known zero selected extent is rejected because it has no valid logical index, with
`IllegalArgumentException` and message `arg-extrema reduction axis must be non-empty, but axis
<normalizedAxis> has static extent 0`. A zero on an unselected axis is valid and may make the
result empty. A dynamic or expression selected extent is accepted structurally and must be
positive when later proven or bound; no sentinel index represents an empty domain. Construction
does not read storage, compare values, select an index, create gradients, [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), lower an
[operation](../../glossary.md#operation), report [backend](../../glossary.md#backend) support, or execute.

Current [Metal](../../glossary.md#term-metal) execution covers the six numeric carriers for fully static
canonical dense ranks `1..16` whose positive geometry and logical indices fit unsigned 32 bits. It
uses one exact custom program, retains [INT64](../../glossary.md#data-type--datatype) outputs, and implements NaN preference and both tie
policies; BOOL, dynamic/empty selected axes, over-limit geometry, and generated backward graphs
remain outside that backend domain.

Failure behavior is deterministic: a null explicit policy fails before input-type or axis
[validation](../../glossary.md#term-validation); BOOL input fails before axis validation; and an axis outside
`[-rank, rank - 1]` fails with `IndexOutOfBoundsException`, including every axis for a scalar
input. Axis validation precedes static-zero inspection. These local failures occur before result
[identity](../../glossary.md#term-identity-meanings) allocation.


---

[Previous topic](reductions.md) · [Next topic](fixed-scans.md)
