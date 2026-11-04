# Masked, Boolean and index reductions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**Reader model.** A **mask** is a true/false tensor marking which positions participate.
Masked reductions exclude false positions before arithmetic; multiplying all data by numeric
zeros and ones is not equivalent when excluded values contain NaN or infinity.
Boolean `all` asks whether every selected truth value is true; `any` asks whether at least one is.
**Arg-min** and **arg-max** return the position of an extremum, not the extremum's value.
A **tie policy** selects which position wins when several candidates are equal.
Mask shapes align from the last axis; they do not automatically align with axes named “batch”
or “time.”

<a id="masked-sum-and-mean-expressions"></a>
### Masked sum and mean expressions

The two masked aggregate overloads build one axis-removing expression from a floating input and
an exact BOOL mask:

| Public form | Operation kind | Result axis |
|---|---|---|
| `sum(axis, mask)` | `AggregateReductionKind.SUM` | Removed |
| `mean(axis, mask)` | `AggregateReductionKind.MEAN` | Removed |

The mask uses the same ordinary right-aligned broadcasting rule as other model expressions, but
the relationship is directional: `broadcast(inputShape, maskShape)` must equal `inputShape`.
The mask may use trailing dimensions and singleton expansion, but it may not add a leading axis
or enlarge an input singleton. A scalar mask broadcasts to the complete input. Equal symbolic or
expression dimensions are accepted when the existing local Shape rule can prove equality.

Right alignment does not infer semantic axis intent. For input `[batch, time, features]`, a mask
shaped `[batch, time]` generally compares `time` with `features` and is rejected. A caller that
means the first two axes inserts a trailing singleton explicitly, producing
`[batch, time, 1]`. That rank-edit expression remains visible as the exact mask producer consumed
by the reduction.

Every valid call returns a fresh Tensor with the input's exact floating data type and gradient
eligibility, unresolved layout, no label, and no host storage. Its operation carries
`MaskedReductionAttrs(normalizedAxis)`, and provenance retains exact ordered references
`[input, alignedMask]`. The mask's gradient eligibility does not affect the result.

The semantic meaning excludes false mask positions. Masked sum produces zero when no value is
selected. Masked mean divides each output by its selected true-count and produces NaN when that
count is zero; no NaN payload or bit pattern is promised. False positions exclude their input
before aggregation, including NaN and infinity. A static zero-sized reduction axis therefore
produces zero for every masked-sum slice and NaN for every masked-mean slice. A runtime zero-sized
or all-false dynamic slice follows the same semantic rule. Expression construction records these
rules but does not align storage, materialize a mask, inspect values, count positions, aggregate,
divide, create a gradient rule, capture a graph, or execute work.

For FLOAT32, masking and the true-count determine the exact contributor set before
arithmetic. Every selected value participates exactly once under the aggregate floor; masked MEAN
still performs the mandatory division by the positive true-count. Existing qualifying final
exact-zero publication freedoms remain local to the final SUM or quotient. False positions stay
excluded even when NaN or infinity, all-false and zero-sized slices remain positive-zero SUM and
NaN MEAN results, and no tolerance, term change, identity insertion, or predicate change is
allowed.

<a id="complete-masked-reduction-example"></a>
#### Complete masked-reduction example

<a id="goal-and-inputs-22"></a>
##### Goal and inputs

Build masked sum and mean metadata for a storage-free `FLOAT32` input shaped
`[batch, time, 4]` and a BOOL mask initially shaped `[batch, time]`. The example makes the mask's
intended axes explicit with `expandDims(2)`, then observes both producer boundaries and the
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

- `expandDims(2)` inserts a trailing singleton, changing the mask Shape from `[batch, time]` to
  `[batch, time, 1]`. The new Tensor records the original mask as its rank-edit producer input.
- Ordinary right alignment now compares `batch` with `batch`, `time` with `time`, and singleton
  `1` with feature extent `4`, so the mask broadcasts exactly to the input Shape.
- Both `1` and `-2` normalize to input axis `1`, the `time` axis.
- Removing input axis `1` produces `[batch, 4]`. The exact `batch` dimension object is retained.
- The masked operation's provenance stores the input first and the exact `alignedMask` Tensor
  second. Only the input supplies result type and gradient eligibility.

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

The output proves the caller-visible rank edit, ordinary broadcast-compatible Shape, local axis
normalization, axis removal, result metadata, and the two producer boundaries. It does not prove
mask materialization, selected values, a sum or mean result, gradient support, compiler capture,
backend support, or execution.

<a id="failures-and-useful-variations-19"></a>
##### Failures and useful variations

- A null mask fails with `NullPointerException`. Integral or BOOL input and every non-BOOL mask
  fail with `IllegalArgumentException`; type checks occur before axis validation.
- An invalid axis fails with `IndexOutOfBoundsException`, including every axis for a scalar input.
  Axis validation occurs before mask broadcasting.
- Incompatible aligned dimensions fail with `IllegalArgumentException`. A compatible broadcast
  that would enlarge the input or add a leading result axis also fails. No cast, symbolic
  constraint, reshape, or rank edit is inserted.
- Scalar masks, equal-rank masks, lower-rank trailing masks, zero extents, equal dynamic symbols,
  exact equal expressions, reused unknown dimensions, and mask-side singleton dimensions are
  supported structurally. Distinct unknown dimensions remain locally unprovable. Repeated valid
  calls remain fresh expressions.

<a id="boolean-aggregate-expressions"></a>
### Boolean aggregate expressions

The six current boolean aggregate methods build conjunction (`all`) or disjunction (`any`)
reduction metadata from one exact BOOL input:

| Family | Full form | Axis removed | Axis optionally retained |
|---|---|---|---|
| Boolean conjunction | `all()` | `all(axis)` | `all(axis, keepDimensions)` |
| Boolean disjunction | `any()` | `any(axis)` | `any(axis, keepDimensions)` |

Every full form selects all input axes, records `NoOperationAttrs.INSTANCE`, and returns a fresh
canonical rank-zero BOOL scalar. An axis form accepts a positive or negative axis and normalizes it
against the input `Shape`. A false `keepDimensions` removes the selected axis; true replaces only
that axis with a new static extent of one. Removing the sole axis of a rank-one input also returns
the canonical scalar shape. Every unaffected static or dynamic `Dimension` object is retained by
exact reference.

Every result has exact `DataType.BOOL`, false gradient eligibility, unresolved layout, no label,
and no host storage. Provenance records `AggregateReductionKind.ALL` or
`AggregateReductionKind.ANY` with exactly the receiver as its one input. This is different from
elementwise `logicalAnd(Tensor)` and `logicalOr(Tensor)`: those methods record
`BooleanLogicalKind.AND` or `BooleanLogicalKind.OR`, retain two ordered inputs, and broadcast
corresponding positions instead of reducing a domain. Equal words such as “and” and “all” describe
related truth concepts, but their typed operation identities and provenance arities are distinct.

Expression construction accepts scalar, ordinary static, zero-extent, and dynamic shapes without
reading truth bytes or element counts. It does not short-circuit, evaluate conjunction or
disjunction, decide whether an empty conjunction is true or an empty disjunction is false, define
gradient behavior, capture or optimize a graph, report backend support, or execute work.

Failure behavior is local and deterministic:

- `all` and `any` accept only exact BOOL input. Every floating or integral input fails with
  `IllegalArgumentException`; no numeric truthiness or implicit cast is used. Type validation
  occurs before axis validation and result identity allocation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`. Every axis is invalid
  for a scalar input, while a full scalar reduction is valid and returns a fresh scalar.
- Repeated and nested calls always create fresh expression tensors; no truth simplification,
  canonicalization, or common-subexpression elimination occurs here.

<a id="arg-extrema-expressions"></a>
### Arg-extrema expressions

The six current `argMin` and `argMax` methods build index-producing aggregate metadata along
exactly one input axis:

| Public forms | Retained dimension | Tie policy |
|---|---|---|
| `argMin(axis)`, `argMax(axis)` | Removed | Explicitly supplies `FIRST_INDEX` |
| `argMin(axis, keepDimensions)`, `argMax(axis, keepDimensions)` | Caller choice | Explicitly supplies `FIRST_INDEX` |
| Complete three-argument forms | Caller choice | Exact non-null caller policy |

The convenience forms choose the smallest logical index among equal extrema by passing
`ArgExtremaTiePolicy.FIRST_INDEX` directly to the shared construction boundary. This is a
public-overload convenience, not an implicit default in `ArgExtremaAttrs`: the semantic attribute
record requires an explicit non-null policy. A complete overload can pass `LAST_INDEX`, which
requests the largest logical index among equal candidates. Logical indices are coordinates along
the normalized selected axis, independent of storage offset, stride, layout, or traversal.

Every form accepts floating and signed-integral input, including FP8, and rejects BOOL;
numeric inputs are neither promoted nor converted. The positive or negative caller axis is
normalized exactly once against the input `Shape`. A false `keepDimensions` removes the selected
axis; true replaces only that axis with a new static extent of one. Removing the sole axis of a
rank-one tensor produces the canonical scalar Shape. Every unaffected static or dynamic
`Dimension` object is retained by exact reference.

Every result is a fresh `DataType.INT64` Tensor with false gradient eligibility, unresolved
layout, no label, and no host storage. Provenance records the exact
`AggregateReductionKind.ARG_MIN` or `ARG_MAX`, one
`ArgExtremaAttrs(normalizedAxis, keepDimensions, tiePolicy)`, exactly the receiver as its sole
ordered input, and output index zero. Repeated or nested calls are not simplified or canonicalized.

Integral candidates use signed order. Floating candidates use one shared total selection policy:
NaN is preferred to every non-NaN for both minimum and maximum; multiple NaNs are ties; negative
zero orders below positive zero; infinities retain their usual order; and otherwise equal numeric
values are ties. The explicit policy chooses the first or last logical index after this ordering.
This fixes semantic results without choosing an algorithm or traversal.

A statically known zero selected extent is rejected because it has no valid logical index, with
`IllegalArgumentException` and message `arg-extrema reduction axis must be non-empty, but axis
<normalizedAxis> has static extent 0`. A zero on an unselected axis is valid and may make the
result empty. A dynamic or expression selected extent is accepted structurally and must be
positive when later proven or bound; no sentinel index represents an empty domain. Construction
does not read storage, compare values, select an index, create gradients, capture a graph, lower an
operation, report backend support, or execute.

Current Metal execution covers the six numeric carriers for fully static
canonical dense ranks `1..16` whose positive geometry and logical indices fit unsigned 32 bits. It
uses one exact custom program, retains INT64 outputs, and implements NaN preference and both tie
policies; BOOL, dynamic/empty selected axes, over-limit geometry, and generated backward graphs
remain outside that backend domain.

Failure behavior is deterministic: a null explicit policy fails before input-type or axis
validation; BOOL input fails before axis validation; and an axis outside
`[-rank, rank - 1]` fails with `IndexOutOfBoundsException`, including every axis for a scalar
input. Axis validation precedes static-zero inspection. These local failures occur before result
identity allocation.


---

[Previous topic](reductions.md) · [Next topic](fixed-scans.md)
