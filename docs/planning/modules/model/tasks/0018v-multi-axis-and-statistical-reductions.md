# Task 0018V: Multi-[axis](../../../../glossary.md#term-axis) and Statistical Reductions

**Editorial reading key (current context, not a status update).**

This completed task records model-expression construction, not execution support. The caller
supplies a receiver Tensor and axes to public reduction methods; the package-private helper
validates them and delegates result/provenance construction to TensorFactory. Axes are copied
descriptions, not element values: normalized order identifies attributes, while membership
determines Shape. For input `[2, 3, 4]`, axes `[2, 0]` leave `[3]`, or `[1, 3, 1]` when retained.
Empty axes mean one-value domains, not full reduction. Correction subtracts a count from the
statistical denominator; point-domain correction one is invalid. Fresh storage-free expressions
record their producing operation and input reference; they neither execute nor own result buffers.
Model semantic responsibility is not storage ownership.

Read the historical exact-real, once-rounded numerical target in its historical scope. The [current
reduction
contract](../../../../reference/tensor/reductions.md#multi-axis-and-statistical-reduction-expressions)
explains today's rounded contributor trees, named-site subnormal permissions, and statistical
validation; its [numeric aggregate
section](../../../../reference/tensor/reductions.md#numeric-aggregate-expressions) details the
limited FLOAT32 final exact-zero sign freedom. These are not blanket reordering or equivalence
licenses: contributor membership, counts, correction, empty-domain identities, and special-value
boundaries remain mandatory. Model semantics defines meaning; execution providers realize supported
routes rather than redefine it.

**Mathematical reading illustration.** For a finite real slice $x_1,\ldots,x_n\in\mathbb{R}$ with
integer count $n>0$, mean $\mu=\frac1n\sum_{i=1}^{n}x_i$ and corrected variance
$v=\frac{\sum_{i=1}^{n}(x_i-\mu)^2}{n-c}$ require correction $c\ge0$ and $n>c$. For $n=2$, $x=(1,3)$
and $c=0$, $\mu=2$ and $v=1$; correction one instead gives two. These are real-number
interpretations, not permissions to replace the recorded or current rounded arithmetic.

## Status

Complete

## Goal

Close the capability-reset reduction frontier with one coherent [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model)
contract for reductions over an ordered set of distinct [axes](../../../../glossary.md#term-axis) and for first-class [log-sum-exp](../../../../glossary.md#log-sum-exp),
[variance](../../../../glossary.md#term-variance), [standard deviation](../../../../glossary.md#term-standard-deviation), [L1 norm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions), and [L2 norm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) expressions.

This task extends the seven existing ordinary aggregate families to multi-axis forms, appends five
first-class meanings to `AggregateReductionKind`, adds [immutable](../../../../glossary.md#term-immutable) normalized multi-axis attributes,
and exposes exact public [`Tensor`](../../../../glossary.md#tensor) construction. It fixes portable result, empty-domain,
special-value, correction, and accumulation-type semantics without reading values, selecting an
execution algorithm, constructing [gradients](../../../../glossary.md#term-gradient), capturing a [graph](../../../../glossary.md#graph), or claiming backend support.

## Why this remains one task

All selected operations share one caller-[axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization) contract, one structural reduced-[Shape](../../../../glossary.md#shape)
rule, one result/[provenance](../../../../glossary.md#provenance) construction boundary, and one floating numerical policy. [Variance](../../../../glossary.md#term-variance) and
[standard deviation](../../../../glossary.md#term-standard-deviation) additionally share correction. Splitting semantic kinds, attributes, and
[Tensor](../../../../glossary.md#tensor) construction would temporarily leave [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature) and public construction
inconsistent.

The implementation changes exactly 17 paths: five production, five test, and seven
documentation/[planning](../../../../glossary.md#term-planning) paths. This is within the planning guide's 12–18-file upper guardrail and
is a justified [atomic](../../../../glossary.md#resource-transaction-and-rollback) extension of one [model](../../../../glossary.md#term-model) family, not a cross-[module](../../../../glossary.md#term-module) exception. Task 0019
remains Draft and receives no detailed specification.

## [Mental model](../../../../glossary.md#term-mental-model) and examples

For input [Shape](../../../../glossary.md#shape) `[2, 3, 4]`:

```java
Tensor totals = input.sum(new int[] {2, 0}, false);
Tensor retained = input.mean(new int[] {-1, 0}, true);
```

Both normalize caller order to `[2, 0]`. `totals` has Shape `[3]`; `retained` has Shape
`[1, 3, 1]`. Shape derivation uses membership, not sorting. Reversing the requested order selects
the same Cartesian domain and Shape but remains a distinct attribute value and [operation](../../../../glossary.md#operation).

For floating input `[1, 2, 3]` reduced over its only [axis](../../../../glossary.md#term-axis):

```text
logSumExp                         ~= 3.407605964
variance(correction = 0)         = 2 / 3
variance(correction = 1)         = 1
standardDeviation(correction=0)  = sqrt(2 / 3)
l1Norm                            = 6
l2Norm                            = sqrt(14)
```

These are mathematical [targets](../../../../glossary.md#term-target). They require numerical [conformance](../../../../glossary.md#term-conformance) to the selected meaning but do
not prescribe max subtraction, pass count, compensation, traversal, vectorization, or a [kernel](../../../../glossary.md#kernel).

## [Scope](../../../../glossary.md#term-scope)

- Append exactly `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`, in
  that order, after `ARG_MIN` in `AggregateReductionKind`.
- Add `MultiAxisReductionAttrs(List<Integer> axes, boolean keepDimensions)` for ordinary
  multi-axis reductions, log-sum-exp, and L1/L2 norm.<span data-glossary-navigation="true"> ([log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [L1 glossary](../../../../glossary.md#term-l1); [L2 norm glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- Add `StatisticalReductionAttrs(List<Integer> axes, boolean keepDimensions, long correction)`
  for variance and standard deviation.<span data-glossary-navigation="true"> ([standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>
- Add a multi-axis operation signature to all seven ordinary kinds while preserving every current
  full, single-axis, masked, and arg-extrema signature.<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [signature glossary](../../../../glossary.md#term-signature))</span>
- Add the exact 26-method public [`Tensor`](../../../../glossary.md#tensor) surface below.
- Normalize caller axes in caller order, reject duplicates after normalization, and retain an
  immutable normalized order in attributes.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Define an empty [axis](../../../../glossary.md#term-axis) list as reduction over no axes, never full reduction or an error.
- Derive exact removal/retention [Shapes](../../../../glossary.md#shape) while preserving unaffected [`Dimension`](../../../../glossary.md#dimension) references.
- Preserve exact input [DataType](../../../../glossary.md#data-type--datatype) and `requiresGrad`, leave layout unresolved, and attach no label or
  storage.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Record a fresh one-output [producer](../../../../glossary.md#term-producer) with exact one-input [provenance](../../../../glossary.md#provenance) and output index zero.
- Fix all numerical policies below, including the previously deferred floating and BOOL ordinary
  empty/special-value policies, without changing construction metadata.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Finalize Javadocs, Tensor/Compile APIs, glossary, capabilities, planning state, and evidence in a
  mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>
- [Run](../../../../glossary.md#term-run) the capability-reset [checkpoint](../../../../glossary.md#term-checkpoint) after executable behavior and documentation stabilize.

## Exact public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor sum(int... axes)
public Tensor sum(int[] axes, boolean keepDimensions)
public Tensor mean(int... axes)
public Tensor mean(int[] axes, boolean keepDimensions)
public Tensor prod(int... axes)
public Tensor prod(int[] axes, boolean keepDimensions)
public Tensor min(int... axes)
public Tensor min(int[] axes, boolean keepDimensions)
public Tensor max(int... axes)
public Tensor max(int[] axes, boolean keepDimensions)
public Tensor all(int... axes)
public Tensor all(int[] axes, boolean keepDimensions)
public Tensor any(int... axes)
public Tensor any(int[] axes, boolean keepDimensions)

public Tensor logSumExp(int... axes)
public Tensor logSumExp(int[] axes, boolean keepDimensions)
public Tensor variance(int... axes)
public Tensor variance(int[] axes, boolean keepDimensions)
public Tensor variance(int[] axes, boolean keepDimensions, long correction)
public Tensor standardDeviation(int... axes)
public Tensor standardDeviation(int[] axes, boolean keepDimensions)
public Tensor standardDeviation(int[] axes, boolean keepDimensions, long correction)
public Tensor l1Norm(int... axes)
public Tensor l1Norm(int[] axes, boolean keepDimensions)
public Tensor l2Norm(int... axes)
public Tensor l2Norm(int[] axes, boolean keepDimensions)
```

Varargs forms use `keepDimensions=false`. [Variance](../../../../glossary.md#term-variance) and standard-deviation forms that omit
correction use population correction zero. The caller array is cloned and never retained.
Existing fixed zero-[argument](../../../../glossary.md#term-argument) ordinary methods still win overload resolution, so `input.sum()` is
the unchanged full reduction while `input.sum(new int[0])` explicitly reduces no [axes](../../../../glossary.md#term-axis). A
zero-argument call to a new varargs family selects an empty axis list.

Add no `std`, generic [`norm`](../../../../glossary.md#term-norm), arbitrary-p overload, collection overload, options object, full
sentinel, static form, output-type override, accuracy mode, or [backend](../../../../glossary.md#backend) hint. The declared public
Tensor method count becomes exactly 156.

## Semantic and attribute contracts

### Ordered normalized [axes](../../../../glossary.md#term-axis)

Create exactly:

```java
public record MultiAxisReductionAttrs(
        List<Integer> axes,
        boolean keepDimensions) implements OperationAttrs
```

Its compact constructor validates `axes`, then each [element](../../../../glossary.md#term-element) in index order, with exact failures:

```text
NullPointerException("axes")
NullPointerException("axes[<index>]")
IllegalArgumentException("axes[<index>] must be non-negative: <axis>")
IllegalArgumentException("axes contains duplicate axis <axis> at index <index>")
```

It finishes with `List.copyOf(axes)`, and its explicit accessor returns that [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot).
An empty list is valid. The record stores no [Shape](../../../../glossary.md#shape)/[rank](../../../../glossary.md#term-rank) and cannot validate an upper bound.

Create exactly:

```java
public record StatisticalReductionAttrs(
        List<Integer> axes,
        boolean keepDimensions,
        long correction) implements OperationAttrs
```

It applies identical axis [validation](../../../../glossary.md#term-validation) and snapshotting first, then rejects negative correction with
exact message `correction must be non-negative: <correction>`. Correction is `long` because
domain [cardinality](../../../../glossary.md#term-cardinality) is not bounded by `int`. Fractional, negative, inferred, per-axis, or
data-type-dependent correction is not selected. Zero means [population variance](../../../../glossary.md#population-variance); one expresses the
usual [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) estimator when the domain has at least two elements.

Public construction normalizes each raw axis exactly once with `Shape.normalizeAxis`, in caller
order, and rejects a duplicate immediately. An earlier duplicate therefore precedes a later
invalid raw axis. The duplicate message uses the normalized value and raw request index. Axis
order is semantic [metadata](../../../../glossary.md#term-metadata) for equality, diagnostics, transformation, and interchange, but does
not prescribe physical traversal or a sequential floating fold.

### Kind/[signature](../../../../glossary.md#term-signature) pairings

Preserve every current [constant](../../../../glossary.md#term-constant) and signature. Extend exact pairings:

- `SUM`/[`MEAN`](../../../../glossary.md#term-mean): current full, single-[axis](../../../../glossary.md#term-axis), and masked variants plus one-input/one-output
  `MultiAxisReductionAttrs`;
- `PROD`/`MIN`/`MAX`/`ALL`/`ANY`: current full and single-axis variants plus
  `MultiAxisReductionAttrs`;
- `ARG_MIN`/`ARG_MAX`: unchanged `ArgExtremaAttrs` only;
- `LOG_SUM_EXP`/`L1_NORM`/`L2_NORM`: exactly one one-input/one-output
  `MultiAxisReductionAttrs` variant; and
- [`VARIANCE`](../../../../glossary.md#term-variance)/`STANDARD_DEVIATION`: exactly one one-input/one-output
  `StatisticalReductionAttrs` variant.

Do not add another kind family, [registry](../../../../glossary.md#term-registry), traits/result flag, compatibility [alias](../../../../glossary.md#aliasing-and-arena), or hidden public
[decomposition](../../../../glossary.md#term-decomposition). [Log-sum-exp](../../../../glossary.md#log-sum-exp) is not stored as [exp](../../../../glossary.md#term-expx)/sum/[log](../../../../glossary.md#term-log), [standard deviation](../../../../glossary.md#term-standard-deviation) is not stored as
variance/sqrt, and norms are not stored as [unary](../../../../glossary.md#term-unary)/power/sum subgraphs.

### [Shape](../../../../glossary.md#shape) and empty-[axis](../../../../glossary.md#term-axis) meaning

When `keepDimensions=false`, remove every selected axis and retain unselected [Dimensions](../../../../glossary.md#dimension) by exact
reference in input order. When true, retain [rank](../../../../glossary.md#term-rank), replace each selected position with a new
`StaticDimension(1)`, and retain every unselected reference. Construct Shape once. Removing every
axis or reducing a [scalar](../../../../glossary.md#term-scalar) over an empty list produces [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`.

An empty list selects a one-value domain at each input position. It creates a fresh [occurrence](../../../../glossary.md#term-occurrence) and
is never simplified to the input:

- ordinary sum/[mean](../../../../glossary.md#term-mean)/product/minimum/maximum/all/any and [log-sum-exp](../../../../glossary.md#log-sum-exp) return the point value;
- [L1](../../../../glossary.md#term-l1)/[L2 norm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) return [absolute value](../../../../glossary.md#norms-elementary-functions-and-probability-distributions);
- [population variance](../../../../glossary.md#population-variance)/[standard deviation](../../../../glossary.md#term-standard-deviation) return [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero); and
- correction at least one is invalid because `N=1`.

`keepDimensions` has no Shape effect for an empty list but remains part of attribute [identity](../../../../glossary.md#term-identity-meanings).
Existing full forms remain distinct reductions over every input axis.

### Data types and result [metadata](../../../../glossary.md#term-metadata)

Multi-[axis](../../../../glossary.md#term-axis) ordinary eligibility exactly matches completed full/single-axis behavior:

| Kind | Accepted exact input/result type |
|---|---|
| `SUM`, `PROD`, `MIN`, `MAX` | [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), [FLOAT64](../../../../glossary.md#term-float64), [INT32](../../../../glossary.md#data-type--datatype), or [INT64](../../../../glossary.md#data-type--datatype) |
| [`MEAN`](../../../../glossary.md#term-mean) | BFLOAT16, FLOAT32, or FLOAT64 |
| `ALL`, `ANY` | [BOOL](../../../../glossary.md#data-type--datatype) |

All five new kinds are floating-only and preserve exact input type. No integral/BOOL acceptance,
cast, promotion, or [widening](../../../../glossary.md#term-widening) occurs.

Every result has the derived [Shape](../../../../glossary.md#shape), unresolved [layout](../../../../glossary.md#layout), no label/storage, and exact input
`requiresGrad`. Floating eligibility is not a [gradient](../../../../glossary.md#term-gradient) rule; integral/BOOL input is already false.
Each valid call uses one `TensorFactory.createDerived` delegation with exact input `[input]`, one
output [descriptor](../../../../glossary.md#term-descriptor), one fresh [producer](../../../../glossary.md#term-producer)/[Tensor](../../../../glossary.md#tensor), and [provenance](../../../../glossary.md#provenance) output index zero. Inputs are not
mutated.

## Numerical policy

### Common floating policy

Result [DataType](../../../../glossary.md#data-type--datatype) is the [accumulator](../../../../glossary.md#accumulator-type) contract. The portable [target](../../../../glossary.md#term-target) is exact real arithmetic plus the
special rules below, rounded once to the result format using [round-to-nearest, ties-to-even](../../../../glossary.md#floating-point-precision-and-rounding). An
implementation may use equal-or-wider intermediates, reassociate, parallelize, compensate, or use
another [stable algorithm](../../../../glossary.md#conditioning-and-numerical-stability) only when its observable result satisfies future [conformance](../../../../glossary.md#term-conformance) [tolerance](../../../../glossary.md#term-tolerance)
and every exact special rule. It must not use narrower accumulation, hidden promotion in the
[model](../../../../glossary.md#term-model), [saturation](../../../../glossary.md#term-saturation), or promise bitwise equality. [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign and signaling preservation are
unspecified.

[Finite](../../../../glossary.md#term-finite) [overflow](../../../../glossary.md#term-overflow) produces signed [infinity](../../../../glossary.md#term-infinity). Finite [underflow](../../../../glossary.md#term-underflow) rounds in the result format, including
[signed zero](../../../../glossary.md#term-signed-zero) where the mathematical result has a sign. [Axis](../../../../glossary.md#term-axis)-list order cannot change the abstract
target. Existing 0018U1 integral semantics are inherited unchanged: exact-type modular sum/product
with [reassociation](../../../../glossary.md#term-reassociation), signed [extrema](../../../../glossary.md#saturation-pole-and-extremum), bounded empty [identities](../../../../glossary.md#term-identity-meanings), and each selected value included
once.

### Ordinary floating and [BOOL](../../../../glossary.md#data-type--datatype) reductions

- `SUM`: NaN propagates; opposite infinities produce NaN; otherwise infinity is preserved; empty
  result is positive zero. An exact-zero non-empty result is negative zero only when every selected
  value is negative zero; cancellation or any positive zero produces positive zero.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- [`MEAN`](../../../../glossary.md#term-mean): exact sum divided by count; NaN propagates; opposite infinities or empty domain produce
  NaN; a sole infinity sign is preserved; zero sign follows the sum rule because the divisor is
  positive.
- `PROD`: NaN propagates; zero times infinity produces NaN; otherwise zero/infinity sign follows
  multiplication parity; empty result is positive one.
- `MIN`/`MAX`: any NaN produces NaN; infinities order normally; minimum selects negative zero and
  maximum positive zero; empty minimum is positive infinity and empty maximum negative infinity.<span data-glossary-navigation="true"> ([positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `ALL`: empty is true. `ANY`: empty is false.

These policies apply equally to existing full/single-[axis](../../../../glossary.md#term-axis) and new multi-axis meanings. They add no
value execution or [metadata](../../../../glossary.md#term-metadata) change to completed public methods. Equal [finite](../../../../glossary.md#term-finite) [extrema](../../../../glossary.md#saturation-pole-and-extremum) produce the
same value, so no tie-selection or traversal-order policy is observable for value-returning
`MIN`/`MAX`; the existing explicit logical-index policy for `ARG_MIN`/`ARG_MAX` is unchanged.

### [Log-sum-exp](../../../../glossary.md#log-sum-exp)

For selected values `x_i`, `LOG_SUM_EXP` means `log(sum_i(exp(x_i)))` as a stable mathematical
[target](../../../../glossary.md#term-target) without selecting an algorithm. Empty domain is [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero). [NaN](../../../../glossary.md#term-nan) produces NaN.
[Positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) produces positive infinity unless NaN exists. An all-negative-[infinity](../../../../glossary.md#term-infinity) domain
produces negative infinity; other negative [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) contribute zero. A [singleton](../../../../glossary.md#term-singleton) [finite](../../../../glossary.md#term-finite) domain,
including an empty-[axis](../../../../glossary.md#term-axis) point domain, returns that value and preserves [signed zero](../../../../glossary.md#term-signed-zero).

### [Variance](../../../../glossary.md#term-variance) and [standard deviation](../../../../glossary.md#term-standard-deviation)

For domain count `N`, correction `c`, and exact [mean](../../../../glossary.md#term-mean) `mu`:

```text
variance = sum_i((x_i - mu)^2) / (N - c)
standardDeviation = sqrt(variance)
```

The denominator must be positive: `N > c`. Reject a statically known selected-domain count at
most correction. Zero extents yield `N=0`. Accept dynamic/expression selected extents; later
[compiler](../../../../glossary.md#term-compiler)/[binding](../../../../glossary.md#term-binding) [validation](../../../../glossary.md#term-validation) must prove `N > c` before execution. No [NaN](../../../../glossary.md#term-nan) fallback, denominator
clamp, or sentinel represents an invalid domain.

NaN or any [infinity](../../../../glossary.md#term-infinity) produces NaN because deviation from the exact mean is undefined. A valid
[constant](../../../../glossary.md#term-constant) [finite](../../../../glossary.md#term-finite) domain, including signed zeros, produces [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). Standard deviation is the
non-negative principal root and never [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero).

Compute static `N` from selected static [Dimensions](../../../../glossary.md#dimension) only with an early zero check and checked
multiplication. If a non-zero product overflows `long`, it exceeds every non-negative `long`
correction and is valid. Empty [axes](../../../../glossary.md#term-axis) have `N=1`.

### [L1 and L2 norms](../../../../glossary.md#norms-elementary-functions-and-probability-distributions)

`L1_NORM = sum_i(abs(x_i))`; `L2_NORM = sqrt(sum_i(x_i*x_i))` under the common abstract policy.
Empty domains produce [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero); point domains produce [absolute value](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [NaN](../../../../glossary.md#term-nan) produces NaN; any
[infinity](../../../../glossary.md#term-infinity) produces [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) unless NaN exists; [finite](../../../../glossary.md#term-finite) results are non-negative and zero is
positive zero.

## [Validation](../../../../glossary.md#term-validation), failure, and construction order

Add one final package-private `TensorMultiAxisReductionExpressions` with no fields, nested types,
state, [registry](../../../../glossary.md#term-registry), or test hook; one private constructor; and package-private entries for ordinary,
uncorrected advanced, and corrected statistical construction. Private methods may own kind/type
validation, [normalization](../../../../glossary.md#term-normalization), [Shape](../../../../glossary.md#shape) derivation, selected-domain count, and common construction.

The package-private entry surface is exactly:

```java
static Tensor applyOrdinary(
        Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions)
static Tensor applyAdvanced(
        Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions)
static Tensor applyStatistical(
        Tensor input, AggregateReductionKind kind, int[] axes,
        boolean keepDimensions, long correction)
```

`applyOrdinary` accepts exactly `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, or `ANY` and otherwise
uses `kind must be SUM, MEAN, PROD, MIN, MAX, ALL, or ANY, but was <kind>`.
`applyAdvanced` accepts exactly `LOG_SUM_EXP`, `L1_NORM`, or `L2_NORM` and otherwise uses
`kind must be LOG_SUM_EXP, L1_NORM, or L2_NORM, but was <kind>`. `applyStatistical` accepts exactly
[`VARIANCE`](../../../../glossary.md#term-variance) or `STANDARD_DEVIATION` and otherwise uses
`kind must be VARIANCE or STANDARD_DEVIATION, but was <kind>`.

Every entry follows this observable order:

1. null-check `input` (`input`);
2. null-check `kind` (`kind`);
3. null-check caller [`axes`](../../../../glossary.md#term-axis) (`axes`);
4. validate the entry-specific permitted kind;
5. validate input [DataType](../../../../glossary.md#data-type--datatype), reusing exact current ordinary messages and using
   `input must have a floating data type, but was <dataType>` for new kinds;
6. for statistics, reject negative correction with the exact correction message;
7. clone caller axes;
8. normalize each raw axis once in order and reject a normalized duplicate immediately;
9. for variance/[standard deviation](../../../../glossary.md#term-standard-deviation), reject a statically invalid denominator with exact message
   `reduction domain count <count> must be greater than correction <correction>`;
10. derive Shape, create exact attributes, descriptor, Operation, producer, and Tensor in order
    through one <span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>`createDerived` call.

Invalid-kind messages list the permitted [constants](../../../../glossary.md#term-constant) for that entry. `Shape.normalizeAxis` supplies
invalid-axis exception type/message. Every failure through static denominator validation consumes
no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) and occurs before attributes/[descriptor](../../../../glossary.md#term-descriptor)/[producer](../../../../glossary.md#term-producer) allocation. Identifier
exhaustion occurs only at final factory delegation after local [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata) and propagates
the existing `IllegalStateException`.

Every public method delegates exactly once with exact kind, array, retention, and correction; it
does not call another public overload.

## Out of [scope](../../../../glossary.md#term-scope)

- arbitrary/zero/infinity p-norm, complex magnitude, weighted statistics, median, quantiles,
  covariance, covariance matrices, or multi-output statistics<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity); [norm glossary](../../../../glossary.md#term-norm); [median glossary](../../../../glossary.md#term-median); [matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>
- cumulative product or another scan; cumulative variance; multi-axis arg extrema or masked
  reductions; changes to cumulative sum, softmax, masking, or arg-extrema<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [variance glossary](../../../../glossary.md#term-variance); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- [normalization](../../../../glossary.md#term-normalization) layers, attention, [loss](../../../../glossary.md#loss), [pooling](../../../../glossary.md#term-pooling), [convolution](../../../../glossary.md#term-channels-first-convolution-layers), sorting, or top-K
- accumulator/output override, integral mean/statistics/norms, BOOL numeric interpretation,
  implicit cast, unsigned/complex/quantized types, or new DataType<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [mean glossary](../../../../glossary.md#term-mean); [BOOL glossary](../../../../glossary.md#data-type--datatype); [DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- algorithm/pass-count/tree/traversal/[vector](../../../../glossary.md#term-vector)/[kernel](../../../../glossary.md#kernel)/[tolerance](../../../../glossary.md#term-tolerance) selection or [eager evaluation](../../../../glossary.md#term-eager-evaluation)
- gradients/backward kinds/autograd, graph capture, compiler validation/canonicalization, planning,
  lowering, prepare, runtime, execution, ONNX, tracing, storage, or materialization<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [materialization glossary](../../../../glossary.md#materialization))</span>
- simplification/[decomposition](../../../../glossary.md#term-decomposition)/[CSE](../../../../glossary.md#term-common-subexpression-elimination-cse)/[producer](../../../../glossary.md#term-producer) sharing for empty-[axis](../../../../glossary.md#term-axis) or other calls
- changes to Shape, Dimension, DataType, TensorFactory, producer/provenance, operation foundations,
  existing helpers, Gradle, dependencies, architecture, another module, or detailed task 0019<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [provenance glossary](../../../../glossary.md#provenance); [operation glossary](../../../../glossary.md#operation); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Task 0016C](0016c-min-and-max-tensor-reduction-expressions.md)
- [Task 0016D](0016d-boolean-all-and-any-tensor-expressions.md)
- [Task 0016E](0016e-arg-max-tensor-expressions.md)
- [Task 0016F](0016f-masked-reduction-semantics-and-axis-mapping.md)
- [Task 0016F1](0016f1-masked-sum-and-mean-tensor-expressions.md)
- [Task 0016G](0016g-cumulative-sum-semantic-kind-and-attributes.md)
- [Task 0016H](0016h-cumulative-sum-tensor-expressions.md)
- [Task 0016I](0016i-softmax-semantic-kinds-and-attributes.md)
- [Task 0016J](0016j-softmax-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Task 0018T1](0018t1-unary-numeric-gaps-and-floating-diagnostics.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0018U1](0018u1-integral-reductions-and-arg-min-normalization.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work remains inside model-owned Tensor, Shape-derived metadata, operation semantics, and
  immutable provenance; Tensor remains public API state, not IR or executable state.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Kinds contain mathematics/[signatures](../../../../glossary.md#term-signature) only, never [backend](../../../../glossary.md#backend), algorithm, [gradient](../../../../glossary.md#term-gradient), cost, or route.
- Attributes are immutable normalized values with no Tensor, Shape, rank, descriptor, storage,
  graph, or execution state.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [descriptor glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph))</span>
- Package direction remains `model.tensor -> model.operation.reduction`, datatype, and shape; the
  reduction package must not import Tensor.
- Compiler later owns graph-wide revalidation, dynamic-domain proof, canonicalization, and
  gradients; backend prepare owns lowering and algorithm choice.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering))</span>
- No dependency, module boundary, lifecycle, architecture contract, or focused architecture change.
  Stop and report if implementation requires one.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>

## Package impact

Existing packages only:

- `io.github.pho001.synaptik.model.operation.reduction` owns aggregate meanings and immutable
  axes/correction.
- `io.github.pho001.synaptik.model.tensor` owns public methods, local validation/Shape/result
  construction, provenance, and factory delegation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.shape` supplies [immutable](../../../../glossary.md#term-immutable) [Dimensions](../../../../glossary.md#dimension)/[Shapes](../../../../glossary.md#shape) and [normalization](../../../../glossary.md#term-normalization).
- `io.github.pho001.synaptik.model.datatype` supplies category eligibility.

No package is added or moved. Type placement:

- `AggregateReductionKind` — extended vocabulary/[signatures](../../../../glossary.md#term-signature).
- `MultiAxisReductionAttrs` — shared ordinary/[log-sum-exp](../../../../glossary.md#log-sum-exp)/[norm](../../../../glossary.md#term-norm) [axes](../../../../glossary.md#term-axis) and retention.
- `StatisticalReductionAttrs` — [variance](../../../../glossary.md#term-variance)/standard-deviation axes, retention, correction.
- [`Tensor`](../../../../glossary.md#tensor) — exact 26-method public surface.
- `TensorMultiAxisReductionExpressions` — package-private construction boundary.
- `ReductionSemanticsTest` — semantic/signature/record coverage.
- `TensorMultiAxisReductionTest` — focused public/helper behavior.
- `TensorTest` — exact public surface/modifiers only.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/MultiAxisReductionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/StatisticalReductionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMultiAxisReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/ReductionSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMultiAxisReductionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inconsistent: Training [API](../../../../glossary.md#term-application-programming-interface-api); [DataType](../../../../glossary.md#data-type--datatype)/[Dimension](../../../../glossary.md#dimension)/[Shape](../../../../glossary.md#shape), [Operation](../../../../glossary.md#operation),
[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance)/factory, full/single/masked/arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum)/[scan](../../../../glossary.md#term-cumulative-scan)/[softmax](../../../../glossary.md#softmax--log-softmax) Javadocs/tests;
architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle) and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most five production, five test, and seven documentation/[planning](../../../../glossary.md#term-planning) files: exactly 17 paths.
[`Tensor`](../../../../glossary.md#tensor)/`TensorTest` may change only for the exact 26 methods, imports/Javadocs, method count/set,
[descriptors](../../../../glossary.md#term-descriptor), and modifiers. `OperationSignatureTest` changes only for the exact new 0018V
[signatures](../../../../glossary.md#term-signature). `TensorBinaryArithmeticTest` changes only for the public Tensor count from 130 to 156.
Every other existing focused helper/test remains unchanged.

This [atomic](../../../../glossary.md#resource-transaction-and-rollback) scope is justified because enum signatures, two attributes, public construction,
focused tests, and documentation must agree in one compilable state. The two-test expansion was
explicitly authorized after the first final [model](../../../../glossary.md#term-model) [run](../../../../glossary.md#term-run) exposed stale global signature and public
method-count assertions. Stop for another Java/test type, documentation file, 18th path,
cross-[module](../../../../glossary.md#term-module) work, or architecture change.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document all five new kinds, formulas, pairings, empty/special policies, types, and boundaries.
- Update ordinary kinds for multi-axis pairing and completed floating/BOOL numerical policy while
  preserving 0018U1 integral semantics.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Document both records' normalization, order, empty-list meaning, snapshot ownership, validation,
  equality, and diagnostic text.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [snapshot glossary](../../../../glossary.md#term-snapshot); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Document every new Tensor method's axes, correction/default, Shape/type/metadata/provenance,
  freshness, numerical target, failures/order/ID effects, and deferred behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [target glossary](../../../../glossary.md#term-target))</span>
- Document helper/member [validation](../../../../glossary.md#term-validation), selected count, construction order, allocation, and effects.
- Review related Javadocs and record reasoned no-change conclusions.

## Acceptance criteria

- Exact five kinds append in order; existing [constants](../../../../glossary.md#term-constant)/[signatures](../../../../glossary.md#term-signature) remain stable.
- Exact signature pairings/cardinalities work and wrong attribute classes fail through [Operation](../../../../glossary.md#operation).
- Both records expose exact components/[API](../../../../glossary.md#term-application-programming-interface-api), [immutable](../../../../glossary.md#term-immutable) snapshots, messages/order, and no extra API.
- Exactly 26 [Tensor](../../../../glossary.md#tensor) methods are added; public count is 156; no unrelated [descriptor](../../../../glossary.md#term-descriptor)/name changes.
- Existing full/single/masked/arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum)/[scan](../../../../glossary.md#term-cumulative-scan)/[softmax](../../../../glossary.md#softmax--log-softmax) behavior remains unchanged.
- [Axes](../../../../glossary.md#term-axis) normalize once in order; duplicates fail; normalized order is retained and mutation-safe.
- Empty/some/all axes and scalar/static-zero/dynamic/expression Shapes derive exact results and
  preserve required Dimension identity.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Ordinary eligibility and 0018U1 integral behavior remain; new families are floating-only.
- Correction defaults/explicit values, static checks, dynamic acceptance, failures/order match.
- Results have exact type/eligibility, unresolved layout, no label/storage, fresh exact producer
  and one-input provenance index zero; inputs remain unchanged.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Numerical policies are consistent across Javadocs, APIs, glossary, and capabilities without
  selecting an algorithm or executing values.
- No decomposition, eager data/storage, gradients, compiler/backend/runtime, registry, dependency,
  Gradle, architecture, or other-module change.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [eager glossary](../../../../glossary.md#term-eager-evaluation); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- Focused tests, one final [model](../../../../glossary.md#term-model) suite, [checkpoint](../../../../glossary.md#term-checkpoint) root tests, [Javadoc](../../../../glossary.md#term-javadoc)/docs [validation](../../../../glossary.md#term-validation) pass.
- Separate clean-context docs pass finalizes authorized documentation and no-change conclusions.
- 0018V becomes Complete only after both passes; 0018U1 stays Complete; 0019+ stay Draft without
  detailed specs.

## Tests / [validation](../../../../glossary.md#term-validation)

Required focused command:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMultiAxisReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorBooleanReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorArgMaxExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorMaskedReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressionTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

Tests cover exact [API](../../../../glossary.md#term-application-programming-interface-api)/helper/enum/record shapes, [signatures](../../../../glossary.md#term-signature), all kinds/types/retention/defaults,
validation/messages/ID effects, caller mutation, [axis](../../../../glossary.md#term-axis) cases, [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) [identity](../../../../glossary.md#term-identity-meanings), result
[metadata](../../../../glossary.md#term-metadata)/[provenance](../../../../glossary.md#provenance)/freshness, contract examples/special policies without storage [evaluation](../../../../glossary.md#term-evaluation), and
unchanged prior behavior.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Capability-reset [checkpoint](../../../../glossary.md#term-checkpoint) after the combined diff:

```bash
./gradlew test
```

Also validate local links/anchors, balanced fences, terminology, generated [Javadoc](../../../../glossary.md#term-javadoc), final newlines,
trailing whitespace, exact 17 paths, package placement, method count, synchronized status,
completed history, 0018U1 Complete, final 0018V Complete, and no task-0019 spec. Use manual
reflection/[bytecode](../../../../glossary.md#term-bytecode) checks only for a concrete risk not covered by stable automated tests.

## Dependencies

- 0001: DataTypes; 0002/0018M: [Dimensions](../../../../glossary.md#dimension), [Shapes](../../../../glossary.md#shape), [normalization](../../../../glossary.md#term-normalization), symbolic extents.
- 0006–0007, 0011–0013, 0018L: [Operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), IDs, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance)/factory.
- 0016A–0016J including 0016F1: reduction, masked, [scan](../../../../glossary.md#term-cumulative-scan), [softmax](../../../../glossary.md#softmax--log-softmax) conventions.
- 0018K: family-owned exact [signatures](../../../../glossary.md#term-signature).
- 0018T1: floating special-value vocabulary.
- 0018U/0018U1: integral arithmetic/reductions and shared arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) policies to preserve.

## Follow-up tasks

- 0019 remains Draft for [linear](../../../../glossary.md#term-linear-meanings) algebra/attention after this [checkpoint](../../../../glossary.md#term-checkpoint).
- 0020–0022 remain Draft consumers of this reduction foundation.
- Compiler later owns capture/revalidation/canonicalization/autograd; backend/conformance later
  owns algorithms, tolerances, lowering, storage, and execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [lowering glossary](../../../../glossary.md#lowering))</span>
- Arbitrary p-[norm](../../../../glossary.md#term-norm), cumulative product, multi-output statistics, and [normalization](../../../../glossary.md#term-normalization) remain deferred.

Do not create another detailed task specification during 0018V.

## Architecture impact

Expected impact: None. If implementation requires `ARCHITECTURE.md`, focused architecture,
[dependency](../../../../glossary.md#term-dependency), boundary, or other-[module](../../../../glossary.md#term-module) changes, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, current architecture,
documentation/planning rules, roadmap, model capabilities/master plan, completed tasks 0016A–J
including 0016F1, tasks 0018K/0018M/0018T1/0018U/0018U1, and task 0018V.

Implement docs/planning/modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md exactly
inside its 17 authorized paths. Preserve current contracts. Stop on architecture uncertainty,
scope overflow, or need for another type/module.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and exact Java evidence to a separate clean-context documentation agent in the same change.
That agent finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and docs checks,
reusing successful Java evidence. Run the root checkpoint after the combined diff. Synchronize
status only after all criteria pass; keep 0019+ Draft without specs.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the separate clean-context documentation agent this task, complete diff, exact focused/final
[model](../../../../glossary.md#term-model) evidence/counts and whether Java changed afterward, all affected contracts/policies, the
seven authorized docs, and required [Javadoc](../../../../glossary.md#term-javadoc)/Markdown/[scope](../../../../glossary.md#term-scope)/status/[checkpoint](../../../../glossary.md#term-checkpoint) [validation](../../../../glossary.md#term-validation).

It independently reads AGENTS, architecture, documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-Javadoc/
[Planning](../../../../glossary.md#term-planning)/Example profiles, this task, actual source/tests/generated Javadoc, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/
Training APIs, glossary, capabilities/master/roadmap, and directly related reduction/[softmax](../../../../glossary.md#softmax--log-softmax)
contracts. It finalizes affected Javadocs/docs and glossary terminology for [multi-axis reduction](../../../../glossary.md#multi-axis-reduction),
correction, [log-sum-exp](../../../../glossary.md#log-sum-exp), and norms.

It does not repeat successful Java tests unless executable Java changes or evidence is stale for a
recorded concrete reason. It records reasoned no-change conclusions for Training API,
[DataType](../../../../glossary.md#data-type--datatype)/[Dimension](../../../../glossary.md#dimension)/[Shape](../../../../glossary.md#shape), [operation](../../../../glossary.md#operation)/[provenance](../../../../glossary.md#provenance)/factory foundations, completed reduction variants,
architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), and other [modules](../../../../glossary.md#term-module).

## Local decisions

- Caller axes are copied, normalized once against the input rank in encounter order, and retained
  as an immutable list. Duplicate detection follows normalization, so equivalent positive and
  negative spellings of one axis conflict.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [immutable glossary](../../../../glossary.md#term-immutable); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- An empty axes list is a point-domain reduction: result Shape equals input Shape and no ordinary
  reduction identity is substituted. Reducing every axis produces a scalar unless retained
  dimensions were requested.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [dimensions glossary](../../../../glossary.md#dimension))</span>
- Ordinary multi-[axis](../../../../glossary.md#term-axis) [signatures](../../../../glossary.md#term-signature) use `MultiAxisReductionAttrs`; variance and standard deviation
  use <span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>`StatisticalReductionAttrs` so non-negative integral correction remains explicit semantic
  data. Log-sum-exp and norms use the ordinary multi-axis attribute shape.<span data-glossary-navigation="true"> ([Log-sum-exp glossary](../../../../glossary.md#log-sum-exp))</span>
- Static reduced element count must exceed correction before a statistical expression is
  constructed. A dynamic count is accepted for later compiler/runtime validation because the
  model cannot prove the boundary.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model))</span>
- All five new families accept floating input only. Their result preserves the exact input type,
  while ordinary reductions retain the eligibility and exact result-type contracts completed by
  0018U1.
- Multi-axis construction is owned by the package-private, field-free
  `TensorMultiAxisReductionExpressions`; the public [`Tensor`](../../../../glossary.md#tensor) surface remains the only caller
  entry point.
- User authorization expanded the original stale-test repair to include both
  `OperationSignatureTest` and `TensorBinaryArithmeticTest`, preserving the exact 17-path [scope](../../../../glossary.md#term-scope).

## Known limitations

- This task constructs storage-free model expressions only. It does not evaluate values, select a
  numerical algorithm, define tolerances, lower operations, claim backend support, or attach
  gradients.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Dynamic statistical domains can be represented even when a later [binding](../../../../glossary.md#term-binding) makes
  `reducedElementCount <= correction`; the [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime) layer must reject that bound case.
- Arbitrary p-norm, cumulative product, multi-output statistics, and normalization remain
  deferred. Task 0019 and all later tasks remain Draft and have no new detailed specification.<span data-glossary-navigation="true"> ([norm glossary](../../../../glossary.md#term-norm); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- No public list-taking overload is added; callers use the exact array and varargs surface defined
  here, while semantic attributes retain immutable lists internally.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context passed the required eight-suite focused command: 78 tests across
  eight suites, with zero failures. Its first final model run exposed only two stale assertions in
  the two user-authorized existing tests; after those assertions were aligned, the single final
  model result passed 744 tests across 91 suites with zero failures, errors, or skips.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- The separate clean-context documentation pass `/root/task_0018u1_docs_recovery` reused that
  stable Java evidence and made no executable Java changes. `./gradlew :modules:model:javadoc`
  succeeded, and generated pages for [`Tensor`](../../../../glossary.md#tensor), `AggregateReductionKind`, and both new public
  attribute records were inspected. The package-private helper correctly has no public page.
- A Java 26 [compile](../../../../glossary.md#compile)-and-[run](../../../../glossary.md#term-run) [metadata](../../../../glossary.md#term-metadata) example produced `Shape[3]`, `Shape[1, 3, 1]`, normalized
  axes <span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>`[2, 0]`, correction `1`, output index `0`, and confirmed caller-array mutation isolation
  and absent storage.
- Reflection verified 156 public `Tensor` methods, the exact 26 new descriptors, five appended
  kinds, both record shapes, and the final/package-private/field-free helper with three package
  entries. <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>`javap -p -s` confirmed both record descriptors and the helper boundary; source/import
  review found no compiler, backend, runtime, reflection, registry, service-loader, or storage
  coupling in the new semantic types.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [registry glossary](../../../../glossary.md#term-registry))</span>
- Markdown validation passed 516 local links including 142 anchors with zero errors. Balanced
  fences, final newlines, trailing whitespace, terminology, generated-Javadoc content, exact
  17-path scope, synchronized Complete status, 0018U1 Complete history, and absence of a task-0019
  specification were checked.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- The required capability-reset [checkpoint](../../../../glossary.md#term-checkpoint) `./gradlew test` succeeded with 36 actionable tasks,
  two executed and 34 up-to-date. `git diff --check` passed after the combined implementation and
  documentation diff.

## Implementation notes

- Appended `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` without
  reordering prior aggregate kinds, and added each exact family-owned signature.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Added [immutable](../../../../glossary.md#term-immutable) `MultiAxisReductionAttrs` and `StatisticalReductionAttrs`, plus focused shared
  construction for normalization, Shape derivation, static correction checks, and fresh
  one-output provenance.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Added the exact 26 public [`Tensor`](../../../../glossary.md#tensor) entries: fourteen ordinary multi-axis overloads and twelve
  log-sum-exp/statistical/norm overloads. Focused tests cover signatures, eligibility, Shape and
  Dimension identity, validation order/messages, mutation safety, metadata, provenance, and
  freshness.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [norm glossary](../../../../glossary.md#term-norm); [Dimension glossary](../../../../glossary.md#dimension); [identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- The documentation pass finalized all five affected production Javadocs, Tensor and Compile API
  references, glossary terminology, capability baseline, task evidence, master plan, and roadmap.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Training API, DataType/Dimension/Shape foundations, operation/provenance/factory foundations,
  completed single-axis/masked/scan/softmax variants, architecture and ADRs, architecture tests,
  backend conformance, integration tests, Gradle configuration, and other modules required no
  changes because this task adds model-only expression metadata within existing boundaries.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [operation glossary](../../../../glossary.md#operation); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

Completed ordered multi-[axis](../../../../glossary.md#term-axis) forms for all seven ordinary reduction families and first-class
floating [log-sum-exp](../../../../glossary.md#log-sum-exp), corrected [variance](../../../../glossary.md#term-variance)/[standard deviation](../../../../glossary.md#term-standard-deviation), and [L1](../../../../glossary.md#term-l1)/[L2 norm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) [model](../../../../glossary.md#term-model) expressions.
The exact 17-path change contains five production files, five test files, and seven
documentation/[planning](../../../../glossary.md#term-planning) files. Focused, final model, root [checkpoint](../../../../glossary.md#term-checkpoint), [Javadoc](../../../../glossary.md#term-javadoc), Java 26 example,
reflection/[bytecode](../../../../glossary.md#term-bytecode)/source, generated-page, Markdown, [scope](../../../../glossary.md#term-scope)/status, formatting, whitespace, and
diff [validation](../../../../glossary.md#term-validation) passed. No unresolved issue or required follow-up remains inside task 0018V;
deferred [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), [gradient](../../../../glossary.md#term-gradient), and later capability work stays in its owning Draft
tasks.

Status: Complete
