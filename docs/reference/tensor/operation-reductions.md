# Reduction, cumulative-[scan](../../glossary.md#term-cumulative-scan), [softmax](../../glossary.md#softmax--log-softmax) and [loss](../../glossary.md#loss) kinds

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** A **reduction domain** is the group of positions combined into one result.
SUM combines the whole group; a **[cumulative scan](../../glossary.md#cumulative-scan)** instead returns a partial result at every
position. Softmax also preserves every position, but each probability depends on the complete
[normalization](../../glossary.md#term-normalization) [slice](../../glossary.md#slice). A **loss** compares [predictions](../../glossary.md#term-prediction) with [targets](../../glossary.md#term-target) and may reduce those comparisons
to one number. The kind names which meaning is requested; attributes record [axes](../../glossary.md#term-axis), correction,
tie rules, or [loss reduction](../../glossary.md#loss-reduction). A **[normalized axis](../../glossary.md#normalized-axis)** is an already checked non-negative position,
not a caller's negative shorthand. See [reductions](reductions.md), [scans](fixed-scans.md),
[normalization](normalization.md), and [losses](statistics-losses.md) for the input-aware APIs.

<a id="aggregate-reduction-semantic-kinds-and-attributes"></a>
### [Aggregate reduction](../../glossary.md#aggregate-reduction) semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly these [constants](../../glossary.md#term-constant) in declaration order:

| Kind | Requested aggregate meaning | Supported semantic forms |
|---|---|---|
| `SUM` | Add values in the selected reduction domain. | Ordinary full/single-/multi-[axis](../../glossary.md#term-axis), masked axis-removing, or [binding](../../glossary.md#term-binding)-aware exact-[target](../../glossary.md#term-target)-[Shape](../../glossary.md#shape) forms. |
| [`MEAN`](../../glossary.md#term-mean) | Compute the arithmetic mean in the selected reduction domain. | Ordinary full/single-/multi-axis or masked axis-removing forms. |
| `PROD` | Multiply values in the selected reduction domain. | Full or ordered normalized axes. |
| `MIN` | Select the minimum value in the selected reduction domain. | Full or ordered normalized axes. |
| `MAX` | Select the maximum value in the selected reduction domain. | Full or ordered normalized axes. |
| `ALL` | Compute [boolean](../../glossary.md#data-type--datatype) conjunction in the selected reduction domain. | Full or ordered normalized axes. |
| `ANY` | Compute boolean disjunction in the selected reduction domain. | Full or ordered normalized axes. |
| `ARG_MAX` | Select a [logical index](../../glossary.md#axis-rank-scalar-vector-and-matrix) of a maximum value along one axis. | One [normalized axis](../../glossary.md#normalized-axis) with an explicit tie policy. |
| `ARG_MIN` | Select a logical index of a minimum value along one axis. | One normalized axis with an explicit tie policy. |
| `LOG_SUM_EXP` | Compute [log](../../glossary.md#term-log) of summed exponentials. | Ordered distinct normalized axes. |
| [`VARIANCE`](../../glossary.md#term-variance) | Compute corrected second central moment. | Ordered distinct normalized axes plus correction. |
| `STANDARD_DEVIATION` | Compute non-negative [square root](../../glossary.md#norms-elementary-functions-and-probability-distributions) of corrected variance. | Ordered distinct normalized axes plus correction. |
| `L1_NORM` | Sum absolute values. | Ordered distinct normalized axes. |
| `L2_NORM` | Compute square root of summed squares. | Ordered distinct normalized axes. |

A full ordinary reduction combines every input axis and pairs the kind with
`NoOperationAttrs.INSTANCE`. A single-axis ordinary reduction pairs the same kind with
`AxisReductionAttrs(axis, keepDimensions)`. This distinction represents "all axes" through the
absence of axis [parameters](../../glossary.md#term-parameter) rather than a negative numeric sentinel:

```java
Operation fullSum = new Operation(
        AggregateReductionKind.SUM,
        NoOperationAttrs.INSTANCE);

Operation axisSum = new Operation(
        AggregateReductionKind.SUM,
        new AxisReductionAttrs(1, true));
```

The axis stored in `AxisReductionAttrs` is already normalized: it is a non-negative index, not a
caller-facing negative axis. Construction accepts any non-negative `int` because these attributes
do not retain a `Shape` or know an input [rank](../../glossary.md#term-rank). A negative value fails with
`IllegalArgumentException` and a message containing the rejected value. The current `sum`, `mean`,
`prod`, reduction `min`, reduction `max`, boolean `all`, and boolean `any` [Tensor](../../glossary.md#tensor) expressions
and the current `argMin`/`argMax` expressions normalize a caller axis through the input Shape before
creating attributes.

When `keepDimensions` is false, the selected axis is removed from the eventual result. When it is
true, the axis remains with [extent](../../glossary.md#term-extent) one. The record stores that request but does not construct an
output Shape. Full reductions in this contract have no `keepDimensions` parameter, and
`ARG_MIN`/`ARG_MAX` have no full form.

Arg-min and arg-max pair only with shared
`ArgExtremaAttrs(axis, keepDimensions, tiePolicy)`. `FIRST_INDEX` requests the smallest logical
index among equal [extrema](../../glossary.md#saturation-pole-and-extremum) along the selected axis; `LAST_INDEX` requests the largest. A logical
index is an axis position rather than a [storage offset](../../glossary.md#term-storage-offset). The policy must be supplied explicitly and
is never defaulted by the semantic value:

```java
Operation argMax = new Operation(
        AggregateReductionKind.ARG_MAX,
        new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX));
```

`ArgExtremaAttrs` rejects a negative axis before checking the tie policy and rejects a null policy
with `NullPointerException("tiePolicy")`. Both attribute records use generated record value
semantics, while their text remains diagnostic rather than [serialization](../../glossary.md#term-serialization) or [dispatch](../../glossary.md#term-dispatch) syntax.
Family-owned [signatures](../../glossary.md#term-signature) enforce these exact attribute variants. Ordinary and target-Shape forms
declare one input; masked SUM and MEAN declare the ordered two-input [occurrence](../../glossary.md#term-occurrence) `[input, mask]`;
every variant has one output.

A binding-aware sum pairs `AggregateReductionKind.SUM` with
`SumToShapeAttrs(targetShape)`. The record retains exactly one non-null, exact target `Shape`
reference; it stores no source, resolved axes, binding, constraint, [descriptor](../../glossary.md#term-descriptor), or execution plan.
The exact fixed one-input/one-output signature is appended after SUM's existing ordinary and
masked variants. No other aggregate kind accepts `SumToShapeAttrs`. Public
`Tensor.sumToShape(Shape)` owns input-aware numeric, rank, and statically provable right-aligned
compatibility checks before constructing this [operation](../../glossary.md#operation) [metadata](../../glossary.md#term-metadata). Unresolved pairs retain an
obligation for later binding [validation](../../glossary.md#term-validation) rather than selecting axes during [model](../../glossary.md#term-model) construction.

A masked sum or mean pairs its existing kind with `MaskedReductionAttrs(axis)`. The axis is
already normalized, non-negative, and removed from the eventual result. The attributes store no
[mask](../../glossary.md#term-boolean-mask), Shape, mapping, or broadcast plan. `Integer.MAX_VALUE` remains structurally valid because
the attribute cannot know the eventual input rank. Public `Tensor.sum(axis, mask)` and
`Tensor.mean(axis, mask)` separately normalize the axis and require one ordinary right-aligned
broadcast whose result equals the input Shape before constructing the attributes.

The public [provenance](../../glossary.md#provenance) order is `[input, mask]`. False mask positions exclude their aligned input
values, including [NaN](../../glossary.md#term-nan) and [infinity](../../glossary.md#term-infinity). When no values are selected, a masked sum produces zero. A
masked mean divides by the true selected-count for each output and produces NaN when that count is
zero, without promising a [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) or bit pattern. These are semantic
requirements for later execution; constructing `MaskedReductionAttrs` does not inspect a mask,
count values, divide, derive an output descriptor, or execute computation. The signature [matrix](../../glossary.md#term-matrix-meanings)
accepts `MaskedReductionAttrs` only for `SUM` and `MEAN`.

Multi-axis ordinary, [log-sum-exp](../../glossary.md#log-sum-exp), and [norm](../../glossary.md#term-norm) occurrences use
`MultiAxisReductionAttrs(axes, keepDimensions)`. Variance and [standard deviation](../../glossary.md#term-standard-deviation) use
`StatisticalReductionAttrs(axes, keepDimensions, correction)`. Both [snapshot](../../glossary.md#term-snapshot) ordered distinct
normalized axes; the statistical value additionally rejects negative correction. An empty axis
list means a point domain, not full reduction.

The public Tensor surface now covers every current aggregate semantic kind. `Tensor.sum`, `prod`,
reduction `min`, and reduction `max` provide floating/integral full and single-axis expression
construction; `mean` remains floating-only. `Tensor.sumToShape` adds the exact-target-Shape SUM
form for numeric input without adding a semantic kind. Public `Tensor.all` and `Tensor.any`
provide the corresponding exact-[BOOL](../../glossary.md#data-type--datatype) forms with fixed false [gradient](../../glossary.md#term-gradient) eligibility. Public `Tensor.argMin` and
`Tensor.argMax` provide axis-only floating/integral construction with fixed [INT64](../../glossary.md#data-type--datatype), false-gradient
results and explicit tie semantics. Twenty-six multi-axis/statistical methods add ordered axes,
optional retention, and population-default/explicit correction. Every current family derives
Shapes locally and records exact one-input provenance. Integral ordinary reductions use exact-type
modular sum/product, signed
minimum/maximum, and bounded empty [identities](../../glossary.md#term-identity-meanings). Arg extrema use shared NaN, [signed-zero](../../glossary.md#nan-infinity-and-signed-zero), infinity,
tie, and invalid-empty-selected-axis policy. Aggregate extrema remain typed separately from equally
named [binary](../../glossary.md#term-binary-descriptor-and-metadata) [elementwise](../../glossary.md#term-elementwise) kinds, while aggregate ALL/ANY remain typed separately from elementwise
AND/OR. These contracts define requested meaning only and do not evaluate values, implement a
gradient or numerical algorithm, [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), lower operations, report [backend](../../glossary.md#backend) availability,
or execute.

<a id="cumulative-scan-semantic-kinds-and-attributes"></a>
### Cumulative-[scan](../../glossary.md#term-cumulative-scan) semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind` implements [`OperationKind`](../../glossary.md#operationkind)
with exactly `CUM_SUM` and `CUM_PROD`, in that order. They identify cumulative addition and
multiplication along one logical input [axis](../../glossary.md#term-axis). Unlike an [aggregate reduction](../../glossary.md#aggregate-reduction), either [cumulative scan](../../glossary.md#cumulative-scan)
preserves one output position for every input position and therefore preserves logical [Shape](../../glossary.md#shape). The
[normalized axis](../../glossary.md#normalized-axis) and scan mode are carried by
`CumulativeScanAttrs(axis, exclusive, reverse)` rather than stored in either kind.

The semantic pairings are explicit:

```java
CumulativeScanAttrs attrs = new CumulativeScanAttrs(0, true, false);
Operation cumulativeSum = new Operation(CumulativeScanKind.CUM_SUM, attrs);
Operation cumulativeProduct = new Operation(CumulativeScanKind.CUM_PROD, attrs);
```

The two results are [immutable](../../glossary.md#term-immutable) semantic [descriptors](../../glossary.md#term-descriptor) for exclusive forward cumulative scans on
normalized axis zero. Each kind's [signature](../../glossary.md#term-signature) enforces the exact shared attributes class and declares
one input and one output, but it does not validate axis bounds for a particular Shape or derive a
result descriptor. `CumulativeScanAttrs` accepts every non-negative `int`, including
`Integer.MAX_VALUE`, and rejects a negative axis with `IllegalArgumentException` and message
`axis must be non-negative: <axis>`. The current Shape-aware `Tensor.cumSum` and `Tensor.cumProd`
methods normalize and validate a caller-facing axis before constructing these attributes.

For logical sum input `[1, 2, 3]`, the complete mode table is:

| `exclusive` | `reverse` | Semantic output | Interpretation |
|---|---|---|---|
| `false` | `false` | `[1, 3, 6]` | Forward traversal includes the current value. |
| `true` | `false` | `[0, 1, 3]` | Forward traversal excludes the current value, so the lowest-index output is additive zero. |
| `false` | `true` | `[6, 5, 3]` | Reverse traversal includes the current value and accumulates from higher indices. |
| `true` | `true` | `[5, 3, 0]` | Reverse traversal excludes the current value, so the highest-index output is additive zero. |

For logical product input `[2, 3, 4]`, the same flags mean:

| `exclusive` | `reverse` | Semantic output | Interpretation |
|---|---|---|---|
| `false` | `false` | `[2, 6, 24]` | Forward traversal includes the current value. |
| `true` | `false` | `[1, 2, 6]` | Forward traversal starts with multiplicative positive one. |
| `false` | `true` | `[24, 12, 4]` | Reverse traversal includes the current value and multiplies from higher indices. |
| `true` | `true` | `[12, 4, 1]` | Reverse traversal ends in input order with multiplicative positive one. |

The tables describe requested mathematics; constructing either semantic pair or a public [Tensor](../../glossary.md#tensor)
expression performs no addition or multiplication. Reverse changes traversal direction, not output
order. A zero-length axis has no output position and therefore emits no [identity](../../glossary.md#term-identity-meanings). A non-empty
exclusive scan emits additive zero or multiplicative positive one at its first traversed position.

Integral `CUM_PROD` uses exact-width [two's-complement](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) modular multiplication. Floating `CUM_PROD`
propagates [NaN](../../glossary.md#term-nan), makes zero times [infinity](../../glossary.md#term-infinity) NaN, and follows multiplication parity for zero and
infinity signs. These selected meanings promise no accumulation algorithm, intermediate [rounding](../../glossary.md#term-rounding),
[NaN payload](../../glossary.md#nan-infinity-and-signed-zero), bitwise cross-[backend](../../glossary.md#backend) result, or [backend route](../../glossary.md#backend-route).

The public methods separately own numeric input [validation](../../glossary.md#term-validation), caller-axis [normalization](../../glossary.md#term-normalization), exact
Shape/type/eligibility retention in an unresolved descriptor, fresh identity, and exact one-input
[provenance](../../glossary.md#provenance). The semantic family and public construction add no value execution, [gradient](../../glossary.md#term-gradient) rule,
[compiler](../../glossary.md#term-compiler) adoption, storage allocation, [runtime](../../glossary.md#term-runtime) behavior, or backend support.

<a id="softmax-semantic-kinds-and-attributes"></a>
### [Softmax](../../glossary.md#softmax--log-softmax) semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind` implements [`OperationKind`](../../glossary.md#operationkind)
with exactly `SOFTMAX` and `LOG_SOFTMAX`, in that order. Both meanings have one logical input and
preserve every logical position. `SoftmaxAttrs(axis)` carries the one already normalized,
non-negative [axis](../../glossary.md#term-axis) shared by both kinds.

Fixing every logical [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) except the selected axis identifies one [normalization](../../glossary.md#term-normalization) [slice](../../glossary.md#slice). For
example, in a [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) input with axis `1`, every row is a separate slice: positions within a row
vary along axis `1`, while the row coordinate remains fixed. Each output position depends on the
complete slice, so these meanings differ from both [aggregate reduction](../../glossary.md#aggregate-reduction), which contracts an axis,
and cumulative sum, which depends on an ordered prefix.

The semantic pairings are explicit:

```java
SoftmaxAttrs attrs = new SoftmaxAttrs(1);
Operation probabilities = new Operation(SoftmaxKind.SOFTMAX, attrs);
Operation logProbabilities = new Operation(SoftmaxKind.LOG_SOFTMAX, attrs);
```

For an all-[finite](../../glossary.md#term-finite) slice, let `m` be the exact represented maximum and let `s` sum every
`exp(x_j-m)` contributor once. SOFTMAX at position `i` is `exp(x_i-m)/s`; LOG_SOFTMAX is
`(x_i-m)-log(s)`. They agree with the familiar unshifted formulas in exact mathematics, but
do not evaluate unshifted [EXP](../../glossary.md#term-expx) for finite slices. A rounded probability can be zero even when
its [log](../../glossary.md#term-log)-probability is finite, so LOG_SOFTMAX is not a subsequent `log(SOFTMAX)` [operation](../../glossary.md#operation).

For the slice `[1, 2, 3]`, the ideal meanings are approximately:

| Kind | Semantic output | Relationship |
|---|---|---|
| `SOFTMAX` | `[0.09003057, 0.24472847, 0.66524096]` | The three probabilities sum to approximately one. |
| `LOG_SOFTMAX` | `[-2.40760596, -1.40760596, -0.40760596]` | Exponentiating each value yields the corresponding SOFTMAX probability. |

The first example explains ideal mathematics. A second, [overflow](../../glossary.md#term-overflow)-sensitive all-finite example,
`[1000, 1000]`, gives `[0.5, 0.5]` and approximately
`[-0.69314718, -0.69314718]`. [NaN](../../glossary.md#term-nan)/[infinity](../../glossary.md#term-infinity) and empty slices retain their distinct [Model](../../glossary.md#term-model) rules;
the [foundational numerical contract](../../architecture/contracts/foundational-modules.md#composite-inheritance)
defines the working [precision](../../glossary.md#term-precision), [narrowing](../../glossary.md#term-narrowing), and special cases.
`SoftmaxAttrs` accepts every non-negative `int`, including `Integer.MAX_VALUE`, because it stores
no [Shape](../../glossary.md#shape) and cannot prove that an axis exists for an eventual input. A negative axis fails with
`IllegalArgumentException` and exact message `axis must be non-negative: <axis>`. Generic
`Operation` retains either kind and the exact attributes reference; their shared [signature](../../glossary.md#term-signature)
enforces the attributes class and one-input, one-output [cardinality](../../glossary.md#term-cardinality), but not [rank](../../glossary.md#term-rank) bounds or result
facts.

The semantic values themselves store no [Tensor](../../glossary.md#tensor), Shape, data-type rule, result [descriptor](../../glossary.md#term-descriptor),
[provenance](../../glossary.md#provenance), numerical policy, [gradient](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) [decomposition](../../glossary.md#term-decomposition), storage, [backend](../../glossary.md#backend) support, or
executable behavior. Public floating `Tensor.softmax` and `Tensor.logSoftmax` now add Shape-aware
caller-axis normalization, exact shape/type/eligibility retention in unresolved descriptors, and
fresh one-input provenance without changing that semantic boundary.

<a id="loss-semantic-kind-reduction-and-attributes"></a>
### [Loss](../../glossary.md#loss) semantic kind, reduction, and attributes

The public `LossKind` enum currently contains three meanings in this order:

```text
MEAN_SQUARED_ERROR
DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
```

Each kind has exactly one fixed two-input, one-output [signature](../../glossary.md#term-signature) with its exact attributes type:

```java
OperationSignature.fixed(MeanSquaredErrorAttrs.class, 2, 1)
OperationSignature.fixed(DenseCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1)
OperationSignature.fixed(IndexCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1)
```

The ordered roles are `[prediction, target]` for [MSE](../../glossary.md#mean-squared-error--mse) and `[logits, target]` for both categorical
meanings; the output role is the selected loss. The public
`LossReduction` enum contains exactly `NONE`, `SUM`, and [`MEAN`](../../glossary.md#term-mean). It is typed configuration rather
than [`OperationAttrs`](../../glossary.md#operationattrs): it stores no [axis](../../glossary.md#term-axis), denominator, default, parser, [mask](../../glossary.md#term-boolean-mask), [Tensor](../../glossary.md#tensor), or executable
behavior. `MeanSquaredErrorAttrs(reduction)` retains one exact non-null reduction and contains no
input, [Shape](../../glossary.md#shape), [element](../../glossary.md#term-element) count, [data type](../../glossary.md#data-type--datatype), algorithm, [gradient](../../glossary.md#term-gradient), [graph](../../glossary.md#graph), [compiler](../../glossary.md#term-compiler), [backend](../../glossary.md#backend), or [runtime](../../glossary.md#term-runtime)
state. `DenseCategoricalCrossEntropyWithLogitsAttrs(axis, reduction)` retains the normalized [class
axis](../../glossary.md#class-axis) and reduction. `IndexCategoricalCrossEntropyWithLogitsAttrs(axis, reduction, ignoreIndex)`
also retains a non-null optional exact [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue); the optional is internal [metadata](../../glossary.md#term-metadata),
not a public Tensor-method [parameter](../../glossary.md#term-parameter) or [producer](../../glossary.md#term-producer) input.

The semantic contract is [elementwise](../../glossary.md#term-elementwise) `(prediction - target)^2`, optionally summed over the
complete logical domain or divided by its complete element count. It fixes [scalar](../../glossary.md#term-scalar) and empty-domain
meaning plus the [NaN](../../glossary.md#term-nan), [infinity](../../glossary.md#term-infinity), [signed-zero](../../glossary.md#nan-infinity-and-signed-zero), [overflow](../../glossary.md#term-overflow), [reassociation](../../glossary.md#term-reassociation), and determinism boundaries
documented in [Mean-squared-error loss expressions](statistics-losses.md#mean-squared-error-loss-expressions). The
categorical contracts are documented in
[Categorical-cross-entropy-with-logits expressions](statistics-losses.md#categorical-cross-entropy-with-logits-expressions).
The kinds and attributes alone do not validate operand [descriptors](../../glossary.md#term-descriptor), derive a result, construct a
Tensor, read values, or add compiler, backend, runtime, gradient, or training behavior.


---

[Previous topic](operation-arithmetic.md) · [Next topic](operation-geometry.md)
