# Task 0021A: [RMS Normalization](../../../../glossary.md#rms-normalization) Semantics and [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (current context, not a status update).**

This completed task records the historical RMS-only expression baseline, not today's execution
coverage. The caller invokes `rmsNorm` on the input Tensor; the Model helper validates Shape/type
metadata and records ordered operands `[input]` or `[input, scale]`. Scale is an explicit
elementwise operand, not an owned parameter or broadcast value. Epsilon is a typed positive scalar
attribute inside the square root. A trailing slice is normalized without centering: for Shape `[2,
3]` and normalized Shape `[3]`, each row uses its own average of squares. Construction passes
descriptions and Tensor references, not evaluated values, producing a storage-free expression that
neither executes nor owns a result buffer.

Read the [current RMS
contract](../../../../reference/tensor/normalization.md#rms-normalization-expressions) for today's
types, numerical rules and downstream boundaries. It includes broader floating types, FLOAT32
epsilon for FLOAT16/FP8 results, and primitive-site arithmetic rather than this record's blanket
wider-intermediate permission. Empty and NaN/infinity/signed-zero boundaries remain distinct from
finite rounding. Model semantics defines meaning; compiler owners handle capture, constraints and
derivative expressions, while execution providers realize supported routes without redefining it.
Those responsibilities do not imply storage ownership or turn this historical completion into a
current support claim.

**Mathematical reading illustration.** For two finite real inputs $x_1,x_2$ and positive real
epsilon $\varepsilon$, the unscaled result is $y_i=x_i/\sqrt{(x_1^2+x_2^2)/2+\varepsilon}$ for
$i\in\{1,2\}$. At $x=(1,1)$ and $\varepsilon=3$, both outputs are one half. Unlike layer
normalization, RMS normalization does not subtract the mean: a constant row need not become zero.
This is a real-number illustration, not an alternative rounded algorithm.

## Status

Complete

## Goal

Add one first-class [backend](../../../../glossary.md#backend)-independent `RMS_NORM` meaning and exactly two public receiver
expressions: one without an affine operand and one with explicit [elementwise](../../../../glossary.md#term-elementwise) scale. Each [operation](../../../../glossary.md#operation)
normalizes a non-empty trailing [slice](../../../../glossary.md#slice) through its [uncentered mean square](../../../../glossary.md#uncentered-mean-square), adds an exact typed
positive [epsilon](../../../../glossary.md#epsilon) inside the [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions), preserves the input [Shape](../../../../glossary.md#shape), and records one output without
evaluating values.

This is an intentionally narrow root-[mean](../../../../glossary.md#term-mean)-square [normalization](../../../../glossary.md#term-normalization) baseline. It keeps scale visible as
an optional logical input, adds no [bias](../../../../glossary.md#term-bias), and does not import framework layer state, defaults,
broadcast options, [accumulator](../../../../glossary.md#accumulator-type) options, or training behavior into the [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api).

## [Scope](../../../../glossary.md#term-scope)

- Add `RmsNormKind.RMS_NORM` in the existing [normalization](../../../../glossary.md#term-normalization) [operation](../../../../glossary.md#operation) package.
- Add one [immutable](../../../../glossary.md#term-immutable) public `RmsNormAttrs(normalizedShape, epsilon)` value shared safely by the
  contiguous valid input counts one and two.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Give `RMS_NORM` exactly one family-owned input-range signature accepting one or two inputs and
  exactly one output.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Add exactly two public receiver methods and no static entry:

  ```java
  public Tensor rmsNorm(Shape normalizedShape, ScalarValue epsilon)
  public Tensor rmsNorm(Shape normalizedShape, Tensor scale, ScalarValue epsilon)
  ```

- Add one package-private stateless `TensorRmsNormExpressions` helper and exactly one final
  `TensorFactory.createDerived` delegation per successful call.
- Normalize the trailing [dimensions](../../../../glossary.md#dimension) described by the exact supplied `normalizedShape`; require an
  optional scale Shape exactly equal to it.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Require floating input and scale, apply ordered floating promotion for the scale form, and
  require epsilon to have the exact result type.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon))</span>
- Select exact uncentered-mean-square, epsilon, accumulation, empty, special-value, signed-zero,
  overflow, reassociation, and determinism policies.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Produce one fresh unlabeled, storage-free, unresolved-layout result with exact ordered
  provenance and output index zero.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update focused Javadocs, Tensor and Compile API references, glossary, capability baseline,
  master plan, roadmap, task evidence, and public-surface tests in the same overall change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- bias, shift, scale-plus-bias, bias-only, broadcast scale, implicit scale constants, parameter
  initialization, modules/layers, parameters, trainable ownership, or a general options object<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [constants glossary](../../../../glossary.md#term-constant); [parameter glossary](../../../../glossary.md#term-parameter); [modules glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- layer, batch, group, instance, local-response, Lp, weight, spectral, partial RMS, or other
  normalization; task 0021 remains complete and tasks 0021B–0021C remain later<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [weight glossary](../../../../glossary.md#term-weight); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- a public axis or axis array, arbitrary non-trailing axes, inferred normalized Shape, scalar or
  empty normalized Shape, default epsilon, untyped binary64 epsilon, or epsilon Tensor input<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [scalar glossary](../../../../glossary.md#term-scalar); [epsilon glossary](../../../../glossary.md#epsilon); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Tensor glossary](../../../../glossary.md#tensor))</span>
- centered mean, variance, standard deviation, subtraction of a mean, correction parameter,
  correction other than the intrinsic divisor <span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>`N`, epsilon outside the square root, denominator
  clamp/sentinel, or configurable result/accumulator type<span data-glossary-navigation="true"> ([square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [accumulator type glossary](../../../../glossary.md#accumulator-type))</span>
- saved [mean](../../../../glossary.md#term-mean) square, [root mean square](../../../../glossary.md#root-mean-square), inverse root mean square, or any auxiliary output
- training/evaluation mode, running statistics, momentum, mutation, hidden flags, or stateful
  services<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [momentum glossary](../../../../glossary.md#term-momentum))</span>
- integral, BOOL, FLOAT16, quantized, sparse, complex, or unsigned inputs; implicit or explicit
  cast-producer insertion<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [producer glossary](../../../../glossary.md#term-producer))</span>
- value reads, eager evaluation, host allocation, result storage, resolved layout, input mutation,
  or materialization<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- algorithms, compensated summation, fixed traversal order, gradients, adjoints, compiler capture,
  graph-wide constraints, binding, decomposition, fusion, backend support, lowering, prepare,
  runtime, execution, tolerances, conformance, or integration<span data-glossary-navigation="true"> ([compensated summation glossary](../../../../glossary.md#term-compensated-summation); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [binding glossary](../../../../glossary.md#term-binding); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to existing layer normalization, softmax, reductions, typed scalars, Shape,
  operation-signature, producer/factory/provenance foundations, architecture, dependencies,
  Gradle, other modules, or detailed specifications for later tasks<span data-glossary-navigation="true"> ([layer normalization glossary](../../../../glossary.md#layer-normalization); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Semantic kind, attributes, and [signature](../../../../glossary.md#term-signature)

`RmsNormKind` contains exactly `RMS_NORM`. It does not extend `LayerNormKind`, reuse
`LayerNormAttrs`, or change any existing [normalization](../../../../glossary.md#term-normalization) family. The kind accepts exactly:

| Attributes type | Ordered inputs | Ordered outputs |
|---|---|---|
| `RmsNormAttrs` | `[input]` or `[input, scale]` | `[output]` |

The family declares this stable signature:

```java
OperationSignature.inputRange(RmsNormAttrs.class, 1, 2, 1)
```

The range is intentional and signature-safe: one and two are both valid consecutive counts, with
the optional second position having exactly the scale role. Zero, three, or any larger input count
is invalid. This differs from [layer normalization](../../../../glossary.md#layer-normalization), whose valid counts one and three contain an
invalid two-input hole and therefore require two exact attribute classes. Do not weaken or change
`OperationSignature`, create a [registry](../../../../glossary.md#term-registry), reuse either layer-normalization attributes class, or add
a second kind for scale.

`RmsNormAttrs` contains exactly:

```java
Shape normalizedShape
ScalarValue epsilon
```

It null-checks `normalizedShape`, then [`epsilon`](../../../../glossary.md#epsilon); requires `normalizedShape.rank() > 0`; requires
floating epsilon; and requires epsilon to be [finite](../../../../glossary.md#term-finite) and strictly greater than [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) in its
exact [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64) representation. It retains both [immutable](../../../../glossary.md#term-immutable) references
unchanged. Exact intrinsic failures are:

```text
NullPointerException("normalizedShape")
NullPointerException("epsilon")
IllegalArgumentException("normalizedShape rank must be positive")
IllegalArgumentException("epsilon must have a floating data type, but was <dataType>")
IllegalArgumentException("epsilon must be finite and positive: <epsilon>")
```

Positive zero, [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), negative finite values, either [infinity](../../../../glossary.md#term-infinity), and every [NaN](../../../../glossary.md#term-nan) encoding fail
the final check. The record stores no [Tensor](../../../../glossary.md#tensor), scale-presence flag, input [rank](../../../../glossary.md#term-rank), [axes](../../../../glossary.md#term-axis), result,
[layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), or execution state. Input [cardinality](../../../../glossary.md#term-cardinality), not duplicated
attributes, identifies whether scale is present.

### Public surface and scale policy

The no-scale method records exact ordered input `[input]`. The scale method records
`[input, scale]`. Scale is a required explicit operand in that overload; there is no nullable
operand, `Optional<Tensor>`, hidden all-ones [Tensor](../../../../glossary.md#tensor), attribute-held [scalar](../../../../glossary.md#term-scalar) scale, [bias](../../../../glossary.md#term-bias), [alias](../../../../glossary.md#aliasing-and-arena), or
attrs-taking public overload.

The methods construct [model](../../../../glossary.md#term-model) expressions, not layer objects. They do not create, initialize, own,
register, or mutate [trainable parameters](../../../../glossary.md#trainable-parameter). There is no `double` [epsilon](../../../../glossary.md#epsilon) convenience, default
epsilon, public static entry, or raw-[axis](../../../../glossary.md#term-axis) overload.

### [Normalized Shape](../../../../glossary.md#normalized-shape), scale [Shape](../../../../glossary.md#shape), and deferred constraints

For input Shape `[D0, ..., D(R-1)]` and normalized Shape `[N0, ..., N(K-1)]`:

- require `K > 0` and `K <= R`;
- [normalized axis](../../../../glossary.md#normalized-axis) `j` corresponds to input [axis](../../../../glossary.md#term-axis) `R - K + j`;
- equal static extents pass and unequal static extents fail locally;
- structurally equal dynamic or expression [Dimensions](../../../../glossary.md#dimension) pass;
- any other pair involving an unresolved Dimension is accepted with an equality obligation for
  later compiler capture/binding because the result Shape is still exactly derivable; and<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- the result retains the exact input Shape reference and every exact Dimension reference.

Exact task-owned [rank](../../../../glossary.md#term-rank) and static mismatch failures are:

```text
rmsNorm normalized rank must not exceed input rank: normalizedRank=<K>, inputRank=<R>
rmsNorm normalized dimension mismatch at normalized axis <j>: input=<inputDimension>, normalized=<normalizedDimension>
```

The scale Shape must be structurally equal to `normalizedShape`; ordinary broadcastability is not
enough. This includes exact rank and static, dynamic, and expression structure. Failure is:

```text
rmsNorm scale Shape must equal normalizedShape: scale=<scaleShape>, normalizedShape=<normalizedShape>
```

A corresponding static zero in the input and normalized Shape is valid. It makes the entire
result empty, so no divisor or [root mean square](../../../../glossary.md#root-mean-square) is evaluated. The same rule applies if an
unresolved normalized [extent](../../../../glossary.md#term-extent) later binds to zero. A zero leading extent also makes the result
empty. [Scalar](../../../../glossary.md#term-scalar) input is invalid because the required positive normalized rank cannot fit rank zero.
No Shape or caller array is copied.

### Formula, correction, and [epsilon](../../../../glossary.md#epsilon) placement

For one non-empty trailing [slice](../../../../glossary.md#slice) containing `N` values `x_i`, define the [uncentered mean square](../../../../glossary.md#uncentered-mean-square) and
[root mean square](../../../../glossary.md#root-mean-square) as:

```text
meanSquare = sum_i(x_i * x_i) / N
rms        = sqrt(meanSquare + epsilon)
normalized_i = x_i / rms
```

The no-scale result is `normalized_i`. The scale form is:

```text
output_i = normalized_i * scale_i
```

Scale [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) span the [normalized Shape](../../../../glossary.md#normalized-shape) in [row-major](../../../../glossary.md#term-row-major) logical coordinate order and are reused
for every leading slice. Epsilon is added to the uncentered mean square inside the [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions).
There is no centering and no [variance](../../../../glossary.md#term-variance). [Statistical correction](../../../../glossary.md#statistical-correction) is not applicable: the [mean](../../../../glossary.md#term-mean) square
always divides by the population count `N`, never `N - c`, and there is no correction attribute.
There is no epsilon outside the root, denominator clamp, sentinel, or replacement [norm](../../../../glossary.md#term-norm).

For `[1, 2, 3]` and [FLOAT64](../../../../glossary.md#term-float64) epsilon `1e-5`, the mean square is `14 / 3`, the root is approximately
`2.1602492140182963`, and the no-scale result is approximately
`[0.4629095539120194, 0.9258191078240388, 1.3887286617360581]`. A scale value `2` at the final
coordinate yields approximately `2.7774573234721163` there. This example illustrates the
mathematical contract only; [model](../../../../glossary.md#term-model) construction does not evaluate it.

The formula and trailing-[dimension](../../../../glossary.md#dimension) interpretation align with the official
[PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) and
[ONNX RMSNormalization](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span> contracts.
Synaptik deliberately requires an explicit typed epsilon, exact scale [Shape](../../../../glossary.md#shape), and no [bias](../../../../glossary.md#term-bias) or
operator-configurable stash type, rather than copying those broader surfaces.

### Data types, promotion, and accumulation

No-scale [RMS normalization](../../../../glossary.md#rms-normalization) accepts [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64) input and retains its exact type.
The scale form requires both inputs to be floating and derives result type with
`DataTypePromotion.promoteFloating(inputType, scaleType)` in ordered input [occurrence](../../../../glossary.md#term-occurrence) order.
Promotion changes result [metadata](../../../../glossary.md#term-metadata) only and inserts no cast.

Exact type failures are:

```text
rmsNorm input must have a floating data type, but was <dataType>
rmsNorm scale must have a floating data type, but was <dataType>
rmsNorm epsilon data type must match result data type: epsilon=<epsilonType>, result=<resultType>
```

[Epsilon](../../../../glossary.md#epsilon) must exactly equal the derived result type. The normalized computation treats input as
the result type before multiplication and accumulation. BFLOAT16/FLOAT32 results square and sum in
FLOAT32; FLOAT64 results square and sum in FLOAT64. Division and optional scale multiplication
occur in the result type. A BFLOAT16 no-scale result therefore accumulates in FLOAT32 and converts
the final normalized value to BFLOAT16. No narrower [accumulator](../../../../glossary.md#accumulator-type) or hidden [model](../../../../glossary.md#term-model) result type is
allowed.

For [finite](../../../../glossary.md#term-finite) values, the semantic [target](../../../../glossary.md#term-target) is the stated real formula rounded to the result format. A
conforming implementation may use equal-or-wider intermediates, reassociate, parallelize, or
choose another [evaluation](../../../../glossary.md#term-evaluation) strategy only if it preserves the exact special-value classes below
and satisfies later [conformance](../../../../glossary.md#term-conformance) tolerances. Model selects no algorithm. Bitwise equality, a fixed
reduction traversal, [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and cross-[backend](../../../../glossary.md#backend) identical [rounding](../../../../glossary.md#term-rounding) are not
promised.

### [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), and determinism

- Any NaN in a non-empty normalized [slice](../../../../glossary.md#slice) makes `meanSquare`, `rms`, and every normalized value in
  that slice NaN. Optional scale cannot suppress that NaN.
- With no NaN and at least one positive or [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero), `meanSquare` and `rms` are positive
  infinity. Each finite input then produces same-signed zero, while each infinite input produces
  NaN from infinity divided by infinity.<span data-glossary-navigation="true"> ([positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- For an all-[finite](../../../../glossary.md#term-finite) slice, `meanSquare` is non-negative and positive [epsilon](../../../../glossary.md#epsilon) makes `rms` strictly
  positive. Positive and negative zero inputs therefore remain positive and negative zero,
  respectively, before scale. A finite constant slice is not centered: a nonzero constant remains
  a nonzero normalized value, and an all-zero slice preserves each input zero sign.<span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [constant glossary](../../../../glossary.md#term-constant))</span>
- Finite squaring or summation may overflow to positive infinity in the selected accumulator. In
  that case every still-finite numerator produces same-signed zero. Other finite overflow and
  underflow follow the selected accumulator and result formats.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [overflow and underflow glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow))</span>
- In the scale form, NaN scale produces NaN at that coordinate. Infinite scale times a normalized
  zero produces NaN; infinite scale times a nonzero finite normalized value produces the
  corresponding signed infinity. Finite scale multiplication follows ordinary floating sign
  rules, including signed zero. No bias addition follows.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [bias glossary](../../../../glossary.md#term-bias))</span>
- Empty results contain no values and expose no mean square, denominator, NaN, infinity, or signed-
  zero result.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean))</span>

These policies make special-value classes independent of physical traversal. Finite nonzero
[rounding](../../../../glossary.md#term-rounding) remains subject to permitted [reassociation](../../../../glossary.md#term-reassociation) and later [tolerance](../../../../glossary.md#term-tolerance). Separately constructed
equal requests remain distinct [producers](../../../../glossary.md#term-producer); there is no interning or [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

### Result [metadata](../../../../glossary.md#term-metadata), [provenance](../../../../glossary.md#provenance), and [identity](../../../../glossary.md#term-identity-meanings) effects

The [descriptor](../../../../glossary.md#term-descriptor) uses the exact or promoted result type, the exact input [Shape](../../../../glossary.md#shape) reference,
unresolved [layout](../../../../glossary.md#layout), and `requiresGrad` equal to input eligibility for no-scale or logical OR of
input and scale eligibility for scaled [RMS normalization](../../../../glossary.md#rms-normalization). Every result is fresh, unlabeled, and
storage-free. Inputs and their descriptors, IDs, labels, storage, and provenance are not mutated.

Each successful call creates one [`Operation`](../../../../glossary.md#operation), one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one output descriptor, one
[Tensor](../../../../glossary.md#tensor) wrapper, and one Tensor ID. Provenance output index is zero and ordered inputs are exactly
`[input]` or `[input, scale]`. No [saved statistic](../../../../glossary.md#saved-statistic), hidden scale, sibling descriptor, wrapper, or ID
is created. Future [compiler](../../../../glossary.md#term-compiler)-generated saved values or adjoints do not change this [producer](../../../../glossary.md#term-producer)'s one-
output contract.

### [Validation](../../../../glossary.md#term-validation) and construction order

`TensorRmsNormExpressions` exposes exactly two package-private entries matching the public methods.
The no-scale entry validates in this order:

1. null-check `input`, `normalizedShape`, then [`epsilon`](../../../../glossary.md#epsilon);
2. validate input floating eligibility;
3. require positive normalized [rank](../../../../glossary.md#term-rank) and validate it against input rank;
4. validate trailing static compatibility in normalized-[axis](../../../../glossary.md#term-axis) order;
5. validate epsilon exact result type;
6. construct `RmsNormAttrs`, [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), and delegate once.

The scale entry validates in this order:

1. null-check `input`, `normalizedShape`, `scale`, then `epsilon`;
2. validate input, then scale floating eligibility;
3. require positive normalized rank and validate it against input rank;
4. validate trailing static compatibility in normalized-axis order;
5. validate exact scale [Shape](../../../../glossary.md#shape);
6. promote input and scale in [occurrence](../../../../glossary.md#term-occurrence) order;
7. validate epsilon exact result type;
8. construct `RmsNormAttrs`, descriptor, operation, and delegate once.

Null failures use the exact [parameter](../../../../glossary.md#term-parameter) names. The attributes constructor repeats its intrinsic
non-null/rank/epsilon validation defensively. Existing Shape, promotion, descriptor, operation,
[producer](../../../../glossary.md#term-producer), and identifier failures retain their current messages. Every local failure happens
before factory delegation and consumes no [Tensor](../../../../glossary.md#tensor) ID, producer, or wrapper.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Layer normalization](0021-layer-normalization-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Layer normalization glossary](../../../../glossary.md#layer-normalization))</span>
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar values](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>

## Architecture constraints

- Work stays in [model](../../../../glossary.md#term-model) plus its documentation/[planning](../../../../glossary.md#term-planning). [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state,
  not graph intermediate representation.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [intermediate representation glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- RMS-normalization operation types contain backend-independent semantic parameters only and do
  not import Tensor, graph, compiler, training, runtime, prepare, or backend types.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [backend glossary](../../../../glossary.md#backend); [parameters glossary](../../../../glossary.md#term-parameter); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Package direction is tensor helper to normalization operation/datatype/shape. No reverse edge,
  registry, service, layer owner, parameter owner, or state owner is introduced.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [registry glossary](../../../../glossary.md#term-registry); [owner glossary](../../../../glossary.md#term-owner))</span>
- Compiler owns capture, deferred equality proof, binding validation, gradients, saved values,
  adjoints, and legal decomposition. Backend prepare owns algorithms, lowering, specialization,
  fusion, kernels, and tolerance satisfaction. Runtime executes prepared work without consuming<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [kernels glossary](../../../../glossary.md#kernel); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
  [`Operation`](../../../../glossary.md#operation) on its [hot path](../../../../glossary.md#term-hot-path).
- No architecture, focused-architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) change.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.normalization` — add distinct RMS-normalization
  semantics beside unchanged softmax and layer-normalization families.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.normalization.RmsNormKind` — public semantic identity and exact contiguous input-
  range signature.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [contiguous glossary](../../../../glossary.md#term-contiguous); [signature glossary](../../../../glossary.md#term-signature))</span>
- `...operation.normalization.RmsNormAttrs` — public trailing-Shape and exact typed epsilon
  parameters shared by the two safe cardinalities.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [epsilon glossary](../../../../glossary.md#epsilon); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `...tensor.TensorRmsNormExpressions` — package-private validation, promotion, Shape, descriptor,
  and provenance owner.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [owner glossary](../../../../glossary.md#term-owner))</span>
- `...tensor.Tensor` — established public fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (8):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signatures and public count 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 181 to 183.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 181 to 183.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime)/Training APIs; [layer normalization](../../../../glossary.md#layer-normalization), [softmax](../../../../glossary.md#softmax--log-softmax),
reductions/statistics, typed-[scalar](../../../../glossary.md#term-scalar), [Shape](../../../../glossary.md#shape), [signature](../../../../glossary.md#term-signature), [Tensor](../../../../glossary.md#tensor)/factory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), and
other [operation](../../../../glossary.md#operation) contracts; architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 19 paths maximum: four production, eight test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. The
two new public methods require six existing public-count assertions to advance together, while one
input-range [signature](../../../../glossary.md#term-signature) avoids a redundant scale-specific attributes class. Stop for path 20,
another production type/test/document, an existing-helper change, architecture, [Gradle](../../../../glossary.md#term-gradle),
cross-[module](../../../../glossary.md#term-module) work, [bias](../../../../glossary.md#term-bias), saved output, state/mode behavior, or a later task specification.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, attrs, helper, and both Tensor methods: uncentered formula, trailing Shape,
  scale role, exact epsilon placement/type, accumulation, empty/special-value behavior,
  metadata/provenance/ID effects, validation/failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every [parameter](../../../../glossary.md#term-parameter), non-void result, and expected failure has complete `@param`, `@return`, and
  `@throws` tags.
- [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) gets both exact [signatures](../../../../glossary.md#term-signature), an operand/[Shape](../../../../glossary.md#shape) table, the [finite](../../../../glossary.md#term-finite) `[1, 2, 3]` example,
  a static mismatch, symbolic deferral, empty/special policies, and current-model versus planned
  compiler/execution boundary.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compile API records RMS-normalization metadata as current after implementation and leaves
  capture, constraint proof, saved values, gradients, decomposition, lowering, and execution
  planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [normalization glossary](../../../../glossary.md#term-normalization); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Review glossary terms normalization, RMS normalization, normalized Shape, uncentered mean
  square, root mean square, epsilon, scale, accumulator, producer, and saved statistic. Add or
  refine only reusable distinctions needed by the public explanation.<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../../glossary.md#rms-normalization); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [uncentered mean square glossary](../../../../glossary.md#uncentered-mean-square); [root mean square glossary](../../../../glossary.md#root-mean-square); [accumulator glossary](../../../../glossary.md#accumulator-type); [producer glossary](../../../../glossary.md#term-producer); [saved statistic glossary](../../../../glossary.md#saved-statistic))</span>
- Synchronize statuses: 0021 remains Complete; 0021A is Ready during implementation and becomes
  Complete only after all criteria pass; 0021B–0021C and 0022–0024 remain Draft without detailed
  specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact `RMS_NORM`, one safe `RmsNormAttrs` input-range signature, and two canonical receiver
  methods exist; layer normalization and softmax remain unchanged and public Tensor count is 183.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [canonical glossary](../../../../glossary.md#term-canonical-form); [layer normalization glossary](../../../../glossary.md#layer-normalization); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Non-empty trailing normalized Shape, exact scale Shape, static mismatch rejection, symbolic
  equality deferral, exact input Shape retention, and empty-result rules match this task.<span data-glossary-navigation="true"> ([normalized Shape glossary](../../../../glossary.md#normalized-shape); [Shape glossary](../../../../glossary.md#shape))</span>
- No-scale exact type retention and scale-form ordered promotion pass; epsilon is exact-result-
  typed, floating, finite, and positive; selected accumulator types are API-locked without value
  evaluation.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [finite glossary](../../../../glossary.md#term-finite); [accumulator glossary](../../../../glossary.md#accumulator-type); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- [Uncentered mean square](../../../../glossary.md#uncentered-mean-square) divided by `N`, epsilon inside the root, no correction option, scale-only
  formula, NaN/infinity/signed-zero, overflow, reassociation, rounding, and determinism policies
  are explicit and internally consistent.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [reassociation glossary](../../../../glossary.md#term-reassociation); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Requires-grad, unresolved layout, freshness, no label/storage/mutation, exact input order, one
  output/index zero, and exact one-ID/producer/wrapper effects pass.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Validation](../../../../glossary.md#term-validation) order and task-owned messages pass; every pre-factory failure consumes no ID.
- No bias, broadcast scale, axes/options/default epsilon, hidden constants, saved outputs,
  state/mode, gradient/compiler/algorithm/backend/runtime, architecture, dependency, build,
  cross-module, or later-spec work is added.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [axes glossary](../../../../glossary.md#term-axis); [constants glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  official-link, exact 19-path/package/public-surface/status audits, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.RmsNormSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.datatype.ScalarValueTest --tests io.github.pho001.synaptik.model.shape.ShapeTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact attrs pairing and input range, public surface, both formulas as
inspectable semantic contracts, static/symbolic/empty [Shapes](../../../../glossary.md#shape), scale Shape, promotion,
accumulation/special-value policy, validation/ID effects, [metadata](../../../../glossary.md#term-metadata), freshness, and exact
one-output [provenance](../../../../glossary.md#provenance) without value execution.

Documentation pass after final [Javadoc](../../../../glossary.md#term-javadoc):

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation-focused pass also runs the repository's existing or targeted Markdown
link/anchor/fence/final-newline checks, verifies the PyTorch and [ONNX](../../../../glossary.md#term-onnx) official links, audits exact
19-path [scope](../../../../glossary.md#term-scope) and package placement, checks public count/[signatures](../../../../glossary.md#term-signature) and synchronized statuses, and
records reused Java evidence. Repository-wide validation is deferred to the selected modern-
[operation](../../../../glossary.md#operation) [checkpoint](../../../../glossary.md#term-checkpoint) after task 0022 and [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [module](../../../../glossary.md#term-module) without
dependencies, architecture, shared build configuration, or executable [backend](../../../../glossary.md#backend) behavior.

## Dependencies

- 0001–0002: [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), static/symbolic, [rank](../../../../glossary.md#term-rank), and [element](../../../../glossary.md#term-element)-count contracts.
- 0005–0007 and 0011–0013: [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), factory, and [provenance](../../../../glossary.md#provenance) foundations.
- 0018K: exact attributes-class [signatures](../../../../glossary.md#term-signature) and inclusive [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality).
- 0018L: [producer](../../../../glossary.md#term-producer)/output-index provenance, used here in one-output form.
- 0018N: exact typed [scalar](../../../../glossary.md#term-scalar) representation for [epsilon](../../../../glossary.md#epsilon).
- 0018V: accumulation, empty-domain, special-value, reassociation, and statistical vocabulary
  precedents; RMS normalization deliberately has no correction parameter.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [RMS normalization glossary](../../../../glossary.md#rms-normalization); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- 0021: trailing normalized-Shape, deferred equality, typed epsilon, and metadata precedents,
  without reusing layer-normalization attrs or centered formula.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [normalization glossary](../../../../glossary.md#term-normalization))</span>

All dependencies are Complete.

## Follow-up tasks

- 0021B (Draft) — batch-normalization inference with explicit running mean/variance and affine
  inputs, one output, and no hidden training flag, mutation, or saved output.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance))</span>
- 0021C (Draft) — batch-normalization training as an explicit state transition after inference:
  batch statistics, running-stat inputs/outputs, and saved statistics use genuine multi-output
  provenance without a stateful service.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- 0022–0024 remain Draft at their established IDs. Task 0023 owns any RMS-normalization adjoint or
  compiler-generated saved-statistic meaning justified by compiler implementation.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [compiler glossary](../../../../glossary.md#term-compiler); [statistic glossary](../../../../glossary.md#term-statistic))</span>

## Architecture impact

Expected impact: None.

The task adds [model](../../../../glossary.md#term-model)-owned semantic values and public expression [metadata](../../../../glossary.md#term-metadata) inside existing package
and [dependency](../../../../glossary.md#term-dependency) boundaries. If implementation requires hidden mode/state, [bias](../../../../glossary.md#term-bias), a cross-[module](../../../../glossary.md#term-module)
type, architecture/dependency change, or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md.
Implement that task exactly as specified and do not implement out-of-scope or later normalization
work. Stop and report any architecture, dependency, or maximum-scope conflict. Do not commit or
push unless separately authorized.

After Java implementation and recorded module-test evidence, hand the final diff and evidence to
a separate documentation-focused agent/thread with clean context. That pass must follow
docs/developer-guide/documentation-rules.md, inspect source/tests, finalize Javadocs, Tensor and
Compile APIs, glossary and planning records, run Javadoc/Markdown/scope/status/whitespace checks,
and reuse successful Java evidence unless executable Java changes.

Update this task's local decisions, limitations, evidence, notes, completion summary, and status.
Do not mark Complete before the documentation pass and all acceptance criteria finish.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide the clean documentation context with this task, the complete implementation diff, exact
focused/final Java commands and results, affected RMS-[normalization](../../../../glossary.md#term-normalization) semantics/public methods,
architecture and 19-path constraints, expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) and glossary/[planning](../../../../glossary.md#term-planning) impact,
official links, and required [Javadoc](../../../../glossary.md#term-javadoc)/Markdown/[scope](../../../../glossary.md#term-scope)/status/whitespace evidence. It must inspect the
actual source/tests, not rely only on the summary, and record reasoned no-change conclusions.

## Local decisions

- Kept [RMS normalization](../../../../glossary.md#rms-normalization) first-class and distinct from [layer normalization](../../../../glossary.md#layer-normalization): one `RMS_NORM` kind,
  one `RmsNormAttrs(normalizedShape, epsilon)` value, and one safe inclusive one-to-two-input
  signature represent exact <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`[input]` and `[input, scale]` occurrences.
- Selected only receiver methods with explicit typed epsilon. The no-scale form retains input type;
  the scaled form promotes input then scale and requires epsilon to match that result exactly.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon))</span>
- Required exact structural scale Shape equality rather than broadcasting. Trailing static input
  mismatches fail locally, while equality involving unresolved Dimensions defers because result
  Shape remains exactly the input Shape.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [broadcasting glossary](../../../../glossary.md#broadcasting); [Dimensions glossary](../../../../glossary.md#dimension))</span>
- Fixed the uncentered population mean-square formula, epsilon inside the square root, FLOAT32
  accumulation for BFLOAT16/FLOAT32 results, FLOAT64 accumulation for FLOAT64, and explicit
  empty/special-value/reassociation boundaries without selecting an algorithm.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Retained the current one-output expression model: one fresh producer, descriptor, wrapper, and
  Tensor ID with output index zero and exact ordered provenance. No hidden scale, bias, saved
  statistic, sibling output, or execution state is created.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [bias glossary](../../../../glossary.md#term-bias); [saved statistic glossary](../../../../glossary.md#saved-statistic))</span>

## Known limitations

- Construction records semantic metadata and reads no values. Compiler capture, deferred-
  constraint representation/proof, saved values, gradients or adjoints, legal decomposition,
  lowering, backend preparation, tolerance enforcement, and runtime execution remain planned.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [tolerance glossary](../../../../glossary.md#term-tolerance); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Fixed traversal, bitwise identity, NaN payload/sign retention, and cross-backend identical
  finite rounding are not promised; later conformance work must select tolerances consistent with
  the documented special-value classes.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [rounding glossary](../../../../glossary.md#term-rounding); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Bias, broadcast scale, defaults, state/mode behavior, saved outputs, and other normalization
  families remain deliberately absent. Tasks 0021B–0021C and 0022–0024 remain Draft without
  detailed specifications.<span data-glossary-navigation="true"> ([Bias glossary](../../../../glossary.md#term-bias); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Repository-wide validation remains deferred to the selected modern-operation checkpoint after
  task 0022 or CI because this change affects only model and its documentation without dependency,
  architecture, shared-build, backend, or execution changes.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0021a_implementation` ran the exact focused command listed in
  this task; it passed `BUILD SUCCESSFUL`. Its first broader module-suite attempt exposed only that
  the exact-name <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`TensorTest` surface set had not yet listed `rmsNorm`; after correcting that
  expected set, the focused `TensorTest` passed. The same context then ran exactly one final stable
  `./gradlew :modules:model:test`; 908 tests passed with `BUILD SUCCESSFUL`. No executable Java
  behavior changed afterward.
- Clean documentation context `/root/task_0021a_implementation/task_0021a_docs` applied the General,
  API/Javadoc, Planning, glossary, and example profiles. It independently reviewed all four
  production contracts, all eight tests, Tensor/Compile/Runtime/Training APIs, glossary,
  capabilities, task, master plan, roadmap, focused architecture, and related layer-normalization,
  softmax, reduction/statistics, typed-scalar, Shape, signature, factory, producer, and provenance
  contracts.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [normalization glossary](../../../../glossary.md#term-normalization); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- `./gradlew :modules:model:javadoc` passed `BUILD SUCCESSFUL` after final Javadoc edits. Generated
  pages contain `RmsNormKind`, `RmsNormAttrs`, and both exact `Tensor.rmsNorm` signatures and their
  reviewed parameter, result, failure, formula, metadata, and lifecycle contracts.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A targeted Markdown checker resolved 607 local links, including 165 heading anchors, across all
  seven documentation/planning paths with zero failures. Separate checks found balanced backtick
  and tilde fences, final newlines, and no trailing whitespace. Manual recalculation confirmed the
  FLOAT64 <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`[1, 2, 3]`, [epsilon](../../../../glossary.md#epsilon) `1e-5` example and scaled final [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix).
- The official PyTorch RMSNorm URL resolves to the current PyTorch RMSNorm reference, and the
  official ONNX RMSNormalization URL resolves to the current ONNX operator reference.<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Compiled-surface inspection found exactly 183 public [`Tensor`](../../../../glossary.md#tensor) methods and exactly these two
  signatures: `rmsNorm(Shape, ScalarValue)` and `rmsNorm(Shape, Tensor, ScalarValue)`. Source and
  test inspection confirmed the sole `RMS_NORM` enum value, exact
  `OperationSignature.inputRange(RmsNormAttrs.class, 1, 2, 1)`, two package-private helper entries,
  and no static public entry or forbidden alias.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Final inventory found exactly 19 authorized paths: four production, eight tests, and seven
  documentation/planning paths. Production types remain in the planned normalization/tensor
  packages; no forbidden cross-module import, architecture/dependency/build edit, later detailed
  specification, bias, saved output, or executable compiler/backend/runtime addition exists.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [bias glossary](../../../../glossary.md#term-bias); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Runtime API remains accurate unchanged because no prepared execution, storage, state, or run
  behavior was added. Training API remains accurate unchanged because gradient eligibility is
  metadata only and no gradient rule, parameter, optimizer, saved statistic, or training mode was
  added. Compile API was updated because current RMS metadata is compiler-visible, while capture,
  proof, saved values, gradients, decomposition, lowering, and execution remain planned.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [run glossary](../../../../glossary.md#term-run); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [saved statistic glossary](../../../../glossary.md#saved-statistic); [capture glossary](../../../../glossary.md#term-capture-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Related layer-normalization, softmax, reduction/statistics, typed-scalar, Shape, signature,
  factory, producer, and provenance contracts remain accurate unchanged because RMS uses their
  established boundaries without changing them. Architecture/ADRs/tests, conformance/integration,
  Gradle, and other modules remain unchanged because no ownership, dependency, build, backend, or
  end-to-end behavior changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Final status audit confirmed tasks 0021 and 0021A Complete; tasks 0021B–0021C and 0022–0024
  Draft; no model task Ready; and no detailed 0021B/0021C or 0022–0024 specification. Final<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Ready glossary](../../../../glossary.md#term-ready))</span>
  `git diff --check` passed with no output.

## Implementation notes

- Added `RmsNormKind`, `RmsNormAttrs`, and package-private `TensorRmsNormExpressions`, then exposed
  exactly two fluent receiver methods in [`Tensor`](../../../../glossary.md#tensor).
- Focused semantics tests lock the kind/signature, typed epsilon, formula and special-value policy.
  Expression tests lock public surface, type promotion, Shape validation and symbolic deferral,
  empty results, metadata, validation order, freshness, identity use, and exact one-output
  provenance. Six existing surface tests advanced their exact public count from 181 to 183.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [epsilon glossary](../../../../glossary.md#epsilon); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- The independent documentation pass finalized all affected Javadocs, added the complete public
  Tensor explanation and compiler lifecycle boundary, refined reusable glossary distinctions, and
  synchronized capability/master/roadmap status without changing executable Java.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Completion summary

- Completed changes: added distinct RMS-normalization semantics and exact no-scale/scale-only
  Tensor expression construction with the specified formula, Shape/type/epsilon policies,
  metadata, validation, freshness, and one-output provenance contracts.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [epsilon glossary](../../../../glossary.md#epsilon); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: exactly 19 authorized paths comprising four production Java files,
  eight model tests, and seven API/glossary/planning documents listed under Affected files.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the exact focused command and final 908-test model suite passed in the
  implementation context; the documentation context reused those stable results, and model
  Javadoc, 607-link/165-anchor Markdown, official-link, generated-page, public-surface, package,
  scope, status, fence/newline/whitespace, terminology, example, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` validation
  passed.
- Documentation review: Tensor and Compile APIs, glossary, capabilities, task, master plan,
  roadmap, and all affected Javadocs are final. Runtime/Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, and other modules need no change for
  the reasons recorded above.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Unresolved issues: None within task [scope](../../../../glossary.md#term-scope).
- Required follow-up: None for task 0021A. Draft tasks 0021B–0021C and 0022–0024 remain future
  work under their established ownership.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Status: Complete
