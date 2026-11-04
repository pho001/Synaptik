# Task 0021: [Layer Normalization](../../../../glossary.md#layer-normalization) Semantics and [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (current context, not a status update).**

This record concerns the original model-only layer-normalization scope; its completion and evidence
do not describe today's execution coverage. The caller invokes `layerNorm` on the input Tensor,
supplying a trailing normalized Shape, typed positive epsilon, and optionally both scale and bias.
The helper validates descriptions and records ordered operand references, not computed values.
“Affine” means scale then bias; both operand Shapes must exactly equal the normalized Shape. For
input Shape `[2, 3]` and normalized Shape `[3]`, each row is standardized independently using
population variance, with epsilon inside the square root.

The [current layer-normalization
contract](../../../../reference/tensor/normalization.md#layer-normalization-expressions) explains
expanded floating-type and epsilon rules beyond this record. The [lifecycle
boundary](../../../../reference/tensor/normalization.md#result-identity-and-lifecycle-boundary)
separates expression construction from compiler and execution-provider responsibilities. Model
semantics defines meaning; providers realize supported routes without redefining it. The
storage-free result neither executes nor owns a result buffer. Semantic responsibility is not
storage ownership, and the single-output producer exposes no saved statistics.

**Mathematical reading illustration.** For finite real row entries $x_1,x_2$, let $\mu=(x_1+x_2)/2$,
population variance $v=((x_1-\mu)^2+(x_2-\mu)^2)/2$, and positive epsilon $\varepsilon$. Without
affine operands, $y_i=(x_i-\mu)/\sqrt{v+\varepsilon}$ for $i\in\{1,2\}$. At $x=(1,3)$ and
$\varepsilon=3$, $\mu=2$, $v=1$, and $y=(-1/2,1/2)$. This illustrates centering and epsilon
placement over real numbers, not actual floating evaluation.

## Status

Complete

## Goal

Add one first-class [backend](../../../../glossary.md#backend)-independent `LAYER_NORM` meaning and two public receiver expressions:
one without affine inputs and one with explicit scale and [bias](../../../../glossary.md#term-bias). The [operation](../../../../glossary.md#operation) normalizes each
non-empty trailing [slice](../../../../glossary.md#slice) described by a non-empty [`Shape`](../../../../glossary.md#shape), uses [population variance](../../../../glossary.md#population-variance) and an exact
typed positive [epsilon](../../../../glossary.md#epsilon), preserves the input Shape, and records one output without evaluating
values or exposing [compiler](../../../../glossary.md#term-compiler)-owned saved statistics.

This is the first [dependency](../../../../glossary.md#term-dependency)-ordered part of the former broad [normalization](../../../../glossary.md#term-normalization) frontier. [RMS
normalization](../../../../glossary.md#rms-normalization) follows because it has a different second-moment formula and affine policy. [Batch
normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) remains later because [inference](../../../../glossary.md#term-inference) consumes explicit running statistics, while training
also transitions running statistics and exposes saved statistics through genuine multi-output
[producer](../../../../glossary.md#term-producer) semantics.

## [Scope](../../../../glossary.md#term-scope)

- Add `LayerNormKind.LAYER_NORM` in the existing [normalization](../../../../glossary.md#term-normalization) [operation](../../../../glossary.md#operation) package.
- Add [immutable](../../../../glossary.md#term-immutable) public `LayerNormAttrs(normalizedShape, epsilon)` for one-input normalization and
  `AffineLayerNormAttrs(normalizedShape, epsilon)` for the exact three-input affine form.
- Give `LAYER_NORM` exactly two family-owned [signatures](../../../../glossary.md#term-signature): one input/one output with
  `LayerNormAttrs`, and three inputs/one output with `AffineLayerNormAttrs`.
- Add exactly two public receiver methods and no static entry:

  ```java
  public Tensor layerNorm(Shape normalizedShape, ScalarValue epsilon)
  public Tensor layerNorm(
          Shape normalizedShape, Tensor scale, Tensor bias, ScalarValue epsilon)
  ```

- Add one package-private stateless `TensorLayerNormExpressions` helper and exactly one final
  `TensorFactory.createDerived` delegation per successful call.
- Normalize the trailing [dimensions](../../../../glossary.md#dimension) whose exact contract is supplied by `normalizedShape`; the
  result retains the exact input Shape reference.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Require floating input and affine operands, apply floating promotion in ordered input occurrence
  order, and require epsilon to have the exact promoted result type.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Select exact population-variance, epsilon, accumulation, empty, special-value, signed-zero,
  reassociation, and determinism policies.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Produce one fresh unlabeled, storage-free, unresolved-layout result with exact ordered
  provenance and output index zero.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update focused API, glossary, capability, master-plan, roadmap, task, Javadoc, and public-surface
  tests in the same overall change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- RMS, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), group, instance, local-response, Lp, [weight](../../../../glossary.md#term-weight), spectral, or other [normalization](../../../../glossary.md#term-normalization)
- a public axis array, first-axis integer, arbitrary non-trailing axes, inferred normalized Shape,
  scalar/empty normalized Shape, or a general normalization options object<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- scale-only, bias-only, broadcast affine operands, implicit affine constants, affine defaults,
  parameter initialization, modules/layers, parameters, stateful services, or trainable ownership<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [constants glossary](../../../../glossary.md#term-constant); [parameter glossary](../../../../glossary.md#term-parameter); [modules glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- saved mean, variance, standard deviation, inverse standard deviation, or any other auxiliary
  output; the public result is one Tensor from one one-output producer<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer))</span>
- training/[evaluation](../../../../glossary.md#term-evaluation) mode, running statistics, [momentum](../../../../glossary.md#term-momentum), state mutation, or hidden mode flags
- correction other than zero, epsilon outside the square root, epsilon Tensor input, untyped
  binary64 epsilon, epsilon defaults, or output/accumulator configuration<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- integral, [BOOL](../../../../glossary.md#data-type--datatype), [FLOAT16](../../../../glossary.md#term-float16), quantized, sparse, complex, or unsigned inputs; explicit cast producers
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  materialization<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- algorithms, compensated summation, exact traversal order, gradients, adjoints, compiler capture,
  binding, graph-wide constraint solving, decomposition, fusion, backend support, lowering,
  prepare, runtime, execution, tolerances, conformance, or integration<span data-glossary-navigation="true"> ([compensated summation glossary](../../../../glossary.md#term-compensated-summation); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to softmax, reductions, typed scalars, Shape foundations, producer/factory contracts,
  architecture, dependencies, Gradle, other modules, or detailed specifications for later tasks<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [Shape glossary](../../../../glossary.md#shape); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Semantic kind, attributes, and [signatures](../../../../glossary.md#term-signature)

`LayerNormKind` contains exactly `LAYER_NORM`. Its declaration does not absorb or rename
`SoftmaxKind`; [softmax](../../../../glossary.md#softmax--log-softmax) and [layer normalization](../../../../glossary.md#layer-normalization) remain distinct families in the same cohesive
package.

The kind accepts exactly these variants, in this order:

| Attributes type | Ordered inputs | Ordered outputs |
|---|---|---|
| `LayerNormAttrs` | `[input]` | `[output]` |
| `AffineLayerNormAttrs` | `[input, scale, bias]` | `[output]` |

Both records contain exactly:

```java
Shape normalizedShape
ScalarValue epsilon
```

They null-check `normalizedShape`, then [`epsilon`](../../../../glossary.md#epsilon); require `normalizedShape.rank() > 0`; require
floating epsilon; and require epsilon to be [finite](../../../../glossary.md#term-finite) and strictly greater than [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) in its
own exact [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64) representation. They retain both [immutable](../../../../glossary.md#term-immutable) references
unchanged. Exact intrinsic failures are:

```text
NullPointerException("normalizedShape")
NullPointerException("epsilon")
IllegalArgumentException("normalizedShape rank must be positive")
IllegalArgumentException("epsilon must have a floating data type, but was <dataType>")
IllegalArgumentException("epsilon must be finite and positive: <epsilon>")
```

[Negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), positive zero, negative values, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and every [NaN](../../../../glossary.md#term-nan) encoding fail the last
check. The records own only [operation](../../../../glossary.md#operation) [parameters](../../../../glossary.md#term-parameter). They store no input, affine [Tensor](../../../../glossary.md#tensor), [rank](../../../../glossary.md#term-rank)-derived
[axis](../../../../glossary.md#term-axis) list, result, [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), or execution state.

Two attribute classes are intentional. The current exact-class signature contract cannot express
the disjoint valid input counts one and three with one attributes class without incorrectly
accepting two inputs. Do not weaken `OperationSignature`, use an input range, add a [registry](../../../../glossary.md#term-registry), or
split affine layer normalization into a second semantic kind.

### Public surface and affine policy

The no-affine method records exact ordered input `[input]`. The affine method records
`[input, scale, bias]`; scale and [bias](../../../../glossary.md#term-bias) are both required. There is no two-input scale-only form,
nullable operand, optional wrapper, hidden ones/zeros construction, or compatibility [alias](../../../../glossary.md#aliasing-and-arena).
Callers that want scale without bias may supply an explicit zero bias [Tensor](../../../../glossary.md#tensor), keeping every
operand visible to [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings).

These receiver methods are [model](../../../../glossary.md#term-model) expression constructors, not layer objects. They do not create,
initialize, own, register, or mutate [parameters](../../../../glossary.md#term-parameter). There is no public static `Tensor.layerNorm`,
attrs-taking overload, raw `int[]` [axes](../../../../glossary.md#term-axis) overload, `double` [epsilon](../../../../glossary.md#epsilon) convenience, or default epsilon.

### [Normalized Shape](../../../../glossary.md#normalized-shape) and deferred constraints

For input [Shape](../../../../glossary.md#shape) `[D0, ..., D(R-1)]` and normalized Shape `[N0, ..., N(K-1)]`:

- require `K > 0` and `K <= R`;
- [normalized axis](../../../../glossary.md#normalized-axis) `j` corresponds to input [axis](../../../../glossary.md#term-axis) `R - K + j`;
- equal static extents pass and unequal static extents fail locally;
- structurally equal dynamic or expression [Dimensions](../../../../glossary.md#dimension) pass;
- any other pair involving an unresolved Dimension is accepted with an equality obligation for
  compiler capture/binding, because the output Shape is still exactly derivable; and<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- the result retains the exact input Shape reference, including every Dimension reference.

The task-owned [rank](../../../../glossary.md#term-rank) and static mismatch failures are:

```text
layerNorm normalized rank must not exceed input rank: normalizedRank=<K>, inputRank=<R>
layerNorm normalized dimension mismatch at normalized axis <j>: input=<inputDimension>, normalized=<normalizedDimension>
```

The affine scale and [bias](../../../../glossary.md#term-bias) Shapes must each be structurally equal to `normalizedShape`; ordinary
broadcastability is deliberately insufficient. These checks include [scalar](../../../../glossary.md#term-scalar), rank, static, dynamic,
and expression structure and use exact failures:

```text
layerNorm scale Shape must equal normalizedShape: scale=<scaleShape>, normalizedShape=<normalizedShape>
layerNorm bias Shape must equal normalizedShape: bias=<biasShape>, normalizedShape=<normalizedShape>
```

No Shape object or caller array is copied. A static zero in `normalizedShape` is valid when the
corresponding input [extent](../../../../glossary.md#term-extent) is zero; the whole input/result then has zero [elements](../../../../glossary.md#term-element), so no
[normalization](../../../../glossary.md#term-normalization) [slice](../../../../glossary.md#slice) produces an output value and no zero divisor is evaluated. The same rule
applies when an unresolved normalized extent later binds to zero. A zero leading extent also makes
the result empty. Scalar input is invalid because a positive normalized rank cannot fit rank zero.

### Formula, correction, and [epsilon](../../../../glossary.md#epsilon)

For one non-empty trailing [slice](../../../../glossary.md#slice) containing `N` values `x_i`, use population correction zero:

```text
mean     = sum_i(x_i) / N
variance = sum_i((x_i - mean) * (x_i - mean)) / N
standardized_i = (x_i - mean) / sqrt(variance + epsilon)
```

The no-affine result is `standardized_i`. The affine result is:

```text
output_i = standardized_i * scale_i + bias_i
```

Scale and [bias](../../../../glossary.md#term-bias) [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) span the [normalized Shape](../../../../glossary.md#normalized-shape) in [row-major](../../../../glossary.md#term-row-major) logical coordinate order and
are reused for every leading slice. Epsilon is added to [population variance](../../../../glossary.md#population-variance) inside the [square
root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). There is no [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) correction, [variance](../../../../glossary.md#term-variance) clamp, denominator sentinel, [norm](../../../../glossary.md#term-norm) replacement,
or epsilon outside the root.

This trailing-[Shape](../../../../glossary.md#shape) and biased-variance contract follows the mainstream functional layer-
[normalization](../../../../glossary.md#term-normalization) [model](../../../../glossary.md#term-model) documented by [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
The first [normalized axis](../../../../glossary.md#normalized-axis) and standardization equations align with the official
[ONNX LayerNormalization operator](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>,
but Synaptik deliberately keeps saved [mean](../../../../glossary.md#term-mean)/inverse-standard-deviation outputs and optional bias
outside this first public model contract.

### Data types, promotion, and accumulation

No-affine [layer normalization](../../../../glossary.md#layer-normalization) accepts [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64) input and retains its exact
type. Affine layer normalization requires all three inputs to be floating and computes the result
type by applying `DataTypePromotion.promoteFloating` to `input` and `scale`, then that result and
[`bias`](../../../../glossary.md#term-bias), preserving ordered [occurrence](../../../../glossary.md#term-occurrence) semantics. Promotion records result [metadata](../../../../glossary.md#term-metadata) only and does
not insert casts or read values.

Exact type failures are:

```text
layerNorm input must have a floating data type, but was <dataType>
layerNorm scale must have a floating data type, but was <dataType>
layerNorm bias must have a floating data type, but was <dataType>
layerNorm epsilon data type must match result data type: epsilon=<epsilonType>, result=<resultType>
```

The exact [epsilon](../../../../glossary.md#epsilon) type must equal the derived result type. The standardized computation treats
the input as the result type before accumulation. BFLOAT16/FLOAT32 results compute [mean](../../../../glossary.md#term-mean) and
[variance](../../../../glossary.md#term-variance) in FLOAT32; FLOAT64 results compute them in FLOAT64. Affine multiply/add occurs in the
result type. A BFLOAT16 no-affine result therefore accumulates in FLOAT32 and converts the final
standardized value to BFLOAT16. No narrower [accumulator](../../../../glossary.md#accumulator-type) or hidden [model](../../../../glossary.md#term-model)-level result type is
allowed.

For [finite](../../../../glossary.md#term-finite) values, the semantic [target](../../../../glossary.md#term-target) is the stated real formula, rounded to the result format;
an implementation may use equal-or-wider intermediates, reassociate, parallelize, or use another
conforming [evaluation](../../../../glossary.md#term-evaluation) strategy only when it satisfies later [conformance](../../../../glossary.md#term-conformance) [tolerance](../../../../glossary.md#term-tolerance) and the exact
policies below. Model does not select an algorithm. Bitwise equality, fixed reduction traversal, [NaN
payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and cross-[backend](../../../../glossary.md#backend) identical [rounding](../../../../glossary.md#term-rounding) are not promised.

### [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), and deterministic classes

- Any NaN or either infinity anywhere in a non-empty normalized slice makes every standardized
  value in that slice NaN. Affine multiplication or addition does not suppress that NaN.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- Every finite constant slice, including any mixture of positive and negative zero, has positive-
  zero variance and exact positive-zero standardized values before affine transformation.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [variance glossary](../../../../glossary.md#term-variance))</span>
- For a nonconstant finite slice, variance is non-negative; epsilon is positive, so the
  denominator is positive. Accumulator overflow may make a centered value or denominator
  infinite: finite divided by positive infinity produces signed zero, while an indeterminate
  infinity divided by infinity produces NaN. Other finite overflow/underflow follows the selected
  accumulator and result formats.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [Accumulator glossary](../../../../glossary.md#accumulator-type); [overflow glossary](../../../../glossary.md#term-overflow); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [underflow glossary](../../../../glossary.md#term-underflow))</span>
- A no-affine exact zero is positive zero for a constant slice. In the affine form, a NaN scale or
  bias produces NaN at that coordinate; infinite scale times standardized zero produces NaN;
  infinite scale times a nonzero finite standardized value produces the corresponding signed
  infinity; finite product plus one infinite bias retains that bias infinity; and opposing
  infinities in the final addition produce NaN. Otherwise zero sign follows the selected floating
  multiply/add operations.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [bias glossary](../../../../glossary.md#term-bias); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Empty results contain no values and therefore expose no mean, variance, NaN, infinity, or signed-
  zero result.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean))</span>

These rules make special-value class behavior independent of physical traversal. [Finite](../../../../glossary.md#term-finite) non-zero
[rounding](../../../../glossary.md#term-rounding) remains subject to permitted [reassociation](../../../../glossary.md#term-reassociation) and later [tolerance](../../../../glossary.md#term-tolerance). Separately constructed
equal requests remain distinct [producers](../../../../glossary.md#term-producer); there is no interning or [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

### Result [metadata](../../../../glossary.md#term-metadata), [provenance](../../../../glossary.md#provenance), and [identity](../../../../glossary.md#term-identity-meanings) effects

The result [descriptor](../../../../glossary.md#term-descriptor) uses the promoted result type, exact input [Shape](../../../../glossary.md#shape) reference, unresolved
[layout](../../../../glossary.md#layout), and `requiresGrad` equal to input eligibility for the no-affine form or the logical OR of
input, scale, and [bias](../../../../glossary.md#term-bias) eligibility for the affine form. Every result is fresh, unlabeled, and
storage-free. Inputs and their labels, storage, provenance, IDs, and descriptors are not mutated.

Each success creates one [`Operation`](../../../../glossary.md#operation), one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one output descriptor, one [Tensor](../../../../glossary.md#tensor)
wrapper, and one Tensor ID. Provenance output index is zero and inputs are exactly `[input]` or
`[input, scale, bias]`. No [saved statistic](../../../../glossary.md#saved-statistic), sibling result, hidden descriptor, or additional ID is
created. A later [compiler](../../../../glossary.md#term-compiler) may derive saved values or an [adjoint](../../../../glossary.md#term-adjoint), but that does not retroactively
change this [producer](../../../../glossary.md#term-producer)'s output count.

### [Validation](../../../../glossary.md#term-validation) and construction order

`TensorLayerNormExpressions` exposes exactly two package-private entries corresponding to the two
public methods. The no-affine entry validates in this order:

1. null-check `input`, `normalizedShape`, then [`epsilon`](../../../../glossary.md#epsilon);
2. validate input floating eligibility;
3. require positive normalized [rank](../../../../glossary.md#term-rank) and validate it against input rank;
4. validate trailing static compatibility in normalized-[axis](../../../../glossary.md#term-axis) order;
5. validate epsilon exact result type;
6. construct `LayerNormAttrs`, [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), and delegate once.

The affine entry validates in this order:

1. null-check `input`, `normalizedShape`, `scale`, [`bias`](../../../../glossary.md#term-bias), then `epsilon`;
2. validate input, scale, then bias floating eligibility;
3. require positive normalized rank and validate it against input rank;
4. validate trailing static compatibility in normalized-axis order;
5. validate scale [Shape](../../../../glossary.md#shape), then bias Shape;
6. promote input/scale/bias in [occurrence](../../../../glossary.md#term-occurrence) order;
7. validate epsilon exact result type;
8. construct `AffineLayerNormAttrs`, descriptor, operation, and delegate once.

Null messages are [parameter](../../../../glossary.md#term-parameter) names. The attribute constructors then repeat their intrinsic
non-null/rank/epsilon validation defensively before retaining values. Existing Shape, promotion,
descriptor, operation, [producer](../../../../glossary.md#term-producer), and identifier failures retain their messages. Every local
failure occurs before factory delegation and consumes no [Tensor](../../../../glossary.md#tensor) ID, producer, or wrapper.

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
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar values](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [Matmul](0019-matmul-semantics-and-tensor-expression.md)
- [Explicit-state dropout](0019b1-explicit-graph-dropout-construction.md)<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- [Scaled dot-product attention](0019e-scaled-dot-product-attention.md)<span data-glossary-navigation="true"> ([Scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention))</span>
- [NCHW Conv2d](0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- [NCHW average pooling](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling))</span>

## Architecture constraints

- Work stays in [model](../../../../glossary.md#term-model) plus its documentation/[planning](../../../../glossary.md#term-planning). [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state,
  not graph intermediate representation.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [intermediate representation glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Normalization operation types record backend-independent semantic parameters only and do not
  import Tensor, graph, compiler, training, runtime, prepare, or backend types.<span data-glossary-navigation="true"> ([Normalization glossary](../../../../glossary.md#term-normalization); [backend glossary](../../../../glossary.md#backend); [parameters glossary](../../../../glossary.md#term-parameter); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Package direction is tensor helper to normalization operation/datatype/shape; no reverse edge,
  service, registry, or state owner is introduced.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [registry glossary](../../../../glossary.md#term-registry); [owner glossary](../../../../glossary.md#term-owner))</span>
- Compiler owns capture, deferred equality proof, binding validation, gradients, saved values,
  adjoints, and legal decomposition. Backend prepare owns conforming algorithms, lowering,
  specialization, fusion, kernels, and tolerance satisfaction. Runtime executes prepared work
  without consuming <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [kernels glossary](../../../../glossary.md#kernel); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>[`Operation`](../../../../glossary.md#operation) on its [hot path](../../../../glossary.md#term-hot-path).
- No architecture, focused-architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) change.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.normalization` — add layer-normalization semantics
  beside, without changing, softmax semantics.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.normalization.LayerNormKind` — public family-owned semantic [identity](../../../../glossary.md#term-identity-meanings)/[signatures](../../../../glossary.md#term-signature).
- `...operation.normalization.LayerNormAttrs` — public no-affine trailing-[Shape](../../../../glossary.md#shape)/[epsilon](../../../../glossary.md#epsilon) [parameters](../../../../glossary.md#term-parameter).
- `...operation.normalization.AffineLayerNormAttrs` — public affine trailing-Shape/epsilon
  parameters whose distinct class selects exact three-input cardinality.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- `...tensor.TensorLayerNormExpressions` — package-private validation, promotion, Shape, descriptor,
  and provenance construction owner.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [owner glossary](../../../../glossary.md#term-owner))</span>
- `...tensor.Tensor` — established public fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages where package-private access is required.

## Affected files

Production (5):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/AffineLayerNormAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signatures and public count 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 179 to 181.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 179 to 181.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime)/Training APIs; [softmax](../../../../glossary.md#softmax--log-softmax), reductions, typed-[scalar](../../../../glossary.md#term-scalar),
[Shape](../../../../glossary.md#shape), [operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature), [Tensor](../../../../glossary.md#tensor)/factory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), matmul, attention, [dropout](../../../../glossary.md#dropout--inverted-dropout),
[convolution](../../../../glossary.md#term-channels-first-convolution-layers), and [pooling](../../../../glossary.md#term-pooling) contracts; architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle);
other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 19 paths maximum: five production, seven test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths.
This one-path exception above the normal 18-path capability guardrail is required because the
existing exact-class [signature](../../../../glossary.md#term-signature) contract needs distinct public no-affine and affine attributes to
represent valid input counts one and three without accepting two; keeping both forms in one
semantic task prevents an unusable partial public capability. `Tensor.java` changes only imports,
two methods/Javadocs, and its [operation](../../../../glossary.md#operation) inventory. Five existing tests change only the stated
signatures/count. Stop for path 20, another type/test/document, an existing-helper change,
architecture, [Gradle](../../../../glossary.md#term-gradle), cross-[module](../../../../glossary.md#term-module) work, saved outputs, or a later task specification.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, both attrs records, helper, and Tensor methods: trailing normalized Shape,
  formula/correction/epsilon, affine roles, Shape constraints, promotion/accumulation, empty and
  special values, metadata/provenance/ID effects, validation/failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [epsilon glossary](../../../../glossary.md#epsilon); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every [parameter](../../../../glossary.md#term-parameter), non-void result, and expected failure has complete `@param`, `@return`, and
  `@throws` tags.
- Tensor API gets the two exact signatures, a Shape/operand table, one finite numerical example,
  one static mismatch example, symbolic deferral, empty/special policy, and current-model versus
  planned compiler/execution boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [signatures glossary](../../../../glossary.md#term-signature); [finite glossary](../../../../glossary.md#term-finite); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compile API records layer-normalization metadata as current after implementation and leaves
  capture, constraint proof, saved values, gradients, legal decomposition, and execution planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [normalization glossary](../../../../glossary.md#term-normalization); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Review glossary terms normalization, layer normalization, normalized Shape, affine transform,
  epsilon, population variance, accumulator, producer, and saved statistic; add/refine only
  reusable distinctions required by the public explanation.<span data-glossary-navigation="true"> ([layer normalization glossary](../../../../glossary.md#layer-normalization); [affine transform glossary](../../../../glossary.md#affine-transform); [population variance glossary](../../../../glossary.md#population-variance); [accumulator glossary](../../../../glossary.md#accumulator-type); [producer glossary](../../../../glossary.md#term-producer); [saved statistic glossary](../../../../glossary.md#saved-statistic))</span>
- Synchronize planning status: 0021 is Ready during implementation and Complete only after every
  criterion passes; 0021A–0021C and 0022–0024 remain Draft without detailed specifications.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact `LAYER_NORM`, both attributes/signature variants, and two canonical receiver methods exist;
  softmax remains unchanged and public Tensor method count is 181.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [canonical glossary](../../../../glossary.md#term-canonical-form); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Non-empty trailing normalized Shape, exact scale/bias Shape, static mismatch rejection, symbolic
  equality deferral, exact input Shape retention, and empty-result rules match this task.<span data-glossary-navigation="true"> ([normalized Shape glossary](../../../../glossary.md#normalized-shape); [bias glossary](../../../../glossary.md#term-bias); [Shape glossary](../../../../glossary.md#shape))</span>
- No-affine exact type retention and affine ordered promotion pass; epsilon is exact-result-typed,
  floating, finite, and positive; selected accumulator types are API-locked without evaluation.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [finite glossary](../../../../glossary.md#term-finite); [accumulator glossary](../../../../glossary.md#accumulator-type); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Population correction zero, epsilon placement, affine formula, NaN/infinity/signed-zero,
  overflow, reassociation, rounding, and determinism policies are explicit and consistent.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [reassociation glossary](../../../../glossary.md#term-reassociation); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Requires-grad, unresolved layout, freshness, no label/storage, no mutation, exact input order,
  one output/index zero, and exact one-ID/producer/wrapper effects pass.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Validation](../../../../glossary.md#term-validation) order and exact task-owned messages pass; every pre-factory failure consumes no ID.
- No axes/options overload, default/untyped epsilon, partial affine form, hidden constants, saved
  outputs, state/mode, gradient/compiler/algorithm/backend/runtime, architecture, dependency,
  build, cross-module, or later-spec work is added.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [constants glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  official-link, exact 19 paths/packages/public surface/statuses, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.LayerNormSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.datatype.ScalarValueTest --tests io.github.pho001.synaptik.model.shape.ShapeTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact attrs pairing/[cardinality](../../../../glossary.md#term-cardinality), public surface, intrinsic [epsilon](../../../../glossary.md#epsilon) validation,
both formulas as inspectable semantic contracts, static/symbolic/empty [Shapes](../../../../glossary.md#shape), affine Shape,
promotion/[accumulator](../../../../glossary.md#accumulator-type)/special-value policy, validation/ID effects, [metadata](../../../../glossary.md#term-metadata), freshness, and exact
[provenance](../../../../glossary.md#provenance) without value execution.

Documentation pass after final [Javadoc](../../../../glossary.md#term-javadoc):

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation-focused pass also runs the repository's existing or targeted Markdown
link/anchor/fence/final-newline checks, verifies both official documentation links, audits exact
19-path [scope](../../../../glossary.md#term-scope) and package placement, checks public method count/[signatures](../../../../glossary.md#term-signature) and synchronized
statuses, and records reused Java evidence. Repository-wide validation is deferred to the selected
modern-[operation](../../../../glossary.md#operation) [checkpoint](../../../../glossary.md#term-checkpoint) after task 0022 and [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [module](../../../../glossary.md#term-module) without
dependencies, architecture, shared build configuration, or executable [backend](../../../../glossary.md#backend) behavior.

## Dependencies

- 0001–0002: [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), [axis](../../../../glossary.md#term-axis)/[rank](../../../../glossary.md#term-rank)/static/symbolic value contracts.
- 0005–0007 and 0011–0013: [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), factory, and [provenance](../../../../glossary.md#provenance) foundations.
- 0018K: exact attributes-class [signatures](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality).
- 0018L: shared [producer](../../../../glossary.md#term-producer)/output-index provenance, used here in its one-output form.
- 0018N: exact typed [scalar](../../../../glossary.md#term-scalar) representation for [epsilon](../../../../glossary.md#epsilon).
- 0018V: [population variance](../../../../glossary.md#population-variance), accumulation, empty, correction, and special-value precedents.
- 0019, 0019E, 0019B1, and 0020–0020A1: current promotion, high-level semantic, explicit-state,
  validation, and numerical-policy precedents.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

All dependencies are Complete.

## Follow-up tasks

- 0021A (Draft) — RMS normalization as a distinct one-output high-level semantic operation. It
  uses root-mean-square rather than centered variance and will independently decide its explicit
  scale-only/no-affine surface; it does not reuse layer-normalization attrs.<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../../glossary.md#rms-normalization); [operation glossary](../../../../glossary.md#operation); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- 0021B (Draft) — batch-normalization inference with explicit running mean/variance and affine
  inputs, one output, no hidden training flag, mutation, or saved outputs.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../../glossary.md#term-inference))</span>
- 0021C (Draft) — batch-normalization training as an explicit state transition after inference
  semantics: batch statistics, running-stat inputs/outputs, and saved statistics use genuine
  multi-output provenance without a stateful service.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- 0022–0024 remain Draft at their established IDs. Task 0023 owns any normalization adjoint or
  compiler-generated saved-statistic semantic operation justified by compiler implementation.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [compiler glossary](../../../../glossary.md#term-compiler); [statistic glossary](../../../../glossary.md#term-statistic))</span>

## Architecture impact

Expected impact: None.

The task adds [model](../../../../glossary.md#term-model)-owned semantic values and public expression [metadata](../../../../glossary.md#term-metadata) inside existing package
and [dependency](../../../../glossary.md#term-dependency) boundaries. If implementation requires hidden mode/state, a cross-[module](../../../../glossary.md#term-module) type,
architecture/dependency change, or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md.
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
focused/final Java commands and results, affected layer-[normalization](../../../../glossary.md#term-normalization) semantics/public methods,
architecture and 19-path constraints, expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) and glossary/[planning](../../../../glossary.md#term-planning) impact,
official links, and required [Javadoc](../../../../glossary.md#term-javadoc)/Markdown/[scope](../../../../glossary.md#term-scope)/status/whitespace evidence. It must inspect the
actual source/tests, not rely only on the summary, and record reasoned no-change conclusions.

## Local decisions

- Implemented the planned two exact attributes classes rather than weakening [signature](../../../../glossary.md#term-signature) matching:
  `LayerNormAttrs` selects one input and `AffineLayerNormAttrs` selects exactly three.
- Reused the task's ordered floating promotion and exact-typed epsilon rules without inserting
  casts or hidden constants. The result always retains the exact input Shape reference.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [constants glossary](../../../../glossary.md#term-constant); [Shape glossary](../../../../glossary.md#shape))</span>
- Kept all deferred unresolved-dimension equality as metadata obligation only. No model constraint
  object, saved output, gradient rule, or execution mechanism was introduced.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [metadata glossary](../../../../glossary.md#term-metadata); [model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Documentation uses “normalized Shape” for the exact trailing Shape and keeps it distinct from a
  normalized integer axis. The glossary adds only reusable layer-normalization distinctions.<span data-glossary-navigation="true"> ([normalized Shape glossary](../../../../glossary.md#normalized-shape); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Known limitations

- Model construction records layer-normalization meaning and metadata but does not evaluate
  values. Compiler capture, deferred-constraint proof, saved-statistic lifetime, gradients or
  adjoints, and legal decomposition remain planned.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [normalization glossary](../../../../glossary.md#term-normalization); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [statistic glossary](../../../../glossary.md#term-statistic); [lifetime glossary](../../../../glossary.md#term-lifetime); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Backend preparation, algorithms, numerical tolerances, kernels, and runtime execution remain
  outside this task. Finite non-zero results therefore have no current execution evidence.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [Finite glossary](../../../../glossary.md#term-finite))</span>
- RMS normalization, batch-normalization inference/training, loss operations, and compiler-
  generated normalization semantics remain Draft tasks 0021A–0023 without detailed specs.<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../../glossary.md#rms-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../../glossary.md#term-inference); [loss glossary](../../../../glossary.md#loss))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0021_implementation` ran the required focused command. Its
  first run failed only because one new assertion expected the abbreviated diagnostic <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`input=3`
  while the established [`Dimension`](../../../../glossary.md#dimension) rendering is `input=StaticDimension[size=3]`. After correcting
  that test expectation, the exact required focused command passed 42 tests with `BUILD
  SUCCESSFUL`.
- After executable Java stabilized, implementation context `/root/task_0021_implementation` ran
  exactly one `./gradlew :modules:model:test`; it passed with `BUILD SUCCESSFUL`. No executable
  Java changed afterward. Documentation-only Javadoc edits do not invalidate that evidence.
- Documentation context `/root/task_0021_implementation/task_0021_docs` applied General, API and
  Javadoc, Planning, and Example profiles; independently reviewed the architecture contract,
  planning contracts, task, complete implementation diff, source, tests, public APIs, and glossary;
  and finalized the five affected production Javadocs plus all seven documentation/planning paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- `./gradlew :modules:model:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits with `BUILD SUCCESSFUL`.
- A targeted Ruby check passed for all seven changed Markdown files: every local link target and
  heading anchor resolved, backtick/tilde fences were balanced, and every file had a final newline.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Official-link checks passed with HTTP 200 for both the PyTorch LayerNorm and ONNX
  LayerNormalization pages.<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>
- Exact-scope audit combined tracked and untracked paths and passed with exactly the prescribed 19
  paths. Package/import review found only the planned normalization, tensor, datatype, shape,
  operation, and JDK dependencies; no new package or cross-module edge exists.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [normalization glossary](../../../../glossary.md#term-normalization); [tensor glossary](../../../../glossary.md#tensor); [shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [module glossary](../../../../glossary.md#term-module))</span>
- `javap` public-surface audit passed with 181 public declared Tensor methods and exactly two
  non-static `layerNorm` signatures. Source and test inspection confirmed no static entry,
  axes/options/default-epsilon overload, partial affine form, saved output, or hidden sibling.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [axes glossary](../../../../glossary.md#term-axis); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Status audit confirmed task 0021 and its synchronized master-plan/roadmap rows are Complete;
  0021A–0021C and 0022–0024 remain Draft and have no detailed specification files.
- `git diff --check` passed on the final combined change. No repository-wide suite was run because
  this single-module task changes no dependency, architecture, shared build, or executable backend
  contract; the modern-operation checkpoint after 0022 and CI retain that responsibility.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Added `LayerNormKind.LAYER_NORM` with ordered exact [signatures](../../../../glossary.md#term-signature)
  `LayerNormAttrs: 1 -> 1` and `AffineLayerNormAttrs: 3 -> 1`.
- Added defensive intrinsic Shape/epsilon attributes validation and one package-private Tensor
  construction helper with the specified validation order, type promotion, exact Shape checks,
  descriptor derivation, factory delegation, freshness, and provenance.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Added the two exact public receiver methods. Focused tests cover signatures, attributes,
  formulas, static/symbolic/empty Shapes, promotion, metadata, validation order and diagnostics,
  identity effects, freshness, and exact one-output provenance. Existing public-count tests were
  updated from 179 to 181 only.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) now contains exact signatures, an operand/[Shape](../../../../glossary.md#shape) table, a [finite](../../../../glossary.md#term-finite) `[1, 2, 3]` example,
  static mismatch, symbolic deferral, empty/special-value rules, provenance, and lifecycle
  boundaries. Compile API identifies layer-normalization metadata as current while keeping
  capture, proof, saved values, gradients, decomposition, lowering, and execution planned.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Compile glossary](../../../../glossary.md#compile); [normalization glossary](../../../../glossary.md#term-normalization); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Runtime and Training APIs remain unchanged because this task adds no prepared/run contract,
  executable state, training mode, parameter ownership, gradient API, or optimizer behavior.
  Related softmax, reduction, typed-scalar, Shape, operation/signature, Tensor/factory/producer,
  matmul, attention, dropout, convolution, and pooling contracts remain accurate because their
  public behavior did not change. Architecture/ADRs/tests remain unchanged because ownership and
  dependency rules did not change. Backend conformance and integration remain unchanged because no
  executable behavior exists. Gradle, other modules, and later task specs remain unchanged because
  the capability stays within <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [optimizer glossary](../../../../glossary.md#term-optimizer); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [dependency glossary](../../../../glossary.md#term-dependency); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>`modules/model` and its existing build/package boundaries.

## Completion summary

- Completed changes: added exact no-affine and affine layer-normalization semantics, public Tensor
  expression construction, validation, result/provenance contracts, focused tests, complete
  Javadocs, API reference, glossary distinctions, and synchronized planning status.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Tensor glossary](../../../../glossary.md#tensor); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the five production, seven test, and seven
  documentation/planning paths listed in this task; no additional path changed.
- Tests and validation: required focused command passed 42 tests after one corrected diagnostic
  expectation; exactly one final model suite passed; final model Javadoc, Markdown
  links/anchors/fences/newlines, official links, package/scope/public-surface/status audits, and<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: completed by clean context
  `/root/task_0021_implementation/task_0021_docs` using the required profiles and reused Java test
  evidence because it changed documentation only.
- Documentation impact: Tensor API, Compile API, glossary, capability baseline, master plan,
  roadmap, and this task are finalized. Runtime/Training APIs and related/architecture/execution/
  build documentation need no change for the reasons recorded above.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: finalized kind, both attributes records, package-private helper, both Tensor
  methods, and Tensor's operation inventory; Javadoc generation passes.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Glossary impact: added layer normalization, normalized Shape, population variance, epsilon,
  affine transform, accumulator type, and saved statistic as reusable public distinctions.<span data-glossary-navigation="true"> ([layer normalization glossary](../../../../glossary.md#layer-normalization); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [population variance glossary](../../../../glossary.md#population-variance); [epsilon glossary](../../../../glossary.md#epsilon); [affine transform glossary](../../../../glossary.md#affine-transform); [accumulator type glossary](../../../../glossary.md#accumulator-type); [saved statistic glossary](../../../../glossary.md#saved-statistic))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0021. Draft tasks 0021A–0024 remain separately scoped future
  work.

Status: Complete
