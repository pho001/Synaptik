# Task 0020A1: [NCHW](../../../../glossary.md#term-nchw) Average Pool2d Semantics and [Tensor](../../../../glossary.md#tensor) Expression

**Editorial reading key (current context, not a status update).**

This record concerns the original rank-four NCHW average-pooling Model expression, not the broader
current pooling surface. NCHW means batch, channel, height, width. The caller supplies a Tensor
receiver and immutable geometry to `averagePool2d`; the helper validates descriptions and delegates
result construction to `TensorFactory.createDerived`. Provenance records the input and producing
operation, not computed samples. The fresh storage-free result describes an average without
executing it or owning a result buffer; semantic responsibility is not storage ownership.

Count-padding means every logical kernel position enters the fixed divisor: one value of one and
three padding positions give a quarter, not one. Dilation spaces samples without changing their
count; literal ceil mode retains terminal all-padding windows, whose mean is positive zero. Follow
[current average-pooling
semantics](../../../../reference/tensor/pooling.md#nchw-average-pooling-expressions) for the
expanded floating-type scope, and [numerical and metadata
policy](../../../../reference/tensor/pooling.md#numerical-empty-and-metadata-policy) for special
values and qualified compiler/execution boundaries. Model defines meaning; execution providers
realize supported routes rather than redefine it. Historical deferrals and evidence remain
historical.

**Mathematical reading illustration.** In a finite real window of $K>0$ logical kernel positions,
including conceptual zero padding, define $a=\frac1K\sum_{j=1}^{K}z_j$, where $z_j$ is the input
sample or padding zero. With $K=4$ and $z=(1,0,0,0)$, $a=1/4$. Dividing by the single in-bounds
sample instead would describe a different operation. The illustration does not override floating
rounding or special-value rules.

## Status

Complete

## Goal

Add one first-class [backend](../../../../glossary.md#backend)-independent `AVERAGE_POOL2D` [operation](../../../../glossary.md#operation) and exactly one public receiver
expression for floating [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) [tensors](../../../../glossary.md#tensor). The result must retain exact [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[channel](../../../../glossary.md#term-channel)
[dimensions](../../../../glossary.md#dimension), derive truthful static or symbolic [spatial](../../../../glossary.md#term-spatial-axes) dimensions, and record one fixed
count-[padding](../../../../glossary.md#padding) average, accumulation, special-value, [metadata](../../../../glossary.md#term-metadata), and [provenance](../../../../glossary.md#provenance) contract without
evaluating values.

This task completes the average-[pooling](../../../../glossary.md#term-pooling) half of the former combined pooling frontier. It follows
completed task 0020A's NCHW [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and literal floor/ceiling grid, but owns a distinct
attributes type because average divisor and accumulation meaning are not max-[extrema](../../../../glossary.md#saturation-pole-and-extremum) [parameters](../../../../glossary.md#term-parameter).

## [Scope](../../../../glossary.md#term-scope)

- Extend `Pool2dKind` with `AVERAGE_POOL2D`, accepting only `AveragePool2dAttrs`, exactly one input,
  and exactly one output. Preserve `MAX_POOL2D` and its [signature](../../../../glossary.md#term-signature) unchanged.
- Add [immutable](../../../../glossary.md#term-immutable) public `AveragePool2dAttrs(kernelHeight, kernelWidth, strideHeight, strideWidth,
  paddingHeight, paddingWidth, dilationHeight, dilationWidth, ceilMode)` using `long` geometry.
- Add exactly `public Tensor averagePool2d(AveragePool2dAttrs attrs)`; add no short [alias](../../../../glossary.md#aliasing-and-arena).
- Add one package-private stateless `TensorAveragePool2dExpressions` helper and one final
  `TensorFactory.createDerived` call for each successful expression.
- Require floating [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) input `[N, C, H, W]` and preserve exact input type, batch,
  channel, and gradient-request metadata.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Derive floor- or ceil-mode static and symbolic spatial extents with the exact completed-0020A
  literal-grid formulas and checked arithmetic.<span data-glossary-navigation="true"> ([spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- For every output window, use the fixed divisor `kernelHeight * kernelWidth`. Every logical kernel
  position counts once. An out-of-bounds position contributes conceptual positive zero to the
  numerator and one count to the denominator.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Select exact accumulation, empty/all-padding, NaN, infinity, signed-zero, and determinism
  policies.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Produce one fresh unlabeled, storage-free result with unresolved layout and exact one-input,
  output-index-zero provenance.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update focused API, glossary, capability, and planning documentation plus public-Tensor
  inventory locks.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- maximum, global, adaptive, one-dimensional, three-dimensional, or [channels-last](../../../../glossary.md#spatial-axes-channels-and-window-sampling) [pooling](../../../../glossary.md#term-pooling)
- configurable count-padding, excluded-padding averages, divisor overrides, asymmetric or
  automatic padding, padding Tensor inputs, configurable padding values, runtime geometry, or a
  broad options framework<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [Tensor glossary](../../../../glossary.md#tensor); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- integral, BOOL, quantized, sparse, complex, unsigned, FLOAT16, or mixed-type input; casts,
  promotion, output-type overrides, or accumulator options<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- terminal-window removal in ceil mode; average pooling uses task 0020A's literal symmetric
  padded grid, including a terminal all-padding window
- reuse or modification of `MaxPool2dAttrs` or `Window2dAttrs`; their extrema and conceptual
  window-transform meanings are not average-divisor semantics<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  input materialization<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- gradients, adjoints, saved values, backward kinds, compiler capture, graph-wide constraint
  solving, canonicalization, decomposition, fusion, or optimization<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- summation or division algorithms, compensated summation, backend capabilities, lowering,
  prepare, runtime, execution, tolerances, conformance, or integration<span data-glossary-navigation="true"> ([compensated summation glossary](../../../../glossary.md#term-compensated-summation); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to Conv2d, max pooling, unfold/fold, pad, aggregate reductions, symbolic-expression
  foundations, promotion, provenance/factory seams, architecture, dependencies, Gradle, or
  another module<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- later task specifications, normalization, loss, compiler-generated semantic work, adaptive or
  global pooling<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Public surface, attributes, and [signature](../../../../glossary.md#term-signature)

The sole new receiver method is:

```java
public Tensor averagePool2d(AveragePool2dAttrs attrs)
```

The full word `average` is the [canonical](../../../../glossary.md#term-canonical-form) public spelling. Do not add `avgPool2d`, overloads,
defaults, static entries, or compatibility [aliases](../../../../glossary.md#aliasing-and-arena).

`AveragePool2dAttrs` validates components in declaration order. [Kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), and [dilation](../../../../glossary.md#term-dilation)
components are positive; [padding](../../../../glossary.md#padding) components are non-negative. Exact constructor failures are:

```text
kernelHeight must be positive: <value>
kernelWidth must be positive: <value>
strideHeight must be positive: <value>
strideWidth must be positive: <value>
paddingHeight must be non-negative: <value>
paddingWidth must be non-negative: <value>
dilationHeight must be positive: <value>
dilationWidth must be positive: <value>
```

`ceilMode` is retained unchanged. The record owns intrinsic average-[pooling](../../../../glossary.md#term-pooling) geometry only. The
fixed count-padding/divisor policy is part of `AVERAGE_POOL2D` meaning and is not configurable
state, so the record has no `countIncludePad`, `divisorOverride`, or padding-value component. It
owns no [Tensor](../../../../glossary.md#tensor), [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype), [layout](../../../../glossary.md#layout), storage, [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), or execution state.

`Pool2dKind` retains exact declaration order `MAX_POOL2D`, `AVERAGE_POOL2D`. Each kind has one
family-owned fixed one-input/one-output signature paired only with its own attributes class. A max
attrs value must not construct average pooling, and an average attrs value must not construct max
pooling.

### [NCHW](../../../../glossary.md#term-nchw) [Shape](../../../../glossary.md#shape) and literal [spatial](../../../../glossary.md#term-spatial-axes) windows

For each [spatial axis](../../../../glossary.md#spatial-axes-channels-and-window-sampling), let input [extent](../../../../glossary.md#term-extent) be `D`, positive [kernel](../../../../glossary.md#kernel) [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) count be `k`, symmetric
[padding](../../../../glossary.md#padding) per side be `p`, positive [dilation](../../../../glossary.md#term-dilation) be `d`, and positive [stride](../../../../glossary.md#term-stride) be `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = D + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

All literal arithmetic uses checked `long` operations. For static `D`, reject a negative
numerator before division. For unresolved `D`, use existing `DimensionExpressions.addConstant`,
then `floorDivide` or `ceilingDivide`, then `addConstant`; the result retains the [binding](../../../../glossary.md#term-binding)
obligation that the numerator is non-negative. Do not require static spatial input, create an
unknown extent, clamp a negative numerator to zero, add a terminal-window correction, or modify
the symbolic-expression [model](../../../../glossary.md#term-model).

The output Shape is `[N, C, H_out, W_out]`. Preserve the exact input `N` and `C` [Dimension](../../../../glossary.md#dimension)
references. Along an [axis](../../../../glossary.md#term-axis), output position `o` starts at `o * s - p` and samples positions
`start + r * d` for `r` from zero through `k - 1`. Floor mode includes every window in the floor
grid. Ceil mode includes every window in the literal ceiling grid, including a final window whose
start lies entirely in trailing padding. These formulas and coordinates intentionally match
completed task 0020A; only the value policy differs.

Exact task-owned Shape failures are:

```text
averagePool2d input rank must be 4: <rank>
averagePool2d effective kernel does not fit padded height: input=<dimension>, effectiveKernel=<value>, padding=<value>
averagePool2d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

Static zero [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) or [channel](../../../../glossary.md#term-channel) extents are valid and make the result empty while retaining spatial
[metadata](../../../../glossary.md#term-metadata). A static zero height or width is valid only when padding makes the corresponding
numerator non-negative. Valid static geometry always produces at least one spatial position;
some or all windows may contain no in-bounds samples. Unresolved geometry is accepted with the
same non-negative-numerator binding obligation.

### [Data type](../../../../glossary.md#data-type--datatype), divisor, and numerical policy

The sole input must be [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64). The result retains the exact input type; no
input promotion occurs. The exact type failure is:

```text
averagePool2d input must have a floating data type, but was <dataType>
```

For each output window, there are exactly `kernelHeight * kernelWidth` logical [kernel](../../../../glossary.md#kernel) positions;
[dilation](../../../../glossary.md#term-dilation) changes their [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), not their count. Every in-bounds position contributes its
input value to the numerator. Every out-of-bounds position contributes conceptual exact [positive
zero](../../../../glossary.md#nan-infinity-and-signed-zero). Every logical position contributes one to the denominator, so the divisor is always the
positive mathematical product `kernelHeight * kernelWidth`. [Padding](../../../../glossary.md#padding) therefore contributes zero
to the numerator and contributes to the denominator. No [runtime](../../../../glossary.md#term-runtime) count, valid-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) count,
divisor override, or zero-divisor case exists. The mathematical product need not be materialized
as a new `long` attribute during [model](../../../../glossary.md#term-model) construction.

BFLOAT16 and FLOAT32 inputs accumulate and divide in FLOAT32; FLOAT64 input accumulates and divides
in FLOAT64. BFLOAT16 converts the final [accumulator](../../../../glossary.md#accumulator-type) result to BFLOAT16. The [operation](../../../../glossary.md#operation)'s selected
meaning is the accumulator-domain sum divided once by the exact positive divisor. [Reassociation](../../../../glossary.md#term-reassociation),
tree reduction, and other conforming summation order are permitted, so fixed traversal order,
bitwise equality, [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and cross-[backend](../../../../glossary.md#backend) identical [rounding](../../../../glossary.md#term-rounding) are not
promised. Later backend [conformance](../../../../glossary.md#term-conformance) owns accuracy tolerances; this task does not select an
algorithm or evaluate values.

The observable special-value classes are fixed:

- any in-bounds [NaN](../../../../glossary.md#term-nan) makes the result NaN;
- positive and [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) both present in one window make the result NaN;
- otherwise any positive infinity makes the result positive infinity and any negative infinity
  makes it negative infinity;<span data-glossary-navigation="true"> ([positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- conceptual padding never introduces NaN or [infinity](../../../../glossary.md#term-infinity);
- an exact-zero finite mean is negative zero only when every divisor contribution is an in-bounds
  negative zero; cancellation, any positive zero, or any conceptual padding produces positive
  zero; and<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- an all-padding window returns exact positive zero in the result type.

Empty [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[channel](../../../../glossary.md#term-channel) outputs contain no values. A [spatial](../../../../glossary.md#term-spatial-axes) window is never divided by zero because
both kernel counts are positive. The special-value and zero rules make observable results
independent of physical traversal; [finite](../../../../glossary.md#term-finite) non-zero rounding remains subject to the permitted
reassociation and later conformance [tolerance](../../../../glossary.md#term-tolerance).

### [Validation](../../../../glossary.md#term-validation) and construction order

The helper validates before its sole factory call in this order:

1. null-check `input`, then `attrs`;
2. validate input floating eligibility;
3. validate input [rank](../../../../glossary.md#term-rank);
4. derive height, then width, including checked effective-kernel, padding, numerator, and
   floor/ceil arithmetic;<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [padding glossary](../../../../glossary.md#padding))</span>
5. create the result [descriptor](../../../../glossary.md#term-descriptor) and exact `AVERAGE_POOL2D` [operation](../../../../glossary.md#operation); and
6. delegate once with exact ordered input `[input]`.

Null messages are [parameter](../../../../glossary.md#term-parameter) names. Attribute construction validates its fields before the helper
is called. Checked-arithmetic and existing symbolic-expression failures retain their existing
messages. There is no separate divisor validation or divisor-[overflow](../../../../glossary.md#term-overflow) message. Every failure
before final delegation consumes no [Tensor](../../../../glossary.md#tensor) ID, [producer](../../../../glossary.md#term-producer), or wrapper. A successful call consumes
exactly one Tensor ID, creates one producer, and creates one output wrapper.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Unfold/fold semantics](0017m-unfold-and-fold-semantics.md)
- [Unfold/fold expressions](0017n-unfold-and-fold-tensor-expressions.md)
- [Signature hardening](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Signature glossary](../../../../glossary.md#term-signature))</span>
- [Shared provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- [Symbolic extents](0018m-symbolic-extent-expressions.md)
- [Typed scalar](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Reduction policies](0018v-multi-axis-and-statistical-reductions.md)
- [NCHW Conv2d](0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- [NCHW max pooling](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling))</span>

## Architecture constraints

- Work stays in model plus its documentation/planning. Tensor remains public mutable state, not
  graph IR.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [mutable glossary](../../../../glossary.md#term-mutable); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Pooling operation types record backend-independent meaning only and do not import Tensor,
  graph, compiler, runtime, prepare, or backend types.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Package direction is [tensor](../../../../glossary.md#tensor) helper to [pooling](../../../../glossary.md#term-pooling) [operation](../../../../glossary.md#operation)/datatype/[shape](../../../../glossary.md#shape); packages remain acyclic.
- Compiler owns capture, binding/deferred proof, gradients, adjoints, and saved values. Backend
  prepare owns conforming algorithms, lowering, kernels, tolerance satisfaction, and
  materialization. Runtime receives prepared work without original operations on its hot path.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [gradients glossary](../../../../glossary.md#term-gradient); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [tolerance glossary](../../../../glossary.md#term-tolerance); [materialization glossary](../../../../glossary.md#materialization); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- No architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), focused-architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) change.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.pooling` — extend the existing public pooling kind
  and add average-specific immutable attributes without weakening max pooling.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.pooling.Pool2dKind` — existing pooling identity/signature owner extended with
  exact <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [signature glossary](../../../../glossary.md#term-signature); [owner glossary](../../../../glossary.md#term-owner))</span>`AVERAGE_POOL2D` pairing.
- `...operation.pooling.AveragePool2dAttrs` — public inspectable average-window geometry; its
  dedicated type prevents max extrema and average divisor contracts from being interchanged.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- `...tensor.TensorAveragePool2dExpressions` — package-private validation, Shape, descriptor, and
  provenance construction owner.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `...tensor.Tensor` — established public fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/AveragePool2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool2dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool2dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signature and public count 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 178 to 179.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 178 to 179.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime)/Training APIs; [Conv2d](../../../../glossary.md#convolution--conv2d), max [pooling](../../../../glossary.md#term-pooling), unfold/fold, pad,
reduction, [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension)-expression, [operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature), [Tensor](../../../../glossary.md#tensor)/factory/[provenance](../../../../glossary.md#provenance) contracts;
architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 18 paths maximum: four production, seven test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths.
`Tensor.java` changes only one import, one method/[Javadoc](../../../../glossary.md#term-javadoc), and its [operation](../../../../glossary.md#operation) inventory. Five
existing tests change only the stated [signature](../../../../glossary.md#term-signature)/count. `Pool2dSemanticsTest` may cover both exact
attrs pairings while preserving all max assertions. Stop for path 19, another type/test/document,
an existing-helper change, architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) work.

This is one cohesive vertical [model](../../../../glossary.md#term-model) capability. Kind, average-specific attrs, public construction,
symbolic [Shape](../../../../glossary.md#shape) derivation, fixed divisor/numerical meaning, [API](../../../../glossary.md#term-application-programming-interface-api) locks, and documentation must agree
in one compilable state.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, attrs, helper, and Tensor method: canonical public spelling, NCHW window
  meaning, Shape formulas, literal ceil grid, fixed count-padding divisor, accumulation,
  special/empty policy, metadata/provenance, validation/failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [canonical glossary](../../../../glossary.md#term-canonical-form); [NCHW glossary](../../../../glossary.md#term-nchw); [Shape glossary](../../../../glossary.md#shape); [padding glossary](../../../../glossary.md#padding); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every [parameter](../../../../glossary.md#term-parameter), result, and expected failure has complete `@param`, `@return`, and `@throws`
  tags as applicable.
- Tensor API gets a geometry/divisor table, static floor/ceil and symbolic examples, one numerical
  example distinguishing count-padding from valid-sample averaging, special-value policy, and
  current-model versus planned compiler/execution boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compile API records current average-pooling metadata and future compiler-owned binding,
  capture, legal decomposition, and gradients without claiming compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [pooling glossary](../../../../glossary.md#term-pooling); [binding glossary](../../../../glossary.md#term-binding); [capture glossary](../../../../glossary.md#term-capture-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Review glossary terms NCHW, pooling, window, padding, dilation, effective kernel, ceil mode,
  count-padding, divisor, and accumulation; add or refine only reusable distinctions needed by the
  public explanation.<span data-glossary-navigation="true"> ([dilation glossary](../../../../glossary.md#term-dilation); [kernel glossary](../../../../glossary.md#kernel))</span>
- Synchronize capability/task/master/roadmap statuses: 0020 and 0020A remain Complete; keep
  0020A1 Ready during implementation and mark it Complete only after every acceptance criterion
  passes; 0021–0024 remain Draft without detailed specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact `AVERAGE_POOL2D` kind, dedicated attrs, one-input/one-output signature, and one canonical
  receiver method exist; max pooling remains unchanged; public Tensor method count is 179.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [canonical glossary](../../../../glossary.md#term-canonical-form); [pooling glossary](../../../../glossary.md#term-pooling); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Static and symbolic floor/ceil Shape derivation, exact N/C reference retention, checked
  geometry, literal terminal windows, and deferred unresolved obligation match this task and
  completed 0020A.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- BFLOAT16/FLOAT32/FLOAT64 retain exact type; BOOL/integral fail; selected accumulator types,
  fixed kernel-count divisor, conceptual padding zeros, and one final division are documented and
  API-locked without evaluation.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BOOL glossary](../../../../glossary.md#data-type--datatype); [accumulator glossary](../../../../glossary.md#accumulator-type); [kernel glossary](../../../../glossary.md#kernel); [padding glossary](../../../../glossary.md#padding); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- NaN, opposing/single infinity, exact signed-zero selection, all-padding positive zero, empty
  axes, reassociation, rounding, and determinism policies are explicit.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [axes glossary](../../../../glossary.md#term-axis); [reassociation glossary](../../../../glossary.md#term-reassociation); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Requires-grad is retained exactly; layout is unresolved; result is fresh, unlabeled, and
  storage-free.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Validation order, exact task-owned failures, no-ID failures, one-ID/one-producer/one-wrapper
  success, and exact one-input/output-zero provenance pass.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- No count-padding flag, valid-sample divisor, divisor override, alias/overload, max/window attrs
  weakening, gradient, compiler, algorithm, backend/runtime, architecture, dependency, build, or
  later-spec work is added.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [alias glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 18 paths/packages/public surface/statuses, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.pooling.Pool2dSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorAveragePool2dExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

The focused tests cover both exact attrs/[signature](../../../../glossary.md#term-signature) pairings, public surface, all static/symbolic
floor/ceil cases, literal terminal/all-[padding](../../../../glossary.md#padding) windows, type/[rank](../../../../glossary.md#term-rank)/geometry failures, [accumulator](../../../../glossary.md#accumulator-type)/
divisor/special-value contracts as inspectable meaning and [Javadoc](../../../../glossary.md#term-javadoc), [metadata](../../../../glossary.md#term-metadata), validation/ID
effects, freshness, and [provenance](../../../../glossary.md#provenance) without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links and anchors, fences, terminology, examples, generated Javadoc,
final newlines and whitespace, exact paths/packages, public count/signature, exact kind/signature
pairing, synchronized statuses, and absence of detailed 0021-or-later specifications. Repository
validation is deferred to the selected-modern-operations [checkpoint](../../../../glossary.md#term-checkpoint) after 0022 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no
repository-wide contract changes.

## Dependencies

- 0001–0002 and 0018M–0018M1: [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimensions](../../../../glossary.md#dimension), [canonical](../../../../glossary.md#term-canonical-form) symbolic arithmetic.
- 0005–0007, 0011–0013, and 0018K–0018L: operation/signature,
  Tensor/descriptor/factory/provenance.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0016A–0016D, 0018U1, and 0018V: floating accumulation, NaN/infinity/signed-zero, empty-domain,
  and reassociation precedents.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- 0017M–0017N: [NCHW](../../../../glossary.md#term-nchw) window vocabulary and the reason not to reuse `Window2dAttrs`.
- 0018N: exact typed positive-zero and floating special-value vocabulary.
- 0020: NCHW naming, checked static/symbolic [spatial](../../../../glossary.md#term-spatial-axes) formula, [metadata](../../../../glossary.md#term-metadata), and [lifecycle](../../../../glossary.md#lifecycle) precedent.
- 0020A: `Pool2dKind`, literal floor/ceil grid, operation-specific attrs, public-surface, and
  one-input provenance precedent.

## Follow-up tasks

- Task 0021 remains Draft for normalization operations and must not be specified or implemented
  here.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- Compiler later owns capture, dynamic binding/constraint proof, gradients, adjoints, and any
  legal decomposition preserving this fixed divisor and special-value meaning.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Backend/conformance/runtime/integration later own algorithms, tolerances, lowering, kernels,
  storage, prepared execution, and numerical evidence.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [runtime glossary](../../../../glossary.md#term-runtime); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>

Do not create another detailed specification during implementation.

## Architecture impact

Expected impact: None.

If this task requires architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), focused-architecture, [Gradle](../../../../glossary.md#term-gradle),
cross-[module](../../../../glossary.md#term-module), or [scope](../../../../glossary.md#term-scope) changes, stop and report the issue.

## Implementation prompt

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, focused architecture,
documentation/planning rules and profiles, roadmap, model capabilities/master, completed
operation/signature/Tensor/provenance/symbolic-extent/typed-scalar/reduction/window/Conv2d/max-
pooling tasks, current related source/tests/APIs/glossary, and task 0020A1.

Implement task 0020A1 exactly inside 18 paths. Update every global public Tensor inventory/count
178 -> 179 up front. Preserve completed max pooling and all architecture contracts; stop on
architecture uncertainty, scope overflow, another type/test/document, existing-helper change,
Gradle, cross-module work, or a need to alter the selected fixed count-padding contract.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and Java evidence to a separate clean documentation-focused agent in the same change; it
finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and documentation checks
while reusing Java evidence. Keep 0020A1 Ready until all criteria pass, then mark it Complete while
0021–0024 remain Draft with no next detailed specification.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide this task, complete diff, exact focused/final Java evidence and post-test Java-change
state, public/[Shape](../../../../glossary.md#shape)/type/ceil/divisor/[padding](../../../../glossary.md#padding)/special-value/[provenance](../../../../glossary.md#provenance) policies, seven documentation
paths, and [validation](../../../../glossary.md#term-validation) requirements. The clean agent reads repository instructions, architecture,
rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning)/Example profiles, task, source/tests/generated Javadoc,
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/[Runtime](../../../../glossary.md#term-runtime)/Training APIs, glossary/planning, and directly related contracts. It
finalizes documentation and records reasoned no-change conclusions without repeating successful
Java tests absent executable change, stale evidence, or a concrete risk.

## Local decisions

- One `AVERAGE_POOL2D` kind with one dedicated `AveragePool2dAttrs` record prevents average
  divisor/accumulation semantics from weakening completed max-pooling extrema semantics.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Count-padding is fixed operation meaning rather than configurable attributes. The mathematical
  divisor is not materialized with checked <span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [operation glossary](../../../../glossary.md#operation))</span>`long` multiplication during model construction,
  because positive kernel counts already establish a positive mathematical product and model
  construction evaluates no window.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [kernel glossary](../../../../glossary.md#kernel))</span>
- Static and symbolic result geometry follows completed max pooling exactly, including literal
  ceil-grid terminal all-padding windows. Only the eventual value policy differs.
- The documentation pass applied the General, API/Javadoc, Planning, and Example profiles. It
  refined the reusable NCHW pooling glossary entry with count-padding, fixed-divisor,
  accumulation-domain, and average-versus-valid-sample distinctions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [NCHW glossary](../../../../glossary.md#term-nchw); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Runtime and Training API pages remain unchanged because this task adds model expression metadata
  only. Runtime will receive prepared work, while compiler/training later own graph capture,
  binding proof, gradients, and backward construction.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [binding glossary](../../../../glossary.md#term-binding); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Related Conv2d, max-pooling, unfold/fold, pad, reduction, Shape/Dimension-expression,
  operation/signature, Tensor/factory/provenance, architecture, ADR, architecture-test,
  backend-conformance, integration, Java 26 Gradle, dependency, and other-module contracts remain
  accurate unchanged because this task composes their current model boundaries without changing
  their APIs, behavior, ownership, build, or executable layers.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Known limitations

- Only floating rank-four NCHW input and static geometry attributes are current. Input spatial
  extents may be symbolic, but their non-negative numerator obligation is not currently bound or
  proved by a compiler.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [NCHW glossary](../../../../glossary.md#term-nchw); [spatial glossary](../../../../glossary.md#term-spatial-axes); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Current construction records result metadata, numerical meaning, and provenance only. It does
  not read values, evaluate pooling, construct gradients, capture or decompose a graph, select a
  backend algorithm, allocate result storage, lower, prepare, or execute.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [pooling glossary](../../../../glossary.md#term-pooling); [gradients glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare))</span>
- Permitted summation reassociation means finite non-zero bitwise results and identical
  cross-backend rounding are deliberately not promised. Later conformance work must select and
  validate tolerances.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [finite glossary](../../../../glossary.md#term-finite); [rounding glossary](../../../../glossary.md#term-rounding); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Repository-wide validation remains deferred to the selected-modern-operations checkpoint after
  task 0022 or CI, as planned.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0020a1_implementation` ran the exact focused command. It
  passed `BUILD SUCCESSFUL`; three Gradle tasks executed. The same context then ran exactly one
  final <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`./gradlew :modules:model:test`; it passed `BUILD SUCCESSFUL` with three actionable tasks,
  one executed and two up-to-date. No executable Java changed afterward.
- Clean documentation context
  `/root/task_0020a1_implementation/documentation_finalization` reused both successful Java
  results and did not rerun Java tests. It independently reviewed the final four production
  contracts, seven tests, Tensor/Compile/Runtime/Training APIs, glossary, capabilities, task,
  master plan, roadmap, and related pooling/window/reduction/Shape/provenance boundaries.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- The documentation context ran `./gradlew :modules:model:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc). It passed
  `BUILD SUCCESSFUL` with two actionable tasks, both executed. Generated pages for `Pool2dKind`,
  `AveragePool2dAttrs`, and `Tensor.averagePool2d` contain the reviewed fixed-divisor,
  accumulation/division, special-value, metadata, failure, and lifecycle contracts.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A targeted local Markdown checker passed 593 links including 160 heading anchors across all
  seven documentation/planning paths. Fence balance and final-newline checks also passed. Manual
  terminology and example review confirmed the literal floor/ceil and symbolic Shape formulas,
  the <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`0.25` count-[padding](../../../../glossary.md#padding) example versus `1.0` valid-sample averaging, and current-versus-planned
  compiler/execution boundaries.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- The final scope and surface audit found exactly 18 authorized paths: four production, seven
  tests, and seven documentation/planning paths. Production packages remain the planned pooling
  and tensor packages; compiled <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>[`Tensor`](../../../../glossary.md#tensor) has exactly 179 public methods and exactly one
  `averagePool2d(AveragePool2dAttrs)` receiver; compiled [pooling](../../../../glossary.md#term-pooling) types expose
  `AVERAGE_POOL2D` and the exact nine-component attributes constructor. No `avgPool2d`,
  configuration flag/override, later detailed specification, cross-module import, or executable
  Java change was introduced.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Final planning audit confirmed tasks 0020, 0020A, and 0020A1 Complete; tasks 0021–0024 Draft;
  no model task Ready; and no detailed 0021-or-later specification. <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Ready glossary](../../../../glossary.md#term-ready))</span>`git diff --check` passed on
  the final combined change.

## Implementation notes

- Added `Pool2dKind.AVERAGE_POOL2D`, [immutable](../../../../glossary.md#term-immutable) `AveragePool2dAttrs`, one package-private stateless
  `TensorAveragePool2dExpressions` helper, and exactly one public [canonical](../../../../glossary.md#term-canonical-form)
  `Tensor.averagePool2d(AveragePool2dAttrs)` receiver method.
- Successful construction performs one final `TensorFactory.createDerived` call, preserving exact
  type, batch/channel Dimensions, gradient request, and one-input/output-zero provenance while
  deriving checked static or canonical symbolic spatial dimensions.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [Dimensions glossary](../../../../glossary.md#dimension); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance); [spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- Focused tests lock the exact kind/attrs/signature surface, canonical receiver spelling, public
  method count 179, static/symbolic literal floor/ceil geometry, local failures and identifier
  effects, metadata, freshness, and provenance without value evaluation.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [metadata glossary](../../../../glossary.md#term-metadata); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- The independent documentation pass finalized the affected production Javadocs, Tensor and
  Compile API references, glossary, capabilities, task, master plan, and roadmap without changing
  executable Java behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Completion summary

- Completed changes: implemented fixed-count NCHW average-pooling semantics, dedicated geometry
  attributes, exact static/symbolic result Shape derivation, one canonical public expression,
  validation, metadata/provenance construction, tests, public-surface locks, Javadocs, API and
  glossary documentation, and planning synchronization.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [pooling glossary](../../../../glossary.md#term-pooling); [Shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Files changed or created: exactly the authorized 18 paths: four production, seven tests, and
  seven documentation/planning paths.
- Tests and validation: focused and final model test evidence passed in the implementation
  context; final documentation validation is recorded above.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Documentation-agent review: clean context
  `/root/task_0020a1_implementation/documentation_finalization` independently finalized the
  required documentation and Javadocs while reusing Java evidence.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, task, master plan, and
  roadmap finalized; Runtime/Training and related contracts remain accurate unchanged for the
  reasons recorded above.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: all four affected production types/members reviewed and finalized.
- Glossary impact: the existing pooling entry now defines count-padding, fixed divisor,
  accumulation domain, and valid-sample contrast.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0020A1; tasks 0021–0024 remain Draft, and later compiler,
  backend, runtime, conformance, and integration work owns the documented boundaries.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

Status: Complete
