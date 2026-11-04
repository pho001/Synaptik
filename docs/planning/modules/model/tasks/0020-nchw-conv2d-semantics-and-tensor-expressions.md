# Task 0020: [NCHW](../../../../glossary.md#term-nchw) [Conv2d](../../../../glossary.md#convolution--conv2d) Semantics and [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (current context, not a status update).**

This record concerns the original Conv2d Model-expression scope, not the broader compiler/execution
descriptions in the current reference. The caller supplies the receiver input, weight, optional bias
and geometry to `Tensor.conv2d`; the helper records Shape, promoted type and ordered provenance, not
computed values. NCHW means batch, channel, height, width; groups isolate contiguous channel
subsets; cross-correlation uses the kernel without reversal. With four input channels and two
groups, each output channel uses two input channels. Follow [Conv2d
expressions](../../../../reference/tensor/convolution.md#grouped-nchw-conv2d-expressions) for
geometry and deferred symbolic obligations.

Model semantics defines meaning, not storage ownership: the storage-free result neither executes
convolution nor owns a result buffer. Compiler capture/proof and provider preparation/execution are
separate responsibilities; providers realize supported routes rather than redefine convolution.
Follow [numerical and metadata
policy](../../../../reference/tensor/convolution.md#numerical-and-metadata-policy) to distinguish
current descriptions from this historical scope: conceptual positive-zero padding still multiplies
infinite weights, bias participates once, and permitted reassociation does not promise bitwise
agreement.

**Geometry reading illustration.** Along one spatial axis, let positive integers $L,K,S,D$ denote
input length, kernel size, stride and dilation, and let non-negative integers $P_0,P_1$ denote
before/after padding. Effective kernel width $E=D(K-1)+1$ gives output length
$O=1+\lfloor(L+P_0+P_1-E)/S\rfloor$ where that window domain is valid. For $L=5$, $K=3$, $S=D=1$ and
no padding, $O=3$. Apply the geometry independently to height and width; it describes locations, not
convolution values.

## Status

Complete

## Goal

Add one first-class [backend](../../../../glossary.md#backend)-independent [`CONV2D`](../../../../glossary.md#convolution--conv2d) [model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) and exactly two public receiver
expressions for grouped [NCHW](../../../../glossary.md#term-nchw) two-dimensional [convolution](../../../../glossary.md#term-channels-first-convolution-layers), with and without [bias](../../../../glossary.md#term-bias). The result must
[carry](../../../../glossary.md#term-carry) truthful static or symbolic [spatial](../../../../glossary.md#term-spatial-axes) extents, promoted floating type, exact ordered
[provenance](../../../../glossary.md#provenance), and the complete selected numerical meaning without performing value work.

This focused task replaces only the convolution portion of the former broad 0020 frontier. Max
and average [pooling](../../../../glossary.md#term-pooling) remain Draft follow-up 0020A because their literal window, [padding](../../../../glossary.md#padding)-value,
average-divisor, ceil-mode, empty, and all-padding-window decisions are distinct.

## [Scope](../../../../glossary.md#term-scope)

- Add `Conv2dKind.CONV2D` with input range two through three and exactly one output.
- Add [immutable](../../../../glossary.md#term-immutable) public `Conv2dAttrs(strideHeight, strideWidth, paddingHeight, paddingWidth,
  dilationHeight, dilationWidth, groups)` using `long` components.
- Add `Conv2dAttrs.defaults()` returning stride one, zero symmetric padding, dilation one, and one
  group.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- Add exactly these receiver methods:

  ```java
  public Tensor conv2d(Tensor weight, Conv2dAttrs attrs)
  public Tensor conv2d(Tensor weight, Tensor bias, Conv2dAttrs attrs)
  ```

- Use one package-private stateless construction helper and one final `TensorFactory.createDerived`
  call for each successful expression.
- Require [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) input `[N, C_in, H, W]` and rank-four [weight](../../../../glossary.md#term-weight)
  `[C_out, C_in/groups, K_h, K_w]`; optional [bias](../../../../glossary.md#term-bias) is rank one `[C_out]`.
- Require floating input, weight, and present bias. Promote input and weight with
  `DataTypePromotion.promoteFloating`, then promote present bias with the intermediate result.
- Require statically known, positive weight kernel height and width. Preserve exact input batch
  and weight output-channel Dimension references in result <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel))</span>`[N, C_out, H_out, W_out]`.
- Validate locally provable group/channel and bias/channel relations; retain unresolved relations
  as compiler/binding obligations represented by exact input descriptors and attributes.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Derive static and dynamic [spatial](../../../../glossary.md#term-spatial-axes) output [Dimensions](../../../../glossary.md#dimension) with the exact floor-mode policy below.
- Produce one fresh unlabeled, storage-free result with unresolved layout, gradient-request OR
  across all actual inputs, and output index zero from exact ordered inputs <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>`[input, weight]` or
  `[input, weight, bias]`.
- Document cross-correlation, padding, promotion, accumulation, empty, special-value,
  determinism, Shape, metadata, and lifecycle boundaries.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Update focused [API](../../../../glossary.md#term-application-programming-interface-api)/glossary/[planning](../../../../glossary.md#term-planning) documentation and public-[Tensor](../../../../glossary.md#tensor) inventory locks.

## Out of [scope](../../../../glossary.md#term-scope)

- max pooling, average pooling, global/adaptive pooling, ceil-mode convolution, transposed
  convolution, depthwise convenience, causal/one-dimensional/three-dimensional convolution, or
  channels-last layouts<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [channels-last glossary](../../../../glossary.md#spatial-axes-channels-and-window-sampling); [layouts glossary](../../../../glossary.md#layout))</span>
- asymmetric or automatic `same`/`valid` padding, padding Tensor inputs, configurable padding
  values, runtime strides/dilations/groups, or a broad options framework<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [Tensor glossary](../../../../glossary.md#tensor); [runtime glossary](../../../../glossary.md#term-runtime); [strides glossary](../../../../glossary.md#term-stride))</span>
- integral, BOOL, quantized, sparse, complex, unsigned, or FLOAT16 inputs; implicit casts, output
  type overrides, or accumulator options<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- dynamic kernel extents: the current positive-coefficient Dimension expression model cannot
  truthfully subtract an unresolved dilated kernel from an unresolved input extent<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [Dimension glossary](../../../../glossary.md#dimension); [model glossary](../../../../glossary.md#term-model); [extent glossary](../../../../glossary.md#term-extent))</span>
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  input materialization<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- gradients, adjoints, backward-operation kinds, saved values, trainable-parameter ownership,
  compiler capture, graph-wide constraint solving, decomposition, fusion, or optimization<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- algorithms such as im2col/GEMM/direct/Winograd/FFT, kernel selection, backend capabilities,
  lowering, prepare, runtime, execution, conformance, or integration<span data-glossary-navigation="true"> ([im2col glossary](../../../../glossary.md#term-im2col); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [Winograd glossary](../../../../glossary.md#alternative-convolution-algorithms); [FFT glossary](../../../../glossary.md#alternative-convolution-algorithms); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to `Window2dAttrs`, unfold/fold, pad, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), attention, `ShapeBroadcast`,
  `DimensionExpression`, `DataTypePromotion`, Tensor provenance/factory seams, architecture,
  dependencies, Gradle, or another module<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Attributes and [signature](../../../../glossary.md#term-signature)

`Conv2dAttrs` validates components in declaration order. [Strides](../../../../glossary.md#term-stride), dilations, and groups are
positive; [padding](../../../../glossary.md#padding) components are non-negative. Exact constructor failures are:

```text
strideHeight must be positive: <value>
strideWidth must be positive: <value>
paddingHeight must be non-negative: <value>
paddingWidth must be non-negative: <value>
dilationHeight must be positive: <value>
dilationWidth must be positive: <value>
groups must be positive: <value>
```

The record owns intrinsic [convolution](../../../../glossary.md#term-channels-first-convolution-layers) geometry only. [Kernel](../../../../glossary.md#kernel) extents come from [weight](../../../../glossary.md#term-weight) [Shape](../../../../glossary.md#shape), and
[bias](../../../../glossary.md#term-bias) presence comes from the [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence)'s input count. It owns no [Tensor](../../../../glossary.md#tensor), Shape,
[DataType](../../../../glossary.md#data-type--datatype), [layout](../../../../glossary.md#layout), storage, [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), or execution state. [`CONV2D`](../../../../glossary.md#convolution--conv2d) accepts two
or three inputs and exactly one output; no second kind or bias flag is added.

### [Shape](../../../../glossary.md#shape) and dynamic extents

For each [spatial axis](../../../../glossary.md#spatial-axes-channels-and-window-sampling), let input [extent](../../../../glossary.md#term-extent) be `D`, static positive [weight](../../../../glossary.md#term-weight) [kernel](../../../../glossary.md#kernel) be `k`, symmetric
[padding](../../../../glossary.md#padding) per side be `p`, positive [dilation](../../../../glossary.md#term-dilation) be `d`, and positive [stride](../../../../glossary.md#term-stride) be `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = D + 2 * p - effectiveKernel
output          = floor(numerator / s) + 1
```

All literal arithmetic uses checked `long` operations. For static `D`, reject negative numerator
before division; zero numerator yields one output position. For unresolved `D`, construct the
[canonical](../../../../glossary.md#term-canonical-form) equivalent through existing `DimensionExpressions.addConstant`, `floorDivide`, and
`addConstant` calls. Do not require a static [spatial](../../../../glossary.md#term-spatial-axes) input, create an [identity](../../../../glossary.md#term-identity-meanings)-based unknown, or
change the symbolic-expression [model](../../../../glossary.md#term-model). The unresolved result retains the [binding](../../../../glossary.md#term-binding) obligation that
the numerator is non-negative. Neutral arithmetic must preserve exact [Dimension](../../../../glossary.md#dimension) references when
the existing [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) contract does so.

Input [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) `N` and weight output-[channel](../../../../glossary.md#term-channel) `C_out` are retained by exact reference. Kernel
Dimensions must be static and positive. Static input channels must be divisible by `groups`;
static output channels must be divisible by `groups`; and when both input channels and weight
channels-per-group are static, `weightChannelsPerGroup * groups == inputChannels` must hold using
checked multiplication. When both [bias](../../../../glossary.md#term-bias) length and output channels are static they must match.
Any corresponding relation involving an unresolved Dimension is deferred; the helper must not
invent a result constraint or reject a truthful result whose exact selected Dimensions are known.

Exact task-owned Shape failures are:

```text
conv2d input rank must be 4: <rank>
conv2d weight rank must be 4: <rank>
conv2d bias rank must be 1: <rank>
conv2d kernel height must be static: <dimension>
conv2d kernel width must be static: <dimension>
conv2d kernel height must be positive: <dimension>
conv2d kernel width must be positive: <dimension>
conv2d input channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv2d output channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv2d weight channels per group do not match input channels: weight=<dimension>, groups=<groups>, input=<dimension>
conv2d bias length must match output channels: bias=<dimension>, output=<dimension>
conv2d effective kernel does not fit padded height: input=<dimension>, effectiveKernel=<value>, padding=<value>
conv2d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

### [Data type](../../../../glossary.md#data-type--datatype), values, and numerical policy

Input, [weight](../../../../glossary.md#term-weight), and present [bias](../../../../glossary.md#term-bias) must each be floating. Role-specific failures are:

```text
conv2d input must have a floating data type, but was <dataType>
conv2d weight must have a floating data type, but was <dataType>
conv2d bias must have a floating data type, but was <dataType>
```

[`CONV2D`](../../../../glossary.md#convolution--conv2d) means [NCHW](../../../../glossary.md#term-nchw) grouped cross-correlation, not mathematical [kernel](../../../../glossary.md#kernel) reversal. For output
[channel](../../../../glossary.md#term-channel) `o`, its group selects the corresponding [contiguous](../../../../glossary.md#term-contiguous) `C_in/groups` input channels. Each
output [element](../../../../glossary.md#term-element) is the optional bias for `o` plus the sum of pairwise input/weight products over
that group and kernel height/width in increasing [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) meaning. Out-of-range [spatial](../../../../glossary.md#term-spatial-axes)
[coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) are conceptual positive-zero input values and participate in ordinary floating
multiplication; no value is read outside the input.

The promoted type is the output type. [FLOAT64](../../../../glossary.md#term-float64) output accumulates in FLOAT64. [FLOAT32](../../../../glossary.md#term-float32) and [BFLOAT16](../../../../glossary.md#term-bfloat16)
output accumulate in FLOAT32, with final conversion for BFLOAT16. Bias participates in that
accumulation domain exactly once. [Reassociation and fused multiply-add](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) are allowed, so fixed
traversal order, bitwise equality, and cross-[backend](../../../../glossary.md#backend) identical [rounding](../../../../glossary.md#term-rounding) are not promised; later
[conformance](../../../../glossary.md#term-conformance) owns tolerances. [IEEE](../../../../glossary.md#term-ieee)-754 [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) behavior follows ordinary
multiplication/addition, including conceptual zero multiplied by an infinite weight. An empty
channel [contraction](../../../../glossary.md#contraction-dimension) yields [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) before optional bias. Empty [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) or output-channel
[axes](../../../../glossary.md#term-axis) are valid; zero output spatial [extent](../../../../glossary.md#term-extent) is not produced by valid static geometry under this
formula, while unresolved geometry is checked when bound.

### [Validation](../../../../glossary.md#term-validation) and construction order

The helper validates before its sole factory call in this order:

1. null-check input, [weight](../../../../glossary.md#term-weight), present [bias](../../../../glossary.md#term-bias), and attrs in [parameter](../../../../glossary.md#term-parameter) order;
2. validate input, weight, then present-bias floating eligibility;
3. promote input/weight and then present bias;
4. validate input [rank](../../../../glossary.md#term-rank), weight rank, then present bias rank;
5. require static positive [kernel](../../../../glossary.md#kernel) height, then width;
6. validate static input-channel divisibility, output-channel divisibility, grouped weight/input
   compatibility, then bias/output compatibility;<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel))</span>
7. derive height then width with checked geometry and existing [Dimension](../../../../glossary.md#dimension) expressions;
8. create the [descriptor](../../../../glossary.md#term-descriptor) and exact [operation](../../../../glossary.md#operation);
9. delegate once with exact ordered inputs.

Null messages are parameter names. `DataTypePromotion` and checked-arithmetic failures retain
their existing messages. Every failure before final delegation consumes no [Tensor](../../../../glossary.md#tensor) ID, [producer](../../../../glossary.md#term-producer),
or result wrapper. A successful call consumes exactly one ID and creates one producer.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Signature hardening](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Signature glossary](../../../../glossary.md#term-signature))</span>
- [Symbolic extents](0018m-symbolic-extent-expressions.md)
- [Dynamic extent adoption](0018m1-dynamic-extent-adoption.md)<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- [Typed scalar](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Multi-axis/statistical reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [MATMUL](0019-matmul-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [Attention](0019e-scaled-dot-product-attention.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. Tensor remains public mutable state, not
  graph IR.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [mutable glossary](../../../../glossary.md#term-mutable); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Convolution operation types record backend-independent meaning only; operation packages do not
  import Tensor, graph, compiler, runtime, prepare, or backend types.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Direction is [tensor](../../../../glossary.md#tensor) helper to [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [operation](../../../../glossary.md#operation)/datatype/[shape](../../../../glossary.md#shape). Packages remain acyclic.
- Compiler owns capture, binding/deferred proof, legal decomposition, gradients, adjoints, and
  saved values. Backend prepare owns conforming algorithms, lowering, kernels, and
  materialization. Runtime receives prepared work without original operations on its hot path.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradients glossary](../../../../glossary.md#term-gradient); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [materialization glossary](../../../../glossary.md#materialization); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- No architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), focused-architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Package added:

- `io.github.pho001.synaptik.model.operation.convolution` — public convolution identity and
  intrinsic immutable attributes only.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [immutable glossary](../../../../glossary.md#term-immutable))</span>

Type placement:

- `...operation.convolution.Conv2dKind` — exact [operation](../../../../glossary.md#operation) [identity](../../../../glossary.md#term-identity-meanings)/[signature](../../../../glossary.md#term-signature) [owner](../../../../glossary.md#term-owner).
- `...operation.convolution.Conv2dAttrs` — public inspectable intrinsic geometry/group semantics.
- `...tensor.TensorConv2dExpressions` — package-private validation, Shape, descriptor, and
  provenance construction owner.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `...tensor.Tensor` — established public fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConv2dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv2dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact two
  signatures and public count 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 175 to 177.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 175 to 177.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime)/Training APIs; [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) expression,
[layout](../../../../glossary.md#layout)/window/pad, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/attention, [operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature), [Tensor](../../../../glossary.md#tensor)/[provenance](../../../../glossary.md#provenance) contracts;
architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 18 paths maximum: four production, seven test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths.
`Tensor.java` changes only one import, two methods/Javadocs, and its [operation](../../../../glossary.md#operation) inventory. Five
existing tests change only the stated [signatures](../../../../glossary.md#term-signature)/counts. Stop for path 19, another type/test/doc,
an existing-helper change, architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) work.

This is one cohesive vertical [model](../../../../glossary.md#term-model) capability inside the planning guide guardrail. Kind, attrs,
public construction, symbolic [Shape](../../../../glossary.md#shape) derivation, promotion/numerical meaning, [API](../../../../glossary.md#term-application-programming-interface-api) locks, and docs
must agree in one compilable state. [Pooling](../../../../glossary.md#term-pooling) would exceed the guardrail and introduces independent
semantic decisions, so it remains 0020A Draft.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, attrs/defaults, helper, and both Tensor methods: NCHW/cross-correlation
  meaning, input/weight/bias Shape, dynamic extent formula, group rules, type/accumulation,
  padding/special/empty policy, metadata/provenance, freshness, validation/failures, and lifecycle
  boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [NCHW glossary](../../../../glossary.md#term-nchw); [weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent); [padding glossary](../../../../glossary.md#padding); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every [parameter](../../../../glossary.md#term-parameter), result, and expected failure has complete `@param`, `@return`, and `@throws`
  tags as applicable.
- Tensor API gets a Shape/attribute table, grouped example, one dynamic-height example, numerical
  policy, and current-model versus planned compiler/execution boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compile API records current convolution expression metadata and future compiler-owned binding,
  capture, decomposition, and gradients without claiming compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [binding glossary](../../../../glossary.md#term-binding); [capture glossary](../../../../glossary.md#term-capture-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Review glossary terms NCHW, convolution, cross-correlation, kernel, dilation, group, and
  effective kernel; add or refine only terms needed by the public explanation.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- Synchronize capabilities/task/master/roadmap: keep 0020 Ready during implementation, then mark
  it Complete only after all criteria pass; keep 0020A and 0021–0024 Draft, exactly one detailed
  post-0019E task, and no Ready model frontier after completion.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact kind, attrs/defaults, two-to-three-input/one-output signature, and two receiver methods
  exist; public Tensor method count is 177.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Static/dynamic Shape derivation, exact Dimension reference retention, checked geometry,
  grouped-channel/bias constraints, and deferred unresolved obligations match this task.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [channel glossary](../../../../glossary.md#term-channel); [bias glossary](../../../../glossary.md#term-bias))</span>
- All floating ordered width combinations promote correctly; BOOL/integral roles fail; selected
  accumulation and conversion policies are documented and tested as semantic metadata without
  evaluation.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [metadata glossary](../../../../glossary.md#term-metadata); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- NCHW grouped cross-correlation, conceptual padding, NaN/infinity/signed-zero, empty contraction,
  reassociation/tolerance/determinism policies are explicit.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [padding glossary](../../../../glossary.md#padding); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [contraction glossary](../../../../glossary.md#contraction-dimension); [reassociation glossary](../../../../glossary.md#term-reassociation); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Validation order, exact task-owned failures, no-ID failures, one-ID success, freshness,
  storage/layout/label, gradient-request OR, and exact ordered provenance pass.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance))</span>
- No pooling, gradients, compiler, algorithm, backend/runtime, architecture, dependency, or build
  work is added.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 18 paths/packages/public surface/statuses, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.convolution.Conv2dSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorConv2dExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

The focused tests cover attributes/[signature](../../../../glossary.md#term-signature), overloads, all static/dynamic [spatial](../../../../glossary.md#term-spatial-axes) cases,
[kernel](../../../../glossary.md#kernel)/group/[channel](../../../../glossary.md#term-channel)/[bias](../../../../glossary.md#term-bias) constraints, floating promotion, validation/ID effects,
[metadata](../../../../glossary.md#term-metadata)/[provenance](../../../../glossary.md#provenance)/freshness, and numerical contracts as meaning/[Javadoc](../../../../glossary.md#term-javadoc) without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links/anchors, fences, terminology, examples, generated Javadoc,
newlines/whitespace, exact paths/packages, public count/signatures, the required completion
frontier state, and no 0020A-or-later detailed spec. Repository validation is deferred to the selected-modern-
operations [checkpoint](../../../../glossary.md#term-checkpoint) after 0022 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no repository-wide contract changes.

## Dependencies

- 0001–0002 and 0018M–0018M1: [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimensions](../../../../glossary.md#dimension), [canonical](../../../../glossary.md#term-canonical-form) symbolic arithmetic.
- 0005–0007, 0011–0013, and 0018K–0018L: operation/signature,
  Tensor/descriptor/factory/provenance.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0018N: exact typed [scalar](../../../../glossary.md#term-scalar) and conceptual positive-zero vocabulary precedent.
- 0018V: final selected reduction/accumulation policy precedent.
- 0019: floating product-sum promotion, accumulation, special-value, and determinism precedent.

Tasks 0017M–0017N and 0019E are table-order/history references, not technical dependencies.

## Follow-up tasks

- 0020A remains Draft for cohesive NCHW max/average Pool2d semantics and expressions. It must
  select literal window/dilation scope, padding value, average divisor, ceil-mode terminal-window,
  empty, all-padding-window, and max tie/special-value contracts before becoming Ready.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [dilation glossary](../../../../glossary.md#term-dilation); [scope glossary](../../../../glossary.md#term-scope); [padding glossary](../../../../glossary.md#padding); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Compiler later owns capture, binding/constraints, legal decomposition, gradients, adjoints, and
  saved values.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradients glossary](../../../../glossary.md#term-gradient))</span>
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
operation/signature/Tensor/provenance/symbolic-extent/typed-scalar/reduction/MATMUL/attention and
layout-window tasks, current related source/tests/APIs/glossary, and task 0020.

Implement task 0020 exactly inside 18 paths. Update every global public Tensor inventory/count
175 -> 177 up front. Preserve all contracts and stop on architecture uncertainty, scope overflow,
another type/test/document, existing-helper change, Gradle, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and Java evidence to a separate clean documentation-focused agent in the same change; it
finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and docs checks while
reusing Java evidence. Keep 0020 Ready until all criteria pass, then mark it Complete while 0020A+
remain Draft with no next Ready model task.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide this task, the complete diff, exact focused/final Java evidence and post-test Java-change
state, [API](../../../../glossary.md#term-application-programming-interface-api)/[Shape](../../../../glossary.md#shape)/type/group/[bias](../../../../glossary.md#term-bias)/numerical/[provenance](../../../../glossary.md#provenance) policies, seven documentation paths, and
[validation](../../../../glossary.md#term-validation) requirements. The clean agent reads repository instructions, architecture, rules and
General/API-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning)/Example profiles, task, source/tests/generated Javadoc,
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/[Runtime](../../../../glossary.md#term-runtime)/Training APIs, glossary/planning, and directly related contracts. It
finalizes documentation and records reasoned no-change conclusions without repeating successful
Java tests absent executable change, stale evidence, or a concrete risk.

## Local decisions

- One [`CONV2D`](../../../../glossary.md#convolution--conv2d) kind with a two-to-three-input signature represents bias presence by occurrence
  input count; no second kind or bias flag was added.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [bias glossary](../../../../glossary.md#term-bias); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Grouped convolution means NCHW cross-correlation with contiguous channel groups and no kernel
  reversal. Kernel extents come from weight Shape and remain statically positive because current
  symbolic extents cannot truthfully represent subtraction of an unresolved dilated kernel.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [NCHW glossary](../../../../glossary.md#term-nchw); [contiguous glossary](../../../../glossary.md#term-contiguous); [channel glossary](../../../../glossary.md#term-channel); [weight glossary](../../../../glossary.md#term-weight); [Shape glossary](../../../../glossary.md#shape))</span>
- Static relations fail locally in the specified order. Relations involving an unresolved channel,
  bias, or spatial input Dimension remain compiler/binding obligations retained by exact
  descriptors and attributes.<span data-glossary-navigation="true"> ([spatial glossary](../../../../glossary.md#term-spatial-axes); [Dimension glossary](../../../../glossary.md#dimension); [compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Conceptual padding is positive zero and participates in ordinary multiplication, including with
  infinity. The selected promotion, accumulation, empty-contraction, special-value,
  reassociation, and determinism policies match the task contract without claiming evaluation.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [contraction glossary](../../../../glossary.md#contraction-dimension); [reassociation glossary](../../../../glossary.md#term-reassociation); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- The documentation pass used General, API/Javadoc, Planning, and Example profiles. It added one
  glossary entry for the reusable convolution/cross-correlation/NCHW/group/dilation/effective-
  kernel distinction and refined <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [dilation glossary](../../../../glossary.md#term-dilation))</span>[`Kernel`](../../../../glossary.md#kernel) to distinguish a weight kernel from an executable
  backend kernel.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Runtime and Training API pages remain unchanged because this task adds model expression metadata
  only: runtime receives prepared work, while training/compiler later own gradients and graph
  expansion. Related Shape/Dimension, layout/window/pad, MATMUL/attention,
  operation/signature/Tensor/provenance contracts remain accurate and unchanged because the new
  helper composes them without changing their behavior.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [gradients glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Architecture, focused architecture, ADRs, architecture tests, backend conformance, integration
  tests, Gradle, dependencies, other modules, and task 0020A-or-later specifications remain
  unchanged because no boundary, executable behavior, build contract, or downstream detailed
  design changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Known limitations

- Dynamic [kernel](../../../../glossary.md#kernel) height and width are rejected; only input [spatial](../../../../glossary.md#term-spatial-axes) extents may remain symbolic.
- Unresolved group/channel, bias/output-channel, and spatial-fit relations are recorded only by
  exact descriptors and attributes; no current compiler or binding API proves them.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [bias glossary](../../../../glossary.md#term-bias); [descriptors glossary](../../../../glossary.md#term-descriptor); [compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Current construction records numerical meaning but does not read values, evaluate convolution,
  construct gradients, capture or decompose a graph, choose an algorithm/backend, allocate
  storage, lower, prepare, or execute.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [gradients glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare))</span>
- Reassociation and fused multiply-add are permitted, so fixed traversal order, bitwise equality,
  and identical cross-backend rounding are deliberately not promised.<span data-glossary-navigation="true"> ([Reassociation and fused multiply-add glossary](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Pooling remains Draft task 0020A. Repository-wide validation remains deferred to the selected-
  modern-operations checkpoint after 0022 or CI as planned.<span data-glossary-navigation="true"> ([Pooling glossary](../../../../glossary.md#term-pooling); [validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0020_implementation` ran the exact focused command. Its
  first attempt exposed a compile-test failure, which was corrected before final evidence. The
  final focused run passed <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>`BUILD SUCCESSFUL`: 7 suites, 41 tests, 0 failures, 0 errors, 0 skips.
- After executable Java stabilized, that implementation context ran exactly one final
  `./gradlew :modules:model:test`; it passed `BUILD SUCCESSFUL`: 109 suites, 860 tests, 0 failures,
  0 errors, 0 skips. Executable Java did not change afterward. Clean documentation context
  `/root/task_0020_implementation/task_0020_docs` reused this evidence and did not repeat either
  Java suite.
- The documentation context ran `./gradlew :modules:model:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc); it passed
  `BUILD SUCCESSFUL` with two actionable tasks, one executed and one up-to-date. Generated pages
  for `Conv2dKind`, `Conv2dAttrs`, and both `Tensor.conv2d` overloads contain the reviewed
  contracts.
- The complete grouped [metadata](../../../../glossary.md#term-metadata) example was compiled with Java against
  `modules/model/build/classes/java/main` and [run](../../../../glossary.md#term-run). It printed the documented [FLOAT64](../../../../glossary.md#term-float64) type,
  `Shape[2, 6, 7, 7]`, true [gradient](../../../../glossary.md#term-gradient) eligibility, true unresolved-[layout](../../../../glossary.md#layout)/storage-free check,
  [`CONV2D`](../../../../glossary.md#convolution--conv2d), and true exact ordered-input check.
- A targeted local Markdown checker passed 580 links including 157 heading anchors across the
  seven documentation/planning paths. Fence balance, terminology, examples, non-empty files,
  final newlines, and generated-Javadoc content checks passed.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- The final path audit found exactly 18 authorized paths: 4 production, 7 tests, and 7
  documentation/planning paths. Production packages, the exact kind/attrs/defaults surface, the
  two receiver signatures, and the public Tensor method count of 177 were verified from compiled
  classes. Task 0020 is Complete; 0020A and 0021–0024 are Draft; no model task is Ready; task 0020
  is the only detailed post-0019E specification; and no 0020A-or-later detailed spec exists.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model); [Ready glossary](../../../../glossary.md#term-ready))</span>
- `git diff --check` passed on the final combined change.

## Implementation notes

- Added public `Conv2dKind.CONV2D`, [immutable](../../../../glossary.md#term-immutable) `Conv2dAttrs`, one package-private stateless
  `TensorConv2dExpressions` helper, and exactly two public `Tensor.conv2d` receiver methods.
- Successful construction performs one final `TensorFactory.createDerived` call and creates one
  fresh output. It preserves exact batch/output-channel Dimensions, derives checked static or
  canonical symbolic spatial Dimensions, applies ordered floating promotion, propagates the
  actual-input gradient-request OR, and records exact ordered provenance at output index zero.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [Dimensions glossary](../../../../glossary.md#dimension); [canonical glossary](../../../../glossary.md#term-canonical-form); [spatial glossary](../../../../glossary.md#term-spatial-axes); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance))</span>
- Focused tests cover surface/signature, attribute validation, static/dynamic Shape cases,
  deferred obligations, promotion and role failures, exact task-owned diagnostics, ID effects,
  checked arithmetic, freshness, metadata, and provenance. Existing Tensor inventory locks now
  expect 177 methods.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The independent documentation pass finalized the four affected production Javadocs plus Tensor
  API, Compile API, glossary, capabilities, task, master plan, and roadmap. It changed no
  executable Java behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Completion summary

- Completed grouped NCHW Conv2d semantic identity, intrinsic immutable attributes, exact
  static/symbolic result Shape derivation, two public biased/unbiased expression methods,
  validation, promotion/numerical policy, fresh metadata/provenance construction, focused tests,
  public-surface locks, Javadocs, API/glossary documentation, and planning synchronization.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Changed exactly the 18 paths listed by this task; no extra type, test, document, module,
  architecture, dependency, Gradle, runtime, training, backend, conformance, or integration change
  was introduced.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Focused 41-test and final 860-test model evidence passed in the implementation context. The
  separate documentation context passed model Javadoc, the compiled example, 580-link/157-anchor
  Markdown validation, generated-content/fence/newline/scope/surface/status checks, and final
  whitespace validation.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Unresolved issues: none. Required follow-up: none for task 0020; Draft task 0020A separately owns
  pooling decisions, and later compiler/backend/runtime tasks own their documented boundaries.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [compiler glossary](../../../../glossary.md#term-compiler))</span>

Status: Complete
