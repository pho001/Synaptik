# Task 0020A: [NCHW](../../../../glossary.md#term-nchw) Max Pool2d Semantics and [Tensor](../../../../glossary.md#tensor) Expression

**Editorial reading key (current context, not a status update).**

This completed record concerns Model expression construction, not execution; its type limits and
deferred downstream work describe that historical scope. NCHW means batch, channel, height, width.
The caller invokes `input.maxPool2d(attrs)`; the Tensor helper validates geometry and delegates to
the factory, returning a storage-free expression. Attributes describe windows; descriptor and
provenance record shape, type, input and producing operation—not computed values or owned result
buffers. Model defines meaning; execution providers realize supported routes rather than redefine
it.

Read [current maximum-pooling
expressions](../../../../reference/tensor/pooling.md#nchw-maximum-pooling-expressions) for today's
floating-type coverage and geometry, and [extrema and padding
policy](../../../../reference/tensor/pooling.md#extrema-and-padding-policy) for selection and
compiler/execution boundaries. These current descriptions do not expand this task retrospectively.
Padding is excluded, not zero-filled: a window containing only an in-bounds negative value selects
it; an all-padding window returns negative infinity. Literal ceil mode retains terminal windows.
NaNs dominate, positive zero outranks negative zero, and equal candidates retain the first
height-then-width sample; NaN representation remains unspecified. Semantic responsibility does not
confer storage ownership.

## Status

Complete

## Goal

Add one first-class [backend](../../../../glossary.md#backend)-independent `MAX_POOL2D` [operation](../../../../glossary.md#operation) and exactly one public receiver
expression for floating [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) [tensors](../../../../glossary.md#tensor). The result must retain exact [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[channel](../../../../glossary.md#term-channel)
[dimensions](../../../../glossary.md#dimension), derive truthful static or symbolic [spatial](../../../../glossary.md#term-spatial-axes) dimensions, and record the selected
[padding](../../../../glossary.md#padding), ceil-mode, [extrema](../../../../glossary.md#saturation-pole-and-extremum), [metadata](../../../../glossary.md#term-metadata), and [provenance](../../../../glossary.md#provenance) semantics without evaluating values.

This task is the max-[pooling](../../../../glossary.md#term-pooling) half of the former combined 0020A row. Average pooling is Draft
follow-up 0020A1 because its divisor, padding-count, accumulation, and invalid-divisor policies
are independent of max ordering, [NaN](../../../../glossary.md#term-nan), tie, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and empty-window semantics.

## [Scope](../../../../glossary.md#term-scope)

- Add `Pool2dKind.MAX_POOL2D` with exactly one input and one output.
- Add [immutable](../../../../glossary.md#term-immutable) public `MaxPool2dAttrs(kernelHeight, kernelWidth, strideHeight, strideWidth,
  paddingHeight, paddingWidth, dilationHeight, dilationWidth, ceilMode)` using `long` geometry.
- Add exactly `public Tensor maxPool2d(MaxPool2dAttrs attrs)`.
- Add one package-private stateless construction helper and one final
  `TensorFactory.createDerived` call for each successful expression.
- Require floating [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) input `[N, C, H, W]` and preserve exact input type, batch,
  channel, and gradient-request metadata.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Derive floor- or ceil-mode static and symbolic [spatial](../../../../glossary.md#term-spatial-axes) extents with checked arithmetic.
- Treat padding positions as excluded from the maximum; an all-padding window returns negative
  infinity in the input type.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Select exact [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and equal-value tie semantics.
- Produce one fresh unlabeled, storage-free result with unresolved layout and exact one-input,
  output-index-zero provenance.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update focused API, glossary, capability, and planning documentation plus public-Tensor
  inventory locks.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- average, global, adaptive, one-dimensional, three-dimensional, or [channels-last](../../../../glossary.md#spatial-axes-channels-and-window-sampling) [pooling](../../../../glossary.md#term-pooling)
- max-pool indices, multiple outputs, unpooling, configurable padding values, asymmetric or
  automatic padding, runtime geometry, or a broad options framework<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- integral, BOOL, quantized, sparse, complex, unsigned, or FLOAT16 input; casts, promotion,
  output-type overrides, or accumulation options<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- PyTorch/ONNX terminal-window removal in ceil mode; this task selects the literal symmetric
  padded window grid defined below so symbolic output extents remain exact<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>
- reuse or modification of `Window2dAttrs`; it owns unfold/fold layout-transform geometry and
  conceptual zero padding, not pooling extrema semantics<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- value reads, eager evaluation, host allocation, result storage, resolved layout, mutation, or
  input materialization<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- gradients, adjoints, saved indices, backward kinds, compiler capture, graph-wide constraint
  solving, canonicalization, decomposition, fusion, or optimization<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- algorithms, backend capabilities, lowering, prepare, runtime, execution, conformance, or
  integration<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to Conv2d, unfold/fold, pad, aggregate reductions, symbolic-expression foundations,
  promotion, provenance/factory seams, architecture, dependencies, Gradle, or another module<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Attributes and [signature](../../../../glossary.md#term-signature)

`MaxPool2dAttrs` validates components in declaration order. [Kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), and [dilation](../../../../glossary.md#term-dilation)
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

`ceilMode` is retained unchanged. The record owns intrinsic max-[pooling](../../../../glossary.md#term-pooling) geometry only and owns no
[Tensor](../../../../glossary.md#tensor), [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype), [layout](../../../../glossary.md#layout), storage, [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), or execution state.
`MAX_POOL2D` accepts only `MaxPool2dAttrs`, exactly one input, and exactly one output.

### [NCHW](../../../../glossary.md#term-nchw) [Shape](../../../../glossary.md#shape) and [spatial](../../../../glossary.md#term-spatial-axes) windows

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
unknown extent, clamp a negative numerator to zero, or modify the symbolic-expression [model](../../../../glossary.md#term-model).

The output Shape is `[N, C, H_out, W_out]`. Preserve the exact input `N` and `C` [Dimension](../../../../glossary.md#dimension)
references. Floor mode includes every window in the floor grid. Ceil mode includes every window
in the literal ceiling grid, including a final window whose start lies entirely in trailing
padding; there is no terminal-window decrement. Along an [axis](../../../../glossary.md#term-axis), output position `o` starts at
`o * s - p` and samples positions `start + r * d` for `r` from zero through `k - 1`.

Exact task-owned Shape failures are:

```text
maxPool2d input rank must be 4: <rank>
maxPool2d effective kernel does not fit padded height: input=<dimension>, effectiveKernel=<value>, padding=<value>
maxPool2d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

Static zero [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) or [channel](../../../../glossary.md#term-channel) extents are valid and make the result empty while retaining spatial
[metadata](../../../../glossary.md#term-metadata). A static zero height or width is valid only when its padding makes the corresponding
numerator non-negative; resulting windows are all-padding where no sampled [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) is in
bounds. Valid static geometry always produces at least one spatial position. Unresolved geometry
is accepted with the same non-negative-numerator binding obligation.

### [Data type](../../../../glossary.md#data-type--datatype), [extrema](../../../../glossary.md#saturation-pole-and-extremum), and numerical policy

The sole input must be [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64). The result retains the exact input type; no
promotion or accumulation type applies. The exact type failure is:

```text
maxPool2d input must have a floating data type, but was <dataType>
```

[Padding](../../../../glossary.md#padding) positions do not participate. If a window has one or more in-bounds sampled positions:

- any [NaN](../../../../glossary.md#term-nan) is greater for selection purposes than every non-NaN, so NaN propagates;
- otherwise ordinary numerical maximum applies, with [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) greater than [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero);
- [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) use ordinary numerical order; and
- equal candidates select the first sampled input in increasing kernel-height then kernel-width
  order.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>

Multiple NaNs use that same first-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) tie rule. [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign and signaling preservation
remain unspecified, matching the current floating-reduction policy; [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) selection is
exact. An all-padding window has no selected input and returns exact [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) in the
input type. These tie rules are semantic evidence for later [compiler](../../../../glossary.md#term-compiler)-owned [gradients](../../../../glossary.md#term-gradient); this task
creates no gradient rule or indices output. The value contract fixes results independently of
physical traversal but does not promise a particular algorithm.

### [Validation](../../../../glossary.md#term-validation) and construction order

The helper validates before its sole factory call in this order:

1. null-check `input`, then `attrs`;
2. validate input floating eligibility;
3. validate input [rank](../../../../glossary.md#term-rank);
4. derive height, then width, including checked effective-kernel, padding, numerator, and
   floor/ceil arithmetic;<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [padding glossary](../../../../glossary.md#padding))</span>
5. create the result [descriptor](../../../../glossary.md#term-descriptor) and exact [operation](../../../../glossary.md#operation); and
6. delegate once with exact ordered input `[input]`.

Null messages are [parameter](../../../../glossary.md#term-parameter) names. Checked-arithmetic and existing symbolic-expression failures
retain their existing messages. Every failure before final delegation consumes no [Tensor](../../../../glossary.md#tensor) ID,
[producer](../../../../glossary.md#term-producer), or wrapper. A successful call consumes exactly one ID and creates one producer.

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

## Architecture constraints

- Work stays in model plus its documentation/planning. Tensor remains public mutable state, not
  graph IR.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [mutable glossary](../../../../glossary.md#term-mutable); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Pooling operation types record backend-independent meaning only and do not import Tensor,
  graph, compiler, runtime, prepare, or backend types.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Package direction is [tensor](../../../../glossary.md#tensor) helper to [pooling](../../../../glossary.md#term-pooling) [operation](../../../../glossary.md#operation)/datatype/[shape](../../../../glossary.md#shape); packages remain acyclic.
- Compiler owns capture, binding/deferred proof, gradients, adjoints, and saved indices. Backend
  prepare owns conforming algorithms, lowering, kernels, and materialization. Runtime receives
  prepared work without original operations on its hot path.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [gradients glossary](../../../../glossary.md#term-gradient); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [materialization glossary](../../../../glossary.md#materialization); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- No architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), focused-architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Package added:

- `io.github.pho001.synaptik.model.operation.pooling` — public pooling identities and
  operation-specific immutable attributes only.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable))</span>

Type placement:

- `...operation.pooling.Pool2dKind` — [pooling](../../../../glossary.md#term-pooling) [identity](../../../../glossary.md#term-identity-meanings)/[signature](../../../../glossary.md#term-signature) [owner](../../../../glossary.md#term-owner), initially MAX only.
- `...operation.pooling.MaxPool2dAttrs` — public inspectable max-window geometry.
- `...tensor.TensorMaxPool2dExpressions` — package-private validation, Shape, descriptor, and
  provenance construction owner.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `...tensor.Tensor` — established public fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/MaxPool2dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool2dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (7):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool2dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signature and public count 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 177 to 178.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 177 to 178.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime)/Training APIs; [Conv2d](../../../../glossary.md#convolution--conv2d), unfold/fold, pad, reduction,
[Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) expression, [operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature), [Tensor](../../../../glossary.md#tensor)/[provenance](../../../../glossary.md#provenance) contracts; architecture/ADRs/
tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 18 paths maximum: four production, seven test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths.
`Tensor.java` changes only one import, one method/[Javadoc](../../../../glossary.md#term-javadoc), and its [operation](../../../../glossary.md#operation) inventory. Five
existing tests change only the stated [signature](../../../../glossary.md#term-signature)/count. Stop for path 19, another type/test/doc,
an existing-helper change, architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) work.

This is one cohesive vertical [model](../../../../glossary.md#term-model) capability. Combining average [pooling](../../../../glossary.md#term-pooling) would add a second
attribute/numerical contract and exceed the established path guardrail; 0020A1 remains Draft.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, attrs, helper, and Tensor method: NCHW window meaning, Shape formulas,
  literal ceil grid, padding exclusion, extrema/special/empty policy, metadata/provenance,
  validation/failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [NCHW glossary](../../../../glossary.md#term-nchw); [Shape glossary](../../../../glossary.md#shape); [padding glossary](../../../../glossary.md#padding); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every [parameter](../../../../glossary.md#term-parameter), result, and expected failure has complete `@param`, `@return`, and `@throws`
  tags as applicable.
- Tensor API gets a geometry table, static floor/ceil and symbolic examples, extrema policy, and
  current-model versus planned compiler/execution boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compile API records current max-pooling metadata and future compiler-owned binding, capture,
  gradient/index-saving, and decomposition without claiming compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [pooling glossary](../../../../glossary.md#term-pooling); [binding glossary](../../../../glossary.md#term-binding); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Review glossary terms NCHW, pooling, window, padding, dilation, effective kernel, and ceil mode;
  add or refine only reusable distinctions needed by the public explanation.<span data-glossary-navigation="true"> ([dilation glossary](../../../../glossary.md#term-dilation); [kernel glossary](../../../../glossary.md#kernel))</span>
- Synchronize capability/task/master/roadmap statuses: 0020 and 0020A are Complete; 0020A1 plus
  0021–0024 remain Draft with no detailed future specs, and no model task is Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact kind, attrs, one-input/one-output signature, and one receiver method exist; public Tensor
  method count is 178.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Static and symbolic floor/ceil Shape derivation, exact N/C reference retention, checked
  geometry, and deferred unresolved obligation match this task.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- BFLOAT16/FLOAT32/FLOAT64 retain exact type; BOOL/integral fail; requiresGrad is retained,
  layout is unresolved, and result is fresh, unlabeled, and storage-free.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BOOL glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout))</span>
- Padding exclusion, all-padding negative infinity, NaN propagation/payload selection,
  infinity, signed-zero, and first logical-kernel tie policies are explicit and API-locked without
  evaluation.<span data-glossary-navigation="true"> ([Padding glossary](../../../../glossary.md#padding); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [kernel glossary](../../../../glossary.md#kernel); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Validation order, exact task-owned failures, no-ID failures, one-ID success, and exact
  one-input/output-zero provenance pass.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- No average pooling, indices output, gradients, compiler, algorithm, backend/runtime,
  architecture, dependency, or build work is added.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 18 paths/packages/public surface/statuses, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes Javadocs/docs, and
  records reasoned no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.pooling.Pool2dSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMaxPool2dExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

The focused tests cover attributes/[signature](../../../../glossary.md#term-signature), public surface, all static/symbolic floor/ceil
cases, literal terminal windows, type/[rank](../../../../glossary.md#term-rank)/geometry failures, [metadata](../../../../glossary.md#term-metadata), validation/ID effects,
freshness/[provenance](../../../../glossary.md#provenance), and numerical contracts as meaning/[Javadoc](../../../../glossary.md#term-javadoc) without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links/anchors, fences, terminology, examples, generated Javadoc,
newlines/whitespace, exact paths/packages, public count/signature, statuses, and absence of a
detailed 0020A1-or-later specification. Repository validation is deferred to the selected-modern-
operations [checkpoint](../../../../glossary.md#term-checkpoint) after 0022 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no repository-wide contract changes.

## Dependencies

- 0001–0002 and 0018M–0018M1: [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimensions](../../../../glossary.md#dimension), [canonical](../../../../glossary.md#term-canonical-form) symbolic arithmetic.
- 0005–0007, 0011–0013, and 0018K–0018L: operation/signature,
  Tensor/descriptor/factory/provenance.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0016A–0016D, 0018U1, and 0018V: floating extrema, NaN/signed-zero, empty-domain, and
  deterministic logical-order precedents.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [NaN glossary](../../../../glossary.md#term-nan))</span>
- 0017M–0017N: [NCHW](../../../../glossary.md#term-nchw) window vocabulary and the explicit reason not to reuse `Window2dAttrs`.
- 0018N: exact typed [infinity](../../../../glossary.md#term-infinity) and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) representation vocabulary.
- 0020: NCHW naming, checked static/symbolic [spatial](../../../../glossary.md#term-spatial-axes) formula, [metadata](../../../../glossary.md#term-metadata), and [lifecycle](../../../../glossary.md#lifecycle) precedent.

## Follow-up tasks

- 0020A1 remains Draft for average pooling with its own attributes, divisor/count-padding,
  accumulation, special-value, empty/all-padding, and validation contracts. It may extend<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [padding glossary](../../../../glossary.md#padding); [validation glossary](../../../../glossary.md#term-validation))</span>
  `Pool2dKind` but must not weaken or [alias](../../../../glossary.md#aliasing-and-arena) `MaxPool2dAttrs`.
- [Compiler](../../../../glossary.md#term-compiler) later owns [capture](../../../../glossary.md#term-capture-meanings), [binding](../../../../glossary.md#term-binding)/constraints, [gradients](../../../../glossary.md#term-gradient), adjoints, and saved max indices.
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
operation/signature/Tensor/provenance/symbolic-extent/typed-scalar/reduction/window/Conv2d tasks,
current related source/tests/APIs/glossary, and task 0020A.

Implement task 0020A exactly inside 18 paths. Update every global public Tensor inventory/count
177 -> 178 up front. Preserve all contracts and stop on architecture uncertainty, scope overflow,
another type/test/document, existing-helper change, Gradle, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and Java evidence to a separate clean documentation-focused agent in the same change; it
finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and documentation checks
while reusing Java evidence. Keep 0020A Ready until all criteria pass, then mark it Complete while
0020A1 and 0021–0024 remain Draft with no next detailed specification.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide this task, complete diff, exact focused/final Java evidence and post-test Java-change
state, public/[Shape](../../../../glossary.md#shape)/type/ceil/[extrema](../../../../glossary.md#saturation-pole-and-extremum)/[padding](../../../../glossary.md#padding)/[provenance](../../../../glossary.md#provenance) policies, seven documentation paths, and
[validation](../../../../glossary.md#term-validation) requirements. The clean agent reads repository instructions, architecture, rules and
General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning)/Example profiles, task, source/tests/generated Javadoc,
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/[Runtime](../../../../glossary.md#term-runtime)/Training APIs, glossary/planning, and directly related contracts. It
finalizes documentation and records reasoned no-change conclusions without repeating successful
Java tests absent executable change, stale evidence, or a concrete risk.

## Local decisions

- Max [pooling](../../../../glossary.md#term-pooling) uses its own `MaxPool2dAttrs` rather than `Window2dAttrs`: identical coordinate
  fields do not make conceptual-zero window transforms and excluded-padding extrema the same
  contract.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [padding glossary](../../../../glossary.md#padding); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Literal ceil mode retains the complete ceiling grid without a terminal-window decrement. This
  keeps static and symbolic formulas identical and allows an all-padding terminal window.
- The operation exposes no indices result. First-logical-sample ties fix forward meaning and
  future gradient evidence without selecting a current gradient or saved-index representation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [gradient glossary](../../../../glossary.md#term-gradient))</span>

## Known limitations

- This task implements model semantics and public expression metadata only. Compiler capture,
  binding proof, gradients/adjoints, saved indices, backend lowering, algorithms, prepared
  execution, conformance, and integration remain planned in their owning layers.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [gradients glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Average pooling remains Draft task 0020A1 because divisor and padding-count semantics require a
  separate contract.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [padding glossary](../../../../glossary.md#padding))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0020a_implementation` ran the specified focused Gradle
  command after correcting its test fixture: passed, 33 tests. The same context then ran exactly
  one final <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [test fixture glossary](../../../../glossary.md#adr-ci-and-qualification))</span>`./gradlew :modules:model:test`: passed, 870 tests across 111 suites. Executable Java
  did not change after that final suite; the documentation pass reused this evidence and did not
  repeat Java tests.
- Documentation-focused clean context
  `/root/task_0020a_implementation/task_0020a_docs` applied the General, API/Javadoc, Planning,
  and Example profiles. It reviewed the complete task diff; all four production files; both
  focused tests; <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>`TensorTest` and count locks; Tensor, Compile, Runtime, and Training API pages;
  glossary and planning state; and related Conv2d, window-transform, reduction, Shape-expression,
  operation/signature, Tensor/provenance, architecture, and lifecycle contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The documentation pass finalized Javadocs in the four task-owned production paths and updated
  the Tensor API with the geometry table, static floor/ceil and symbolic examples, extrema policy,
  and lifecycle boundary. It updated the Compile API with current metadata and planned compiler
  ownership, and refined reusable NCHW pooling/window/padding/dilation/effective-kernel/ceil-mode
  glossary distinctions.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [NCHW glossary](../../../../glossary.md#term-nchw); [pooling glossary](../../../../glossary.md#term-pooling); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [kernel glossary](../../../../glossary.md#kernel))</span>
- Documentation validation: `./gradlew :modules:model:javadoc` passed. The static Tensor API
  example compiled and ran with the documented two output Shapes. Local Markdown links and
  anchors, fences, final newlines, whitespace, generated Javadoc content, exact 18 paths,
  production/test packages, public Tensor count 178, `MAX_POOL2D` one-input/one-output signature,
  synchronized statuses, and absence of detailed 0020A1-or-later specifications were checked and
  passed. `git diff --check` passed.
- Runtime and Training API pages require no change: they describe planned prepared execution and
  autograd/training boundaries, while this task adds no runtime or training API. Related Conv2d,
  unfold/fold, pad, reduction, Shape/Dimension-expression, operation/signature, and
  Tensor/provenance contracts remain accurate and unchanged because max pooling owns a new
  operation-specific type and construction helper. Architecture documents/tests, backend
  conformance, integration tests, Gradle, and other modules require no change because there is no
  dependency, lifecycle, backend, execution, build, or cross-module change.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [autograd glossary](../../../../glossary.md#autograd); [Dimension glossary](../../../../glossary.md#dimension); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

- Added `Pool2dKind.MAX_POOL2D`, exact [immutable](../../../../glossary.md#term-immutable) `MaxPool2dAttrs`, one package-private construction
  helper, and exactly `Tensor.maxPool2d(MaxPool2dAttrs)`.
- Construction accepts exactly BFLOAT16/FLOAT32/FLOAT64 rank-four NCHW input, retains exact type,
  batch/channel references and gradient request, derives checked static or canonical symbolic
  floor/ceil spatial Dimensions, and creates one fresh unresolved-layout, unlabeled, storage-free
  output with exact one-input/output-zero provenance.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [rank glossary](../../../../glossary.md#term-rank); [NCHW glossary](../../../../glossary.md#term-nchw); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [gradient glossary](../../../../glossary.md#term-gradient); [canonical glossary](../../../../glossary.md#term-canonical-form); [spatial glossary](../../../../glossary.md#term-spatial-axes); [Dimensions glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Public and planning documentation records excluded padding, all-padding negative infinity, NaN
  dominance, positive-zero ordering, ordinary infinity behavior, and first height-major kernel
  sample ties without claiming evaluation or downstream lifecycle support.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [padding glossary](../../../../glossary.md#padding); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [kernel glossary](../../../../glossary.md#kernel); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [evaluation glossary](../../../../glossary.md#term-evaluation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Completion summary

- Completed changes: added NCHW maximum-pooling semantics, attributes, public Tensor expression,
  focused tests, complete Javadocs, public API explanations, glossary distinctions, and
  synchronized capability/planning status.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly four production, seven test, and seven documentation/planning
  paths listed in this specification.
- Tests and validation: reused focused 33-test and final 870-test/111-suite implementation
  evidence; documentation Javadoc, runnable example, links/anchors/fences/newlines/whitespace,
  generated output, scope/public/signature/status/no-spec checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [signature glossary](../../../../glossary.md#term-signature))</span>`git diff --check` passed.
- Documentation-agent review: completed in clean context
  `/root/task_0020a_implementation/task_0020a_docs` using the required profiles.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap finalized; Runtime/Training and related-contract no-change conclusions are recorded.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: all four production paths reviewed and finalized; generated model Javadoc
  passed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Glossary impact: added reusable NCHW pooling window distinctions without redefining ordinary
  programming terms.
- Unresolved issues: None.
- Follow-up required: None for task 0020A; Draft 0020A1 remains future average-[pooling](../../../../glossary.md#term-pooling) work.

Status: Complete
