# Task 0025G: [NCW](../../../../glossary.md#term-ncw) [Conv1d Composition](../../../../glossary.md#conv1d-composition)

**Editorial reading key (current context, not a status update).**

Read this completed task as historical Model scope, not a current execution-status statement. NCW
means batch, channel, width; groups isolate channel subsets, and cross-correlation does not reverse
the kernel. The caller supplies input, weight and optional bias Tensor descriptions to
`Tensor.conv1d`; its helper delegates to two expansions, Conv2d and squeeze. `Conv1dAttrs` becomes
`Conv2dAttrs`, not a retained Conv1d operation. The [current composition
reference](../../../../reference/tensor/convolution.md#grouped-ncw-conv1d-composition) explains this
mapping: inserting height one leaves width as the varying spatial axis; bias passes directly to
Conv2d.

Model semantics defines meaning; supported execution providers realize it rather than redefine it.
Compiler consumes provenance and descriptors, not computed convolution values. These storage-free
expressions neither execute nor own result buffers; convolution’s semantic responsibility is not
storage ownership. Conceptual positive-zero padding still multiplies infinite weights. Follow the
[current numerical and metadata
policy](../../../../reference/tensor/convolution.md#numerical-and-metadata-policy) for that boundary
and qualified execution descriptions; this record’s unavailable-execution and future-Conv3d
statements remain historical.

## Status

Complete

## Goal

Add public one-dimensional grouped cross-correlation for [tensors](../../../../glossary.md#tensor) in [NCW](../../../../glossary.md#term-ncw) order: [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), [channel](../../../../glossary.md#term-channel),
width. The [API](../../../../glossary.md#term-application-programming-interface-api) must be a [rank](../../../../glossary.md#term-rank)-specific convenience over current [Model](../../../../glossary.md#term-model) operations, not a new
semantic [operation](../../../../glossary.md#operation). Every successful call therefore returns the exact visible composition:

```text
NCW input   --EXPAND_DIMS(axis 2)--\
                                      CONV2D --SQUEEZE(axis 2)--> NCW result
1-D weight --EXPAND_DIMS(axis 2)--/
optional bias --------------------/
```

The two inserted [axes](../../../../glossary.md#term-axis) are [singleton](../../../../glossary.md#term-singleton) height axes. The resulting [`CONV2D`](../../../../glossary.md#convolution--conv2d) [occurrence](../../../../glossary.md#term-occurrence) remains the
single [owner](../../../../glossary.md#term-owner) of [convolution](../../../../glossary.md#term-channels-first-convolution-layers) meaning, [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference) and [gradients](../../../../glossary.md#term-gradient), and later Conv2d
execution.

## [Scope](../../../../glossary.md#term-scope)

- Add [immutable](../../../../glossary.md#term-immutable) public `Conv1dAttrs(stride, padding, dilation, groups)` with declaration-order
  validation and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`defaults()` equal to `(1, 0, 1, 1)`.
- Add exactly these public receiver methods to [`Tensor`](../../../../glossary.md#tensor):

  ```java
  public Tensor conv1d(Tensor weight, Conv1dAttrs attrs)

  public Tensor conv1d(Tensor weight, Tensor bias, Conv1dAttrs attrs)
  ```

- Add one package-private, final, field-free `TensorConv1dExpressions` construction [owner](../../../../glossary.md#term-owner).
- Accept floating input `[N, C_in, W]`, [weight](../../../../glossary.md#term-weight) `[C_out, C_in/groups, K_w]`, and optional [bias](../../../../glossary.md#term-bias)
  `[C_out]`; derive `[N, C_out, W_out]`.
- Validate the rank-one-dimensional contract locally before constructing the visible composition:
  nulls, floating roles, promotion, ranks, positive static kernel width, group divisibility,
  grouped input/weight compatibility, bias length, and checked static/symbolic output geometry.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- Expand input and weight at [axis](../../../../glossary.md#term-axis) `2`, map the [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) geometry to
  `Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`, call the existing biased or unbiased
  [`conv2d`](../../../../glossary.md#convolution--conv2d), and squeeze the Conv2d height axis `2`.
- Preserve current Conv2d grouped cross-correlation, conceptual padding, floating promotion,
  accumulation, special-value, reassociation, and determinism policy.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Test exact Shapes, validation/effects, public surface, four-producer provenance, canonical
  wrappers, freshness, and static/symbolic Dimension relations.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [canonical glossary](../../../../glossary.md#term-canonical-form); [Dimension glossary](../../../../glossary.md#dimension))</span>
- Finalize affected Javadocs, Tensor/Compile APIs, glossary, Model capabilities, and planning in a
  separate clean documentation-focused context before completion.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- A `CONV1D` kind, signature, producer, attributes occurrence, compiler inventory row, captured
  node, backend operation, or capability advertisement.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence); [compiler glossary](../../../../glossary.md#term-compiler); [node glossary](../../../../glossary.md#node); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- [`Conv3d`](../../../../glossary.md#conv3d--ncdhw), public or private `ConvNd`, arbitrary rank, geometry arrays, asymmetric intrinsic
  padding, padding modes, transposed convolution, causal convolution, depthwise-specific APIs,
  separable convolution, quantized convolution, or another layout such as NWC.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [padding glossary](../../../../glossary.md#padding); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [layout glossary](../../../../glossary.md#layout))</span>
- A neural-network layer, [module](../../../../glossary.md#term-module), [parameter](../../../../glossary.md#term-parameter), initializer, factory recipe, state, or training [API](../../../../glossary.md#term-application-programming-interface-api).
- Compiler, autograd, graph verification, Planning, Prepare, Runtime, Engine, backend, lowering,
  fusion, generated code, execution, conformance, or integration implementation.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Hidden fusion, producer elision, expand/squeeze cancellation, direct construction of a Conv2d
  producer, a synthetic provenance wrapper, or a specialized factory seam.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [provenance glossary](../../../../glossary.md#provenance))</span>
- Changes to `Conv2dKind`, `Conv2dAttrs`, `TensorConv2dExpressions`, rank-editing operations,
  TensorFactory, Shape/Dimension arithmetic, current convolution-gradient formulas, or operation
  signatures.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- A detailed specification for 0025H, 0026, Compiler 0006B, CPU 0008/0008A, Engine 0004, or NN
  0025.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Legacy source, package structure, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), or implementation reuse.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Expand-dimensions and squeeze expressions](0017f1-expand-dimensions-and-squeeze-tensor-expressions.md)<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>
- [NCHW Conv2d semantics and Tensor expressions](0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Tensor glossary](../../../../glossary.md#tensor))</span>
- [Canonical TensorProducer outputs](0025-canonical-tensor-producer-outputs.md)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [TensorProducer glossary](../../../../glossary.md#producer--tensorproducer))</span>
- [Compiler convolution gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [gradients glossary](../../../../glossary.md#term-gradient))</span>

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable Tensor
  expression metadata; it does not capture graphs, select execution, or depend on downstream
  modules.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [modules glossary](../../../../glossary.md#term-module))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains the public fluent receiver facade. The helper direction is Tensor to existing
  operation/datatype/shape contracts, preserving acyclic package dependencies.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [operation glossary](../../../../glossary.md#operation); [shape glossary](../../../../glossary.md#shape))</span>
- The constructed [graph](../../../../glossary.md#graph) is exactly two `EXPAND_DIMS` occurrences, one [`CONV2D`](../../../../glossary.md#convolution--conv2d) [occurrence](../../../../glossary.md#term-occurrence), and one
  `SQUEEZE` occurrence. No code path may replace, conceal, tag, or fuse that structure.
- Each derived Tensor is the canonical wrapper created by its existing factory call. Provenance
  remains per-producer, ordered, immutable, and output-index zero.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Provenance glossary](../../../../glossary.md#provenance))</span>
- `Conv1dAttrs` is a [rank](../../../../glossary.md#term-rank)-specific [immutable](../../../../glossary.md#term-immutable) semantic-[parameter](../../../../glossary.md#term-parameter) value. It implements
  [`OperationAttrs`](../../../../glossary.md#operationattrs), but no [`OperationKind`](../../../../glossary.md#operationkind) accepts or retains it: construction translates it to a
  fresh `Conv2dAttrs`, which is the exact attributes value on the sole [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [producer](../../../../glossary.md#term-producer).
- Compiler owns capture, deferred Shape proof, gradients, and legal rewrites. Backend Prepare owns
  algorithms, lowering, fusion, allocation, and materialization. Runtime receives only prepared
  work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient); [Backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- No architecture, module-boundary, dependency, lifecycle, build, or architecture-test rule
  changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.convolution`
- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model.tensor`

Packages added or changed:

- change `io.github.pho001.synaptik.model.operation.convolution` only by adding the rank-one
  immutable parameter record;<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [immutable glossary](../../../../glossary.md#term-immutable); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- change `io.github.pho001.synaptik.model.tensor` only by adding the package-private composition
  owner and two receiver overloads;<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- add no package and widen no existing type or factory seam.

Type placement:

- `...operation.convolution.Conv1dAttrs` owns only validated one-axis stride, symmetric padding,
  dilation, and group parameters.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- `...tensor.TensorConv1dExpressions` owns rank-specific prevalidation, exact parameter mapping,
  and calls to the existing public Tensor compositions.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `...tensor.Tensor` exposes the two natural single-output receiver entries, parallel to current
  [`conv2d`](../../../../glossary.md#convolution--conv2d) and [`linear`](../../../../glossary.md#term-linear-meanings) composition entries.
- Existing `Conv2dKind`, `Conv2dAttrs`, `AxisTransformKind`, `AxisTransformAttrs`, and their
  expression owners remain unchanged and own every actual producer.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>

## Exact semantic and expression contract

### [Shapes](../../../../glossary.md#shape) and geometry

```text
input:   [N, C_in, W]
weight:  [C_out, C_in/groups, K_w]
bias:    [C_out]                              optional
result:  [N, C_out, W_out]

effectiveKernel = dilation * (K_w - 1) + 1
numerator       = W + 2 * padding - effectiveKernel
W_out           = floor(numerator / stride) + 1
```

`K_w` must be statically known and positive. Literal arithmetic is checked in signed `long`.
A static negative numerator fails locally; zero yields one output position. For unresolved `W`,
use the same [canonical](../../../../glossary.md#term-canonical-form) `DimensionExpressions.addConstant`, `floorDivide`, and `addConstant`
sequence as [Conv2d](../../../../glossary.md#convolution--conv2d) width derivation. The future concrete [binding](../../../../glossary.md#term-binding) retains the obligation that the
numerator is non-negative.

The result reuses the exact input [batch Dimension](../../../../glossary.md#batch-dimension) and [weight](../../../../glossary.md#term-weight) output-[channel](../../../../glossary.md#term-channel) [Dimension](../../../../glossary.md#dimension) references.
Static input and output channels must each be divisible by `groups`. When both sides are static,
`weightChannelsPerGroup * groups == inputChannels` is checked with exact multiplication. Static
[bias](../../../../glossary.md#term-bias) length must equal static output channels. Relations involving an unresolved Dimension are
deferred exactly as for Conv2d and remain visible through the expanded [descriptors](../../../../glossary.md#term-descriptor) and mapped
Conv2d attributes.

### Attributes and mapping

`Conv1dAttrs` components are validated in declaration order:

```text
stride must be positive: <value>
padding must be non-negative: <value>
dilation must be positive: <value>
groups must be positive: <value>
```

Every call creates this exact mapping:

```java
new Conv2dAttrs(1, attrs.stride(), 0, attrs.padding(), 1, attrs.dilation(), attrs.groups())
```

The new `Conv2dAttrs` is retained by exact reference on the [Conv2d](../../../../glossary.md#convolution--conv2d) [operation](../../../../glossary.md#operation). `Conv1dAttrs` is not
retained by a [producer](../../../../glossary.md#term-producer) because there is no Conv1d operation [occurrence](../../../../glossary.md#term-occurrence).

### Values and numerical policy

Conv1d means grouped [NCW](../../../../glossary.md#term-ncw) cross-correlation. It does not reverse the stored [kernel](../../../../glossary.md#kernel). Output [channel](../../../../glossary.md#term-channel)
`o` selects its [contiguous](../../../../glossary.md#term-contiguous) channel group and sums products across the group's input channels and
kernel width positions. The optional [bias](../../../../glossary.md#term-bias) for `o` participates once. Out-of-range width positions
are conceptual positive-zero inputs and still participate in ordinary floating multiplication.

Input, [weight](../../../../glossary.md#term-weight), and present bias must each be floating. Promotion processes input and weight, then
bias. [FLOAT64](../../../../glossary.md#term-float64) output accumulates in FLOAT64; [FLOAT32](../../../../glossary.md#term-float32) and [BFLOAT16](../../../../glossary.md#term-bfloat16) output accumulate in FLOAT32,
with final BFLOAT16 conversion. [NaN, infinity, and signed zero](../../../../glossary.md#nan-infinity-and-signed-zero) follow the current [Conv2d](../../../../glossary.md#convolution--conv2d) meaning,
including conceptual zero multiplied by [infinity](../../../../glossary.md#term-infinity). Empty channel [contraction](../../../../glossary.md#contraction-dimension) begins with [positive
zero](../../../../glossary.md#nan-infinity-and-signed-zero) before optional bias. Empty [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) or output-channel [axes](../../../../glossary.md#term-axis) are valid. [Reassociation and fused
multiply-add](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) are allowed, so bitwise or cross-[backend](../../../../glossary.md#backend)-identical [rounding](../../../../glossary.md#term-rounding) is not promised.

### Exact visible [provenance](../../../../glossary.md#provenance)

Successful construction occurs in this order:

1. `input.expandDims(2)` produces `[N, C_in, 1, W]` with `EXPAND_DIMS` and
   `AxisTransformAttrs(2)`.
2. `weight.expandDims(2)` produces `[C_out, C_in/groups, 1, K_w]` with a distinct
   `EXPAND_DIMS` and `AxisTransformAttrs(2)`.
3. The biased or unbiased [`conv2d`](../../../../glossary.md#convolution--conv2d) call produces `[N, C_out, 1, W_out]` with mapped
   `Conv2dAttrs` and ordered inputs `[expandedInput, expandedWeight]` or
   `[expandedInput, expandedWeight, bias]`. [Bias](../../../../glossary.md#term-bias) is not expanded.
4. `conv2dResult.squeeze(2)` produces `[N, C_out, W_out]` with `SQUEEZE` and
   `AxisTransformAttrs(2)`.

The returned [Tensor](../../../../glossary.md#tensor)'s immediate [producer](../../../../glossary.md#term-producer) is `SQUEEZE`, not `CONV2D`. Each success consumes exactly
four Tensor IDs and creates four [identity](../../../../glossary.md#term-identity-meanings)-distinct one-output producers and [canonical](../../../../glossary.md#term-canonical-form) wrappers.
Equivalent calls remain fresh. Final type and [gradient](../../../../glossary.md#term-gradient) eligibility are the Conv2d promoted type
and logical OR over original input, [weight](../../../../glossary.md#term-weight), and optional bias. Final label and [host storage](../../../../glossary.md#host-storage) are
empty, and final [layout](../../../../glossary.md#layout) is unresolved because current Conv2d construction produces unresolved
layout.

### [Validation](../../../../glossary.md#term-validation) and effects

Each overload null-checks in declaration order: `input`, [`weight`](../../../../glossary.md#term-weight), optional [`bias`](../../../../glossary.md#term-bias), then `attrs`,
using the [parameter](../../../../glossary.md#term-parameter) name as the `NullPointerException` message. Before the first composition call,
the helper validates in this order:

1. input, weight, then present-bias floating eligibility;
2. input/weight and then present-bias floating promotion;
3. input [rank](../../../../glossary.md#term-rank) three, weight rank three, then present-bias rank one;
4. positive static [kernel](../../../../glossary.md#kernel) width;
5. static input-channel divisibility, output-channel divisibility, grouped weight/input equality,
   then bias/output equality;<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel))</span>
6. checked effective-kernel and output-width geometry; and
7. construct the mapped `Conv2dAttrs`.

Task-owned failures use `conv1d` and the [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) role names:

```text
conv1d input must have a floating data type, but was <dataType>
conv1d weight must have a floating data type, but was <dataType>
conv1d bias must have a floating data type, but was <dataType>
conv1d input rank must be 3: <rank>
conv1d weight rank must be 3: <rank>
conv1d bias rank must be 1: <rank>
conv1d kernel width must be static: <dimension>
conv1d kernel width must be positive: <dimension>
conv1d input channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv1d output channels must be divisible by groups: channels=<dimension>, groups=<groups>
conv1d weight channels per group do not match input channels: weight=<dimension>, groups=<groups>, input=<dimension>
conv1d bias length must match output channels: bias=<dimension>, output=<dimension>
conv1d effective kernel does not fit padded width: input=<dimension>, effectiveKernel=<value>, padding=<value>
```

Every task-owned validation failure occurs before an ID is consumed. After composition begins,
existing rank-editing, [Conv2d](../../../../glossary.md#convolution--conv2d), factory, and ID-exhaustion effects remain visible: a later delegated
failure may leave earlier successful intermediate IDs consumed. Do not add [rollback](../../../../glossary.md#term-rollback), hidden
preallocation, direct [producer](../../../../glossary.md#term-producer) construction, or duplicated rank-editing [layout](../../../../glossary.md#layout) implementation to
make the composition appear [atomic](../../../../glossary.md#resource-transaction-and-rollback).

## Affected files

Expected production source (3):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv1dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConv1dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected [Model](../../../../glossary.md#term-model) tests (16):

- add `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/convolution/Conv1dAttrsTest.java`
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv1dExpressionTest.java`
- update the exact [Tensor](../../../../glossary.md#tensor) public-surface locks in `TensorTest`, `RecurrentScanExpressionTest`,
  `TensorBatchNormInferenceExpressionTest`, `TensorBinaryArithmeticTest`,
  `TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest`,
  `TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest`, `TensorLayerNormExpressionTest`,
  `TensorLinearExpressionTest`, `TensorMatmulExpressionTest`, `TensorMeanSquaredErrorExpressionTest`,
  `TensorRmsNormExpressionTest`, `TensorScaledDotProductAttentionExpressionTest`,
  `TensorSlicePlacementExpressionTest`, and `TensorSumToShapeExpressionTest`, changing only the
  expected method count from 202 to 204 except for the direct `conv1d` signature/name assertions
  owned by `TensorTest` and `TensorConv1dExpressionTest`.

Expected documentation/[planning](../../../../glossary.md#term-planning) (8):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`
- `docs/planning/backends/cpu/master-plan.md` — planning-stage-only correction of the one stale
  sentence that called completed CPU 0007F2 Ready; implementation must preserve all other CPU
  evidence and wording.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Ready glossary](../../../../glossary.md#term-ready))</span>

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime) and Training APIs; [Conv2d](../../../../glossary.md#convolution--conv2d)/window/[rank](../../../../glossary.md#term-rank)-editing,
[operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature), TensorFactory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) and [gradient](../../../../glossary.md#term-gradient) contracts;
architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 27 paths maximum: three production, sixteen [Model](../../../../glossary.md#term-model) test, and eight
documentation/[planning](../../../../glossary.md#term-planning) paths. No production or test path outside Model may change. The [CPU](../../../../glossary.md#term-cpu) master
plan may contain only the planning-stage [`Ready`](../../../../glossary.md#term-ready) to `Complete` correction already named above.
Stop for path 28, another type/test/document, an existing-helper modification, architecture,
[Gradle](../../../../glossary.md#term-gradle), [dependency](../../../../glossary.md#term-dependency), or cross-[module](../../../../glossary.md#term-module) executable work.

This is one cohesive Model capability despite four visible [producers](../../../../glossary.md#term-producer): the public [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
contract, attributes, prevalidation, exact existing-[operation](../../../../glossary.md#operation) composition, public locks, tests,
and documentation must land together.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document `Conv1dAttrs`, its [canonical](../../../../glossary.md#term-canonical-form) constructor and `defaults()`, the package-private
  helper, and both Tensor methods. Every parameter, result, nullability rule, failure, and
  arithmetic/ID effect receives the required tags.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Tensor API explains NCW, Shapes, mapping, groups, formula, symbolic relations, numerical policy,
  exact four-node provenance, and a complete biased grouped metadata example.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [NCW glossary](../../../../glossary.md#term-ncw); [Shapes glossary](../../../../glossary.md#shape); [node glossary](../../../../glossary.md#node); [provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Compile API states that generic capture, current Conv2d inference and Compiler 0005D gradients
  see ordinary expand/Conv2d/squeeze nodes; it makes no fusion, execution, or capability claim.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [inference glossary](../../../../glossary.md#term-inference); [Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Glossary distinguishes Conv1d composition from first-class Conv2d and planned Conv3d while
  reusing cross-correlation, group, dilation, effective-kernel, and conceptual-padding terms.<span data-glossary-navigation="true"> ([Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [dilation glossary](../../../../glossary.md#term-dilation); [kernel glossary](../../../../glossary.md#kernel); [padding glossary](../../../../glossary.md#padding))</span>
- Model capabilities distinguishes the current public convenience from a new operation-kind
  inventory entry.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- Synchronize this task and Model master to Complete only after implementation, documentation,
  and validation pass. Keep 0025H and 0026 Draft without detailed specifications.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture/tests, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `Conv1dAttrs` has exactly four record components, implements [`OperationAttrs`](../../../../glossary.md#operationattrs), validates in the
  specified order/messages, and returns exact defaults.
- Exactly two `conv1d` Tensor receiver overloads exist; public Tensor method count is 204. There is
  no alias, static namespace, nullable-bias form, default-attrs overload, or public ConvNd.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena); [bias glossary](../../../../glossary.md#term-bias))</span>
- Static and symbolic Shape derivation, exact Dimension-reference retention, checked geometry,
  group/channel/bias relations, and deferred unresolved relations match this task.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [channel glossary](../../../../glossary.md#term-channel))</span>
- Every floating ordered width combination promotes correctly; BOOL and integral roles fail.
  Numerical meaning is documented/tested as metadata without Model value evaluation.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [metadata glossary](../../../../glossary.md#term-metadata); [Model glossary](../../../../glossary.md#term-model); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- [Provenance](../../../../glossary.md#provenance) is exactly `EXPAND_DIMS(input,2)`, `EXPAND_DIMS(weight,2)`, mapped [`CONV2D`](../../../../glossary.md#convolution--conv2d), then
  `SQUEEZE(2)`, with optional bias direct at Conv2d input two. Four IDs/producers/wrappers,
  freshness, storage/layout/label, requires-grad OR, and delegated-failure effects pass.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Current Compiler capture/inference/gradient applicability follows only from existing operation
  support. No Compiler inventory, source, test, or capability changes.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation))</span>
- No `CONV1D`, Conv3d, compiler/backend/runtime/NN behavior, generated code, hidden fusion,
  architecture, dependency, or build work is added.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [fusion glossary](../../../../glossary.md#term-fusion); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests, exactly one final Model suite, Javadoc/docs checks, exact 27 paths/public surface,
  statuses/order/dependencies, no 0025H spec, and whitespace validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [validation glossary](../../../../glossary.md#term-validation))</span>
- A separate clean documentation-focused pass finalizes Javadocs/docs and records no-change
  conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrsTest --tests io.github.pho001.synaptik.model.tensor.TensorConv1dExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorConv2dExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorRankEditingExpressionTest --tests io.github.pho001.synaptik.model.shape.DimensionExpressionsTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links/anchors, balanced fences, final newlines, trailing whitespace,
exact paths/packages/[signatures](../../../../glossary.md#term-signature)/counts, no `CONV1D` or ConvNd production [symbol](../../../../glossary.md#term-symbol), task/master/roadmap
status consistency, hard 0025G-before-0025H ordering, and absence of a 0025H task file. Repository-
wide tests are deferred to the next recorded cross-[module](../../../../glossary.md#term-module) [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task
changes one module without [dependency](../../../../glossary.md#term-dependency) or architecture boundaries.

Generated-code validation is explicitly non-applicable. [Model](../../../../glossary.md#term-model) builds [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata) and the
ordinary four-[node](../../../../glossary.md#node) expression only; it generates no Java source, Class-File, executable [kernel](../../../../glossary.md#kernel),
or hot loop. Generated/direct [oracle](../../../../glossary.md#term-test-oracle), [bytecode](../../../../glossary.md#term-bytecode) inspection, and performance evidence belong to the
later [CPU](../../../../glossary.md#term-cpu) execution tasks.

## Dependencies

- Completed [Model 0020](0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span> supplies grouped
  NCHW cross-correlation, mapped geometry, floating policy, Shape relations, and optional bias.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Shape glossary](../../../../glossary.md#shape); [bias glossary](../../../../glossary.md#term-bias))</span>
- Completed [Model 0017F1](0017f1-expand-dimensions-and-squeeze-tensor-expressions.md) supplies the
  exact public singleton-axis producers and canonical wrappers used on both sides of Conv2d.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [canonical glossary](../../../../glossary.md#term-canonical-form); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Completed [Model 0025](0025-canonical-tensor-producer-outputs.md) supplies canonical producer
  output wrappers and immutable provenance identity.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Complete [Compiler](../../../../glossary.md#term-compiler) 0004A [rank](../../../../glossary.md#term-rank)-edit [gradients](../../../../glossary.md#term-gradient) and
  [Compiler 0005D](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
  supply the current downstream gradient coverage inherited by the visible composition; this
  task changes neither contract.
- Model 0025F was Complete when implementation began. Task 0025G was then the next unfinished
  Model row and sole Ready frontier, so no out-of-order exception was used.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Follow-up tasks

- Model 0025H remains Draft without a detailed task specification. It depends on completed 0025G
  and separately owns first-class grouped NCDHW Conv3d because current operations cannot express
  it with graph size independent of spatial extents.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [graph glossary](../../../../glossary.md#graph); [spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- Compiler 0006B follows Model 0025H for Conv3d forward adoption. Conv1d needs no Compiler task or
  inventory row because it is ordinary existing-operation composition.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [operation glossary](../../../../glossary.md#operation))</span>
- CPU 0008 later implements Conv2d execution; CPU 0008A then validates this visible Conv1d
  composition end to end and adds Conv3d execution. Fusion remains later CPU 0008B–0008E work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Fusion glossary](../../../../glossary.md#term-fusion))</span>
- Engine 0004 and NN 0025 remain separate lifecycle and layer owners after the ordered
  Model/Compiler/CPU prerequisites.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Asymmetric padding, other layouts, transposed/depthwise-specific APIs, and mixed precision need
  separately justified future tasks.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [layouts glossary](../../../../glossary.md#layout); [mixed precision glossary](../../../../glossary.md#term-mixed-precision))</span>

Do not create another detailed specification during implementation.

## Architecture impact

Expected impact: None.

This task uses existing public [Model](../../../../glossary.md#term-model) expressions and adds a bounded public convenience with
[rank](../../../../glossary.md#term-rank)-specific [parameters](../../../../glossary.md#term-parameter). It changes no semantic-kind inventory, [ownership](../../../../glossary.md#term-ownership-meanings), package [dependency](../../../../glossary.md#term-dependency),
[graph](../../../../glossary.md#graph) representation, [lifecycle](../../../../glossary.md#lifecycle), or executable boundary. If implementation requires any such
change, stop and report the conflict.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik Model task 0025G in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit, stage, or push.

Read AGENTS.md, ARCHITECTURE.md, the focused architecture and documentation rules/profiles,
planning guide/roadmap, Model capabilities/master plan, task 0025G, completed tasks 0017F1, 0020,
0025, and 0025F, and the complete current Conv2d/rank-edit/Tensor/provenance source, tests,
Javadocs, APIs, glossary, and Compiler 0005D boundary before editing.

Implement task 0025G exactly inside 27 paths. Preserve the planning-only CPU 0007F2 status
correction and every other CPU evidence line. Update all exact Tensor method locks 202 -> 204 up
front. Build the literal input-expand, weight-expand, Conv2d, squeeze composition; do not add a
CONV1D kind, direct producer, hidden fusion, ConvNd, compiler/backend/runtime/NN work, or another
detailed task. Stop on architecture uncertainty, scope overflow, another path, existing-helper
change, Gradle, dependency, or cross-module executable work.

Run focused validation and exactly one final Model suite after Java stabilizes. Then hand the
actual diff and exact Java evidence to a separate clean documentation-focused agent in the same
overall change. That agent must independently finalize affected Javadocs, Tensor/Compile APIs,
glossary, capabilities, and planning; record reasoned no-change conclusions; reuse successful
Java evidence unless executable code changes; and run final Javadoc/documentation checks.

Keep 0025G Ready until every implementation, documentation, and validation criterion passes.
Then mark the task/master/roadmap entry Complete while 0025H and 0026 remain Draft without task
files. Update validation evidence, implementation notes, and completion summary before reporting
Status: Complete.
```

## Local decisions

- `Tensor.conv1d` is selected because this is a natural single-output algebraic composition with
  the receiver as input, matching [`conv2d`](../../../../glossary.md#convolution--conv2d) and [`linear`](../../../../glossary.md#term-linear-meanings). A domain namespace is unnecessary.
- Bias presence uses two overloads. There is no nullable bias, bias flag, default-attributes
  overload, or alias.<span data-glossary-navigation="true"> ([Bias glossary](../../../../glossary.md#term-bias); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [Axis](../../../../glossary.md#term-axis) `2` is the inserted and removed [singleton](../../../../glossary.md#term-singleton) height axis. Inserting at axis `3` would model a
  singleton width and move convolution over the wrong spatial axis.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [spatial axis glossary](../../../../glossary.md#spatial-axes-channels-and-window-sampling))</span>
- `Conv1dAttrs` remains rank-specific and implements the semantic-parameter marker, but the actual
  producer retains only a freshly mapped <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [producer glossary](../../../../glossary.md#term-producer))</span>`Conv2dAttrs`. This makes inspection truthful: there is
  no Conv1d operation whose attributes could retain the convenience value.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [Rank](../../../../glossary.md#term-rank)-specific [validation](../../../../glossary.md#term-validation) occurs before composition so callers receive `conv1d` diagnostics and
  task-owned failures consume no ID. Once the first existing operation succeeds, ordinary
  delegated partial-ID effects remain visible rather than being hidden by an atomic facade.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [facade glossary](../../../../glossary.md#term-facade))</span>
- Current Compiler rank-edit and Conv2d inference/gradient rules apply structurally without
  knowing the <span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient))</span>`conv1d` Java method name. No [Compiler](../../../../glossary.md#term-compiler) change or test is warranted.
- Symmetric [padding](../../../../glossary.md#padding) is intrinsic. Asymmetric padding remains explicit [`PAD`](../../../../glossary.md#padding) composition rather
  than a silent change to the completed Conv2d contract.

## Known limitations

- Kernel width must be statically positive; only input width and channel/bias relations may remain
  symbolic under the current Shape contract.<span data-glossary-navigation="true"> ([Kernel glossary](../../../../glossary.md#kernel); [channel glossary](../../../../glossary.md#term-channel); [bias glossary](../../../../glossary.md#term-bias); [Shape glossary](../../../../glossary.md#shape))</span>
- Construction records meaning and provenance only. It does not evaluate values, bind unresolved
  extents, lower, prepare, allocate storage, choose an algorithm, fuse, or execute.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [prepare glossary](../../../../glossary.md#prepare))</span>
- Execution remains unavailable until the ordered Conv2d CPU route is implemented; no current
  backend capability follows from this Model API.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [CPU glossary](../../../../glossary.md#term-cpu); [backend capability glossary](../../../../glossary.md#backend-capability); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Final layout is unresolved even when original operands have resolved layouts because the
  existing Conv2d result layout is unresolved.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- The visible four-node composition may allocate earlier intermediate IDs before a later
  delegated layout, factory, or ID-space failure. No rollback is promised.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context supplied the final executable evidence. Its focused task command
passed, and its single final `./gradlew :modules:model:test` [run](../../../../glossary.md#term-run) passed 1,055 tests with zero
failures, errors, or skips. No executable Java changed afterward: the documentation context added
only [Javadoc](../../../../glossary.md#term-javadoc) and Markdown. The documentation pass therefore reused those results as required by
the [planning](../../../../glossary.md#term-planning) and documentation rules and did not rerun Java tests.

The mandatory separate clean documentation-focused context was `/root`. It applied the General,
[API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc, Planning, and Example profiles and independently reviewed the complete implementation
diff, all three production paths, all sixteen [Model](../../../../glossary.md#term-model) test paths, rendered affected public Javadoc,
[Tensor](../../../../glossary.md#tensor) and [Compile](../../../../glossary.md#compile) APIs, glossary, Model capabilities, planning status, completed Model
0017F1/0020/0025/0025F contracts, and the relevant [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference) and [convolution](../../../../glossary.md#term-channels-first-convolution-layers)-[gradient](../../../../glossary.md#term-gradient)
boundaries.

- `./gradlew :modules:model:javadoc` passed after the final Javadoc edits. Inspection of generated
  `Conv1dAttrs.html` and `Tensor.html` confirmed complete constructor, parameter, return, failure,
  Shape, provenance, numerical-policy, and identifier-effect contracts for the record and exactly
  two public receiver overloads. The package-private helper is intentionally absent from public
  generated Javadoc; its source Javadoc was reviewed in full.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- The documented Java 26 biased grouped example compiled and ran against Model classes. It printed
  `Shape[2, 6, 4]`, [`FLOAT64`](../../../../glossary.md#term-float64), `SQUEEZE`, [`CONV2D`](../../../../glossary.md#convolution--conv2d), and expanded [weight](../../../../glossary.md#term-weight)
  `Shape[6, 2, 1, 3]`, confirming the documented [metadata](../../../../glossary.md#term-metadata) and visible [producer](../../../../glossary.md#term-producer) boundary.
- `javap -public` showed exactly the two required `conv1d` methods and exactly 204 public Tensor
  methods. `javap -p`, source/reflection tests, import scans, and manual provenance review confirmed
  four record components, a final package-private field-free helper, exact two overloads, the two
  distinct axis-2 expansions, mapped <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`,
  direct optional bias, and axis-2 squeeze. Production/test scans found no <span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>`CONV1D` or `ConvNd`.
- A read-only documentation check validated 893 local Markdown links including target anchors,
  balanced fences, final newlines, no carriage returns, and no trailing whitespace across the
  eight documentation/planning paths. <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- Combined tracked/untracked scope inspection found exactly 27 permitted paths: three production,
  sixteen Model tests, and eight documentation/planning paths. The staging area is empty. No
  0025H specification exists. Task/master/roadmap mark 0025G Complete; 0025H and 0026 remain Draft
  without detailed specifications. The CPU master-plan diff remains the single requested 0007F2<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
  [`Ready`](../../../../glossary.md#term-ready) to `Complete` correction and preserves every other [CPU](../../../../glossary.md#term-cpu) evidence line.
- Final independent diff review found no stale execution, fusion, backend, or performance claim,
  no architecture or dependency change, and no work outside task 0025G.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

- Added [immutable](../../../../glossary.md#term-immutable) `Conv1dAttrs` and a bounded `TensorConv1dExpressions` construction owner. The
  two Tensor receiver methods validate the NCW contract and visibly construct<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Tensor glossary](../../../../glossary.md#tensor); [NCW glossary](../../../../glossary.md#term-ncw))</span>
  `EXPAND_DIMS(2) -> EXPAND_DIMS(2) -> CONV2D -> SQUEEZE(2)` with optional direct [bias](../../../../glossary.md#term-bias).
- The input becomes `[N, C_in, 1, W]`; [rank](../../../../glossary.md#term-rank)-three [weight](../../../../glossary.md#term-weight) becomes
  `[C_out, C_in/groups, 1, K_w]`; `Conv1dAttrs` maps to
  `Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`. Floating promotion and numerical
  meaning are inherited from Conv2d, while the final Tensor is the canonical squeeze output.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Focused tests cover attributes, API shape, static/symbolic dimensions, grouping, bias,
  promotion, failures and ID effects, exact provenance, canonical wrappers, and freshness. The
  fourteen existing exact Tensor method-count locks now expect 204.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [dimensions glossary](../../../../glossary.md#dimension); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tensor API now contains the complete current NCW contract and biased grouped metadata example.
  Compile API explains that generic capture, inference, and gradients reuse ordinary rank-edit and
  Conv2d nodes. The glossary distinguishes the composition from first-class Conv2d and future
  Conv3d, and Model capabilities records no new operation-kind inventory entry.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient); [nodes glossary](../../../../glossary.md#node); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- Runtime and Training APIs remain accurate unchanged because this task adds no runtime or
  training surface. Existing Conv2d, window, rank-editing, operation/signature, Tensor factory and
  provenance, Shape/Dimension, and gradient contracts remain accurate because the new API composes
  them without changing their meanings. Architecture/ADRs/tests, conformance/integration, Gradle,
  dependencies, and other modules require no change because no boundary, build, executable, or
  cross-module behavior changed.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [signature glossary](../../../../glossary.md#term-signature); [Tensor factory glossary](../../../../glossary.md#tensor-factory); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Completion summary

- Completed changes: Added the exact NCW Conv1d convenience, immutable geometry, local validation,
  four-producer composition, public locks/tests, complete Javadocs and explanatory documentation,
  and synchronized completion status.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [immutable glossary](../../../../glossary.md#term-immutable); [validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer))</span>
- Files changed or created: Exactly 27 permitted paths: three production, sixteen Model test, and
  eight documentation/planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Reused the implementation context's passing focused command and final
  1,055-test Model suite; final Model Javadoc, rendered inspection, Java 26 example,
  reflection/`javap`/import/manual structural checks, 893 local links/anchors, fences/newlines/
  whitespace, exact scope/status/staging checks, and `git diff --check` passed.
- Documentation-agent review: Clean documentation context `/root` independently finalized all
  affected Javadocs, APIs, glossary/capabilities impact, planning evidence, and status.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, task, master plan, and
  roadmap now explain the current composition and its boundaries; the sole CPU status correction
  is preserved.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `Conv1dAttrs`, its constructor/defaults, the helper and all helper methods, and
  both Tensor methods have complete accurate contracts; directly related Javadocs remain accurate
  unchanged.
- Glossary impact: Added [`Conv1d composition`](../../../../glossary.md#conv1d-composition) to distinguish the rank-specific convenience from
  first-class Conv2d and planned Conv3d while reusing existing convolution terminology.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Unresolved issues: None within task [scope](../../../../glossary.md#term-scope).
- Follow-up required: None for 0025G. [Model](../../../../glossary.md#term-model) 0025H and 0026 remain Draft without specifications.

Status: Complete
