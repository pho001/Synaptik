# Task 0025J: First-class [NCDHW](../../../../glossary.md#conv3d--ncdhw) max/average [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) semantics

## Status

Complete

## Goal

Add first-class, [rank](../../../../glossary.md#term-rank)-specific maximum and fixed-count average [pooling](../../../../glossary.md#term-pooling) for floating [NCDHW](../../../../glossary.md#conv3d--ncdhw)
[Tensors](../../../../glossary.md#tensor). Each successful receiver call creates one fresh one-input, one-output semantic [occurrence](../../../../glossary.md#term-occurrence):

```text
[N, C, D, H, W] -> MAX_POOL3D or AVERAGE_POOL3D -> [N, C, D_out, H_out, W_out]
```

[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) is first-class because enumerating depth windows through current operations would make the
[producer](../../../../glossary.md#term-producer) [graph](../../../../glossary.md#graph) depend on the input depth and cannot give one bounded, [Shape](../../../../glossary.md#shape)-independent
composition that preserves depth [padding](../../../../glossary.md#padding), [dilation](../../../../glossary.md#term-dilation), and literal ceil windows. This task owns [Model](../../../../glossary.md#term-model)
[metadata](../../../../glossary.md#term-metadata) and [provenance](../../../../glossary.md#provenance) only. [Compiler](../../../../glossary.md#term-compiler) adoption, [gradients](../../../../glossary.md#term-gradient), [lowering](../../../../glossary.md#lowering), [backend capability](../../../../glossary.md#backend-capability),
generated code, storage, [Runtime](../../../../glossary.md#term-runtime) behavior, and execution remain downstream work.

## [Scope](../../../../glossary.md#term-scope)

- Add [immutable](../../../../glossary.md#term-immutable) public `MaxPool3dAttrs` and `AveragePool3dAttrs` records with depth, height, and
  width kernel, stride, symmetric padding, and dilation components plus literal <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>`ceilMode`.
- Add one `Pool3dKind` family with exactly `MAX_POOL3D` and `AVERAGE_POOL3D`, each paired only with
  its own attributes class and an exact one-input, one-output signature.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Add `Tensor.maxPool3d(MaxPool3dAttrs)` and
  `Tensor.averagePool3d(AveragePool3dAttrs)`.
- Add separate package-private, final, field-free `TensorMaxPool3dExpressions` and
  `TensorAveragePool3dExpressions` construction [owners](../../../../glossary.md#term-owner).
- Accept only BFLOAT16, FLOAT32, and FLOAT64 rank-five NCDHW input, retain its exact type, batch and
  channel Dimension references, and gradient-request metadata, and derive exact static or
  canonical-symbolic depth, height, and width output Dimensions.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [rank glossary](../../../../glossary.md#term-rank); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [Dimension glossary](../../../../glossary.md#dimension); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Preserve the selected maximum, average, exceptional-value, rounding, failure-order, canonical-
  output, and provenance contracts below.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalize affected Javadocs and explanatory documentation in a distinct clean documentation-
  focused context before marking the task Complete.

## Out of [scope](../../../../glossary.md#term-scope)

- `unfold3d`, `fold3d`, Pool3d adjoints, saved maximum indices, pooling-specific gradient
  primitives, backward state, or any other Model gradient behavior.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [pooling glossary](../../../../glossary.md#term-pooling); [gradient glossary](../../../../glossary.md#term-gradient); [Model glossary](../../../../glossary.md#term-model))</span>
- Compiler forward adoption or descriptor validation, graph inventory changes, derivative
  allocation, lowering, Planning capability, Prepare behavior, CPU generated code, Runtime
  interpretation, execution, or performance claims.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [graph glossary](../../../../glossary.md#graph); [derivative glossary](../../../../glossary.md#term-derivative); [lowering glossary](../../../../glossary.md#lowering); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- A public or private `PoolNd`, dynamic-rank API, geometry arrays, adaptive/global pooling,
  asymmetric intrinsic padding, valid-sample average division, divisor override, or NDHWC layout.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [padding glossary](../../../../glossary.md#padding); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [layout glossary](../../../../glossary.md#layout))</span>
- Changes to Pool1d composition, Pool2d semantics, Conv3d, Shape/Dimension contracts, factory
  seams, storage, dependencies, Gradle, architecture contracts, or architecture tests.<span data-glossary-navigation="true"> ([Pool1d composition glossary](../../../../glossary.md#pool1d-composition); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Detailed specifications or implementation for Model 0025K, Compiler 0006B1/0006B2, CPU
  0008G1, or Model 0026.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capabilities](../capabilities.md)
- [NCHW Max Pool2d semantics](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw))</span>
- [NCHW Average Pool2d semantics](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [NCDHW Conv3d precedent](0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- [NCW Pool1d composition](0025i-ncw-max-average-pool1d-composition.md)<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [Pool1d composition glossary](../../../../glossary.md#pool1d-composition))</span>
- [Compiler Pool2d gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- [CPU Pool2d generated execution](../../../backends/cpu/tasks/0008g-portable-max-average-pool2d-execution.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable
  backend-independent operation semantics, Tensor descriptors, expression provenance, and
  construction-time validation; it does not capture graphs, create gradients, select algorithms,
  advertise backend support, allocate result storage, or execute.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [capture glossary](../../../../glossary.md#term-capture-meanings); [graphs glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains the fluent facade and delegates to package-private construction owners. The two
  numerical families remain separate; no broad pooling manager, helper, public facade, or shared
  mutable state is added.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [pooling glossary](../../../../glossary.md#term-pooling); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Every successful call creates one [`Operation`](../../../../glossary.md#operation), one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), and its one canonical output
  atomically through the existing factory seam. The returned Tensor is exactly producer<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
  `output(0)`.
- Each kind owns an [immutable](../../../../glossary.md#term-immutable) stable [singleton](../../../../glossary.md#term-singleton) [signature](../../../../glossary.md#term-signature) list. `MAX_POOL3D` accepts exactly
  `MaxPool3dAttrs`; `AVERAGE_POOL3D` accepts exactly `AveragePool3dAttrs`; each signature accepts
  exactly one input and one output.
- `requiresGrad` remains passive Tensor metadata. This task defines no derivative and makes no
  claim that Compiler can adopt, validate, differentiate, lower, prepare, or execute either kind.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [derivative glossary](../../../../glossary.md#term-derivative); [Compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare))</span>
- Compiler 0006B1 remains the sole forward-adoption and fail-closed-backward owner; Model 0025K and
  Compiler 0006B2 remain the ordered prerequisites for exact gradients; CPU 0008G1 remains the
  generated execution owner.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- No architecture, dependency, module-boundary, lifecycle, build, conformance, or integration-test
  rule changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Package impact

- Existing packages used:
  - `io.github.pho001.synaptik.model.operation` for [`OperationAttrs`](../../../../glossary.md#operationattrs), [`OperationKind`](../../../../glossary.md#operationkind),
    `OperationSignature`, and [`Operation`](../../../../glossary.md#operation);
  - `io.github.pho001.synaptik.model.operation.pooling` for the [rank](../../../../glossary.md#term-rank)-specific [pooling](../../../../glossary.md#term-pooling) family;
  - `io.github.pho001.synaptik.model.tensor` for Tensor receivers, descriptors, factory creation,
    producer ownership, and provenance;<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
  - `io.github.pho001.synaptik.model.shape` for static and [canonical](../../../../glossary.md#term-canonical-form) symbolic output geometry; and
  - `io.github.pho001.synaptik.model.datatype` for the existing floating-type boundary.
- Packages added or changed:
  - no package is added;
  - `io.github.pho001.synaptik.model.operation.pooling` gains two public attrs records and one
    public semantic-kind enum; and
  - `io.github.pho001.synaptik.model.tensor` gains two package-private construction owners and two
    public Tensor receiver methods.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- Type placement:
  - rank-specific [immutable](../../../../glossary.md#term-immutable) intrinsic geometry belongs beside existing Pool2d attrs;
  - operation [identity](../../../../glossary.md#term-identity-meanings) and exact [signatures](../../../../glossary.md#term-signature) belong to `Pool3dKind`; and
  - input-aware rank/type/geometry validation plus descriptor/provenance construction belongs in
    the two family-specific tensor helpers, not in attrs, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>[`Tensor`](../../../../glossary.md#tensor), or a generic utility.

## Exact public surface

`MaxPool3dAttrs` and `AveragePool3dAttrs` are public records implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with
exactly these components in declaration and [validation](../../../../glossary.md#term-validation) order:

```text
long kernelDepth
long kernelHeight
long kernelWidth
long strideDepth
long strideHeight
long strideWidth
long paddingDepth
long paddingHeight
long paddingWidth
long dilationDepth
long dilationHeight
long dilationWidth
boolean ceilMode
```

[Kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), and [dilation](../../../../glossary.md#term-dilation) components must be positive. [Padding](../../../../glossary.md#padding) components must be non-negative.
[Canonical](../../../../glossary.md#term-canonical-form) constructors validate in exact declaration order and use the established messages
`<name> must be positive: <value>` and `<name> must be non-negative: <value>`. The records expose
no defaults, builder, nested type, array geometry, [normalization](../../../../glossary.md#term-normalization), divisor option, or [mutable](../../../../glossary.md#term-mutable) state.

`Pool3dKind.values()` is exactly `[MAX_POOL3D, AVERAGE_POOL3D]`. Its two [signatures](../../../../glossary.md#term-signature) are respectively
`OperationSignature.fixed(MaxPool3dAttrs.class, 1, 1)` and
`OperationSignature.fixed(AveragePool3dAttrs.class, 1, 1)`. Cross-pairing either kind with the
other attrs class fails through the existing exact-class signature contract.

The only new [Tensor](../../../../glossary.md#tensor) declarations are:

```java
public Tensor maxPool3d(MaxPool3dAttrs attrs)
public Tensor averagePool3d(AveragePool3dAttrs attrs)
```

There is no [alias](../../../../glossary.md#aliasing-and-arena) such as `avgPool3d`, no overload, no defaults receiver, and no PoolNd spelling.
These methods increase the exact public `Tensor` declared-method count from 208 to 210 and add only
the names `maxPool3d` and `averagePool3d` to the locked public-name set.

## Output geometry

The input [Shape](../../../../glossary.md#shape) is `[N, C, D, H, W]`; the output Shape is
`[N, C, D_out, H_out, W_out]`. Output [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and [channel](../../../../glossary.md#term-channel) positions retain the exact input [Dimension](../../../../glossary.md#dimension)
references. For each [spatial](../../../../glossary.md#term-spatial-axes) input [extent](../../../../glossary.md#term-extent) `X` and its matching [kernel](../../../../glossary.md#kernel) count `k`, symmetric [padding](../../../../glossary.md#padding)
per side `p`, [dilation](../../../../glossary.md#term-dilation) `d`, and [stride](../../../../glossary.md#term-stride) `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = X + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

`ceilMode == true` selects the literal ceiling formula for depth, height, and width.
It does not decrement a terminal output whose window begins wholly in trailing padding; an
all-padding terminal window remains part of the output grid.

All literal geometry [constants](../../../../glossary.md#term-constant) and static calculations use checked signed-`long` arithmetic.
For a static input extent, a negative numerator fails before [producer](../../../../glossary.md#term-producer) construction. Zero spatial
input is valid when symmetric padding makes the numerator non-negative; zero batch or channel is
always valid and contains no values. For an unresolved spatial Dimension, construction retains
the [canonical](../../../../glossary.md#term-canonical-form) `addConstant`, matching floor-or-ceiling divide, then `addConstant(1)` expression.
The exact obligation that a later concrete [binding](../../../../glossary.md#term-binding) make each numerator non-negative remains
visible for [Compiler](../../../../glossary.md#term-compiler) or binding [validation](../../../../glossary.md#term-validation). Spatial derivation and failure order are depth, then
height, then width.

## Maximum numerical semantics

- Each output window enumerates logical kernel coordinates in increasing depth, then height, then
  width order. The corresponding input coordinate is<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
  `output * stride - padding + kernelCoordinate * dilation` on each [axis](../../../../glossary.md#term-axis).
- Out-of-bounds padding positions are excluded from selection and are never eligible winners. A
  window with no in-bounds sampled coordinate has exact negative infinity in the input/result
  floating type.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Any in-bounds NaN dominates every non-NaN. Among multiple NaNs, the first eligible logical
  depth-height-width occurrence wins. NaN payload, sign, quiet/signaling form, and preservation are
  unspecified.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [occurrence glossary](../../../../glossary.md#term-occurrence); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Otherwise ordinary numerical maximum applies: positive zero ranks above negative zero,
  infinities use ordinary order, and numerically equal eligible candidates retain the first
  depth-height-width occurrence.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Maximum pooling retains the selected represented non-NaN value in the exact input type; it has
  no average accumulator, mixed-type promotion, saved-index output, hidden winner buffer, or
  algorithm/traversal promise beyond the semantic first-winner order.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [accumulator glossary](../../../../glossary.md#accumulator-type); [buffer glossary](../../../../glossary.md#term-buffer))</span>

## Average numerical semantics

- Every logical [kernel](../../../../glossary.md#kernel) position counts in the fixed positive divisor, whose mathematical value is
  `kernelDepth * kernelHeight * kernelWidth`. Dilation changes sampled coordinates, not the
  divisor. There is no valid-sample count, divisor override, count-padding option, or zero-divisor
  case.<span data-glossary-navigation="true"> ([Dilation glossary](../../../../glossary.md#term-dilation); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [padding glossary](../../../../glossary.md#padding))</span>
- Each in-bounds position contributes its input value once. Each out-of-bounds position
  contributes conceptual exact positive zero and still counts once in the divisor.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- BFLOAT16 and FLOAT32 accumulate and divide in FLOAT32. FLOAT64 accumulates and divides in
  FLOAT64. The sum is divided once by the fixed divisor, and BFLOAT16 narrows only once after that
  division. Construction must not materialize the three-factor divisor in <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`long`; its mathematical
  product may exceed `Long.MAX_VALUE` while every per-[axis](../../../../glossary.md#term-axis) geometry calculation remains valid.
- Any in-bounds NaN produces NaN. Positive and negative infinity together produce NaN; otherwise a
  present infinity retains its sign. Padding introduces no NaN or infinity.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- An exact-zero finite mean is negative zero only when every divisor position is an in-bounds
  negative zero. Cancellation, any positive zero, or any padding contribution produces positive
  zero. An all-padding window therefore has exact positive zero.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [mean glossary](../../../../glossary.md#term-mean); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Finite accumulation may be reassociated, so no fixed traversal, bitwise result, NaN payload/sign
  preservation, or cross-backend rounding identity is promised. This permission does not weaken
  the specified accumulator type, one final division, BFLOAT16 narrowing, or exceptional-value
  classes.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend); [rounding glossary](../../../../glossary.md#term-rounding); [identity glossary](../../../../glossary.md#term-identity-meanings); [accumulator type glossary](../../../../glossary.md#accumulator-type); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>

## Construction and failure contract

Each family helper validates and constructs in this order:

1. require non-null `input`, then non-null `attrs`, with messages `input` and `attrs`;
2. require the input type to be [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64) and retain it exactly;
3. require [rank](../../../../glossary.md#term-rank) five;
4. derive and validate depth, then height, then width geometry using checked arithmetic;
5. create the exact result [descriptor](../../../../glossary.md#term-descriptor) with unresolved [layout](../../../../glossary.md#layout) and unchanged `requiresGrad`;
6. create the paired [`Operation`](../../../../glossary.md#operation) with the exact attrs reference; and
7. call the existing one-output derived factory seam with ordered inputs `[input]`.

The established family messages are `maxPool3d ...` and `averagePool3d ...`. Geometry failures use
`<family> effective kernel does not fit padded <axis>: input=<dimension>,`
`effectiveKernel=<value>, padding=<value>`.
Every locally decidable null, type, rank, geometry, arithmetic, descriptor, operation/[signature](../../../../glossary.md#term-signature), or
input-list failure occurs before [Tensor](../../../../glossary.md#tensor) identifier allocation. A successful invocation consumes
exactly one fresh ID. Identifier exhaustion remains the existing factory failure and is not
reinterpreted.

Each result is a fresh, unlabeled, storage-free [canonical](../../../../glossary.md#term-canonical-form) Tensor with unresolved layout. Its
[provenance](../../../../glossary.md#provenance) has a fresh [producer](../../../../glossary.md#term-producer), the paired [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) kind, the exact supplied attrs reference,
ordered exact inputs `[input]`, output index zero, and the exact output descriptor; the returned
Tensor is the exact object from producer `output(0)`. Repeated calls with equal inputs and attrs
create [identity](../../../../glossary.md#term-identity-meanings)-distinct Operations, producers, descriptors, IDs, and canonical outputs.

## Affected files

Production/[Javadoc](../../../../glossary.md#term-javadoc), exactly six paths:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/MaxPool3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/AveragePool3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool3dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool3dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool3dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Focused tests, exactly three new paths:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/pooling/Pool3dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool3dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool3dExpressionTest.java`

Existing exact [Tensor](../../../../glossary.md#tensor) public-method-count [owners](../../../../glossary.md#term-owner), exactly seventeen paths, change only from 208 to
210 plus the central `TensorTest` name-set additions:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/RecurrentScanExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv1dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv3dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPool1dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Documentation/[planning](../../../../glossary.md#term-planning), exactly seven paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only with an explicit no-change conclusion unless a contradiction requires stopping:
`ARCHITECTURE.md`, current architecture plan, ADRs, Training [API](../../../../glossary.md#term-application-programming-interface-api), Pool1d/Pool2d/[Conv3d](../../../../glossary.md#conv3d--ncdhw) contracts,
[Compiler](../../../../glossary.md#term-compiler)/[CPU](../../../../glossary.md#term-cpu) master plans and production/tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration, architecture
tests, [Gradle](../../../../glossary.md#term-gradle)/build files, and every other [module](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

- Hard ceiling: 33 paths total — six production/Javadoc, twenty test, and seven
  documentation/planning paths listed above.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- The seventeen existing count-[owner](../../../../glossary.md#term-owner) test files receive only the mechanical `208 -> 210` update,
  except `TensorTest` also adds the two exact method names. Any other change in those owners is out
  of scope.
- No compatibility alias, default overload, shared geometry abstraction, unrelated refactor, or
  downstream implementation is permitted within the ceiling.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Stop and request clarification before exceeding 33 paths or changing architecture, Shape/
  Dimension contracts, factory seams, operation-signature infrastructure, dependencies, build
  configuration, Compiler/CPU code, or tests outside the exact list.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Acceptance criteria

- The two attrs records have the exact thirteen-component surface, declaration-order validation,
  equality/hash behavior, and family-specific Javadocs specified above.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [hash glossary](../../../../glossary.md#term-hash))</span>
- `Pool3dKind` has exactly two constants in the specified order, stable immutable exact signatures,
  accepted own-attrs Operations, and rejected cross-family attrs.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [immutable glossary](../../../../glossary.md#term-immutable); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Tensor exposes exactly the two new receiver methods, increasing its public method count from 208
  to 210 without another public name or overload.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Both helpers accept every current floating type, reject integral/BOOL and wrong-rank input, and
  derive exact NCDHW static floor, literal-ceil, terminal all-padding, and canonical symbolic
  geometry while retaining exact N/C references.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [rank glossary](../../../../glossary.md#term-rank); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [padding glossary](../../../../glossary.md#padding); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Static and symbolic geometry uses checked arithmetic, validates depth/height/width in order,
  accepts empty axes where the formula is valid, and never materializes the average divisor.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>
- Max semantics are documented exactly as excluded padding, all-padding negative infinity, NaN
  dominance, positive-over-negative zero, ordinary infinities, and first eligible
  depth-height-width winner.<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Average semantics are documented exactly as fixed three-factor count-padding, specified
  accumulator/division/narrowing domains, permitted reassociation, and the Pool2d-compatible NaN,
  infinity, signed-zero, and all-padding policies.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [narrowing glossary](../../../../glossary.md#term-narrowing); [reassociation glossary](../../../../glossary.md#term-reassociation); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Null/type/rank/geometry/arithmetic failures precede factory allocation; every success creates
  exactly one fresh canonical output and one exact one-input Pool3d producer occurrence.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Javadocs cover every public type/method, every parameter, result, constraint, nullability,
  failure, metadata/provenance effect, and current-versus-planned boundary without restating code.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tensor API, Compile API, glossary, Model capabilities, task, master plan, and roadmap agree that
  Model construction is current while Compiler 0006B1, Model 0025K, Compiler 0006B2, and CPU
  0008G1 remain Draft downstream owners. Training API and architecture contracts remain accurate
  without edits.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- 0025J is the sole detailed Ready Model task before implementation. 0025K and 0026 remain Draft
  without task files; Compiler 0006B1/0006B2 and CPU 0008G1 remain Draft without task files.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation tests:

```text
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.pooling.Pool3dSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMaxPool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorAveragePool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Required once after executable Java stabilizes:

```text
./gradlew :modules:model:test
```

Required in the final clean documentation pass, without repeating the successful Java suite when
only documentation/Javadocs change:

```text
./gradlew :modules:model:javadoc
git diff --check
```

Also record:

- reflection and `javap` inspection of the exact records, kind/signatures, Tensor declarations,
  helper visibility/field-free shape, and public-method count 210;<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [shape glossary](../../../../glossary.md#shape))</span>
- exact static/symbolic depth-height-width formulas, terminal all-padding grids, checked overflow,
  unmaterialized three-factor average divisor, failure ordering, ID deltas, producer freshness,
  exact attrs/input/output-index provenance, and canonical <span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [overflow glossary](../../../../glossary.md#term-overflow); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`producer.output(0)` [identity](../../../../glossary.md#term-identity-meanings);
- source/Javadoc inspection for max and average exceptional-value/rounding contracts, because
  Model construction evaluates no tensor values;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [rounding glossary](../../../../glossary.md#term-rounding); [Model glossary](../../../../glossary.md#term-model))</span>
- Markdown headings, links, anchors, fences, and status/order/[dependency](../../../../glossary.md#term-dependency) checks;
- scans proving 0025J is the sole detailed Ready Model task and that no task files exist for Model
  0025K/0026, Compiler 0006B1/0006B2, or CPU 0008G1;<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- exact 33-path ceiling, exact package placement, no forbidden downstream/build/architecture
  changes, and whitespace validation.

Repository-wide validation is deferred to the next recorded capability [checkpoint](../../../../glossary.md#term-checkpoint) and [CI](../../../../glossary.md#term-continuous-integration-ci). No
architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, [Compiler](../../../../glossary.md#term-compiler), [CPU](../../../../glossary.md#term-cpu), or [Gradle](../../../../glossary.md#term-gradle) suite is required because
this task changes one [module](../../../../glossary.md#term-module) without changing a shared boundary or executable backend behavior.

## Dependencies

- Complete Model 0020A and 0020A1 define the exact maximum and fixed-count average Pool2d
  numerical and geometry policies extended by one spatial axis.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [spatial axis glossary](../../../../glossary.md#spatial-axes-channels-and-window-sampling))</span>
- Complete Model 0025H supplies the first-class NCDHW rank-specific attrs, helper, symbolic
  geometry, validation-order, and canonical-provenance precedent.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Complete Model 0025I closes the rank-specific pooling naming and the reason Pool3d, unlike
  Pool1d, must be first-class.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling))</span>
- Complete Model 0018K–0018N and 0018V supply current immutable operation signatures, floating
  type boundaries, Tensor expression ownership, and capability-reset constraints.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Current [canonical](../../../../glossary.md#term-canonical-form) [TensorProducer](../../../../glossary.md#producer--tensorproducer)/factory and [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) expression contracts are stable.

## Follow-up tasks

- [Model](../../../../glossary.md#term-model) 0025K remains Draft and will add the smallest public [NCDHW](../../../../glossary.md#conv3d--ncdhw) `unfold3d`/`fold3d` algebra
  needed to express exact Pool3d adjoints. It must not add pooling-specific gradient primitives.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Compiler 0006B1 remains Draft and will independently infer/final-validate both Pool3d forward
  occurrences as ordinary flat nodes while rejecting any backward-capable request containing one
  before derivative allocation.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [nodes glossary](../../../../glossary.md#node); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Compiler 0006B2 remains Draft after 0025K/0006B1 and will close exact gradients: average divides
  by logical `kernelDepth * kernelHeight * kernelWidth`; maximum reconstructs the first eligible
  depth-height-width winner with exact NaN and signed-zero occurrence matching.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- CPU 0008G1 remains Draft after the Model and Compiler chain. Its direct NCDHW Pool3d generated
  scalar/parallel-scalar code must support only proved static resolved current-floating cases,
  report truthful kind capability, allocate zero workspace/materialization, and match a specialized
  optimal clean Java oracle in semantic algorithm, hot-loop/dataflow shape, avoidable-overhead
  profile, Class-File/decompilation/forbidden-structure checks, and five-fork generated/direct<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [materialization glossary](../../../../glossary.md#materialization); [oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [decompilation glossary](../../../../glossary.md#term-decompilation); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
  `<= 1.15x` evidence.
- [CPU](../../../../glossary.md#term-cpu) 0008H remains after that complete inserted [pooling](../../../../glossary.md#term-pooling) chain.

## Architecture impact

None. The task adds one [rank](../../../../glossary.md#term-rank)-specific [operation](../../../../glossary.md#operation) family and two [Tensor](../../../../glossary.md#tensor) expression constructors
inside existing [Model](../../../../glossary.md#term-model) [ownership](../../../../glossary.md#term-ownership-meanings). `ARCHITECTURE.md`, the current architecture plan, ADRs,
[dependency](../../../../glossary.md#term-dependency) rules, [module](../../../../glossary.md#term-module) [descriptors](../../../../glossary.md#term-descriptor), and architecture tests remain unchanged.

## Implementation prompt

### Clean implementation task prompt

Implement [Model](../../../../glossary.md#term-model) task 0025J exactly within this specification in a separate clean implementation
context. Read root `AGENTS.md`, architecture contracts, [planning](../../../../glossary.md#term-planning) guide/roadmap, this task, Model
master/capabilities, completed 0020A/0020A1/0025H/0025I, and directly relevant Pool2d/[Conv3d](../../../../glossary.md#conv3d--ncdhw)
source and tests in full. Do not use a GSD workflow, [commit](../../../../glossary.md#term-commit), or push.

Add only the six production paths, three focused suites, and mechanical changes to the seventeen
exact public-method-count [owners](../../../../glossary.md#term-owner) listed here. Preserve exact attrs ordering, kind/[signature](../../../../glossary.md#term-signature) pairing,
[NCDHW](../../../../glossary.md#conv3d--ncdhw) geometry, max/average numerical contracts, [validation](../../../../glossary.md#term-validation) and ID ordering, fresh [canonical](../../../../glossary.md#term-canonical-form)
[provenance](../../../../glossary.md#provenance), and absence of execution/[gradient](../../../../glossary.md#term-gradient) claims. [Run](../../../../glossary.md#term-run) the focused command and one full Model
test command. Inspect reflection/`javap`, exact paths, and downstream absence. Do not edit the
seven documentation/planning paths; production Javadocs may remain provisional for the required
clean documentation review. Hand the stabilized diff and exact test evidence to that distinct
documentation-focused context. If an
architecture contradiction or need beyond the 33-path ceiling appears, stop and request
clarification. Return the required completion summary and do not mark Complete until the clean
documentation pass finishes.

### Clean documentation task prompt

In a distinct clean documentation-focused context, review the stabilized 0025J diff and recorded
implementation evidence. Read root `AGENTS.md`, documentation rules, General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/
[Planning](../../../../glossary.md#term-planning)/Example profiles, architecture contracts, this task, [Model](../../../../glossary.md#term-model) master/capabilities,
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, directly affected production Javadocs/tests, and downstream
[Compiler](../../../../glossary.md#term-compiler)/[CPU](../../../../glossary.md#term-cpu) rows in full. Do not repeat a successful Java test suite unless executable Java
changes or concrete evidence requires it; do not [commit](../../../../glossary.md#term-commit) or push.

Independently finalize the six affected production Javadocs and exactly the seven documentation/
planning paths. Clearly distinguish current Model [metadata](../../../../glossary.md#term-metadata) from Draft Compiler 0006B1, Model
0025K, Compiler 0006B2, and CPU 0008G1 behavior; record reasoned no-change conclusions for
Training API, architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), downstream production/tests,
and other [modules](../../../../glossary.md#term-module). [Run](../../../../glossary.md#term-run) Model Javadoc, Markdown links/anchors/fences, exact status/[dependency](../../../../glossary.md#term-dependency)/order,
task-file absence, exact 33-path, and `git diff --check` gates. Mark 0025J Complete only when all
criteria pass; otherwise return the exact incomplete follow-up.

## Local decisions

- Pool3d is first-class rather than a repeated Pool2d composition because the number of depth
  slices would depend on Shape. One operation occurrence is the bounded metadata representation.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [occurrence glossary](../../../../glossary.md#term-occurrence); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Max and average have separate attrs and construction owners because excluded-padding extrema and
  fixed-divisor accumulation are distinct semantic contracts; sharing coordinates does not make
  their numerical policies interchangeable.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Public geometry is rank-specific primitive fields, matching Conv3d and Pool2d precedent. There
  is no dynamic-rank array contract or generic PoolNd abstraction.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Symmetric intrinsic padding is the current contract. Asymmetric behavior remains explicit PAD
  composition or later separately justified semantics.<span data-glossary-navigation="true"> ([PAD glossary](../../../../glossary.md#padding))</span>
- Literal ceil mode is preserved without framework-style terminal-window suppression.
- Average's divisor is a positive mathematical product and is intentionally not calculated in
  Model construction, avoiding a false <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>`long` representability restriction.
- Max first-winner order extends Pool2d height-width order by making depth the outer logical kernel
  coordinate. Average finite accumulation remains reassociable and does not acquire that traversal
  promise.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [finite glossary](../../../../glossary.md#term-finite))</span>
- Separate family helpers are deliberately small and field-free; no generic geometry helper is
  introduced for only two implementations.
- Documentation uses General, API/Javadoc, Planning, and Example profiles. The glossary adds the
  reusable NCDHW Pool3d distinction; architecture documents remain unchanged because ownership and
  boundaries do not change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Known limitations

- Only [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) [NCDHW](../../../../glossary.md#conv3d--ncdhw) input with symmetric intrinsic [padding](../../../../glossary.md#padding) is included.
- Average pooling is fixed count-padding only; valid-sample division and divisor override are
  absent.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- No saved maximum indices, adaptive/global pooling, asymmetric intrinsic padding, NDHWC, PoolNd,
  or dynamic-rank geometry exists.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Model does not evaluate values. Compiler adoption, exact gradients, CPU generated execution,
  other backends, NN layers, and performance remain downstream.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient); [CPU glossary](../../../../glossary.md#term-cpu); [backends glossary](../../../../glossary.md#backend); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The stabilized implementation evidence was reused without rerunning Java tests because this
  clean documentation context changed no executable Java. The final Model XML reports 138 suites
  and 1,098 tests with zero skips, failures, or errors. The four focused owners named by this task
  account for 31 passing tests: 4 Pool3d semantics, 6 maximum-expression, 6 average-expression,
  and 15 Tensor surface tests.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Clean documentation context `/root` independently reviewed the complete 33-path diff, all six
  affected production source/Javadocs, all three new focused suites, the seventeen mechanical
  public-count owners, completed Pool2d/Pool1d/Conv3d precedents, Tensor/Compile/Training APIs,
  glossary, Model capabilities/master plan, and downstream Compiler/CPU planning boundaries. It
  changed no executable Java.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Compile glossary](../../../../glossary.md#compile); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- `./gradlew :modules:model:javadoc` passed after the Javadoc review with two tasks up to date.
  Rendered <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`MaxPool3dAttrs.html`, `AveragePool3dAttrs.html`, `Pool3dKind.html`, and both Pool3d
  sections in `Tensor.html` contain the reviewed geometry, numerical, provenance, parameter,
  result, failure, and current-versus-planned contracts. Package-private helper Javadocs were
  reviewed directly in source.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- The documented Java 26 [metadata](../../../../glossary.md#term-metadata) example compiled and ran against [Model](../../../../glossary.md#term-model) classes. It printed
  `Shape[1, 2, 3, 3, 3]` twice, `MAX_POOL3D`, `AVERAGE_POOL3D`, and `true`, confirming exact output
  Shape metadata, kinds, and one-input provenance without evaluating values.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Reflection reported exactly 210 public Tensor methods, exactly two Pool3d receivers, thirteen
  record components in each attrs type, kind order `[MAX_POOL3D, AVERAGE_POOL3D]`, and exact
  singleton one-input/one-output signature pairings. <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [signature glossary](../../../../glossary.md#term-signature))</span>`javap -public` confirmed the two receiver
  declarations and exact record/kind surfaces. `javap -private` confirmed both helpers are
  package-private final, field-free, privately constructed, and expose only package-private
  `apply` plus a private geometry method.
- A runtime inventory scan over compiled Model operation enums reported exactly 40 families, 113
  constants, and 134 signatures. Source scans found no Pool3d references in Compiler or CPU
  production/tests, architecture tests, backend conformance, or integration tests, confirming
  that current support stops at Model metadata.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan); [operation glossary](../../../../glossary.md#operation); [constants glossary](../../../../glossary.md#term-constant); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The targeted Markdown validator passed all seven documentation/planning files: 1,060 local
  links, 792 target anchors, balanced fences, final newlines, no carriage returns, and no trailing
  whitespace. The exact-scope check passed 33 authorized paths: six production/Javadoc, twenty
  Model tests, and seven documentation/planning files. The staging area is empty.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [scope glossary](../../../../glossary.md#term-scope))</span>
- Status/dependency/order checks set 0025J Complete in this task, Model master plan, and roadmap;
  preserve Model 0025K, Compiler 0006B1/0006B2, and CPU 0008G1 as Draft; preserve the ordered<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
  `0025I -> 0025J -> 0025K -> 0006B1 -> 0006B2 -> 0008G1 -> 0008H` chain; and find no detailed
  task files for Model 0025K/0026, Compiler 0006B1/0006B2, or CPU 0008G1.
- `git diff --check` passed on the final combined change.
- Training API remains unchanged because Pool3d adds no Training-owned optimizer, session,
  orchestration, gradient, or execution surface. <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient))</span>`ARCHITECTURE.md`, focused architecture pages,
  ADRs, and architecture tests remain unchanged because module ownership, dependencies, and
  lifecycle rules did not change. Backend conformance and integration tests remain unchanged
  because no backend or end-to-end behavior exists. Gradle/build files remain unchanged because
  no dependency or build contract changed. Downstream Compiler/CPU production and tests remain
  unchanged because Draft 0006B1/0006B2/0008G1 own adoption, gradients, and execution. Other
  modules remain unchanged because this task is confined to Model semantics and directly affected
  documentation/planning.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Implementation notes

- Added public [immutable](../../../../glossary.md#term-immutable) `MaxPool3dAttrs` and `AveragePool3dAttrs`, exact `Pool3dKind` semantics,
  separate field-free construction owners, and exactly two Tensor receivers. Construction retains
  exact floating type, N/C Dimensions, <span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Dimensions glossary](../../../../glossary.md#dimension))</span>`requiresGrad`, checked static or canonical-symbolic NCDHW
  geometry, and one fresh canonical output with exact provenance.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [provenance glossary](../../../../glossary.md#provenance))</span>
- Added three focused suites and mechanically changed the seventeen existing Tensor public-count
  owners from 208 to 210; only `TensorTest` additionally adds the two exact method names.
- The documentation pass finalized all six affected Javadocs and updated Tensor/Compile APIs,
  glossary, Model capabilities, task, Model master plan, and roadmap. It keeps current Model
  metadata distinct from Draft Compiler 0006B1, Model 0025K, Compiler 0006B2, and CPU 0008G1.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Completion summary

- Completed changes: Added first-class NCDHW maximum and fixed-count average Pool3d Model
  semantics, exact rank-specific geometry, numerical policies, canonical provenance, focused
  tests/public locks, finalized Javadocs, and synchronized explanatory/planning documentation.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [Model glossary](../../../../glossary.md#term-model); [rank glossary](../../../../glossary.md#term-rank); [canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Exactly 33 authorized paths: six production/Javadoc, twenty Model
  tests, and seven documentation/planning files.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Tests and validation: Reused the stabilized 138-suite/1,098-test Model result with 31 passing
  focused-owner tests; passed Model Javadoc, rendered inspection, Java 26 example,
  reflection/<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [owner glossary](../../../../glossary.md#term-owner))</span>`javap`, 40/113/134 inventory and downstream-absence scans, 1,060-link/792-anchor
  Markdown validation, exact scope/status/order/task-file/staging checks, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: Clean context `/root` independently finalized affected Javadocs,
  APIs, glossary/capability impact, planning evidence, and status without executable Java edits.
- Documentation impact: Tensor and Compile APIs, glossary, Model capabilities, task, Model master
  plan, and roadmap now describe current Model Pool3d metadata and exact downstream ownership.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: Both attrs records and constructors, both kind constants/signatures, both
  package-private construction owners, both Tensor methods, and the affected Tensor overview have
  complete contracts; no additional production Javadoc change was necessary after review.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Glossary impact: Added the reusable Pool3d/NCDHW distinction and synchronized Model-versus-
  Compiler inventory counts and downstream boundaries.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Unresolved issues: None within task 0025J.
- Follow-up required: Model 0025K remains the next Draft Model frontier; Compiler 0006B1/0006B2
  and CPU 0008G1 remain the ordered Draft consumers.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

Status: Complete
